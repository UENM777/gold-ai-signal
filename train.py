# %% [markdown]
# # Gold AI Signal — DES423 Section 1 · Group GIVE ME A!!
# Model A (direction): will XAUUSD be higher after the chosen horizon?   -> BUY / SELL + probability
# Model B (volatility): will it move more than usual over that horizon?  -> sets TP/SL distance
# Backtest: how often does TP get hit before SL on unseen data?          -> "hit %" shown in the app
#
#   python train.py                    -> default profile (15 min)
#   python train.py "1 hour" "1 day"   -> these profiles
#   python train.py --all              -> every profile (slow: ~10-30 min each)
#   add --no-extras                    -> gold only (compare with/without EURUSD, USDJPY, silver)
#   add --jobs=4                       -> use 4 CPU cores (when running 2 trainings at once)
#   add --resume                       -> skip profiles already trained (continue after a crash)
#   add --fast                         -> ~3x faster (3 folds, 8 tuning rounds, 7 models)

# %%
import os, sys, json, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # save figures without popping up windows
import matplotlib.pyplot as plt
from pycaret.classification import ClassificationExperiment
from common import *

POINT = 0.01   # XAUUSD point size on most brokers (spread column is in points)
NO_EXTRAS = "--no-extras" in sys.argv   # train without related markets (for a before/after comparison)
FAST = "--fast" in sys.argv          # fewer folds/tuning rounds, skip the 2 weakest+slowest models
MODELS = ["lr", "nb", "rf", "et", "ada", "lightgbm", "xgboost"] + ([] if FAST else ["knn", "dt"])
FOLDS, N_ITER = (3, 8) if FAST else (5, 20)
JOBS = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--jobs=")), -1)   # CPU cores per run (-1 = all)
RESUME = "--resume" in sys.argv       # skip profiles that already have saved models


def load(profile):
    P = paths(profile)
    df = pd.read_csv(P["data"], parse_dates=["time"])
    if os.path.exists(P["live_log"]):
        cols = ["time", "open", "high", "low", "close", "tick_volume", "spread"]
        with open(P["live_log"], newline="") as fh:              # rows may have different lengths across app versions
            rows = [r[:7] for r in csv.reader(fh) if len(r) >= 7 and r[0] != "time"]
        live = pd.DataFrame(rows, columns=cols)
        live["time"] = pd.to_datetime(live["time"], errors="coerce")
        live[cols[1:]] = live[cols[1:]].apply(pd.to_numeric, errors="coerce")
        live = live.dropna()
        df = pd.concat([df, live]).drop_duplicates("time").sort_values("time")
        print(f"+ {len(live)} live bars merged")
    return df.reset_index(drop=True)


def eda(df, profile, fig_dir):
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    tf = PROFILES[profile]["tf"]
    ax[0, 0].plot(df.time, df.close, lw=.7); ax[0, 0].set_title(f"XAUUSD close ({tf})")
    df.close.pct_change().hist(bins=200, ax=ax[0, 1]); ax[0, 1].set_title(f"{tf} returns (fat tails)")
    q = df.close.pct_change().abs().quantile(.999); ax[0, 1].set_xlim(-q, q)
    df.assign(r=df.close.pct_change().abs()).groupby(df.time.dt.hour).r.mean().plot.bar(ax=ax[1, 0])
    ax[1, 0].set_title("Avg absolute move by hour (server time)")
    r = df.close.pct_change()
    pd.Series([r.abs().autocorr(k) for k in range(1, 50)], index=range(1, 50)).plot(ax=ax[1, 1], marker=".")
    pd.Series([r.autocorr(k) for k in range(1, 50)], index=range(1, 50)).plot(ax=ax[1, 1], marker=".")
    ax[1, 1].legend(["|return| autocorr (volatility clusters)", "return autocorr (~0 = hard to predict)"])
    ax[1, 1].set_title("Why direction is hard but volatility is predictable")
    plt.tight_layout(); plt.savefig(f"{fig_dir}/eda.png", dpi=130); plt.close()


def run(train_df, test_df, target, name, optimize, fig_dir, feats):
    print(f"[{name}] comparing models ...", flush=True)
    exp = ClassificationExperiment()
    exp.setup(data=train_df[feats + [target]], test_data=test_df[feats + [target]], target=target,
              data_split_shuffle=False, data_split_stratify=False,
              fold_strategy="timeseries", fold=FOLDS, n_jobs=JOBS, normalize=True, session_id=42, verbose=False)
    best = exp.compare_models(include=MODELS, sort="AUC", errors="ignore")
    table = exp.pull(); table.to_csv(f"{fig_dir}/compare_{name}.csv"); print(table)
    tuned = exp.tune_model(best, optimize=optimize, n_iter=N_ITER, choose_better=True)
    print(exp.pull())
    base_name = type(tuned).__name__
    # calibrate so "70%" really means right ~70% of the time (raw XGB/Ada probabilities are not)
    # single-core here: nested parallel jobs (CV x calibration) can freeze forever on Windows
    exp.set_config("n_jobs_param", 1)
    print(f"[{name}] calibrating {base_name} ...", flush=True)
    model = exp.calibrate_model(tuned, method="sigmoid")
    exp.set_config("n_jobs_param", JOBS)
    print("calibrated:"); print(exp.pull())
    exp.predict_model(model)                      # holdout (test) metrics
    hold = exp.pull(); hold.to_csv(f"{fig_dir}/holdout_{name}.csv"); print(hold)
    p_test = model.predict_proba(test_df[feats])[:, 1]
    for plot in ["confusion_matrix", "auc"]:
        try:
            exp.plot_model(model, plot=plot, save=fig_dir)
        except Exception as e:
            print("plot skipped:", plot, e)
    try:
        exp.plot_model(tuned, plot="feature", save=fig_dir)
    except Exception as e:
        print("plot skipped: feature", e)
    return exp, model, p_test, hold, base_name


def backtest(t, p_up, p_vol, q_lo, q_hi, h, strong_only=True):
    hold_bars = max_hold(h)
    hi, lo, cl, sp, at = t.high.values, t.low.values, t.close.values, t.spread.values * POINT, t.atr.values
    rows = []
    for i in range(len(t) - hold_bars):
        side, strong = decide(p_up[i], q_lo, q_hi)
        if strong_only and not strong:
            continue
        conf = p_up[i] if side == "BUY" else 1 - p_up[i]
        regime = "HIGH" if p_vol[i] >= 0.5 else "LOW"
        entry = cl[i] + (sp[i] if side == "BUY" else 0)
        tp, sl = tp_sl(entry, at[i], side, regime, h)
        outcome, exit_p = "TIMEOUT", cl[i + hold_bars]
        for j in range(i + 1, i + 1 + hold_bars):
            hit_sl = lo[j] <= sl if side == "BUY" else hi[j] >= sl
            hit_tp = hi[j] >= tp if side == "BUY" else lo[j] <= tp
            if hit_sl:                     # both in same bar -> assume SL (conservative)
                outcome, exit_p = "SL", sl; break
            if hit_tp:
                outcome, exit_p = "TP", tp; break
        pnl = (exit_p - entry) if side == "BUY" else (entry - exit_p - sp[i])
        rows.append(dict(time=t.time.values[i], side=side, regime=regime, conf=conf, outcome=outcome, pnl=pnl))
    return pd.DataFrame(rows, columns=["time", "side", "regime", "conf", "outcome", "pnl"])


def train_profile(profile):
    print(f"\n==================== {profile} ====================")
    P, h = paths(profile), PROFILES[profile]["h"]
    os.makedirs(P["dir"], exist_ok=True); os.makedirs(P["figures"], exist_ok=True)

    df = load(profile)
    print(df.shape, df.time.min(), "->", df.time.max())
    extras = {} if NO_EXTRAS else load_extras(PROFILES[profile]["tf"])
    feats = FEATURES + extra_features(list(extras))
    print("related markets:", list(extras) or "none", f"| {len(feats)} features")
    eda(df, profile, P["figures"])

    data = make_labels(make_features(df, h, extras), h).dropna(subset=feats + ["y_dir", "y_vol"]).reset_index(drop=True)
    print(len(data), "rows | y_dir balance:", data.y_dir.mean().round(3), "| y_vol balance:", data.y_vol.mean().round(3))
    plt.figure(figsize=(12, 10)); cols = feats + ["y_dir", "y_vol"]
    plt.imshow(data[cols].corr(), cmap="RdBu_r", vmin=-1, vmax=1)
    plt.xticks(range(len(cols)), cols, rotation=90); plt.yticks(range(len(cols)), cols); plt.colorbar()
    plt.title("Feature correlation"); plt.tight_layout(); plt.savefig(f"{P['figures']}/corr.png", dpi=130); plt.close()

    # time-ordered split (NO shuffling -> no peeking at the future)
    split = int(len(data) * 0.8)
    train_df, test_df = data.iloc[:split], data.iloc[split:]     # keep original index (PyCaret needs unique ids)
    print("train:", train_df.time.min(), "->", train_df.time.max())
    print("test :", test_df.time.min(), "->", test_df.time.max())

    exp_dir, dir_model, p_up, hold_dir, dir_name = run(train_df, test_df, "y_dir", "dir", "MCC", P["figures"], feats)  # MCC punishes "always BUY"
    exp_vol, vol_model, p_vol, hold_vol, vol_name = run(train_df, test_df, "y_vol", "vol", "AUC", P["figures"], feats)

    # signals = 10% highest / lowest P(up) that also clear the 52% / 48% minimum edge
    q_lo = float(np.quantile(p_up, SIGNAL_TOP_PCT / 2))
    q_hi = float(np.quantile(p_up, 1 - SIGNAL_TOP_PCT / 2))
    print(f"P(up) on test: min {p_up.min():.3f}  max {p_up.max():.3f}  | BUY >= {max(q_hi, .5 + MIN_EDGE):.3f}  SELL <= {min(q_lo, .5 - MIN_EDGE):.3f}")
    bt_df = test_df.reset_index(drop=True)
    bt = backtest(bt_df, p_up, p_vol, q_lo, q_hi, h)
    closed = bt[bt.outcome != "TIMEOUT"]
    tp_rate = (closed.outcome == "TP").mean() if len(closed) else float("nan")
    print(f"STRONG signals: {len(bt)}  TP hit rate: {tp_rate:.1%}  total pnl: {bt.pnl.sum():.1f} $/oz")
    bt_all = backtest(bt_df, p_up, p_vol, q_lo, q_hi, h, strong_only=False)      # every bar (weak signals too)
    c_all = bt_all[bt_all.outcome != "TIMEOUT"]
    print(f"ALL signals:    {len(bt_all)}  TP hit rate: {(c_all.outcome == 'TP').mean():.1%}  total pnl: {bt_all.pnl.sum():.1f} $/oz")

    rng = np.random.default_rng(0)
    rand_p = rng.random(len(test_df))
    bt_rand = backtest(bt_df, rand_p, p_vol, SIGNAL_TOP_PCT / 2, 1 - SIGNAL_TOP_PCT / 2, h)
    plt.figure(figsize=(11, 4))
    plt.plot(bt.time, bt.pnl.cumsum(), label=f"AI signals ({len(bt)})")
    plt.plot(bt_rand.time, bt_rand.pnl.cumsum(), label=f"Random direction, same TP/SL ({len(bt_rand)})", alpha=.7)
    plt.title(f"Backtest {profile} on unseen test period ($ per 1 oz)"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{P['figures']}/backtest.png", dpi=130); plt.close()

    stats = {"profile": profile, "features": feats, "extras": list(extras), "tp_rate": {}, "tp_rate_all": {}, "q_lo": q_lo, "q_hi": q_hi, "overall_tp_rate": float(tp_rate),
             "n_signals": int(len(bt)), "pnl": float(bt.pnl.sum()), "random_pnl": float(bt_rand.pnl.sum()),
             "test_from": str(test_df.time.min()), "test_to": str(test_df.time.max()),
             "dir_model": dir_name, "vol_model": vol_name,
             "dir_auc": float(hold_dir["AUC"].iloc[0]), "dir_acc": float(hold_dir["Accuracy"].iloc[0]),
             "vol_auc": float(hold_vol["AUC"].iloc[0]), "vol_acc": float(hold_vol["Accuracy"].iloc[0])}
    for (s, r), g in bt.groupby(["side", "regime"]):
        stats["tp_rate"][f"{s}_{r}"] = float((g.outcome == "TP").mean())
    for (s, r), g in bt_all.groupby(["side", "regime"]):
        stats["tp_rate_all"][f"{s}_{r}"] = float((g.outcome == "TP").mean())
    print(json.dumps(stats, indent=2))

    # finalize (refit on ALL data) and save for the app
    exp_dir.save_model(exp_dir.finalize_model(dir_model), P["dir_model"], verbose=False)
    exp_vol.save_model(exp_vol.finalize_model(vol_model), P["vol_model"], verbose=False)
    save_json(stats, P["stats"])     # written last: models + stats always match
    print("saved models ->", P["dir"])

    # one row per profile -> slide table "how predictable is each horizon?"
    stats["extras_used"] = "+".join(stats["extras"]) or "none"
    row = pd.DataFrame([{k: stats[k] for k in ["profile", "extras_used", "dir_model", "dir_auc", "dir_acc", "vol_model", "vol_auc",
                                               "vol_acc", "n_signals", "overall_tp_rate", "pnl", "random_pnl"]}])
    os.makedirs("figures/summary_rows", exist_ok=True)               # one file per run -> safe when runs overlap
    row.to_csv(f"figures/summary_rows/{slug(profile)}__{stats['extras_used']}.csv", index=False)
    build_summary()


def build_summary():
    import glob
    parts = [pd.read_csv(f) for f in sorted(glob.glob("figures/summary_rows/*.csv"))]
    if parts:
        order = {p: i for i, p in enumerate(PROFILES)}
        out = pd.concat(parts).sort_values("profile", key=lambda c: c.map(order))
        out.to_csv("figures/summary.csv", index=False)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--") or a == "--all"]
    todo = list(PROFILES) if "--all" in args else (args or [DEFAULT_PROFILE])
    for p in todo:
        if p not in PROFILES:
            raise SystemExit(f"unknown profile '{p}'. choose from: {list(PROFILES)}")
    for p in todo:
        if not os.path.exists(paths(p)["data"]):
            print(f"skip {p}: {paths(p)['data']} missing -> run export_data.py"); continue
        if RESUME and trained(p):
            print(f"skip {p}: already trained (--resume)"); continue
        train_profile(p)
    if os.path.exists("figures/summary.csv"):
        print("\n", pd.read_csv("figures/summary.csv").to_string(index=False))
