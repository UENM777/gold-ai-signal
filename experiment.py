"""Can the models get more accurate? Quick, honest test of candidate features BEFORE changing the app.

    python experiment.py                     -> default horizon (15 min)
    python experiment.py "1 hour" "1 day"    -> these horizons

Same rules as train.py: time-ordered 80/20 split, test period never seen in training.
For each feature set it trains two quick models (LightGBM + Logistic Regression) and prints the test AUC
(0.50 = coin flip) for direction and for volatility. "min4" = worst of 4 consecutive test blocks (stability).
Results -> figures/<horizon>/experiment.csv
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from common import *

warnings.filterwarnings("ignore")

VOL_CTX = list(VOL_FEATURES)        # now part of the app's features (common.py); "before" = without them
TREND_CTX = ["ret_60", "ret_240", "don_60", "don_240", "close_sma200", "efficiency_60"]
SEASON = ["tod_act", "tod_fut"]


def add_candidates(f, h, tf):
    """Candidate features. All past-only (no look-ahead)."""
    f = f.copy()
    c, r = f["close"], f["close"].pct_change()
    # --- volatility context: the volatility label compares the future range with `typical`,
    #     but the current models are never told what `typical` is ---
    f["typical"] = f["past_range_h"].rolling(500, min_periods=100).median()
    f["range_vs_typical"] = f["past_range_h"] / f["typical"]
    f["range_4h_vs_typical"] = (f["high"].rolling(4 * h).max() - f["low"].rolling(4 * h).min()) / c / f["typical"]
    f["std_60"], f["std_240"] = r.rolling(60).std(), r.rolling(240).std()
    f["vol_trend"] = f["std_20"] / f["std_240"]
    f["atr_ratio"] = f["atr"] / atr(f, 100)
    f["recent_bar_range"] = f["range_pct"].rolling(5).mean() / f["range_pct"].rolling(500, min_periods=100).mean()
    # --- longer trend context ---
    f["ret_60"], f["ret_240"] = c.pct_change(60), c.pct_change(240)
    for w in (60, 240):
        lo, hi = f["low"].rolling(w).min(), f["high"].rolling(w).max()
        f[f"don_{w}"] = (c - lo) / (hi - lo).replace(0, np.nan)          # where is price inside its recent range
    f["close_sma200"] = c / c.rolling(200).mean() - 1
    f["efficiency_60"] = f["ret_60"].abs() / (r.abs().rolling(60).sum()).replace(0, np.nan)   # 1 = straight line
    # --- time-of-day seasonality (intraday only) ---
    if tf != "D1" and h * TF_SECONDS[tf] <= 86400:
        tod = f["time"].dt.hour * 60 + f["time"].dt.minute
        g = f.groupby(tod)
        f["tod_act"] = g["range_pct"].transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean()) \
            / f["range_pct"].rolling(500, min_periods=100).mean()
        fut = (f["high"][::-1].rolling(h).max()[::-1].shift(-1) - f["low"][::-1].rolling(h).min()[::-1].shift(-1)) / c
        idx = pd.Series(np.arange(len(f)), index=f.index)
        src = idx.groupby(tod).shift(2)                                   # newest occurrence used
        tf_fut = fut.groupby(tod).transform(lambda s: s.shift(2).rolling(10, min_periods=5).mean()) / f["typical"]
        f["tod_fut"] = tf_fut.where(src + h <= idx)                       # its future window must be over already
    else:
        f["tod_act"], f["tod_fut"] = 0.0, 0.0
    return f


def auc_blocks(y, p, k=4):
    out = []
    for yy, pp in zip(np.array_split(y, k), np.array_split(p, k)):
        if len(np.unique(yy)) == 2:
            out.append(roc_auc_score(yy, pp))
    return min(out) if out else np.nan


def evaluate(data, feats, target, h):
    d = data.dropna(subset=feats + [target])
    split = int(len(d) * 0.8)
    tr, te = d.iloc[:split - h], d.iloc[split:]                          # gap of h bars: labels don't overlap the test
    lgb = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, min_child_samples=100, random_state=42, verbose=-1)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.1))
    res = {}
    for name, m in [("lgbm", lgb), ("logreg", lr)]:
        Xtr, Xte = tr[feats].clip(-1e6, 1e6), te[feats].clip(-1e6, 1e6)
        p = m.fit(Xtr, tr[target]).predict_proba(Xte)[:, 1]
        res[name] = roc_auc_score(te[target], p)
        res[name + "_min4"] = auc_blocks(te[target].to_numpy(), p)
    return res, len(tr), len(te)


def run(df, h, tf, extras=None, label=""):
    base = [f for f in FEATURES if f not in VOL_CTX] + extra_features(list(extras or {}))
    data = add_candidates(make_labels(make_features(df, h, extras), h), h, tf)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=base + VOL_CTX + TREND_CTX + SEASON + ["y_dir", "y_vol"]) \
        .reset_index(drop=True)                                           # same rows for every feature set
    sets = {"before (no vol context)": base, "+ volatility context": base + VOL_CTX, "+ trend context": base + TREND_CTX,
            "+ time-of-day": base + SEASON, "+ all three": base + VOL_CTX + TREND_CTX + SEASON}
    rows = []
    for name, feats in sets.items():
        row = dict(horizon=label, features=name, n_features=len(feats))
        for target, key in [("y_vol", "vol"), ("y_dir", "dir")]:
            r, ntr, nte = evaluate(data, feats, target, h)
            row.update({f"{key}_auc_lgbm": r["lgbm"], f"{key}_min4_lgbm": r["lgbm_min4"], f"{key}_auc_logreg": r["logreg"]})
        row.update(train=ntr, test=nte)
        rows.append(row)
    # no-ML baseline for volatility: "it was wild recently -> it stays wild"
    te = data.iloc[int(len(data) * 0.8):]
    rows.append(dict(horizon=label, features="(no ML) recent range vs typical", n_features=1,
                     vol_auc_lgbm=roc_auc_score(te["y_vol"], te["range_vs_typical"])))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    todo = [a for a in sys.argv[1:] if not a.startswith("--")] or [DEFAULT_PROFILE]
    for p in todo:
        if p not in PROFILES:
            raise SystemExit(f"unknown profile '{p}'. choose from: {list(PROFILES)}")
        P, h, tf = paths(p), PROFILES[p]["h"], PROFILES[p]["tf"]
        if not os.path.exists(P["data"]):
            print(f"skip {p}: {P['data']} missing"); continue
        print(f"\n=== {p} ===", flush=True)
        res = run(pd.read_csv(P["data"], parse_dates=["time"]), h, tf, load_extras(tf), p)
        pd.set_option("display.width", 250)
        print(res.round(4).to_string(index=False))
        os.makedirs(P["figures"], exist_ok=True)
        res.to_csv(f"{P['figures']}/experiment.csv", index=False)
