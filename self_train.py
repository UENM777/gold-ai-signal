"""Self-training: refit the current models on the latest bars, keep the new one only if it is not worse."""
import os, json, shutil
from datetime import datetime
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from common import *

RETRAIN_EVERY_BARS = 60     # auto retrain every 60 new closed bars of the selected profile
RETRAIN_BARS = 30_000       # how much recent history to learn from (capped by what exists)
KEEP_BACKUPS = 3            # old model files kept per model in models/<profile>/backup (older ones deleted)


def valid_bars(n_rows):
    return int(min(1_440, max(100, n_rows * 0.1)))     # newest ~10% (max 1 day of M1) judges old vs new


def retrain(df, dir_model, vol_model, stats, profile, extras=None):
    """df: recent closed bars. Returns (new_dir, new_vol, new_stats, report). Runs in a background thread."""
    h = PROFILES[profile]["h"]
    feats = [c for c in getattr(dir_model, "feature_names_in_", []) if c not in ("y_dir", "y_vol")] or model_features(stats)
    data = make_labels(make_features(df, h, extras), h).dropna(subset=feats + ["y_dir", "y_vol"]).reset_index(drop=True)
    if len(data) < 1_000:
        raise ValueError(f"not enough data ({len(data)} rows)")
    nv = valid_bars(len(data))
    fit, val = data.iloc[:-nv], data.iloc[-nv:]
    out, report = {}, {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "profile": profile, "rows": len(data)}

    for key, target, old in [("dir", "y_dir", dir_model), ("vol", "y_vol", vol_model)]:
        if val[target].nunique() < 2:
            out[key] = old
            report.update({f"{key}_auc_old": float("nan"), f"{key}_auc_new": float("nan"), f"{key}_swap": False})
            continue
        new = clone(old).fit(fit[feats], fit[target])
        auc_old = roc_auc_score(val[target], old.predict_proba(val[feats])[:, 1])
        auc_new = roc_auc_score(val[target], new.predict_proba(val[feats])[:, 1])
        swap = auc_new >= auc_old
        if swap:                                   # accepted -> refit on everything incl. validation
            new = clone(old).fit(data[feats], data[target])
        out[key] = new if swap else old
        report.update({f"{key}_auc_old": round(auc_old, 4), f"{key}_auc_new": round(auc_new, 4), f"{key}_swap": swap})

    new_stats = dict(stats)
    if report["dir_swap"]:                         # signal thresholds follow the new direction model
        p = out["dir"].predict_proba(val[feats])[:, 1]
        new_stats["q_lo"] = float(np.quantile(p, SIGNAL_TOP_PCT / 2))
        new_stats["q_hi"] = float(np.quantile(p, 1 - SIGNAL_TOP_PCT / 2))
    new_stats["last_retrain"] = report
    return out["dir"], out["vol"], new_stats, report


def save(dir_model, vol_model, stats, report):
    """Back up the previous models, then write the accepted ones where app.py / train.py expect them."""
    P = paths(report["profile"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(f"{P['dir']}/backup", exist_ok=True)
    for path, model, key in [(P["dir_model"], dir_model, "dir"), (P["vol_model"], vol_model, "vol")]:
        if report[f"{key}_swap"]:
            if os.path.exists(path + ".pkl"):
                shutil.copy(path + ".pkl", f"{P['dir']}/backup/{os.path.basename(path)}_{stamp}.pkl")
            joblib.dump(model, path + ".pkl")
            olds = sorted(f for f in os.listdir(f"{P['dir']}/backup") if f.startswith(os.path.basename(path) + "_"))
            for f in olds[:-KEEP_BACKUPS]:
                os.remove(f"{P['dir']}/backup/{f}")
    json.dump(stats, open(P["stats"], "w"), indent=2)
    hist = f"{P['dir']}/retrain_history.csv"
    new = not os.path.exists(hist)
    with open(hist, "a") as fh:
        if new:
            fh.write(",".join(report) + "\n")
        fh.write(",".join(str(v) for v in report.values()) + "\n")
