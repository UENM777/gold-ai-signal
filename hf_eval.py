"""Ready-made vs trained: Chronos-Bolt (Hugging Face, zero-shot) vs our PyCaret direction model.
Same unseen test period as train.py, same question: "will gold be higher after the horizon?"

    python hf_eval.py                   -> default horizon (15 min)
    python hf_eval.py "1 hour" "1 day"  -> these horizons
    add --n=2000                        -> number of test points to score (default 1500; more = slower)

Writes figures/<horizon>/chronos_vs_pycaret.csv and figures/hf_comparison.csv (one row per horizon).
"""
import os, sys, glob
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, accuracy_score
from pycaret.classification import load_model
from common import *
import hf_models

N = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--n=")), 1500)
BATCH = 64


def evaluate(profile):
    P, h, tf = paths(profile), PROFILES[profile]["h"], PROFILES[profile]["tf"]
    if not trained(profile):
        print(f"skip {profile}: not trained"); return
    stats = load_json(P["stats"])
    dir_model = load_model(P["dir_model"], verbose=False)
    feats = [c for c in getattr(dir_model, "feature_names_in_", []) if c not in ("y_dir", "y_vol")] or model_features(stats)
    extras = {k: v for k, v in load_extras(tf).items() if any(f.startswith(k + "_") for f in feats)}

    df = pd.read_csv(P["data"], parse_dates=["time"])
    data = make_labels(make_features(df, h, extras), h).dropna(subset=feats + ["y_dir"]).reset_index(drop=True)
    split = int(len(data) * 0.8)                                  # same split as train.py
    test_idx = np.linspace(split, len(data) - 1, min(N, len(data) - split)).astype(int)
    closes = data["close"].to_numpy()
    if split < hf_models.CONTEXT_BARS:
        print(f"note {profile}: only {split} bars of context available for Chronos")

    print(f"\n=== {profile}: scoring {len(test_idx)} test points ({data.time[split]} -> {data.time.iloc[-1]}) ===")
    p_chr = []
    for b in range(0, len(test_idx), BATCH):
        idx = test_idx[b:b + BATCH]
        ctx = [closes[max(0, i - hf_models.CONTEXT_BARS + 1):i + 1] for i in idx]   # past closes only, up to bar i
        p_chr.extend(hf_models.forecast_many(ctx, h)[1])
        print(f"  Chronos {min(b + BATCH, len(test_idx))}/{len(test_idx)}", end="\r", flush=True)
    p_chr = np.array(p_chr)
    # The saved model was finalized on ALL data (incl. this test period) -> refit a copy on the train part only,
    # so PyCaret is judged on data it has never seen, exactly like Chronos.
    print(f"\n  refitting {stats.get('dir_model', 'PyCaret model')} on the train period only (no peeking) ...", flush=True)
    fair = clone(dir_model).fit(data.loc[:split - 1, feats], data.loc[:split - 1, "y_dir"])
    p_pyc = fair.predict_proba(data.loc[test_idx, feats])[:, 1]
    y = data.loc[test_idx, "y_dir"].to_numpy()

    rows = []
    for name, p in [("Chronos-Bolt (HF, ready-made)", p_chr), (f"PyCaret {stats.get('dir_model', '')} (trained, out-of-sample)", p_pyc),
                    ("Average of both", (p_chr + p_pyc) / 2)]:
        rows.append(dict(profile=profile, model=name, n=len(y), accuracy=accuracy_score(y, p >= .5),
                         auc=roc_auc_score(y, p), share_up_predicted=float((p >= .5).mean())))
    res = pd.DataFrame(rows)
    print("\n" + res.to_string(index=False))
    os.makedirs(P["figures"], exist_ok=True)
    res.to_csv(f"{P['figures']}/chronos_vs_pycaret.csv", index=False)
    out = pd.concat([pd.read_csv(f) for f in glob.glob("figures/*/chronos_vs_pycaret.csv")])
    out.to_csv("figures/hf_comparison.csv", index=False)


if __name__ == "__main__":
    if not hf_models.AVAILABLE:
        raise SystemExit(f"Hugging Face packages missing ({hf_models.IMPORT_ERROR}).\n"
                         f"Run:  pip install -r requirements-hf.txt")
    todo = [a for a in sys.argv[1:] if not a.startswith("--")] or [DEFAULT_PROFILE]
    for p in todo:
        if p not in PROFILES:
            raise SystemExit(f"unknown profile '{p}'. choose from: {list(PROFILES)}")
        evaluate(p)
