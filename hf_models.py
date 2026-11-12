"""Ready-made Hugging Face models (no training needed).

    Chronos-Bolt (amazon/chronos-bolt-small)  -> zero-shot price forecast from past closes
    FinBERT      (ProsusAI/finbert)           -> sentiment of the latest gold / USD / Fed headlines

Optional: if torch / transformers / chronos-forecasting are not installed, AVAILABLE is False and the app
simply hides these panels.  Install with:  pip install -r requirements-hf.txt
The first use downloads the models (~200 MB + ~440 MB) into the Hugging Face cache; after that it runs offline.
"""
import re
import threading
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import numpy as np

try:
    import torch
    from chronos import BaseChronosPipeline
    from transformers import pipeline as hf_pipeline
    AVAILABLE, IMPORT_ERROR = True, ""
except Exception as e:                       # not installed -> app runs without the HF panels
    AVAILABLE, IMPORT_ERROR = False, str(e)

CHRONOS_ID = "amazon/chronos-bolt-small"
FINBERT_ID = "ProsusAI/finbert"
CONTEXT_BARS = 512                            # how many past closes Chronos reads
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NEWS_QUERIES = ["gold price", "XAUUSD", "Federal Reserve interest rates", "US dollar index"]
NEWS_MAX_AGE_H = 24
NEWS_MAX_ITEMS = 40


# Loading two models at the same moment from different threads breaks one of them
# ("Tensor.item() cannot be called on meta tensors"): transformers/accelerate load weights through a global
# "empty weights" switch. So every model is loaded under one lock.
_load_lock = threading.Lock()

# ---------------- Chronos ----------------
_chronos = None


def chronos():
    global _chronos
    with _load_lock:
        if _chronos is None:
            _chronos = BaseChronosPipeline.from_pretrained(CHRONOS_ID, device_map="cpu", torch_dtype=torch.float32)
    return _chronos


def p_up_from_quantiles(q, last):
    """q: forecast quantiles (QUANTILES order) at the horizon. P(price > last) by interpolating the forecast CDF."""
    q = np.maximum.accumulate(np.asarray(q, dtype=float))      # guard against tiny crossings
    cdf = np.interp(last, q, QUANTILES, left=0.05, right=0.95)  # outside the 10-90% band -> clip
    return float(1 - cdf)


def forecast_many(contexts, h):
    """contexts: list of 1-D arrays of past closes. Returns (median[h-ahead], p_up, q10, q90) arrays."""
    pipe = chronos()
    inputs = [torch.tensor(np.asarray(c[-CONTEXT_BARS:], dtype=np.float32)) for c in contexts]
    quant, _mean = pipe.predict_quantiles(inputs, prediction_length=h, quantile_levels=QUANTILES)
    q = quant[:, h - 1, :].float().numpy()                     # (batch, n_quantiles) at the horizon
    last = np.array([c[-1] for c in contexts], dtype=float)
    p_up = np.array([p_up_from_quantiles(q[i], last[i]) for i in range(len(contexts))])
    return q[:, QUANTILES.index(0.5)], p_up, q[:, 0], q[:, -1]


def forecast(closes, h):
    med, p_up, lo, hi = forecast_many([np.asarray(closes, dtype=float)], h)
    last = float(closes[-1])
    return dict(median=float(med[0]), p_up=float(p_up[0]), q10=float(lo[0]), q90=float(hi[0]),
                change_pct=float(med[0] / last - 1), last=last)


# ---------------- FinBERT + news ----------------
_finbert = None


def finbert():
    global _finbert
    with _load_lock:
        if _finbert is None:
            _finbert = hf_pipeline("text-classification", model=FINBERT_ID, top_k=None, device=-1)
    return _finbert


def fetch_headlines(queries=NEWS_QUERIES, max_age_h=NEWS_MAX_AGE_H, limit=NEWS_MAX_ITEMS):
    """Latest English headlines from Google News RSS. Returns [(datetime_utc, title, source)] newest first."""
    seen, out = set(), []
    now = datetime.now(timezone.utc)
    for q in queries:
        url = ("https://news.google.com/rss/search?q=" + urllib.parse.quote(q)
               + "&hl=en-US&gl=US&ceid=US:en")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (gold-ai-signal)"})
        try:
            xml = urllib.request.urlopen(req, timeout=10).read()
        except Exception:
            continue
        for item in ET.fromstring(xml).iter("item"):
            title = (item.findtext("title") or "").strip()
            src = (item.findtext("source") or "").strip()
            if src and title.endswith(" - " + src):           # Google appends " - Source"
                title = title[: -len(src) - 3]
            try:
                t = parsedate_to_datetime(item.findtext("pubDate"))
            except Exception:
                continue
            key = re.sub(r"\W+", " ", title.lower()).strip()
            if not title or key in seen or (now - t).total_seconds() > max_age_h * 3600:
                continue
            seen.add(key)
            out.append((t, title, src))
    out.sort(key=lambda x: x[0], reverse=True)
    return out[:limit]


def score_headlines(titles):
    """Returns [(label, positive - negative)] per title. FinBERT labels: positive / negative / neutral."""
    if not titles:
        return []
    res = finbert()(list(titles), truncation=True, batch_size=16)
    out = []
    for r in res:
        p = {d["label"].lower(): d["score"] for d in r}
        out.append((max(p, key=p.get), p.get("positive", 0) - p.get("negative", 0)))
    return out


def news_mood():
    """Average FinBERT sentiment of the last 24h of gold / USD / Fed headlines, in [-1, 1].
    Note: this is mood about the *news*, not a price forecast. Good news for the US dollar is often bad for gold."""
    items = fetch_headlines()
    scored = score_headlines([t for _, t, _ in items])
    if not scored:
        return dict(score=0.0, n=0, label="no news", top=[])
    w = np.array([0.5 ** ((items[0][0] - t).total_seconds() / 21600) for t, _, _ in items])  # half-life 6 h
    s = np.array([x[1] for x in scored])
    score = float((w * s).sum() / w.sum())
    label = "Positive" if score > 0.15 else "Negative" if score < -0.15 else "Neutral"
    top = [(items[i][1], scored[i][0]) for i in range(min(5, len(items)))]
    return dict(score=score, n=len(items), label=label, top=top)


# ---------------- YOLOv8 chart patterns (image) ----------------
try:
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download, list_repo_files
    YOLO_AVAILABLE, YOLO_ERROR = True, ""
except Exception as e:
    YOLO_AVAILABLE, YOLO_ERROR = False, str(e)

YOLO_ID = "foduucom/stockmarket-pattern-detection-yolov8"
YOLO_BARS = 100                      # how many candles go into the chart picture
YOLO_CONF = 0.25
CHART_PNG = "figures/live_chart.png"
CHART_ANNOTATED = "figures/live_chart_patterns.png"
PATTERN_INFO = {                     # model class -> (friendly name, what traders usually read into it)
    "Head and shoulders top": ("Head & Shoulders top", "often bearish"),
    "Head and shoulders bottom": ("Head & Shoulders bottom", "often bullish"),
    "M_Head": ("Double Top (M)", "often bearish"),
    "W_Bottom": ("Double Bottom (W)", "often bullish"),
    "Triangle": ("Triangle", "breakout pending"),
    "StockLine": ("Trend line", "trend in progress"),
}
_yolo = None


def yolo():
    global _yolo
    with _load_lock:
        if _yolo is None:
            weights = next(f for f in list_repo_files(YOLO_ID) if f.endswith(".pt"))
            _yolo = YOLO(hf_hub_download(YOLO_ID, weights))
    return _yolo


def render_chart(bars, path=CHART_PNG):
    """Draw a plain candlestick chart (like a trading-platform screenshot) for the YOLO model to look at."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    d = bars.tail(YOLO_BARS).reset_index(drop=True)
    fig = Figure(figsize=(8, 6.4), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.02, 0.04, 0.9, 0.94])
    up = d["close"] >= d["open"]
    for i, r in d.iterrows():
        c = "#26a69a" if up[i] else "#ef5350"
        ax.vlines(i, r["low"], r["high"], color=c, linewidth=1)
        ax.bar(i, abs(r["close"] - r["open"]) or (r["high"] - r["low"]) * 0.02,
               bottom=min(r["open"], r["close"]), width=0.7, color=c)
    ax.set_xlim(-1, len(d)); ax.yaxis.tick_right(); ax.set_xticks([])
    ax.grid(color="#eeeeee", linewidth=0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, facecolor="white")
    return path


def detect_patterns(bars):
    """Returns (patterns, annotated_png). patterns: [dict(name, hint, conf, recent)] strongest first.
    recent=True when the box reaches the newest 25% of candles (what matters for the next move)."""
    path = render_chart(bars)
    res = yolo()(path, conf=YOLO_CONF, verbose=False)[0]
    w = res.orig_shape[1]
    out = []
    for cls, conf, box in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist()):
        raw = res.names[int(cls)]
        name, hint = PATTERN_INFO.get(raw, (raw, ""))
        out.append(dict(name=name, hint=hint, conf=float(conf), recent=box[2] >= 0.72 * w))   # chart area ends at ~92% width
    out.sort(key=lambda p: (p["recent"], p["conf"]), reverse=True)
    from PIL import Image
    Image.fromarray(res.plot()[..., ::-1]).save(CHART_ANNOTATED)       # plot() is BGR
    return out, CHART_ANNOTATED
