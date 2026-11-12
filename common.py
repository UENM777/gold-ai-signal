"""Shared config + feature engineering. Used by export_data.py, train.py, self_train.py and app.py."""
import os
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
SYMBOL_CANDIDATES = ["XAUUSD", "XAUUSDm", "XAUUSD.", "GOLD", "XAUUSD.a", "XAUUSDc"]

# Prediction horizons shown in the app's dropdown.
# tf = candle timeframe the model reads, h = how many candles ahead it predicts, n_bars = history to train on
PROFILES = {
    "1 min":   dict(tf="M1",  h=1,  n_bars=100_000),
    "5 min":   dict(tf="M1",  h=5,  n_bars=100_000),
    "15 min":  dict(tf="M1",  h=15, n_bars=100_000),
    "1 hour":  dict(tf="M5",  h=12, n_bars=100_000),
    "4 hours": dict(tf="M15", h=16, n_bars=50_000),
    "1 day":   dict(tf="H1",  h=24, n_bars=30_000),
    "1 week":  dict(tf="D1",  h=5,  n_bars=5_000),
}
DEFAULT_PROFILE = "15 min"
TF_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "H1": 3600, "D1": 86400}

SIGNAL_TOP_PCT = 0.20      # candidates = 10% highest + 10% lowest P(up)
MIN_EDGE = 0.02            # ...and P(up) must be >= 52% (BUY) or <= 48% (SELL), else WAIT
ATR_PERIOD = 14
# TP/SL in "horizon ATR" units (ATR x sqrt(h)/2), chosen by the volatility model
TP_SL = {"HIGH": (2.0, 1.0), "LOW": (1.5, 1.0)}

PREDICT_EVERY_MS = 2000    # live: check for a new closed bar every 2 s
USE_FORMING_BAR = False    # True = predict on the unfinished bar (fast but features are distorted)
DEFAULT_LOT = 0.01
# AUTO mode (app checkbox): opens trades by itself, no confirmation popup. DEMO accounts only.
AUTO_ONLY_GO = True            # trade only when the traffic light is GO (False = also on CAUTION)
AUTO_MAX_OPEN = 10              # max positions opened by AUTO at the same time
AUTO_MAX_TRADES_24H = 1000       # max new AUTO trades per 24 hours
AUTO_SHORT_PROFILE = "1 min"   # "SHORT" tick: AUTO scalps on this horizon
AUTO_SHORT_MAX_TRADES_24H = 1000 # trade limit while SHORT is ticked (instead of AUTO_MAX_TRADES_24H)
AUTO_MAX_LOSS_PCT_24H = 10.0    # AUTO switches itself off when XAUUSD P/L of the last 24 h <= -3% of balance
MAGIC = 423001             # tags orders placed by this app (DES423)

# Related markets pulled from MT5 alongside gold (skipped automatically if the broker doesn't have one)
#   EURUSD ~ inverse of the US dollar (EUR is ~57% of the DXY index)   USDJPY ~ follows US bond yields
#   XAGUSD = silver, moves with gold
EXTRA_SYMBOLS = ["EURUSD", "USDJPY", "XAGUSD"]
SERVER_UTC_OFFSET = 2      # MT5 server time = UTC+2 (MetaQuotes / most brokers; +3 in northern summer)

OLD_FEATURES = [           # features of models trained before related markets were added
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20",
    "rsi_14", "close_sma20", "close_sma50", "sma20_sma50",
    "atr_pct", "std_20", "range_pct", "past_range_h", "vol_ratio",
    "body_pct", "upper_wick", "lower_wick", "hour", "dow",
]
SESSION_FEATURES = ["sess_asia", "sess_london", "sess_ny"]
# Volatility context (added after experiment.py showed +0.015-0.025 AUC for the volatility model on short horizons)
VOL_FEATURES = ["typical", "range_vs_typical", "range_4h_vs_typical", "std_60", "std_240", "vol_trend",
                "atr_ratio", "recent_bar_range"]
FEATURES = OLD_FEATURES + SESSION_FEATURES + VOL_FEATURES   # base set; extras are appended per model
LIVE_BARS = 1500           # bars the app loads per prediction (the 500-bar "typical range" needs history)


def extra_features(symbols):
    return [f"{s}_{k}" for s in symbols for k in ("ret_1", "ret_5", "ret_20", "vs_gold_20")]


def model_features(stats):
    """The exact feature list a saved model was trained with."""
    return stats.get("features", OLD_FEATURES)


def extra_csv(sym, tf):
    return f"data/EXTRA_{sym}_{tf}.csv"


def load_extras(tf):
    out = {}
    for s in EXTRA_SYMBOLS:
        if os.path.exists(extra_csv(s, tf)):
            out[s] = pd.read_csv(extra_csv(s, tf), parse_dates=["time"])
    return out


def slug(profile):
    return profile.replace(" ", "")


def paths(profile):
    p = PROFILES[profile]
    d = f"models/{slug(profile)}"
    return dict(dir=d, dir_model=f"{d}/dir_model", vol_model=f"{d}/vol_model", stats=f"{d}/stats.json",
                figures=f"figures/{slug(profile)}", data=data_csv(p["tf"]), live_log=f"data/live_log_{p['tf']}.csv")


def data_csv(tf):
    return f"data/XAUUSD_{tf}.csv"


def max_hold(h):
    return max(10, 4 * h)      # backtest: a trade expires after 4x the horizon


def trained(profile):
    return os.path.exists(paths(profile)["dir_model"] + ".pkl") and os.path.exists(paths(profile)["vol_model"] + ".pkl")


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int = ATR_PERIOD) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def make_features(df: pd.DataFrame, h: int, extras: dict = None) -> pd.DataFrame:
    """Only uses data up to and including each bar (no look-ahead).
    extras: {symbol: DataFrame(time, close)} of related markets on the same timeframe."""
    df = df.copy().sort_values("time").reset_index(drop=True)
    c = df["close"]
    for k in [1, 2, 3, 5, 10, 20]:
        df[f"ret_{k}"] = c.pct_change(k)
    df["rsi_14"] = rsi(c)
    sma20, sma50 = c.rolling(20).mean(), c.rolling(50).mean()
    df["close_sma20"] = c / sma20 - 1
    df["close_sma50"] = c / sma50 - 1
    df["sma20_sma50"] = sma20 / sma50 - 1
    df["atr"] = atr(df)
    df["atr_pct"] = df["atr"] / c
    df["std_20"] = c.pct_change().rolling(20).std()
    df["range_pct"] = (df["high"] - df["low"]) / c
    df["past_range_h"] = (df["high"].rolling(h).max() - df["low"].rolling(h).min()) / c
    df["vol_ratio"] = df["tick_volume"] / df["tick_volume"].rolling(20).mean()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_pct"] = (df["close"] - df["open"]) / rng
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / rng
    df["hour"] = df["time"].dt.hour
    df["dow"] = df["time"].dt.dayofweek
    utc_h = (df["hour"] - SERVER_UTC_OFFSET) % 24                 # trading sessions in UTC
    df["sess_asia"] = ((utc_h >= 0) & (utc_h < 8)).astype(int)    # Tokyo / Sydney / Shanghai
    df["sess_london"] = ((utc_h >= 7) & (utc_h < 16)).astype(int)
    df["sess_ny"] = ((utc_h >= 12) & (utc_h < 21)).astype(int)    # overlap with London = most active
    # volatility context: the volatility label asks "bigger than the typical range?" -> tell the model what typical is
    r = c.pct_change()
    df["typical"] = df["past_range_h"].rolling(500, min_periods=100).median()
    df["range_vs_typical"] = df["past_range_h"] / df["typical"]
    df["range_4h_vs_typical"] = (df["high"].rolling(4 * h).max() - df["low"].rolling(4 * h).min()) / c / df["typical"]
    df["std_60"], df["std_240"] = r.rolling(60).std(), r.rolling(240).std()
    df["vol_trend"] = df["std_20"] / df["std_240"]
    df["atr_ratio"] = df["atr"] / atr(df, 100)
    df["recent_bar_range"] = df["range_pct"].rolling(5).mean() / df["range_pct"].rolling(500, min_periods=100).mean()
    df[VOL_FEATURES] = df[VOL_FEATURES].replace([np.inf, -np.inf], np.nan)
    for s, ex in (extras or {}).items():
        ex = ex[["time", "close"]].rename(columns={"close": f"{s}_close"}).sort_values("time")
        df = pd.merge_asof(df, ex, on="time", direction="backward")   # last known price at or before this bar
        x = df[f"{s}_close"]
        df[f"{s}_ret_1"], df[f"{s}_ret_5"], df[f"{s}_ret_20"] = x.pct_change(1), x.pct_change(5), x.pct_change(20)
        df[f"{s}_vs_gold_20"] = df[f"{s}_ret_20"] - df["ret_20"]      # is gold lagging / leading this market?
    return df


def make_labels(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Targets look h bars into the future -> last h rows have no label and are dropped."""
    df = df.copy()
    # Model A: direction (close in h bars higher than now?)
    df["y_dir"] = (df["close"].shift(-h) > df["close"]).astype(float)
    # Model B: volatility (next-h-bar range bigger than the typical PAST range?)
    fut_range = (df["high"][::-1].rolling(h).max()[::-1].shift(-1)
                 - df["low"][::-1].rolling(h).min()[::-1].shift(-1)) / df["close"]
    typical = df["past_range_h"].rolling(500, min_periods=100).median()   # past only
    df["y_vol"] = (fut_range > typical).astype(float)
    df.loc[df.index[-h:], ["y_dir", "y_vol"]] = np.nan
    return df


def decide(p_up, q_lo, q_hi):
    """Always BUY or SELL (whichever side P(up) leans to). strong=True only for the most extreme
    predictions that also have a real edge over 50/50 (>= 52% / <= 48%)."""
    side = "BUY" if p_up >= 0.5 else "SELL"
    strong = p_up >= max(q_hi, 0.5 + MIN_EDGE) if side == "BUY" else p_up <= min(q_lo, 0.5 - MIN_EDGE)
    return side, bool(strong)


def tp_sl(price: float, atr_val: float, side: str, vol_regime: str, h: int):
    move = atr_val * max(1.0, np.sqrt(h) / 2)          # expected move grows with the horizon
    tp_m, sl_m = TP_SL[vol_regime]
    if side == "BUY":
        return price + tp_m * move, price - sl_m * move
    return price - tp_m * move, price + sl_m * move


def save_json(obj, path):
    """Write JSON safely: numpy values converted, and written to a temp file first so a crash never leaves half a file."""
    import json
    def fix(o):
        if isinstance(o, dict):
            return {str(k): fix(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [fix(v) for v in o]
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return o
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(fix(obj), fh, indent=2)
    os.replace(tmp, path)


def load_json(path):
    """Read stats.json; if an older version left it cut off, recover everything before the broken part."""
    import json
    if not os.path.exists(path):
        return {"tp_rate": {}}
    text = open(path).read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cut = text.rfind('"last_retrain"')
        if cut > 0:
            fixed = json.loads(text[:cut].rstrip().rstrip(",") + "\n}")
            save_json(fixed, path)
            return fixed
        raise


# ---------------- "Should I trade now?" traffic light ----------------
# Built from the backtest on unseen data: which SITUATIONS made or lost money, not from gut feeling.
FILTER_MIN_N = 30          # a situation needs at least this many past trades before we trust its result
LIGHTS = {2: "GO", 1: "CAUTION", 0: "NO TRADE"}


def session_name(asia, london, ny):
    if london and ny:
        return "London+NY"
    if ny:
        return "New York"
    if london:
        return "London"
    if asia:
        return "Asia"
    return "Off-hours"


def situation(side, regime, rsi_v, trend_v, asia, london, ny):
    """Describe the market situation of one signal with a few simple, explainable tags."""
    trend_up = trend_v > 0                                         # SMA20 above SMA50
    align = "with trend" if (side == "BUY") == trend_up else "against trend"
    rsi_zone = "RSI high" if rsi_v > 70 else "RSI low" if rsi_v < 30 else "RSI mid"
    return dict(regime=regime, align=align, session=session_name(asia, london, ny), rsi_zone=rsi_zone, side=side)


def cell_key(s):
    return f"{s['regime']}|{s['align']}|{s['session']}"


def rsi_key(s):
    return f"{s['side']}|{s['rsi_zone']}"


def build_filter(trades):
    """trades: DataFrame with regime, align, session, rsi_zone, side, pnl. Returns the lookup table."""
    def table(keys):
        out = {}
        for k, g in trades.groupby(keys):
            k = k if isinstance(k, tuple) else (k,)
            out["|".join(map(str, k))] = dict(n=int(len(g)), avg_pnl=float(g.pnl.mean()),
                                              tp_rate=float((g.outcome == "TP").mean()))
        return out
    return dict(cells=table(["regime", "align", "session"]), rsi=table(["side", "rsi_zone"]), min_n=FILTER_MIN_N)


def traffic_light(s, filt, chronos_agrees=None, check=None):
    """Returns (level 0/1/2, [reasons]). Uses the backtest table; Chronos agreement is a live-only extra check.
    check: stats["filter_check"]. If the out-of-sample check says GO trades did NOT beat average trades on this
    horizon, GO is capped to CAUTION (the light has no proven value there)."""
    reasons, mn = [], filt.get("min_n", FILTER_MIN_N)
    c = filt["cells"].get(cell_key(s))
    where = f"{s['regime'].lower()} volatility, {s['align']}, {s['session']}"
    if not c or c["n"] < mn:
        level = 1
        reasons.append(f"? {where}: too few past trades ({c['n'] if c else 0}) to judge")
    elif c["avg_pnl"] > 0:
        level = 2
        reasons.append(f"✔ {where}: made money in backtest ({c['n']} trades, avg {c['avg_pnl']:+.2f} $/oz, TP {c['tp_rate']:.0%})")
    else:
        level = 0
        reasons.append(f"✘ {where}: lost money in backtest ({c['n']} trades, avg {c['avg_pnl']:+.2f} $/oz, TP {c['tp_rate']:.0%})")
    r = filt["rsi"].get(rsi_key(s))
    if s["rsi_zone"] != "RSI mid" and r and r["n"] >= mn:
        if r["avg_pnl"] < 0:
            level = max(0, level - 1)
            reasons.append(f"✘ {s['side']} with {s['rsi_zone']}: lost in backtest ({r['n']} trades, avg {r['avg_pnl']:+.2f})")
        else:
            reasons.append(f"✔ {s['side']} with {s['rsi_zone']}: ok in backtest ({r['n']} trades, avg {r['avg_pnl']:+.2f})")
    if check is not None and not check.get("helped", False) and level == 2:
        level = 1
        reasons.append("⚠ GO capped to CAUTION: on this horizon, GO trades did not beat average trades "
                       "in the out-of-sample check")
    if chronos_agrees is False:
        level = max(0, level - 1)
        reasons.append("✘ Chronos (Hugging Face) points the other way  [live check, not backtested]")
    elif chronos_agrees is True:
        reasons.append("✔ Chronos (Hugging Face) agrees  [live check, not backtested]")
    return level, reasons
