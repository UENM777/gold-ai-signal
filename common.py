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
FEATURES = OLD_FEATURES + SESSION_FEATURES          # base set; extras are appended per model


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
