"""Step 1: download XAUUSD history from MT5 for every timeframe the profiles use (run with MT5 open).
    python export_data.py            -> all timeframes
    python export_data.py M1 H1      -> only these
"""
import os, sys, time
import MetaTrader5 as mt5
import pandas as pd
from common import SYMBOL_CANDIDATES, PROFILES, EXTRA_SYMBOLS, data_csv, extra_csv

TF = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
      "H1": mt5.TIMEFRAME_H1, "D1": mt5.TIMEFRAME_D1}


def find_symbol():
    for s in SYMBOL_CANDIDATES:
        if mt5.symbol_info(s) is not None:
            mt5.symbol_select(s, True)
            return s
    gold = [s.name for s in mt5.symbols_get() if "XAU" in s.name.upper() or "GOLD" in s.name.upper()]
    if gold:
        mt5.symbol_select(gold[0], True)
        return gold[0]
    raise RuntimeError("Gold symbol not found in MT5")


def find_like(name):
    """Broker symbols may carry a suffix (EURUSDm, EURUSD.a ...)."""
    if mt5.symbol_info(name) is not None:
        mt5.symbol_select(name, True); return name
    for s in mt5.symbols_get() or []:
        if s.name.upper().startswith(name):
            mt5.symbol_select(s.name, True); return s.name
    return None


def download(sym, tf, n_bars):
    for n in [n_bars, int(n_bars * .8), int(n_bars * .6), int(n_bars * .4), int(n_bars * .2)]:  # MT5 caps by "Max bars in chart"
        for _ in range(3):                                  # first call may still be downloading history
            rates = mt5.copy_rates_from_pos(sym, TF[tf], 1, n)   # start=1 -> skip unfinished bar
            if rates is not None and len(rates) > 0:
                return rates
            time.sleep(2)
        print(f"  {tf}: {n} bars failed ({mt5.last_error()}) -> trying fewer")
    return None


if __name__ == "__main__":
    wanted = {}
    for p in PROFILES.values():
        wanted[p["tf"]] = max(wanted.get(p["tf"], 0), p["n_bars"])
    if len(sys.argv) > 1:
        wanted = {tf: n for tf, n in wanted.items() if tf in sys.argv[1:]}
    if not mt5.initialize():
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
    sym = find_symbol()
    os.makedirs("data", exist_ok=True)
    for tf, n in wanted.items():
        rates = download(sym, tf, n)
        if rates is None:
            print(f"{tf}: no data. Open an XAUUSD {tf} chart, press Home to scroll back, "
                  f"set Tools > Options > Charts > Max bars in chart = Unlimited, then retry.")
            continue
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df[["time", "open", "high", "low", "close", "tick_volume", "spread"]].to_csv(data_csv(tf), index=False)
        print(f"{sym} {tf}: saved {len(df)} bars {df.time.min()} -> {df.time.max()} to {data_csv(tf)}")
        for name in EXTRA_SYMBOLS:                       # related markets on the same timeframe
            s = find_like(name)
            r = download(s, tf, n) if s else None
            if r is None:
                print(f"   {name} {tf}: not available -> skipped"); continue
            x = pd.DataFrame(r); x["time"] = pd.to_datetime(x["time"], unit="s")
            x[["time", "close"]].to_csv(extra_csv(name, tf), index=False)
            print(f"   + {s} {tf}: {len(x)} bars")
    mt5.shutdown()
