"""AUI window.
    python app.py            -> live from MT5 (market must be open)
    python app.py --replay   -> replays recent history from the CSVs (for recording the demo on weekends)
"""
import os, sys, json, csv, threading
import tkinter as tk
from tkinter import messagebox
from datetime import datetime, timedelta
import pandas as pd
from PIL import Image, ImageTk
from pycaret.classification import load_model
from common import *
import self_train
import hf_models          # Chronos + FinBERT (optional; hidden if not installed)

REPLAY = "--replay" in sys.argv
ALLOW_REAL_ACCOUNT = False          # safety: the BUY/SELL buttons only work on a DEMO account
POINT = 0.01
WIN_W, WIN_H, STATS_W = 640, 600, 420       # main window; STATS_W = extra width when the stats panel is open
LOGO = os.path.join("assets", "siit_logo.png")

BG, FG, DIM = "#0b0b0f", "#f2f2f2", "#7a7a85"
GREEN, RED, GOLD, GREY = "#22c55e", "#ef4444", "#f5c542", "#9ca3af"
PANEL = "#1f2937"


# ---------------- data sources ----------------
class MT5Feed:
    def __init__(self):
        import MetaTrader5 as mt5
        from export_data import find_symbol, TF
        self.mt5, self.TF = mt5, TF
        if not mt5.initialize():
            raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
        self.sym = find_symbol()
        from export_data import find_like
        self.extra_syms = {n: find_like(n) for n in EXTRA_SYMBOLS}
        acc = mt5.account_info()
        self.is_demo = acc is not None and acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
        self.account = f"{'DEMO' if self.is_demo else 'REAL'} {acc.login}" if acc else "no account"
        self.currency = acc.currency if acc else "$"

    def tick(self):
        t = self.mt5.symbol_info_tick(self.sym)
        return t.bid, t.ask

    def _rates(self, tf, start, n):
        df = pd.DataFrame(self.mt5.copy_rates_from_pos(self.sym, self.TF[tf], start, n))
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def bars(self, tf, n=300):
        return self._rates(tf, 0 if USE_FORMING_BAR else 1, n)

    def history(self, tf, n):
        return self._rates(tf, 1, n)

    def extras(self, tf, n, names):
        out = {}
        for name in names:
            s = self.extra_syms.get(name)
            r = self.mt5.copy_rates_from_pos(s, self.TF[tf], 0 if USE_FORMING_BAR else 1, n) if s else None
            if r is not None and len(r):
                x = pd.DataFrame(r); x["time"] = pd.to_datetime(x["time"], unit="s")
                out[name] = x[["time", "close"]]
        return out

    def place(self, side, lot, regime, atr_val, h, comment="GIVE ME A!! AI"):
        mt5 = self.mt5
        if not self.is_demo and not ALLOW_REAL_ACCOUNT:
            return False, "Blocked: this is not a DEMO account."
        term = mt5.terminal_info()
        if term is not None and not term.trade_allowed:
            return False, "Turn on 'Algo Trading' in the MT5 toolbar first."
        info, t = mt5.symbol_info(self.sym), mt5.symbol_info_tick(self.sym)
        price = t.ask if side == "BUY" else t.bid
        tp, sl = tp_sl(price, atr_val, side, regime, h)
        min_d = (info.trade_stops_level + info.spread + 5) * info.point       # broker's minimum stop distance
        if side == "BUY":
            tp, sl = max(tp, price + min_d), min(sl, price - min_d)
        else:
            tp, sl = min(tp, price - min_d), max(sl, price + min_d)
        fill = (mt5.ORDER_FILLING_FOK if info.filling_mode & 1 else
                mt5.ORDER_FILLING_IOC if info.filling_mode & 2 else mt5.ORDER_FILLING_RETURN)
        req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=self.sym, volume=float(lot),
                   type=mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
                   price=price, sl=round(sl, info.digits), tp=round(tp, info.digits),
                   deviation=30, magic=MAGIC, comment=comment, type_time=mt5.ORDER_TIME_GTC,
                   type_filling=fill)
        res = mt5.order_send(req)
        if res is None:
            return False, f"order_send failed: {mt5.last_error()}"
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"Rejected ({res.retcode}): {res.comment}"
        return True, f"{side} {lot} lot @ {res.price:,.2f}\nTP {req['tp']:,.2f}   SL {req['sl']:,.2f}\nticket #{res.order}"

    def positions(self):
        return list(self.mt5.positions_get(symbol=self.sym) or [])

    def trade_history(self, days=90):
        """Closed XAUUSD positions of the last `days`: [dict(kind='auto'|'app'|'manual', pnl)]."""
        mt5, now, pos = self.mt5, datetime.now(), {}
        for d in mt5.history_deals_get(now - timedelta(days=days), now + timedelta(days=2)) or []:
            if d.symbol != self.sym:
                continue
            p = pos.setdefault(d.position_id, dict(pnl=0.0, closed=False, kind="manual", side=""))
            p["pnl"] += d.profit + d.swap + d.commission
            if d.entry == mt5.DEAL_ENTRY_IN:
                cm = d.comment or ""
                p["kind"] = ("short" if "SHORT" in cm else "auto" if "AUTO" in cm else "app") if d.magic == MAGIC else "manual"
                p["side"] = "BUY" if d.type == mt5.DEAL_TYPE_BUY else "SELL"
                p["time"] = d.time                                   # server time (epoch seconds)
                p["regime"] = "HIGH" if cm.endswith(" H") else "LOW" if cm.endswith(" L") else ""
                by_slug = {slug(k): k for k in PROFILES}
                p["profile"] = (AUTO_SHORT_PROFILE if "SHORT" in cm else
                                next((by_slug[t] for t in cm.split() if t in by_slug), ""))
                p["ticket"] = d.position_id
            else:
                p["closed"] = True
        return [p for p in pos.values() if p["closed"]]

    def auto_stats(self):
        """(open positions made by this app, AUTO trades in last 24 h, XAUUSD P/L of last 24 h incl. open, balance)"""
        mt5, pos = self.mt5, self.positions()
        t = mt5.symbol_info_tick(self.sym).time                      # server clock
        now = datetime.now()
        deals = [d for d in (mt5.history_deals_get(now - timedelta(days=3), now + timedelta(days=2)) or [])
                 if d.symbol == self.sym and d.time >= t - 86400]
        ins = [d for d in deals if d.entry == mt5.DEAL_ENTRY_IN and d.magic == MAGIC and "AUTO" in (d.comment or "")]
        n = (sum(1 for d in ins if "SHORT" not in d.comment), sum(1 for d in ins if "SHORT" in d.comment))
        pnl = sum(d.profit + d.swap + d.commission for d in deals) + sum(p.profit + p.swap for p in pos)
        mine = [p for p in pos if p.magic == MAGIC and "AUTO" in (p.comment or "")]
        n_open = (sum(1 for p in mine if "SHORT" not in p.comment), sum(1 for p in mine if "SHORT" in p.comment))
        return n_open, n, pnl, mt5.account_info().balance          # open counted per mode: SHORT and normal AUTO

    def close_positions(self, which):
        """which: 'profit' | 'loss' | 'all'. Closes this symbol's open positions at market. Returns a report string."""
        mt5 = self.mt5
        if not self.is_demo and not ALLOW_REAL_ACCOUNT:
            return False, "Blocked: this is not a DEMO account."
        pos = self.positions()
        pick = {"profit": lambda p: p.profit + p.swap > 0, "loss": lambda p: p.profit + p.swap < 0,
                "all": lambda p: True}[which]
        pos = [p for p in pos if pick(p)]
        if not pos:
            return True, "Nothing to close."
        info = mt5.symbol_info(self.sym)
        fill = (mt5.ORDER_FILLING_FOK if info.filling_mode & 1 else
                mt5.ORDER_FILLING_IOC if info.filling_mode & 2 else mt5.ORDER_FILLING_RETURN)
        ok, total, errs = 0, 0.0, []
        for p in pos:
            t = mt5.symbol_info_tick(self.sym)
            is_buy = p.type == mt5.POSITION_TYPE_BUY
            req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=self.sym, volume=p.volume, position=p.ticket,
                       type=mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
                       price=t.bid if is_buy else t.ask, deviation=30, magic=MAGIC,
                       comment="GIVE ME A!! close", type_time=mt5.ORDER_TIME_GTC, type_filling=fill)
            res = mt5.order_send(req)
            if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
                ok += 1; total += p.profit + p.swap
            else:
                errs.append(f"#{p.ticket}: {res.comment if res else mt5.last_error()}")
        msg = f"Closed {ok}/{len(pos)} positions · P/L {total:+,.2f} {self.currency}"
        return not errs, msg + ("\n\n" + "\n".join(errs) if errs else "")


class ReplayFeed:
    sym, account, is_demo = "XAUUSD (replay)", "REPLAY", False

    def __init__(self):
        self.dfs, self.i, self.cur = {}, {}, None

    def _df(self, tf):
        if tf not in self.dfs:
            self.dfs[tf] = pd.read_csv(data_csv(tf), parse_dates=["time"])
            self.i[tf] = max(min(LIVE_BARS, len(self.dfs[tf]) - 1), len(self.dfs[tf]) - 1500)
        self.cur = tf
        return self.dfs[tf]

    def tick(self):
        if self.cur is None:
            raise RuntimeError("no data yet")
        r = self.dfs[self.cur].iloc[self.i[self.cur]]
        return r.close, r.close + r.spread * POINT

    def bars(self, tf, n=300):
        df = self._df(tf)
        self.i[tf] = min(self.i[tf] + 1, len(df) - 1)       # advance one bar per call
        return df.iloc[max(0, self.i[tf] - n + 1):self.i[tf] + 1].reset_index(drop=True)

    def history(self, tf, n):
        df = self._df(tf)
        return df.iloc[max(0, self.i[tf] - n + 1):self.i[tf] + 1].reset_index(drop=True)

    def extras(self, tf, n, names):
        if tf not in self.dfs:
            return {}
        t1 = self.dfs[tf]["time"].iloc[self.i[tf]]
        if not hasattr(self, "ex"):
            self.ex = {}
        if tf not in self.ex:
            self.ex[tf] = load_extras(tf)
        return {k: v[v.time <= t1].tail(n) for k, v in self.ex[tf].items() if k in names}

    def place(self, *a):
        return False, "Orders are disabled in replay mode."

    currency = "$"

    def trade_history(self, days=90):
        return []

    def positions(self):
        return []

    def close_positions(self, which):
        return False, "Orders are disabled in replay mode."


# ---------------- intro ----------------
def fade(win, a0, a1, ms, done, steps=30):
    def step(k=0):
        win.attributes("-alpha", a0 + (a1 - a0) * k / steps)
        if k < steps:
            win.after(ms // steps, step, k + 1)
        else:
            done()
    step()


def intro(root, on_done):
    W, H = 760, 560
    sp = tk.Toplevel(root, bg="black")
    sp.overrideredirect(True)
    sp.attributes("-alpha", 0.0)
    sp.geometry(f"{W}x{H}+{(sp.winfo_screenwidth() - W) // 2}+{(sp.winfo_screenheight() - H) // 2}")
    frame = tk.Frame(sp, bg="black"); frame.pack(expand=True, fill="both")

    img = Image.open(LOGO).convert("RGBA")
    img.thumbnail((440, 440))
    bg = Image.new("RGBA", img.size, (0, 0, 0, 255)); bg.alpha_composite(img)
    sp.logo = ImageTk.PhotoImage(bg.convert("RGB"))

    finished = {"v": False}

    def finish(*_):
        if not finished["v"]:
            finished["v"] = True
            sp.destroy(); on_done()

    sp.bind("<Button-1>", finish); sp.bind("<Escape>", finish)

    def scene_logo():
        tk.Label(frame, image=sp.logo, bg="black").pack(expand=True)
        fade(sp, 0, 1, 1400, lambda: sp.after(1300, lambda: fade(sp, 1, 0, 1000, scene_text)))

    def scene_text():
        if finished["v"]:
            return
        for w in frame.winfo_children():
            w.destroy()
        box = tk.Frame(frame, bg="black"); box.pack(expand=True)
        tk.Label(box, text="DES423  ·  SECTION 1  PROJECT", fg=DIM, bg="black",
                 font=("Segoe UI", 16, "bold")).pack(pady=(0, 18))
        tk.Label(box, text="by group", fg=FG, bg="black", font=("Segoe UI", 14, "italic")).pack()
        tk.Label(box, text="GIVE ME A!!", fg=GOLD, bg="black", font=("Impact", 64)).pack()
        tk.Label(box, text="AUI", fg=FG, bg="black", font=("Segoe UI Light", 18)).pack(pady=(14, 0))
        fade(sp, 0, 1, 1200, lambda: sp.after(1800, lambda: fade(sp, 1, 0, 1000, finish)))

    sp.after(300, scene_logo)


# ---------------- main window ----------------
class App:
    def __init__(self, root):
        self.root = root
        self.feed = ReplayFeed() if REPLAY else MT5Feed()
        self.models = {}                       # profile -> (dir_model, vol_model, stats)
        self.profile = DEFAULT_PROFILE if trained(DEFAULT_PROFILE) else next((p for p in PROFILES if trained(p)), DEFAULT_PROFILE)
        self.last_bar = None
        self.signal = None
        self.bars_since_train = 0
        self.training = False
        self.train_result = None
        self.hf_busy, self.hf_result, self.hf_profile = False, None, None
        self.news_busy, self.news_result, self.news_next = False, None, 0
        self.yolo_busy, self.yolo_result, self.patterns = False, None, None
        self.ticks = 0
        self.stats_open, self.sliding = False, False
        self.build()

    # ---------- models ----------
    def current(self):
        if self.profile not in self.models:
            if not trained(self.profile):
                return None
            P = paths(self.profile)
            stats = load_json(P["stats"])
            dm, vm = load_model(P["dir_model"], verbose=False), load_model(P["vol_model"], verbose=False)
            feats = [c for c in getattr(dm, "feature_names_in_", []) if c not in ("y_dir", "y_vol")]  # trust the model file
            if feats and feats != model_features(stats):
                stats = dict(stats, features=feats, tp_rate_all=stats.get("tp_rate_all", {}),
                             extras=[s for s in EXTRA_SYMBOLS if any(f.startswith(s + "_") for f in feats)])
            self.models[self.profile] = (dm, vm, stats)
        return self.models[self.profile]

    def pick_profile(self, name):
        if getattr(self, "var_short", None) is not None and self.var_short.get() and name != AUTO_SHORT_PROFILE:
            self.var_short.set(False)                          # leaving 1 min ends SHORT mode
        self.set_profile(name)

    def set_profile(self, name):
        self.profile = name
        self.last_bar, self.signal, self.bars_since_train = None, None, 0
        self.show_wait()
        if hf_models.AVAILABLE and hasattr(self, "lbl_chr"):
            self.lbl_chr.config(text="🤗 Chronos: waiting for the next bar…", fg=DIM)
        if self.current() is None:
            self.lbl_foot.config(text=f'"{name}" is not trained yet → run:  python train.py "{name}"', fg=GOLD)
        self.show_train_status()

    # ---------- layout ----------
    def build(self):
        r = self.root
        r.title("AUI — GIVE ME A!!")
        r.configure(bg=BG); r.geometry(f"{WIN_W}x{WIN_H}"); r.resizable(False, False)
        r.attributes("-topmost", True)
        L = lambda parent, txt="", size=12, fg=FG, bold=False: tk.Label(
            parent, text=txt, fg=fg, bg=BG, font=("Segoe UI", size, "bold" if bold else "normal"), anchor="w")

        # top bar: horizon dropdown + account
        top = tk.Frame(r, bg=BG); top.place(x=24, y=10, width=596, height=34)
        L(top, "Predict next", 10, DIM).pack(side="left")
        self.var_profile = tk.StringVar(value=self.profile)
        labels = [p if trained(p) else f"{p}  (not trained)" for p in PROFILES]
        menu = tk.OptionMenu(top, self.var_profile, *labels, command=lambda v: self.pick_profile(v.split("  (")[0]))
        menu.config(bg=PANEL, fg=GOLD, activebackground="#374151", activeforeground=GOLD, relief="flat",
                    highlightthickness=0, font=("Segoe UI", 10, "bold"), width=11, cursor="hand2")
        menu["menu"].config(bg=PANEL, fg=FG, activebackground="#374151", font=("Segoe UI", 10))
        menu.pack(side="left", padx=8)
        L(top, self.feed.account, 9, GREEN if self.feed.is_demo else DIM).pack(side="right")
        self.var_auto = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="AUTO", variable=self.var_auto, command=self.on_auto, bg=BG, fg=GOLD,
                       selectcolor=PANEL, activebackground=BG, activeforeground=GOLD,
                       font=("Segoe UI", 10, "bold"), cursor="hand2").pack(side="left", padx=(4, 0))
        self.var_short, self.var_nolight = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        for txt, var, cmd in [("SHORT 1m", self.var_short, self.on_short), ("Ignore light", self.var_nolight, self.on_nolight)]:
            tk.Checkbutton(top, text=txt, variable=var, command=cmd, bg=BG, fg=FG, selectcolor=PANEL,
                           activebackground=BG, activeforeground=FG, font=("Segoe UI", 9), cursor="hand2").pack(side="left")
        self.lbl_auto = tk.Label(r, text="", fg=DIM, bg=BG, font=("Segoe UI", 8), anchor="w")
        self.lbl_auto.place(x=100, y=52, width=260)

        left = tk.Frame(r, bg=BG); left.place(x=24, y=50, width=330, height=270)
        right = tk.Frame(r, bg=BG); right.place(x=370, y=50, width=250, height=270)

        self.lbl_sym = L(left, self.feed.sym, 10, DIM); self.lbl_sym.pack(anchor="w")
        self.lbl_price = L(left, "—", 34, FG, True); self.lbl_price.pack(anchor="w")
        self.lbl_ba = L(left, "", 10, DIM); self.lbl_ba.pack(anchor="w")
        self.lbl_conf = L(left, "—", 44, FG, True); self.lbl_conf.pack(anchor="w", pady=(6, 0))
        sig_row = tk.Frame(left, bg=BG); sig_row.pack(anchor="w", fill="x")
        self.lbl_sig = L(sig_row, "…", 38, GREY, True); self.lbl_sig.pack(side="left")
        self.lbl_strength = L(sig_row, "", 12, GREY, True); self.lbl_strength.pack(side="left", padx=(12, 0), pady=(14, 0))
        self.lbl_strength.config(cursor="hand2")
        self.lbl_strength.bind("<Button-1>", lambda e: self.show_light_reasons())

        L(right, "TP", 14, DIM, True).pack(anchor="w")
        self.lbl_tp = L(right, "—", 22, GREEN, True); self.lbl_tp.pack(anchor="w")
        self.lbl_tp_d = L(right, "", 11); self.lbl_tp_d.pack(anchor="w")
        self.lbl_tp_p = L(right, "", 18, GOLD, True); self.lbl_tp_p.pack(anchor="w")
        L(right, "SL", 14, DIM, True).pack(anchor="w", pady=(8, 0))
        self.lbl_sl = L(right, "—", 18, RED, True); self.lbl_sl.pack(anchor="w")
        self.lbl_sl_d = L(right, "", 11); self.lbl_sl_d.pack(anchor="w")
        self.lbl_vol = L(right, "", 11, GOLD, True); self.lbl_vol.pack(anchor="w", pady=(8, 0))

        # trade row
        row = tk.Frame(r, bg=BG); row.place(x=24, y=328, width=596, height=40)
        L(row, "Lot", 10, DIM).pack(side="left")
        self.var_lot = tk.StringVar(value=str(DEFAULT_LOT))
        tk.Spinbox(row, from_=0.01, to=1.0, increment=0.01, textvariable=self.var_lot, width=5,
                   font=("Segoe UI", 11), bg=PANEL, fg=FG, buttonbackground=PANEL, relief="flat",
                   insertbackground=FG).pack(side="left", padx=(6, 14))
        self.btn_buy = tk.Button(row, text="BUY", command=lambda: self.order("BUY"), bg="#14532d", fg=FG,
                                 activebackground=GREEN, relief="flat", font=("Segoe UI", 11, "bold"), cursor="hand2")
        self.btn_buy.pack(side="left", fill="y", expand=True, padx=(0, 6))
        self.btn_sell = tk.Button(row, text="SELL", command=lambda: self.order("SELL"), bg="#7f1d1d", fg=FG,
                                  activebackground=RED, relief="flat", font=("Segoe UI", 11, "bold"), cursor="hand2")
        self.btn_sell.pack(side="left", fill="y", expand=True)

        self.lbl_foot = tk.Label(r, text="", fg=DIM, bg=BG, font=("Segoe UI", 8), anchor="w")
        hf = tk.Frame(r, bg="#111827"); hf.place(x=24, y=376, width=596, height=76)
        self.lbl_chr = tk.Label(hf, text="", fg=FG, bg="#111827", font=("Segoe UI", 10), anchor="w")
        self.lbl_chr.place(x=10, y=4, width=580)
        self.lbl_news = tk.Label(hf, text="", fg=FG, bg="#111827", font=("Segoe UI", 10), anchor="w", cursor="hand2")
        self.lbl_news.place(x=10, y=27, width=580)
        self.lbl_news.bind("<Button-1>", lambda e: self.show_headlines())
        self.lbl_pat = tk.Label(hf, text="", fg=FG, bg="#111827", font=("Segoe UI", 10), anchor="w", cursor="hand2")
        self.lbl_pat.place(x=10, y=50, width=580)
        self.lbl_pat.bind("<Button-1>", lambda e: self.show_pattern_image())
        if hf_models.AVAILABLE and not hf_models.YOLO_AVAILABLE:
            self.lbl_pat.config(text="📈 Chart patterns off — pip install -r requirements-hf.txt", fg=DIM)

        # open positions + close buttons
        pr = tk.Frame(r, bg=BG); pr.place(x=24, y=460, width=596, height=32)
        self.lbl_pos = tk.Label(pr, text="Open: —", fg=DIM, bg=BG, font=("Segoe UI", 10, "bold"), anchor="w")
        self.lbl_pos.pack(side="left")
        for txt, which, col in [("Close all", "all", "#374151"), ("Close loss", "loss", "#7f1d1d"),
                                ("Close profit", "profit", "#14532d")]:
            tk.Button(pr, text=txt, command=lambda w=which: self.close(w), bg=col, fg=FG, activebackground=GREY,
                      relief="flat", font=("Segoe UI", 9, "bold"), cursor="hand2", width=11).pack(side="right", padx=(6, 0))
        if not hf_models.AVAILABLE:
            self.lbl_chr.config(text="🤗 Hugging Face models off — pip install -r requirements-hf.txt", fg=DIM)
        self.lbl_foot.place(x=24, y=500, width=596)
        self.lbl_train = tk.Label(r, text="", fg=DIM, bg=BG, font=("Segoe UI", 9), anchor="w")
        self.lbl_train.place(x=24, y=536, width=450)
        self.btn_train = tk.Button(r, text="⟳  Retrain now", command=self.start_retrain, bg=PANEL, fg=FG,
                                   activebackground="#374151", activeforeground=FG, relief="flat",
                                   font=("Segoe UI", 9, "bold"), cursor="hand2")
        self.btn_train.place(x=490, y=530, width=130, height=30)
        tk.Label(r, text="Decision support for a demo account — not financial advice.", fg="#4b5563", bg=BG,
                 font=("Segoe UI", 8), anchor="w").place(x=24, y=572, width=596)

        # slide-out statistics panel (arrow on the right edge)
        self.stats_frame = tk.Frame(r, bg="#0f172a"); self.stats_frame.place(x=WIN_W, y=0, width=STATS_W - 18, height=WIN_H)
        self.btn_arrow = tk.Button(r, text="›", command=self.toggle_stats, bg=PANEL, fg=GOLD, activebackground="#374151",
                                   activeforeground=GOLD, relief="flat", font=("Segoe UI", 14, "bold"), cursor="hand2")
        self.btn_arrow.place(x=WIN_W - 18, y=WIN_H // 2 - 34, width=18, height=68)

        self.lbl_auto.lift()                                  # status sits over the left panel (was hidden behind it)

        self.set_profile(self.profile)
        self.update_bar()
        self.update_tick()

    # ---------- prediction loop ----------
    def update_bar(self):
        try:
            if self.current() is not None:
                tf = PROFILES[self.profile]["tf"]
                bars = self.feed.bars(tf, LIVE_BARS)
                self.ex = self.feed.extras(tf, LIVE_BARS + 30, self.current()[2].get("extras", []))
                t = bars["time"].iloc[-1]
                if USE_FORMING_BAR and not REPLAY:
                    if self.last_bar is not None and t != self.last_bar:
                        closed = self.compute(bars.iloc[:-1])
                        if closed:
                            self.log(*closed)
                    self.last_bar = t
                    out = self.compute(bars)
                    if out:
                        self.show(*out)
                        self.start_hf(bars)
                elif t != self.last_bar:                           # closed-bar mode: once per new bar
                    first = self.last_bar is None
                    self.last_bar = t
                    out = self.compute(bars)
                    if out:
                        self.show(*out)
                        self.start_hf(bars)
                        if not REPLAY and not first:
                            self.log(*out)
                    if self.stats_open:
                        self.refresh_stats()
                    if not first:
                        self.auto_trade()
                        self.bars_since_train += 1
                        if self.bars_since_train >= self_train.RETRAIN_EVERY_BARS:
                            self.start_retrain()
                        elif not self.training:
                            self.show_train_status()
        except Exception as e:
            self.lbl_foot.config(text=f"error: {e}", fg=RED)
        self.check_hf()
        self.check_yolo()
        try:
            self.check_retrain()
        except Exception as e:
            self.training = False
            self.btn_train.config(state="normal", text="⟳  Retrain now")
            self.lbl_train.config(text=f"self-train failed: {e}", fg=RED)
        self.root.after(1000 if REPLAY else PREDICT_EVERY_MS, self.update_bar)

    def compute(self, bars):
        dir_model, vol_model, stats = self.current()
        h = PROFILES[self.profile]["h"]
        feats = model_features(stats)
        f = make_features(bars, h, getattr(self, "ex", None)).iloc[[-1]]
        missing = [c for c in feats if c not in f]
        if missing:
            raise RuntimeError(f"model needs '{missing[0]}' which is not available → retrain this horizon")
        if f[feats].isna().any(axis=1).iloc[0]:
            return None
        p_up = float(dir_model.predict_proba(f[feats])[0, 1])
        p_vol = float(vol_model.predict_proba(f[feats])[0, 1])
        q_lo, q_hi = stats.get("q_lo", 0.5), stats.get("q_hi", 0.5)
        regime = "HIGH" if p_vol >= 0.5 else "LOW"
        side, strong = decide(p_up, q_lo, q_hi)
        conf = p_up if side == "BUY" else 1 - p_up
        entry, a = float(f["close"].iloc[0]), float(f["atr"].iloc[0])
        tp, sl = tp_sl(entry, a, side, regime, h)
        return f, p_up, p_vol, side, conf, tp, sl, regime, strong

    def show_wait(self):
        self.lbl_conf.config(text="—"); self.lbl_sig.config(text="…", fg=GREY); self.lbl_strength.config(text="")
        for w in (self.lbl_tp, self.lbl_sl):
            w.config(text="—")
        for w in (self.lbl_tp_d, self.lbl_sl_d, self.lbl_tp_p, self.lbl_vol):
            w.config(text="")

    def show(self, f, p_up, p_vol, side, conf, tp, sl, regime, strong):
        _, _, stats = self.current()
        self.signal = dict(side=side, strong=strong, p_up=p_up, regime=regime, tp=tp, sl=sl, atr=float(f["atr"].iloc[0]))
        q_lo, q_hi = stats.get("q_lo", 0.5), stats.get("q_hi", 0.5)
        col = {"BUY": GREEN, "SELL": RED}[side]
        dim = {"BUY": "#86efac", "SELL": "#fca5a5"}[side]
        self.lbl_conf.config(text=f"{conf:.1%}", fg=FG if strong else GREY)
        self.lbl_sig.config(text=side, fg=col if strong else dim)
        self.sit = situation(side, regime, float(f["rsi_14"].iloc[0]), float(f["sma20_sma50"].iloc[0]),
                             int(f["sess_asia"].iloc[0]), int(f["sess_london"].iloc[0]), int(f["sess_ny"].iloc[0]))
        self.update_light()
        vol_conf = p_vol if regime == "HIGH" else 1 - p_vol
        self.lbl_vol.config(text=f"{regime} VOLATILITY  {vol_conf:.1%}")
        self.lbl_tp.config(text=f"{tp:,.2f}", fg=GREEN)
        self.lbl_sl.config(text=f"{sl:,.2f}", fg=RED)
        table = stats.get("tp_rate" if strong else "tp_rate_all", {})
        rate = table.get(f"{side}_{regime}")
        self.lbl_tp_p.config(text=f"hit {rate:.0%}" if rate is not None else "")
        self.btn_buy.config(bg=(GREEN if strong else "#15803d") if side == "BUY" else "#14532d")
        self.btn_sell.config(bg=(RED if strong else "#b91c1c") if side == "SELL" else "#7f1d1d")
        mode = "replay" if REPLAY else f"live · {datetime.now():%H:%M:%S}"
        buy_at, sell_at = max(q_hi, .5 + MIN_EDGE), min(q_lo, .5 - MIN_EDGE)
        ex = "+".join(stats.get("extras", [])) or "gold only"
        self.lbl_foot.config(fg=DIM, text=f"{mode}  ·  {PROFILES[self.profile]['tf']} bar {self.last_bar:%m-%d %H:%M}  ·  "
                             f"P(up) {p_up:.1%}  ·  STRONG if ≥ {buy_at:.1%} / ≤ {sell_at:.1%}  ·  "
                             f"A: {stats.get('dir_model', '?')}  B: {stats.get('vol_model', '?')}  ·  data: {ex}")

    # ---------- traffic light: should I trade now? ----------
    def update_light(self):
        s, sig = getattr(self, "sit", None), self.signal
        cur = self.current()
        if not s or not sig or not cur:
            return
        filt = cur[2].get("filter")
        if not filt:                                       # model trained before the traffic light existed
            self.light = (None, ["Retrain this horizon to get the traffic light:  python train.py \"%s\" --fast" % self.profile])
            self.lbl_strength.config(text=("STRONG" if sig["strong"] else "WEAK · risky") + "  ⓘ",
                                     fg=GOLD if sig["strong"] else GREY)
            return
        ch = getattr(self, "chronos_last", None)
        agrees = None
        if ch and ch["profile"] == self.profile and ch["bar"] == self.last_bar:
            agrees = (ch["p_up"] >= .5) == (sig["side"] == "BUY")
        level, reasons = traffic_light(s, filt, agrees, cur[2].get("filter_check") or {})
        reasons.insert(0, f"{'STRONG' if sig['strong'] else 'WEAK'} lean: P(up) {sig['p_up']:.1%}  "
                          f"(direction model is close to a coin flip on every horizon)")
        chk = cur[2].get("filter_check") or {}
        if chk.get("go_n"):
            reasons.append(f"\nOut-of-sample check (2nd half of test): GO trades avg {chk['go_avg_pnl']:+.2f} $/oz "
                           f"vs all trades {chk['all_avg_pnl']:+.2f}  ({chk['go_n']} GO trades) → "
                           f"{'the light helped' if chk['helped'] else 'the light did NOT clearly help on this horizon'}")
        self.light = (level, reasons)
        icon = {2: "🟢 GO", 1: "🟡 CAUTION", 0: "🔴 NO TRADE"}[level]
        self.lbl_strength.config(text=icon + "  ⓘ", fg={2: GREEN, 1: GOLD, 0: RED}[level])

    def show_light_reasons(self):
        lt = getattr(self, "light", None)
        if lt:
            title = {2: "GO", 1: "CAUTION", 0: "NO TRADE", None: "Signal"}[lt[0]]
            messagebox.showinfo(f"Should I trade now? — {title}", "\n".join(lt[1]))

    # ---------- statistics panel ----------
    def toggle_stats(self):
        if self.sliding:
            return
        self.stats_open, self.sliding = not self.stats_open, True
        if self.stats_open:
            self.refresh_stats()
        start = self.root.winfo_width()
        target = WIN_W + STATS_W if self.stats_open else WIN_W

        def step(k=1, n=12):
            w = int(start + (target - start) * (1 - (1 - k / n) ** 2))      # ease-out
            self.root.geometry(f"{w}x{WIN_H}")
            self.btn_arrow.place(x=w - 18, y=WIN_H // 2 - 34, width=18, height=68)
            if k < n:
                self.root.after(15, step, k + 1)
            else:
                self.sliding = False
                self.btn_arrow.config(text="‹" if self.stats_open else "›")
        step()

    def prediction_stats(self, prof=None):
        """Compare what the app predicted (live log) with what really happened. Returns (rows, note)."""
        prof = prof or self.profile
        P = paths(prof)
        h, tf = PROFILES[prof]["h"], PROFILES[prof]["tf"]
        if not os.path.exists(P["live_log"]):
            return [], "No predictions logged yet for this horizon (the log fills while the app runs live)."
        log = pd.read_csv(P["live_log"], parse_dates=["time"], on_bad_lines="skip")
        if "profile" in log:
            log = log[log["profile"] == prof]
        log = log.dropna(subset=["time", "p_up"]).drop_duplicates("time", keep="last")
        if log.empty:
            return [], "No predictions logged yet for this horizon."
        hist = self.feed.history(tf, min(20000, PROFILES[prof]["n_bars"]))
        truth = make_labels(make_features(hist, h), h)[["time", "y_dir", "y_vol"]]
        m = log.merge(truth, on="time", how="left")
        pending = int(m["y_dir"].isna().sum())
        m = m.dropna(subset=["y_dir"])
        if m.empty:
            return [], f"{pending} prediction(s) waiting — the {prof} horizon has not passed yet."
        ok = (m["p_up"] >= .5) == (m["y_dir"] == 1)
        rows = [("Direction · all", int(ok.sum()), len(m))]
        if "light" in m:
            for lt in ["GO", "CAUTION", "NO TRADE"]:
                sel = m["light"] == lt
                rows.append((f"Direction · {lt}", int(ok[sel].sum()), int(sel.sum())))
        if "strong" in m:
            sel = m["strong"].astype(str) == "True"
            rows.append(("Direction · STRONG", int(ok[sel].sum()), int(sel.sum())))
        v = m.dropna(subset=["y_vol", "p_vol"])
        rows.append(("Volatility high/low", int(((v["p_vol"] >= .5) == (v["y_vol"] == 1)).sum()), len(v)))
        return rows, (f"{m['time'].min():%d %b %H:%M} → {m['time'].max():%d %b %H:%M} (server time) · "
                      f"{pending} still waiting for the result")

    def refresh_stats(self):
        f = self.stats_frame
        for w in f.winfo_children():
            w.destroy()
        SB = "#0f172a"
        pct = lambda a, b: f"{a * 100 / b:.1f}%" if b else "—"

        RH = 20                                               # row height

        def table(title, head, rows, y, widths):
            tk.Label(f, text=title, fg=GOLD, bg=SB, font=("Segoe UI", 10, "bold"), anchor="w").place(x=12, y=y, width=380)
            y += 22
            for r_i, row in enumerate([head] + rows):
                x = 12
                for c_i, (cell, w) in enumerate(zip(row, widths)):
                    txt, col = cell if isinstance(cell, tuple) else (cell, DIM if r_i == 0 else FG)
                    tk.Label(f, text=txt, fg=col, bg=SB if r_i % 2 == 0 else "#111c33", font=("Segoe UI", 9, "bold" if r_i == 0 else "normal"),
                             anchor="w" if c_i == 0 else "e").place(x=x, y=y, width=w, height=RH)
                    x += w
                y += RH
            return y + 4

        def trade_row(name, t):
            win, pl = sum(1 for p in t if p["pnl"] > 0), sum(p["pnl"] for p in t)
            return [name, str(win), str(len(t) - win), str(len(t)), pct(win, len(t)),
                    (f"{pl:+,.2f}", GREEN if pl > 0 else RED if pl < 0 else FG)]

        tk.Label(f, text="Statistics", fg=FG, bg=SB, font=("Segoe UI", 13, "bold"), anchor="w").place(x=12, y=10, width=250)
        tk.Button(f, text="⟳", command=self.refresh_stats, bg=PANEL, fg=FG, relief="flat", cursor="hand2",
                  font=("Segoe UI", 10, "bold")).place(x=354, y=10, width=32, height=26)
        y = 48
        # --- trades ---
        try:
            trades = self.feed.trade_history()
            auto = [p for p in trades if p["kind"] == "auto"]
            self.fill_auto_profile(auto)
            rows = [trade_row(f"AUTO · {pr}", [p for p in auto if p["profile"] == pr])
                    for pr in PROFILES if any(p["profile"] == pr for p in auto)]
            if any(not p["profile"] for p in auto):
                rows.append(trade_row("AUTO · horizon ?", [p for p in auto if not p["profile"]]))
            rows = (rows or [trade_row("AUTO", [])]) + [trade_row(name, [p for p in trades if p["kind"] in kinds]) for name, kinds in
                    [("App buttons", ("app",)), ("By hand in MT5", ("manual",)),
                     ("All XAUUSD", ("auto", "short", "app", "manual"))]]
            y = table(f"Closed trades · last 90 days ({self.feed.currency})",
                      ["", "Win", "Loss", "Total", "Win %", "Profit/Loss"], rows, y, [112, 44, 44, 48, 56, 78])
            # --- SHORT (1 min scalping) ---
            sh = [p for p in trades if p["kind"] == "short"]
            self.fill_regime(sh, AUTO_SHORT_PROFILE)
            rows = [trade_row("SHORT · all", sh), trade_row("SHORT · BUY", [p for p in sh if p["side"] == "BUY"]),
                    trade_row("SHORT · SELL", [p for p in sh if p["side"] == "SELL"]),
                    trade_row("SHORT · HIGH vol", [p for p in sh if p["regime"] == "HIGH"]),
                    trade_row("SHORT · LOW vol", [p for p in sh if p["regime"] == "LOW"])]
            unknown = sum(1 for p in sh if not p["regime"])
            if unknown:
                rows.append(trade_row("SHORT · vol unknown", [p for p in sh if not p["regime"]]))
            y = table("Short mode · AUTO 1 min", ["", "Win", "Loss", "Total", "Win %", "Profit/Loss"], rows, y + 4,
                      [112, 44, 44, 48, 56, 78])
            try:
                pr, _ = self.prediction_stats(AUTO_SHORT_PROFILE)
                a1, b1 = (pr[0][1], pr[0][2]) if pr else (0, 0)
                y = self.stats_note(f"1-min direction right: {a1}/{b1} = {pct(a1, b1)}", y, 18)
            except Exception:
                pass
            if REPLAY:
                y = self.stats_note("Replay mode: no real trades.", y, 18)
        except Exception as e:
            y = self.stats_note(f"Trades: {e}", y)
        # --- predictions ---
        y += 2
        try:
            rows, note = self.prediction_stats()
            rows = [[n, str(a), str(b - a), str(b), (pct(a, b), FG if not b else GREEN if a * 2 > b else RED)] for n, a, b in rows]
            y = table(f"Was the app right? · {self.profile}", ["", "Right", "Wrong", "Total", "Right %"], rows, y,
                      [150, 52, 52, 56, 72])
            y = self.stats_note(note, y)
        except Exception as e:
            y = self.stats_note(f"Predictions: {e}", y)
        self.stats_note("% = count × 100 ÷ total · small totals are mostly luck, judge after 100+", WIN_H - 26, 20)

    def fill_auto_profile(self, trades):
        """AUTO trades opened before the horizon was written into the order comment: find it in data/auto_trades*.csv
        (the log keeps the ticket; in MT5 a position's id is the ticket of the order that opened it)."""
        todo = [p for p in trades if not p.get("profile")]
        if not todo:
            return
        import glob, re
        lut = {}
        for fpath in glob.glob("data/auto_trades*.csv"):
            try:
                with open(fpath, newline="") as fh:
                    for row in csv.DictReader(fh):
                        m = re.search(r"ticket #(\d+)", row.get("message") or "")
                        if m and row.get("profile"):
                            lut[int(m.group(1))] = row["profile"]
            except OSError:
                continue
        for p in todo:
            p["profile"] = lut.get(p.get("ticket"), "")

    def fill_regime(self, trades, prof):
        """Older AUTO trades have no H/L tag: look up the volatility the app showed for the bar they opened on."""
        todo = [p for p in trades if not p.get("regime") and p.get("time")]
        if not todo:
            return
        path = paths(prof)["live_log"]
        if not os.path.exists(path):
            return
        log = pd.read_csv(path, on_bad_lines="skip")
        if "time" not in log or "p_vol" not in log:
            return
        reg = log["regime"] if "regime" in log else None
        lut = {}
        for i, t in enumerate(pd.to_datetime(log["time"], errors="coerce")):
            if pd.isna(t):
                continue
            r = reg.iloc[i] if reg is not None and isinstance(reg.iloc[i], str) else None
            lut[int(t.timestamp())] = r or ("HIGH" if float(log["p_vol"].iloc[i]) >= .5 else "LOW")
        step = TF_SECONDS[PROFILES[prof]["tf"]]
        for p in todo:
            bar = (p["time"] // step) * step - step            # trade opens just after this bar closed
            for b in (bar, bar - step, bar + step):
                if b in lut:
                    p["regime"] = lut[b]; break

    def stats_note(self, text, y, h=34):
        tk.Label(self.stats_frame, text=text, fg=DIM, bg="#0f172a", font=("Segoe UI", 8), anchor="nw", justify="left",
                 wraplength=376).place(x=12, y=y, width=380, height=h)
        return y + h + 2

    # ---------- AUTO mode ----------
    def on_auto(self):
        if not self.var_auto.get():
            self.lbl_auto.config(text="", fg=DIM); return
        cur = self.current()
        problem = ("AUTO is not available in replay mode." if REPLAY else
                   "AUTO works on DEMO accounts only." if not self.feed.is_demo else
                   "This horizon is not trained yet." if not cur else
                   f"Retrain this horizon first (AUTO needs the traffic light):\npython train.py \"{self.profile}\" --fast"
                   if not cur[2].get("filter") and not self.var_nolight.get() else None)
        if problem:
            self.var_auto.set(False); messagebox.showinfo("AUTO", problem); return
        nolight = self.var_nolight.get()
        need = "any light (Ignore light is ticked)" if nolight else "GO" if AUTO_ONLY_GO else "GO or CAUTION"
        max_tr = AUTO_SHORT_MAX_TRADES_24H if self.var_short.get() else AUTO_MAX_TRADES_24H
        if not nolight and not (cur[2].get("filter_check") or {}).get("helped") and AUTO_ONLY_GO:
            messagebox.showinfo("AUTO", f"On \"{self.profile}\" the traffic light did not prove useful in the "
                                f"out-of-sample check, so it never shows GO and AUTO would never trade.\n\n"
                                f"Tick \"Ignore light\" to trade anyway, or use a horizon where it passed: " + (", ".join(
                                    p for p in PROFILES if trained(p) and
                                    (load_json(paths(p)["stats"]).get("filter_check") or {}).get("helped")) or "none"))
            self.var_auto.set(False); return
        if not messagebox.askokcancel("Turn on AUTO?",
                f"The app will open trades by itself, without asking:\n\n"
                f"• mode: {'SHORT (1 min scalping)' if self.var_short.get() else self.profile}\n"
                f"• once per new bar, when the light is {need}\n"
                f"• lot = the Lot box ({self.var_lot.get()}), TP/SL set automatically\n"
                f"• max {AUTO_MAX_OPEN} open position(s), max {max_tr} trades per 24 h\n"
                f"• switches itself off if XAUUSD loses {AUTO_MAX_LOSS_PCT_24H:g}% of balance in 24 h\n\n"
                f"Account: {self.feed.account}\n\n"
                f"The backtests show no proven edge for direction. Treat this as a test."):
            self.var_auto.set(False); return
        self.lbl_auto.config(text="on · waiting for next bar", fg=GOLD)

    def on_short(self):
        if not self.var_short.get():
            self.lbl_auto.config(text="SHORT off", fg=DIM); return
        if not trained(AUTO_SHORT_PROFILE):
            self.var_short.set(False)
            messagebox.showinfo("SHORT", f'Train it first:  python train.py "{AUTO_SHORT_PROFILE}" --fast'); return
        self.var_profile.set(AUTO_SHORT_PROFILE)
        self.set_profile(AUTO_SHORT_PROFILE)
        self.lbl_auto.config(text=f"SHORT: 1 min scalping · limit {AUTO_SHORT_MAX_TRADES_24H} trades / 24 h", fg=GOLD)

    def on_nolight(self):
        if not self.var_nolight.get():
            return
        if not messagebox.askokcancel("Ignore traffic light?",
                "AUTO will trade every new bar in the direction the model leans, even on NO TRADE.\n\n"
                "In the backtest this lost money on every short horizon (1 min: about -0.32 $ per trade at 0.01 lot).\n"
                "Use it on Demo to collect live statistics, not to make money."):
            self.var_nolight.set(False)

    def auto_trade(self):
        if not self.var_auto.get() or not self.signal:
            return
        try:
            (open_auto, open_short), (n_auto, n_short), pnl, bal = self.feed.auto_stats()
        except Exception as e:
            self.lbl_auto.config(text=f"error: {e}", fg=RED); return
        short, nolight = self.var_short.get(), self.var_nolight.get()
        n_tr, max_tr = (n_short, AUTO_SHORT_MAX_TRADES_24H) if short else (n_auto, AUTO_MAX_TRADES_24H)
        n_open = open_short if short else open_auto            # SHORT and normal AUTO don't block each other
        tail = f"{'SHORT ' if short else ''}{n_tr}/{max_tr} · 24h {pnl:+.2f}"
        if pnl <= -bal * AUTO_MAX_LOSS_PCT_24H / 100:
            self.var_auto.set(False)
            self.lbl_auto.config(text=f"STOPPED: 24h loss limit · {pnl:+.2f}", fg=RED); return
        lt = getattr(self, "light", (None,))[0]
        skip = ("light not GO" if not nolight and (lt is None or lt < (2 if AUTO_ONLY_GO else 1)) else
                "position open" if n_open >= AUTO_MAX_OPEN else
                "max trades" if n_tr >= max_tr else None)
        if skip:
            hint = " · resets as old trades pass 24 h" if skip == "max trades" else ""
            self.lbl_auto.config(text=f"on · skip ({skip}) · {tail}{hint}", fg=RED if skip == "max trades" else GOLD); return
        try:
            lot = round(float(self.var_lot.get()), 2)
        except ValueError:
            self.lbl_auto.config(text="bad lot value", fg=RED); return
        s = self.signal
        ok, msg = self.feed.place(s["side"], lot, s["regime"], s["atr"], PROFILES[self.profile]["h"],
                                  comment="GIVE ME A!! AUTO" + (" SHORT" if short else " " + slug(self.profile)) + (" NL" if nolight else "")
                                  + (" H" if s["regime"] == "HIGH" else " L"))       # volatility at entry, for the stats
        self.lbl_auto.config(text=f"{'opened ' + s['side'] if ok else 'order failed'} · {tail}", fg=GREEN if ok else RED)
        os.makedirs("data", exist_ok=True)
        path, head = "data/auto_trades.csv", ["time", "profile", "side", "lot", "p_up", "regime", "light", "short",
                                              "ignore_light", "ok", "message"]
        if os.path.exists(path):
            with open(path, newline="") as fh:
                old_head = next(csv.reader(fh), [])
            if old_head != head:                                 # older column layout -> keep it aside (file closed first:
                try:                                             #  Windows can't rename an open file)
                    os.replace(path, path.replace(".csv", f"_old_{datetime.now():%Y%m%d%H%M%S}.csv"))
                except OSError:                                  # e.g. open in Excel -> write to a side file instead
                    path = path.replace(".csv", "_v2.csv")
        new = not os.path.exists(path)
        try:
            fh = open(path, "a", newline="")
        except OSError as e:                                     # file locked (e.g. open in Excel): skip the log, keep trading
            self.lbl_auto.config(text=f"{self.lbl_auto.cget('text')} · log skipped ({e.__class__.__name__})"); return
        with fh:
            w = csv.writer(fh)
            if new:
                w.writerow(head)
            w.writerow([datetime.now(), self.profile, s["side"], lot, s["p_up"], s["regime"], LIGHTS.get(lt, ""),
                        short, nolight, ok, msg.replace("\n", " | ")])

    # ---------- orders ----------
    def order(self, side):
        if not self.signal:
            messagebox.showinfo("AUI", "No prediction yet."); return
        try:
            lot = round(float(self.var_lot.get()), 2)
        except ValueError:
            messagebox.showerror("AUI", "Lot must be a number, e.g. 0.01"); return
        s, h = self.signal, PROFILES[self.profile]["h"]
        bid, ask = self.feed.tick()
        price = ask if side == "BUY" else bid
        tp, sl = tp_sl(price, s["atr"], side, s["regime"], h)
        note = f"\n\n⚠ The AI leans {s['side']} right now, not {side}." if side != s["side"] else ""
        lt = getattr(self, "light", (None, []))[0]
        if side == s["side"] and lt == 0:
            note += "\n\n🔴 Traffic light says NO TRADE — situations like this lost money in the backtest."
        elif side == s["side"] and lt == 1:
            note += "\n\n🟡 Traffic light says CAUTION."
        elif lt is None and not s["strong"]:
            note += "\n\n⚠ WEAK signal — close to 50/50, higher risk."
        if not messagebox.askyesno("Confirm order",
                                   f"{side} {lot} lot XAUUSD @ ~{price:,.2f}\n\nTP {tp:,.2f}   ({abs(tp - price):,.2f} $)\n"
                                   f"SL {sl:,.2f}   ({abs(sl - price):,.2f} $)\n\nAccount: {self.feed.account}{note}"):
            return
        ok, msg = self.feed.place(side, lot, s["regime"], s["atr"], h)
        (messagebox.showinfo if ok else messagebox.showerror)("Order sent" if ok else "Order failed", msg)

    # ---------- Hugging Face models (background threads; UI updated in check_hf) ----------
    def start_hf(self, bars):
        if not hf_models.AVAILABLE:
            return
        if not self.hf_busy:
            closes, prof = bars["close"].to_numpy(dtype=float), self.profile
            h = PROFILES[prof]["h"]
            self.hf_busy = True
            if hf_models._chronos is None:
                self.lbl_chr.config(text="🤗 Chronos: loading model (first time downloads ~200 MB)…", fg=GOLD)

            def work():
                try:
                    self.hf_result = ("ok", prof, hf_models.forecast(closes, h))
                except Exception as e:
                    self.hf_result = ("err", prof, e)
            threading.Thread(target=work, daemon=True).start()
        if hf_models.YOLO_AVAILABLE and not self.yolo_busy:
            snap, prof = bars.copy(), self.profile
            self.yolo_busy = True
            if hf_models._yolo is None:
                self.lbl_pat.config(text="📈 YOLO: loading chart-pattern model…", fg=GOLD)

            def pat():
                try:
                    self.yolo_result = ("ok", prof, hf_models.detect_patterns(snap))
                except Exception as e:
                    self.yolo_result = ("err", prof, e)
            threading.Thread(target=pat, daemon=True).start()
        if not self.news_busy and datetime.now().timestamp() >= self.news_next:
            self.news_busy = True
            if hf_models._finbert is None:
                self.lbl_news.config(text="📰 FinBERT: loading model + reading news…", fg=GOLD)

            def news():
                try:
                    self.news_result = ("ok", hf_models.news_mood())
                except Exception as e:
                    self.news_result = ("err", e)
            threading.Thread(target=news, daemon=True).start()

    def check_hf(self):
        if self.hf_result is not None:
            status, prof, r = self.hf_result
            self.hf_result, self.hf_busy = None, False
            if prof == self.profile:
                if status == "err":
                    self.lbl_chr.config(text=f"🤗 Chronos error: {r}", fg=RED)
                else:
                    up = r["p_up"] >= .5
                    self.chronos_last = dict(profile=prof, bar=self.last_bar, p_up=r["p_up"])
                    self.update_light()
                    self.lbl_chr.config(fg=GREEN if up else RED, text=(
                        f"🤗 Chronos (ready-made) next {prof}: {'▲' if up else '▼'} {r['median']:,.2f} "
                        f"({r['change_pct']:+.2%})  ·  range {r['q10']:,.0f}–{r['q90']:,.0f}  ·  P(up) {r['p_up']:.0%}"))
        if self.news_result is not None:
            status, r = self.news_result
            self.news_result, self.news_busy = None, False
            self.news_next = datetime.now().timestamp() + 600          # refresh news every 10 min
            if status == "err":
                self.lbl_news.config(text=f"📰 FinBERT error: {r}", fg=RED)
            else:
                self.news = r
                col = GREEN if r["label"] == "Positive" else RED if r["label"] == "Negative" else GREY
                self.lbl_news.config(fg=col, text=f"📰 News mood (FinBERT): {r['label']} {r['score']:+.2f}  ·  "
                                                  f"{r['n']} headlines, last 24 h  ·  click to read")

    def check_yolo(self):
        if self.yolo_result is None:
            return
        status, prof, r = self.yolo_result
        self.yolo_result, self.yolo_busy = None, False
        if prof != self.profile:
            return
        if status == "err":
            self.lbl_pat.config(text=f"📈 YOLO error: {r}", fg=RED); return
        pats, img = r
        self.patterns = (pats, img)
        if not pats:
            self.lbl_pat.config(text="📈 Chart pattern (YOLO): none found in the last 100 candles · click to view", fg=GREY)
            return
        p = pats[0]
        col = GREEN if "bullish" in p["hint"] else RED if "bearish" in p["hint"] else GOLD
        more = f"  (+{len(pats) - 1} more)" if len(pats) > 1 else ""
        self.lbl_pat.config(fg=col if p["recent"] else GREY, text=(
            f"📈 Chart pattern (YOLO): {p['name']} {p['conf']:.0%} → {p['hint']}"
            f"{'' if p['recent'] else ' (older part of chart)'}{more} · click to view"))

    def show_pattern_image(self):
        if not self.patterns:
            return
        pats, img = self.patterns
        w = tk.Toplevel(self.root, bg=BG); w.title("Chart patterns — YOLOv8 (Hugging Face)")
        w.attributes("-topmost", True)
        im = Image.open(img); im.thumbnail((800, 640))
        w.photo = ImageTk.PhotoImage(im)
        tk.Label(w, image=w.photo, bg=BG).pack()
        txt = "\n".join(f"{p['name']}  {p['conf']:.0%}  · {p['hint']}{'' if p['recent'] else '  (older)'}" for p in pats) or "No pattern found"
        tk.Label(w, text=txt + "\n\nPatterns are hints, not guarantees.", fg=FG, bg=BG, font=("Segoe UI", 10),
                 justify="left").pack(padx=10, pady=8, anchor="w")

    # ---------- close positions ----------
    def close(self, which):
        pos = self.feed.positions()
        pick = {"profit": lambda p: p.profit + p.swap > 0, "loss": lambda p: p.profit + p.swap < 0,
                "all": lambda p: True}[which]
        sel = [p for p in pos if pick(p)]
        if not sel:
            messagebox.showinfo("AUI", f"No {'' if which == 'all' else which + ' '}positions to close."); return
        total = sum(p.profit + p.swap for p in sel)
        label = {"profit": "in PROFIT", "loss": "in LOSS", "all": "ALL"}[which]
        if not messagebox.askyesno("Confirm close",
                                   f"Close {len(sel)} XAUUSD position(s) {label}?\n\nP/L now: {total:+,.2f} {self.feed.currency}\n"
                                   f"Account: {self.feed.account}"):
            return
        ok, msg = self.feed.close_positions(which)
        (messagebox.showinfo if ok else messagebox.showerror)("Close positions", msg)
        self.update_positions()

    def update_positions(self):
        try:
            pos = self.feed.positions()
        except Exception:
            return
        if not pos:
            self.lbl_pos.config(text="Open: none", fg=DIM); return
        pl = sum(p.profit + p.swap for p in pos)
        nb = sum(1 for p in pos if p.type == 0)
        self.lbl_pos.config(fg=GREEN if pl >= 0 else RED,
                            text=f"Open: {len(pos)} ({nb} buy / {len(pos) - nb} sell) · P/L {pl:+,.2f} {self.feed.currency}")

    def show_headlines(self):
        r = getattr(self, "news", None)
        if not r or not r["top"]:
            return
        messagebox.showinfo("Latest headlines (FinBERT)",
                            "\n\n".join(f"[{lab}] {t}" for t, lab in r["top"]) +
                            "\n\nPositive news is about the market/economy, not always good for gold.")

    # ---------- self-training ----------
    def start_retrain(self):
        if self.training or self.current() is None:
            return
        prof, p = self.profile, PROFILES[self.profile]
        try:
            df = self.feed.history(p["tf"], min(self_train.RETRAIN_BARS, p["n_bars"]))   # MT5 read on the UI thread
        except Exception as e:
            self.lbl_train.config(text=f"self-train: could not get data ({e})", fg=RED); return
        self.training, self.bars_since_train = True, 0
        self.btn_train.config(state="disabled", text="training…")
        self.lbl_train.config(text=f"self-train ({prof}): learning from {len(df):,} latest bars…", fg=GOLD)
        dir_model, vol_model, stats = self.current()
        try:
            ex = self.feed.extras(p["tf"], len(df) + 50, stats.get("extras", []))
        except Exception:
            ex = {}

        def work():
            try:
                self.train_result = ("ok", prof, self_train.retrain(df, dir_model, vol_model, stats, prof, ex))
            except Exception as e:
                self.train_result = ("err", prof, e)
        threading.Thread(target=work, daemon=True).start()

    def check_retrain(self):
        if not self.training or self.train_result is None:
            return
        status, prof, payload = self.train_result
        self.train_result, self.training = None, False
        self.btn_train.config(state="normal", text="⟳  Retrain now")
        if status == "err":
            self.lbl_train.config(text=f"self-train failed: {payload}", fg=RED); return
        new_dir, new_vol, new_stats, report = payload
        self.models[prof] = (new_dir, new_vol, new_stats)
        self_train.save(new_dir, new_vol, new_stats, report)
        if prof == self.profile:
            self.last_bar = None                                # re-predict right away with the new models
        self.show_train_status()

    def show_train_status(self):
        cur = self.current()
        r = cur[2].get("last_retrain") if cur else None
        if not r:
            self.lbl_train.config(fg=DIM, text=f"self-train: auto every {self_train.RETRAIN_EVERY_BARS} bars · "
                                               f"{self_train.RETRAIN_EVERY_BARS - self.bars_since_train} to go")
            return
        part = lambda k, n: (f"{n} {r[k + '_auc_old']:.3f}→{r[k + '_auc_new']:.3f} "
                             f"{'✔ updated' if r[k + '_swap'] else '✘ kept old'}")
        self.lbl_train.config(text=f"self-train {r['time'][11:16]} · AUC {part('dir', 'A')} · {part('vol', 'B')}",
                              fg=GREEN if (r["dir_swap"] or r["vol_swap"]) else DIM)

    # ---------- live price ----------
    def update_tick(self):
        try:
            bid, ask = self.feed.tick()
            self.lbl_price.config(text=f"{bid:,.2f}")
            self.lbl_ba.config(text=f"BID {bid:,.2f}   ASK {ask:,.2f}   SPREAD {(ask - bid) / POINT:.0f} pts")
            s = self.signal
            if s:
                px = bid if s["tp"] > s["sl"] else ask        # price at which the trade would close
                self.lbl_tp_d.config(text=f"{abs(s['tp'] - px):,.2f} $ away")
                self.lbl_sl_d.config(text=f"{abs(s['sl'] - px):,.2f} $ away")
        except Exception:
            pass
        self.ticks += 1
        if self.ticks % 4 == 0:
            self.update_positions()
        self.root.after(250, self.update_tick)

    # collect new closed bars while running (merged by train.py next time you retrain)
    def log(self, f, p_up, p_vol, side, conf, tp, sl, regime=None, strong=None):
        os.makedirs("data", exist_ok=True)
        path = paths(self.profile)["live_log"]
        row = f[["time", "open", "high", "low", "close", "tick_volume", "spread"]].iloc[0].to_dict()
        row.update(p_up=p_up, p_vol=p_vol, signal=side, strong=strong, regime=regime, conf=conf, tp=tp, sl=sl,
                   logged_at=datetime.now(), profile=self.profile,
                   light=LIGHTS.get(getattr(self, "light", (None,))[0], ""))
        if os.path.exists(path):                                   # header from an older app version -> start a new file
            with open(path, newline="") as fh:
                header = next(csv.reader(fh), [])
            if header != list(row):
                os.replace(path, path.replace(".csv", f"_old_{datetime.now():%Y%m%d%H%M%S}.csv"))
        new = not os.path.exists(path)
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)


if __name__ == "__main__":
    root = tk.Tk(); root.withdraw()
    def start():
        root.deiconify(); App(root)
    intro(root, start)
    root.mainloop()
