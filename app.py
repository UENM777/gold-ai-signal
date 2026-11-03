"""Gold AI Signal window.
    python app.py            -> live from MT5 (market must be open)
    python app.py --replay   -> replays recent history from the CSVs (for recording the demo on weekends)
"""
import os, sys, json, csv, threading
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import pandas as pd
from PIL import Image, ImageTk
from pycaret.classification import load_model
from common import *
import self_train

REPLAY = "--replay" in sys.argv
ALLOW_REAL_ACCOUNT = False          # safety: the BUY/SELL buttons only work on a DEMO account
POINT = 0.01
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

    def place(self, side, lot, regime, atr_val, h):
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
                   deviation=30, magic=MAGIC, comment="GIVE ME A!! AI", type_time=mt5.ORDER_TIME_GTC,
                   type_filling=fill)
        res = mt5.order_send(req)
        if res is None:
            return False, f"order_send failed: {mt5.last_error()}"
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"Rejected ({res.retcode}): {res.comment}"
        return True, f"{side} {lot} lot @ {res.price:,.2f}\nTP {req['tp']:,.2f}   SL {req['sl']:,.2f}\nticket #{res.order}"


class ReplayFeed:
    sym, account, is_demo = "XAUUSD (replay)", "REPLAY", False

    def __init__(self):
        self.dfs, self.i, self.cur = {}, {}, None

    def _df(self, tf):
        if tf not in self.dfs:
            self.dfs[tf] = pd.read_csv(data_csv(tf), parse_dates=["time"])
            self.i[tf] = max(300, len(self.dfs[tf]) - 1500)
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
        return df.iloc[self.i[tf] - n + 1:self.i[tf] + 1].reset_index(drop=True)

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
        tk.Label(box, text="Gold AI Signal", fg=FG, bg="black", font=("Segoe UI Light", 18)).pack(pady=(14, 0))
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

    def set_profile(self, name):
        self.profile = name
        self.last_bar, self.signal, self.bars_since_train = None, None, 0
        self.show_wait()
        if self.current() is None:
            self.lbl_foot.config(text=f'"{name}" is not trained yet → run:  python train.py "{name}"', fg=GOLD)
        self.show_train_status()

    # ---------- layout ----------
    def build(self):
        r = self.root
        r.title("Gold AI Signal — GIVE ME A!!")
        r.configure(bg=BG); r.geometry("640x480"); r.resizable(False, False)
        r.attributes("-topmost", True)
        L = lambda parent, txt="", size=12, fg=FG, bold=False: tk.Label(
            parent, text=txt, fg=fg, bg=BG, font=("Segoe UI", size, "bold" if bold else "normal"), anchor="w")

        # top bar: horizon dropdown + account
        top = tk.Frame(r, bg=BG); top.place(x=24, y=10, width=596, height=34)
        L(top, "Predict next", 10, DIM).pack(side="left")
        self.var_profile = tk.StringVar(value=self.profile)
        labels = [p if trained(p) else f"{p}  (not trained)" for p in PROFILES]
        menu = tk.OptionMenu(top, self.var_profile, *labels, command=lambda v: self.set_profile(v.split("  (")[0]))
        menu.config(bg=PANEL, fg=GOLD, activebackground="#374151", activeforeground=GOLD, relief="flat",
                    highlightthickness=0, font=("Segoe UI", 10, "bold"), width=16, cursor="hand2")
        menu["menu"].config(bg=PANEL, fg=FG, activebackground="#374151", font=("Segoe UI", 10))
        menu.pack(side="left", padx=8)
        L(top, self.feed.account, 9, GREEN if self.feed.is_demo else DIM).pack(side="right")

        left = tk.Frame(r, bg=BG); left.place(x=24, y=50, width=330, height=270)
        right = tk.Frame(r, bg=BG); right.place(x=370, y=50, width=250, height=270)

        self.lbl_sym = L(left, self.feed.sym, 10, DIM); self.lbl_sym.pack(anchor="w")
        self.lbl_price = L(left, "—", 34, FG, True); self.lbl_price.pack(anchor="w")
        self.lbl_ba = L(left, "", 10, DIM); self.lbl_ba.pack(anchor="w")
        self.lbl_conf = L(left, "—", 44, FG, True); self.lbl_conf.pack(anchor="w", pady=(6, 0))
        sig_row = tk.Frame(left, bg=BG); sig_row.pack(anchor="w", fill="x")
        self.lbl_sig = L(sig_row, "…", 38, GREY, True); self.lbl_sig.pack(side="left")
        self.lbl_strength = L(sig_row, "", 11, GREY, True); self.lbl_strength.pack(side="left", padx=(12, 0), pady=(14, 0))

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
        self.lbl_foot.place(x=24, y=378, width=596)
        self.lbl_train = tk.Label(r, text="", fg=DIM, bg=BG, font=("Segoe UI", 9), anchor="w")
        self.lbl_train.place(x=24, y=416, width=450)
        self.btn_train = tk.Button(r, text="⟳  Retrain now", command=self.start_retrain, bg=PANEL, fg=FG,
                                   activebackground="#374151", activeforeground=FG, relief="flat",
                                   font=("Segoe UI", 9, "bold"), cursor="hand2")
        self.btn_train.place(x=490, y=410, width=130, height=30)
        tk.Label(r, text="Decision support for a demo account — not financial advice.", fg="#4b5563", bg=BG,
                 font=("Segoe UI", 8), anchor="w").place(x=24, y=452, width=596)

        self.set_profile(self.profile)
        self.update_bar()
        self.update_tick()

    # ---------- prediction loop ----------
    def update_bar(self):
        try:
            if self.current() is not None:
                tf = PROFILES[self.profile]["tf"]
                bars = self.feed.bars(tf, 300)
                self.ex = self.feed.extras(tf, 330, self.current()[2].get("extras", []))
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
                elif t != self.last_bar:                           # closed-bar mode: once per new bar
                    first = self.last_bar is None
                    self.last_bar = t
                    out = self.compute(bars)
                    if out:
                        self.show(*out)
                        if not REPLAY and not first:
                            self.log(*out)
                    if not first:
                        self.bars_since_train += 1
                        if self.bars_since_train >= self_train.RETRAIN_EVERY_BARS:
                            self.start_retrain()
                        elif not self.training:
                            self.show_train_status()
        except Exception as e:
            self.lbl_foot.config(text=f"error: {e}", fg=RED)
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
        self.lbl_strength.config(text="STRONG" if strong else "WEAK · risky", fg=GOLD if strong else GREY)
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

    # ---------- orders ----------
    def order(self, side):
        if not self.signal:
            messagebox.showinfo("Gold AI", "No prediction yet."); return
        try:
            lot = round(float(self.var_lot.get()), 2)
        except ValueError:
            messagebox.showerror("Gold AI", "Lot must be a number, e.g. 0.01"); return
        s, h = self.signal, PROFILES[self.profile]["h"]
        bid, ask = self.feed.tick()
        price = ask if side == "BUY" else bid
        tp, sl = tp_sl(price, s["atr"], side, s["regime"], h)
        note = (f"\n\n⚠ The AI leans {s['side']} right now, not {side}." if side != s["side"] else
                "" if s["strong"] else "\n\n⚠ WEAK signal — close to 50/50, higher risk.")
        if not messagebox.askyesno("Confirm order",
                                   f"{side} {lot} lot XAUUSD @ ~{price:,.2f}\n\nTP {tp:,.2f}   ({abs(tp - price):,.2f} $)\n"
                                   f"SL {sl:,.2f}   ({abs(sl - price):,.2f} $)\n\nAccount: {self.feed.account}{note}"):
            return
        ok, msg = self.feed.place(side, lot, s["regime"], s["atr"], h)
        (messagebox.showinfo if ok else messagebox.showerror)("Order sent" if ok else "Order failed", msg)

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
        self.root.after(250, self.update_tick)

    # collect new closed bars while running (merged by train.py next time you retrain)
    def log(self, f, p_up, p_vol, side, conf, tp, sl, regime=None, strong=None):
        os.makedirs("data", exist_ok=True)
        path = paths(self.profile)["live_log"]
        row = f[["time", "open", "high", "low", "close", "tick_volume", "spread"]].iloc[0].to_dict()
        row.update(p_up=p_up, p_vol=p_vol, signal=side, strong=strong, regime=regime, conf=conf, tp=tp, sl=sl,
                   logged_at=datetime.now(), profile=self.profile)
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
