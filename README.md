# Gold AI Signal — DES423 Section 1 · GIVE ME A!!

## Setup (Windows, Python 3.11) — once per computer
```
cd /d C:\gold_ai                   # your actual folder path, e.g. D:\Projects\gold_ai
py -3.11 -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```
Every time you open a new cmd window:
```
cd /d C:\gold_ai
venv\Scripts\activate.bat          # the line should now start with (venv)
```

## Quick start
```
python export_data.py                                   # 1. download data (MT5 open)
python train.py "1 day" "1 week" "4 hours" "1 hour" --fast --resume   # 2. light horizons first (~10-20 min)
python train.py "15 min" "5 min" "1 min" --fast --resume              #    heavy M1 horizons (slow)
python app.py                                           # 3. open the window
```

## All commands

### export_data.py — download history from MT5
| Command | What it does |
|---|---|
| `python export_data.py` | All timeframes (M1, M5, M15, H1, D1) + EURUSD, USDJPY, XAGUSD |
| `python export_data.py M1 H1` | Only these timeframes |

Run again only when you want newer data or a file is missing. Updating code does **not** need it.

### train.py — compare models, tune, calibrate, backtest, save
| Command | What it does |
|---|---|
| `python train.py` | Default horizon (15 min) |
| `python train.py "15 min" "1 hour"` | Only these horizons (names must match the table below, in quotes) |
| `python train.py --all` | Every horizon (several hours) |

Options (combine freely, e.g. `python train.py --all --fast --resume`):
| Option | What it does |
|---|---|
| `--fast` | ~2-3x faster: 3 folds, 8 tuning rounds, 7 models (skips KNN, Decision Tree) |
| `--resume` | Skip horizons that already have saved models (continue after a crash / shutdown) |
| `--jobs=4` | Use only 4 CPU cores — for running 2 cmd windows at once (give each about half your cores) |
| `--no-extras` | Train on gold only (no EURUSD / USDJPY / silver) — for the before/after comparison |

Retrain when: features or labels changed in `common.py` / `train.py`, or you want fresh results for the slides.
If a line like `[vol] calibrating ...` sits for more than 15 minutes, it's stuck: Ctrl+C and rerun with `--resume`.

### app.py — the signal window
| Command | What it does |
|---|---|
| `python app.py` | Live from MT5 (market must be open) |
| `python app.py --replay` | Replays saved history 1 bar per second (record the demo on weekends; orders disabled) |

Click or press Esc to skip the intro. Only restart the app when just `app.py` / `self_train.py` changed — no retraining needed.

## Horizons (dropdown "Predict next")
| Profile | Reads | Predicts |
|---|---|---|
| 1 min | M1 candles | 1 candle ahead |
| 5 min | M1 | 5 ahead |
| 15 min | M1 | 15 ahead |
| 1 hour | M5 | 12 ahead |
| 4 hours | M15 | 16 ahead |
| 1 day | H1 | 24 ahead |
| 1 week | D1 | 5 ahead |

Each profile has its own models in `models/<profile>/` and figures in `figures/<profile>/`.
`figures/summary.csv` compares all trained profiles in one table (slide: "which horizon is predictable?").

## Extra data (related markets)
- `export_data.py` also downloads EURUSD (US-dollar proxy), USDJPY (follows US yields) and XAGUSD (silver)
  for every timeframe; missing symbols are skipped. Edit `EXTRA_SYMBOLS` in `common.py` to change them
- Session features (Asia / London / New York) come from the bar time; set `SERVER_UTC_OFFSET` if your broker differs
- Before/after for the slides: `python train.py "1 hour" --no-extras --fast` then `python train.py "1 hour" --fast`
  -> both rows in `figures/summary.csv` (column `extras_used`)

## Signals
- Always BUY or SELL (the side P(up) leans to)
- **STRONG** when P(up) is in the top/bottom 10% **and** at least 52% / at most 48%; otherwise **WEAK · risky**
- TP/SL distance = ATR x sqrt(horizon)/2, wider when the volatility model says HIGH
- "hit %" = how often TP was reached before SL in the backtest on unseen data (strong-only, or all bars for weak)

## Orders
- BUY / SELL buttons send a market order with TP/SL to MT5 after a confirmation popup
- Works on DEMO accounts only (`ALLOW_REAL_ACCOUNT = False` in `app.py`)
- Turn on **Algo Trading** in the MT5 toolbar first
- Orders are tagged with magic number 423001 and comment "GIVE ME A!! AI"

## Self-training
- Retrains the selected profile every 60 new bars, or press **Retrain now**
- Refits on the latest bars, tests old vs new on the newest 10% of data, switches only if AUC is not worse
- Keeps the last 3 old models in `models/<profile>/backup/`, history in `models/<profile>/retrain_history.csv`

## Sharing with a teammate (no retraining needed)
1. Zip the whole `gold_ai` folder **except `venv`** (keep `models/`, `data/`, `figures/`)
2. Teammate unzips, then runs the Setup section above (Python 3.11 + `pip install -r requirements.txt`)
3. `python app.py` (needs MT5 + Demo login) or `python app.py --replay` (no MT5 needed)

Library versions are pinned in `requirements.txt` — trained models only load with the same versions.

## Troubleshooting
| Error | Fix |
|---|---|
| `'py' is not recognized` | Reinstall Python 3.11 and tick "Add python.exe to PATH" |
| `The system cannot find the path specified` (activate) | You're in the wrong folder — `cd /d` to the gold_ai folder first |
| `MT5 init failed` | Open MT5 and log in |
| `Gold symbol not found` / `no data` | Open the XAUUSD chart for that timeframe, press Home, set Tools > Options > Charts > Max bars = Unlimited |
| `... not trained yet` in the app | `python train.py "<that horizon>" --fast` |
| Order: "Turn on Algo Trading" | Click **Algo Trading** in the MT5 toolbar (turns green) |
| Order: "Blocked: not a DEMO account" | Log in to a Demo account |

## Team workflow (GitHub + Drive)
- Code lives on GitHub. `venv/`, `models/`, `data/`, `figures/` are in `.gitignore`
- Trained models + data: zip `models/` and `data/` and put them on the team Google Drive
- New teammate: `git clone <repo>` → unzip `models.zip` and `data.zip` into the folder → Setup section above
- Get the latest code: `git pull` (restart `app.py`; retrain only if `common.py` / `train.py` changed)
