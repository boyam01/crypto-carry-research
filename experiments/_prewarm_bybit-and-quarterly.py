"""Data prewarm worker: Bybit funding+klines for top-8 coins, Binance USDT-M dated
quarterlies for BTC/ETH, and a dapi COIN-M quarterly availability probe for XRP/BNB/SOL.
Does NOT compute strategies. Just warms the parquet cache and reports row counts.

Run:  cd "D:/量化交易CLAUDE" && python experiments/_prewarm_bybit-and-quarterly.py
"""
import sys, time, json, traceback
import requests
sys.path.insert(0, '.')
from engine import fetch_binance as fb
from engine import fetch_funding as ff

def ms(y, m, d):
    import datetime as dt
    return int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp() * 1000)

START = ms(2023, 1, 1)
END   = int(time.time() * 1000)

report = {"bybit_funding": {}, "bybit_klines_4h": {}, "binance_quarterly_1d": {},
          "dapi_coinm_probe": {}, "errors": []}

# ---- top 8 coins (strip USDT suffix to get base symbols) ----
try:
    uni = fb.exchange_universe(min_quote_vol=5e7, top=40)
except Exception as e:
    uni = []
    report["errors"].append(f"exchange_universe: {e}")

# Fallback to a sane top-8 if universe call is thin
fallback = ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","BNBUSDT","DOGEUSDT","ADAUSDT","LINKUSDT"]
top8 = (uni[:8] if len(uni) >= 8 else fallback)
report["top8"] = top8
print("TOP8:", top8)

# ---- 1) Bybit funding + 4h klines for top 8 (Bybit perp symbols == USDT perps) ----
for sym in top8:
    # funding
    try:
        df = ff.bybit_funding(sym, START, END)
        report["bybit_funding"][sym] = int(len(df))
        print(f"[by_fr] {sym}: {len(df)} rows  {df.index.min()}..{df.index.max()}" if len(df) else f"[by_fr] {sym}: 0 rows")
    except Exception as e:
        report["bybit_funding"][sym] = f"ERR {e}"
        report["errors"].append(f"bybit_funding {sym}: {e}")
        print(f"[by_fr] {sym}: ERR {e}")
    # 4h klines (interval_min=240)
    try:
        dfk = ff.bybit_klines(sym, 240, START, END)
        report["bybit_klines_4h"][sym] = int(len(dfk))
        print(f"[by_kl] {sym}: {len(dfk)} rows  {dfk.index.min()}..{dfk.index.max()}" if len(dfk) else f"[by_kl] {sym}: 0 rows")
    except Exception as e:
        report["bybit_klines_4h"][sym] = f"ERR {e}"
        report["errors"].append(f"bybit_klines {sym}: {e}")
        print(f"[by_kl] {sym}: ERR {e}")

# ---- 2) Binance USDT-M dated quarterly 1d klines for BTC & ETH ----
CODES = [250328, 250627, 250926, 251226, 260327, 260626, 260925]
for coin in ["BTCUSDT", "ETHUSDT"]:
    for code in CODES:
        s = f"{coin}_{code}"
        try:
            df = fb.klines(s, "1d", START, END, futures=True)
            report["binance_quarterly_1d"][s] = int(len(df))
            rng = f"{df.index.min()}..{df.index.max()}" if len(df) else "-"
            print(f"[q1d] {s}: {len(df)} rows  {rng}")
        except Exception as e:
            report["binance_quarterly_1d"][s] = f"ERR {e}"
            print(f"[q1d] {s}: ERR {e}")

# ---- 3) Probe dapi COIN-M quarterly availability for XRP/BNB/SOL (one code each) ----
# COIN-M dated symbols look like e.g. XRPUSD_250627
DAPI = "https://dapi.binance.com/dapi/v1/klines"
for coin in ["XRP", "BNB", "SOL"]:
    found = None
    for code in CODES:
        s = f"{coin}USD_{code}"
        try:
            r = requests.get(DAPI, params=dict(symbol=s, interval="1d", limit=10), timeout=20)
            ok = r.status_code == 200
            n = len(r.json()) if ok and isinstance(r.json(), list) else 0
            if ok and n > 0:
                found = {"symbol": s, "rows": n}
                print(f"[dapi] {s}: {n} rows (AVAILABLE)")
                break
            else:
                print(f"[dapi] {s}: status={r.status_code} rows={n}")
        except Exception as e:
            print(f"[dapi] {s}: ERR {e}")
        time.sleep(0.25)
    report["dapi_coinm_probe"][coin] = found or "none of probed codes returned rows"

print("\n===== JSON REPORT =====")
print(json.dumps(report, indent=2, default=str))
