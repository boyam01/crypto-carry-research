"""Prewarm short-history microstructure data.

For the top-10 USDT-M perp coins, fetch Binance:
  - openInterestHist  period=1h limit=500  (~20-30d history only)
  - takerlongshortRatio period=1h limit=500 (~30d)
Save raw to data/cache/oi_<coin>.parquet and taker_<coin>.parquet.
Also fetch Coinbase BTC-USD & ETH-USD daily candles and Kraken XBTUSDT daily.
Report actual date spans. NO strategy computation.
"""
from __future__ import annotations
import sys, time, pathlib
sys.path.insert(0, ".")
import requests
import pandas as pd

from engine import fetch_binance as fb

CACHE = pathlib.Path("data/cache")
CACHE.mkdir(parents=True, exist_ok=True)
FUT = "https://fapi.binance.com"
S = requests.Session()
S.headers.update({"User-Agent": "quant-research/0.1"})


def _get(url, params, tries=5):
    for i in range(tries):
        r = S.get(url, params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (418, 429):
            time.sleep(2 ** i)
            continue
        r.raise_for_status()
    r.raise_for_status()


def span(df, col="ts"):
    if df is None or not len(df):
        return "EMPTY"
    a, b = df[col].iloc[0], df[col].iloc[-1]
    days = (b - a).total_seconds() / 86400
    return f"{a:%Y-%m-%d %H:%M} -> {b:%Y-%m-%d %H:%M} UTC ({len(df)} rows, {days:.1f}d)"


def fetch_oi(sym):
    data = _get(FUT + "/futures/data/openInterestHist",
                dict(symbol=sym, period="1h", limit=500))
    if not data:
        return None
    df = pd.DataFrame(data)
    for c in ("sumOpenInterest", "sumOpenInterestValue"):
        if c in df:
            df[c] = pd.to_numeric(df[c])
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def fetch_taker(sym):
    data = _get(FUT + "/futures/data/takerlongshortRatio",
                dict(symbol=sym, period="1h", limit=500))
    if not data:
        return None
    df = pd.DataFrame(data)
    for c in ("buySellRatio", "buyVol", "sellVol"):
        if c in df:
            df[c] = pd.to_numeric(df[c])
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def fetch_coinbase(product):
    # granularity 86400 = 1d ; returns [time, low, high, open, close, volume]
    data = _get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                dict(granularity=86400))
    if not data:
        return None
    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def fetch_kraken(pair="XBTUSDT", interval=1440):
    j = _get("https://api.kraken.com/0/public/OHLC",
             dict(pair=pair, interval=interval))
    if j.get("error"):
        print("  kraken error:", j["error"])
    res = j.get("result", {})
    key = next((k for k in res if k != "last"), None)
    if not key:
        return None
    cols = ["time", "open", "high", "low", "close", "vwap", "volume", "count"]
    df = pd.DataFrame(res[key], columns=cols)
    df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def coin(sym):
    # strip USDT suffix for filename
    return sym[:-4] if sym.endswith("USDT") else sym


def main():
    top = fb.exchange_universe(top=10)
    print("TOP-10:", top)
    print("=" * 70)
    rep = {}
    for sym in top:
        c = coin(sym)
        oi = fetch_oi(sym)
        time.sleep(0.15)
        tk = fetch_taker(sym)
        time.sleep(0.15)
        if oi is not None and len(oi):
            oi.to_parquet(CACHE / f"oi_{c}.parquet")
        if tk is not None and len(tk):
            tk.to_parquet(CACHE / f"taker_{c}.parquet")
        oi_s = span(oi)
        tk_s = span(tk)
        rep[c] = (oi_s, tk_s)
        print(f"{c:>8}  OI : {oi_s}")
        print(f"{'':>8}  TLS: {tk_s}")
    print("=" * 70)
    cb_btc = fetch_coinbase("BTC-USD")
    cb_btc.to_parquet(CACHE / "coinbase_BTC-USD_1d.parquet")
    print("Coinbase BTC-USD 1d:", span(cb_btc))
    cb_eth = fetch_coinbase("ETH-USD")
    cb_eth.to_parquet(CACHE / "coinbase_ETH-USD_1d.parquet")
    print("Coinbase ETH-USD 1d:", span(cb_eth))
    kr = fetch_kraken("XBTUSDT", 1440)
    if kr is not None and len(kr):
        kr.to_parquet(CACHE / "kraken_XBTUSDT_1d.parquet")
    print("Kraken  XBTUSDT 1d:", span(kr))
    print("=" * 70)
    print("DONE. Saved oi_/taker_ per coin + coinbase/kraken daily to data/cache/")


if __name__ == "__main__":
    main()
