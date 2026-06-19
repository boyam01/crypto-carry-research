"""Funding-rate history from Bybit and OKX (USDT perps), parquet-cached.
Pairs with engine.fetch_binance.funding_rate for cross-venue work. All public.
"""
from __future__ import annotations
import time, hashlib, pathlib
import requests
import pandas as pd

CACHE = pathlib.Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
_S = requests.Session(); _S.headers.update({"User-Agent": "quant-research/0.1"})


def _get(url, params, tries=5):
    for i in range(tries):
        r = _S.get(url, params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
        time.sleep(1.5 ** i)
    r.raise_for_status()


def _cached(tag, builder):
    p = CACHE / f"{hashlib.md5(tag.encode()).hexdigest()[:16]}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    df = builder()
    if df is not None and len(df):
        df.to_parquet(p)
    return df


def bybit_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    tag = f"by_fr|{symbol}|{start_ms}|{end_ms}"

    def build():
        # Bybit returns newest-first, capped 200/page -> walk BACKWARD via endTime.
        rows, end_cur = [], end_ms
        for _ in range(400):
            j = _get("https://api.bybit.com/v5/market/funding/history",
                     dict(category="linear", symbol=symbol, startTime=start_ms,
                          endTime=end_cur, limit=200))
            lst = j.get("result", {}).get("list", [])
            if not lst:
                break
            rows.extend(lst)
            oldest = min(int(x["fundingRateTimestamp"]) for x in lst)
            if oldest <= start_ms or len(lst) < 200:
                break
            end_cur = oldest - 1
            time.sleep(0.15)
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["fundingRate"] = pd.to_numeric(df["fundingRate"])
        df.index = pd.to_datetime(df["fundingRateTimestamp"].astype("int64"), unit="ms", utc=True)
        return df[["fundingRate"]][~df.index.duplicated()].sort_index()

    return _cached(tag, build)


def okx_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """symbol like BTCUSDT -> instId BTC-USDT-SWAP. OKX paginates via `after`."""
    inst = symbol[:-4] + "-USDT-SWAP"
    tag = f"okx_fr|{inst}|{start_ms}|{end_ms}"

    def build():
        rows, after = [], end_ms
        for _ in range(400):                      # hard page cap
            j = _get("https://www.okx.com/api/v5/public/funding-rate-history",
                     dict(instId=inst, after=after, limit=100))
            data = j.get("data", [])
            if not data:
                break
            rows.extend(data)
            oldest = min(int(d["fundingTime"]) for d in data)
            if oldest <= start_ms or len(data) < 100:
                break
            after = oldest
            time.sleep(0.12)
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["fundingRate"] = pd.to_numeric(df["fundingRate"])
        df.index = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
        df = df[(df.index >= pd.to_datetime(start_ms, unit="ms", utc=True))]
        return df[["fundingRate"]][~df.index.duplicated()].sort_index()

    return _cached(tag, build)


def bybit_klines(symbol: str, interval_min: int, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Bybit linear-perp klines. interval_min e.g. 480 for 8h. Newest-first, walk back."""
    tag = f"by_kl|{symbol}|{interval_min}|{start_ms}|{end_ms}"

    def build():
        rows, end_cur = [], end_ms
        for _ in range(400):
            j = _get("https://api.bybit.com/v5/market/kline",
                     dict(category="linear", symbol=symbol, interval=interval_min,
                          start=start_ms, end=end_cur, limit=1000))
            lst = j.get("result", {}).get("list", [])
            if not lst:
                break
            rows.extend(lst)
            oldest = min(int(x[0]) for x in lst)
            if oldest <= start_ms or len(lst) < 1000:
                break
            end_cur = oldest - 1
            time.sleep(0.15)
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v", "to"])
        df["close"] = pd.to_numeric(df["c"])
        df.index = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True)
        return df[["close"]][~df.index.duplicated()].sort_index()

    return _cached(tag, build)


if __name__ == "__main__":
    end = int(time.time() * 1000); start = end - 90 * 24 * 3600 * 1000
    b = bybit_funding("BTCUSDT", start, end)
    o = okx_funding("BTCUSDT", start, end)
    print("bybit:", b.shape, "okx:", o.shape)
    j = b.join(o, lsuffix="_by", rsuffix="_okx").dropna()
    sp = (j["fundingRate_okx"] - j["fundingRate_by"]) * 1e4
    print("aligned rows:", len(j), "| OKX-Bybit funding spread bp/8h: mean %.2f std %.2f" % (sp.mean(), sp.std()))
    assert len(b) > 100 and len(o) > 100 and len(j) > 80
    print("SELF-CHECK OK")
