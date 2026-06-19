"""Binance public-data fetcher with on-disk parquet cache.

All endpoints used here are PUBLIC (no API key, read-only market data).
ponytail: caching is keyed by (kind, symbol, interval, range) -> one parquet file.
"""
from __future__ import annotations
import time, hashlib, pathlib
import requests
import pandas as pd

SPOT = "https://api.binance.com"
FUT  = "https://fapi.binance.com"          # USDT-M futures
CACHE = pathlib.Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "quant-research/0.1"})


def _get(url: str, params: dict, tries: int = 5):
    for i in range(tries):
        r = _SESSION.get(url, params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (418, 429):          # rate-limited -> back off
            time.sleep(2 ** i)
            continue
        r.raise_for_status()
    r.raise_for_status()


def _cache_path(tag: str) -> pathlib.Path:
    h = hashlib.md5(tag.encode()).hexdigest()[:16]
    return CACHE / f"{h}.parquet"


def _cached(tag: str, builder):
    p = _cache_path(tag)
    if p.exists():
        return pd.read_parquet(p)
    df = builder()
    if df is not None and len(df):
        df.to_parquet(p)
    return df


def klines(symbol: str, interval: str, start_ms: int, end_ms: int,
           futures: bool = False) -> pd.DataFrame:
    """OHLCV klines, paginated. Returns DataFrame indexed by open_time (UTC)."""
    tag = f"kl|{'F' if futures else 'S'}|{symbol}|{interval}|{start_ms}|{end_ms}"

    def build():
        base = FUT if futures else SPOT
        path = "/fapi/v1/klines" if futures else "/api/v3/klines"
        rows, cur = [], start_ms
        while cur < end_ms:
            data = _get(base + path, dict(symbol=symbol, interval=interval,
                                          startTime=cur, endTime=end_ms, limit=1000))
            if not data:
                break
            rows.extend(data)
            last = data[-1][0]
            if last <= cur:
                break
            cur = last + 1
            if len(data) < 1000:
                break
            time.sleep(0.12)
        if not rows:
            return None
        cols = ["open_time", "open", "high", "low", "close", "volume",
                "close_time", "qvol", "trades", "tbbav", "tbqav", "ignore"]
        df = pd.DataFrame(rows, columns=cols)
        for c in ["open", "high", "low", "close", "volume", "qvol", "tbbav", "tbqav"]:
            df[c] = pd.to_numeric(df[c])
        df["trades"] = pd.to_numeric(df["trades"])
        df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df[["open", "high", "low", "close", "volume", "qvol", "trades", "tbbav", "tbqav"]]
        return df[~df.index.duplicated()].sort_index()

    return _cached(tag, build)


def funding_rate(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Realized perpetual funding-rate history (paid every 8h). USDT-M."""
    tag = f"fr|{symbol}|{start_ms}|{end_ms}"

    def build():
        rows, cur = [], start_ms
        while cur < end_ms:
            data = _get(FUT + "/fapi/v1/fundingRate",
                        dict(symbol=symbol, startTime=cur, endTime=end_ms, limit=1000))
            if not data:
                break
            rows.extend(data)
            last = data[-1]["fundingTime"]
            if last <= cur:
                break
            cur = last + 1
            if len(data) < 1000:
                break
            time.sleep(0.12)
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["fundingRate"] = pd.to_numeric(df["fundingRate"])
        df.index = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        return df[["fundingRate"]][~df.index.duplicated()].sort_index()

    return _cached(tag, build)


def exchange_universe(min_quote_vol: float = 5e7, top: int = 40) -> list[str]:
    """Top USDT-M perpetual symbols by 24h quote volume (liquidity prefilter)."""
    data = _get(FUT + "/fapi/v1/ticker/24hr", {})
    rows = [d for d in data if d["symbol"].endswith("USDT")
            and float(d["quoteVolume"]) >= min_quote_vol]
    rows.sort(key=lambda d: float(d["quoteVolume"]), reverse=True)
    return [d["symbol"] for d in rows[:top]]


if __name__ == "__main__":
    # demo / self-check: small real fetch, assert shapes
    end = int(time.time() * 1000)
    start = end - 30 * 24 * 3600 * 1000          # 30 days
    k = klines("BTCUSDT", "1h", start, end, futures=True)
    f = funding_rate("BTCUSDT", start, end)
    u = exchange_universe(top=10)
    print("klines BTCUSDT 1h:", k.shape, k.index[0], "->", k.index[-1])
    print(k.tail(2).to_string())
    print("funding rows:", f.shape, "mean 8h funding:", round(f.fundingRate.mean(), 6))
    print("universe top10:", u)
    assert k.shape[0] > 600 and {"open", "close", "volume"} <= set(k.columns)
    assert f.shape[0] > 50 and f.fundingRate.abs().mean() < 0.05
    assert len(u) == 10 and "BTCUSDT" in u
    print("SELF-CHECK OK")
