"""Tick microstructure edge: trade-sign OFI + VPIN from Binance aggTrades.
The genuine NEW-INFORMATION dimension (not formulas on klines). Tests whether
order-flow imbalance predicts short-horizon returns, NET of realistic fees
(the make-or-break for HF), OOS.

Trade sign: is_buyer_maker=True => aggressor is SELLER (-1); False => BUYER (+1).
OFI(bar) = net signed volume / total volume.  VPIN = bulk-volume order toxicity.
"""
from __future__ import annotations
import sys, io, zipfile, urllib.request, ssl, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import backtest as bt

CACHE = pathlib.Path(__file__).resolve().parent.parent / "data" / "cache"
CTX = ssl.create_default_context()
SYM = "BTCUSDT"
DAYS = ["2026-06-08", "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12",
        "2026-06-13", "2026-06-14", "2026-06-15"]


def load_day(sym, d):
    cf = CACHE / f"ofibars_{sym}_{d}.parquet"
    if cf.exists():
        return pd.read_parquet(cf)
    url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{sym}/{sym}-aggTrades-{d}.zip"
    raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "M"}), timeout=60, context=CTX).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    with zf.open(zf.namelist()[0]) as fh:
        head = fh.read(200).decode("utf-8", "ignore")
    hdr = 0 if "price" in head.split("\n")[0] else None
    cols = ["a", "price", "qty", "f", "l", "T", "m"]
    df = pd.read_csv(io.BytesIO(raw), compression="zip",
                     names=None if hdr == 0 else cols, header=hdr)
    df.columns = [c.strip().lower() for c in df.columns]
    pcol = "price" if "price" in df.columns else df.columns[1]
    qcol = "quantity" if "quantity" in df.columns else ("qty" if "qty" in df.columns else df.columns[2])
    tcol = [c for c in df.columns if c in ("transact_time", "t", "time")][0]
    mcol = [c for c in df.columns if "maker" in c or c == "m"][0]
    px = pd.to_numeric(df[pcol]); q = pd.to_numeric(df[qcol])
    sign = np.where(df[mcol].astype(str).str.lower().isin(["true", "1"]), -1.0, 1.0)
    ts = pd.to_datetime(pd.to_numeric(df[tcol]), unit="ms", utc=True)
    t = pd.DataFrame({"px": px.values, "q": q.values, "sv": (q.values * sign)}, index=ts)
    bars = pd.DataFrame({
        "close": t["px"].resample("1min").last(),
        "vol": t["q"].resample("1min").sum(),
        "ofi": t["sv"].resample("1min").sum(),
        "ntr": t["px"].resample("1min").count(),
    }).dropna()
    bars["ofi_norm"] = bars["ofi"] / bars["vol"].replace(0, np.nan)
    bars.to_parquet(cf)
    return bars


def vpin(bars, bucket_frac=1 / 50.0, win=50):
    """volume-bucket VPIN (Easley-Lopez de Prado), bulk-classified by ret/std."""
    V = bars["vol"].sum() * bucket_frac
    cum, b_buy, b_sell, buckets = 0.0, 0.0, 0.0, []
    dpx = bars["close"].diff().fillna(0).values
    sd = np.std(dpx) + 1e-9
    from scipy.stats import norm
    for i in range(len(bars)):
        v = bars["vol"].iloc[i]; z = dpx[i] / sd
        frac_buy = norm.cdf(z)
        b_buy += v * frac_buy; b_sell += v * (1 - frac_buy); cum += v
        while cum >= V and V > 0:
            buckets.append(abs(b_buy - b_sell) / (b_buy + b_sell + 1e-9))
            b_buy = b_sell = cum = 0.0
    if len(buckets) < win:
        return np.nan
    return float(np.mean(buckets[-win:]))


def main():
    print(f"loading {len(DAYS)} days of {SYM} aggTrades (tick) -> 1-min OFI bars ...")
    bars = pd.concat([load_day(SYM, d) for d in DAYS]).sort_index()
    bars = bars[~bars.index.duplicated()]
    print(f"  {len(bars)} 1-min bars  {bars.index[0]} .. {bars.index[-1]}")
    print(f"  VPIN (last 50 buckets) = {vpin(bars):.3f}")

    r1 = np.log(bars["close"]).diff().shift(-1).values          # next-min return (target)
    ofi = bars["ofi_norm"].values
    n = len(bars); cut = int(n * 0.6)
    # predictive IC at horizons 1/5/15 min, OOS
    print(f"\n{'horizon':8} {'OOS_IC':>8} {'gross_Sh':>9} {'maker1.8bp':>11} {'taker4.5bp':>11}")
    for H in (1, 5, 15):
        fwd = (np.log(bars["close"]).shift(-H) - np.log(bars["close"])).values  # t->t+H
        sig = pd.Series(ofi).rolling(H).mean().shift(1).values                   # avg OFI, lagged (no look-ahead)
        m = np.isfinite(sig) & np.isfinite(fwd)
        ic = np.corrcoef(sig[m][cut:], fwd[m][cut:])[0, 1] if m[cut:].sum() > 50 else np.nan
        # strategy: position = sign(OFI), hold H min (rebalance every H), per-min return
        pos = np.sign(sig)
        perbar = pos * r1                                       # next-min pnl from position
        # only trade every H bars (hold)
        held = pos.copy()
        for i in range(1, n):
            if i % H != 0:
                held[i] = held[i - 1]
        turn = np.abs(np.diff(held, prepend=0))
        for fee, name in [(0.0, "gross"), (1.8, "maker"), (4.5, "taker")]:
            net = held * r1 - fee / 1e4 * turn
            mo = bt.metrics(net[cut:], 365 * 1440 / 1)          # per-minute ann factor
            if name == "gross":
                gs = mo["sharpe_ann"]
            elif name == "maker":
                mk = mo["sharpe_ann"]
            else:
                tk = mo["sharpe_ann"]
        print(f"{H:>4}min  {ic:8.4f} {gs:9.2f} {mk:11.2f} {tk:11.2f}")
    print("\nNOTE: ann factor uses 1-min bars; high Sharpe scale is per-minute annualized.")
    print("Read: gross IC/Sharpe = does OFI predict? maker/taker = survivable net of fee?")
    print("If gross strong but taker<<0 => real microstructure signal, retail-untradeable on fees (needs maker/rebate/colo).")


if __name__ == "__main__":
    main()
