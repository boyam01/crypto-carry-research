"""Execution-grade OFI: order-flow imbalance as a MARKET-MAKING lean signal.

Taker OFI died on fees (maker -44 / taker -105 Sharpe). The legitimate use of OFI
is as a MAKER: post passive quotes, earn spread + rebate, and use OFI to skew the
quotes AWAY from the side about to adversely-select you. This tests the DIRECTION
(does OFI reduce adverse selection enough to help MM), with a stylized fill model
(no queue/latency) -- so it judges OFI's value-add to MM, not live PnL.

Adverse selection logic:
  buy pressure  (OFI>0) -> price likely UP -> aggressors hit your ASK (you go short)
                 -> adverse -> PULL/REDUCE ASK participation when OFI>0.
  sell pressure (OFI<0) -> price likely DOWN -> aggressors hit your BID (you go long)
                 -> adverse -> PULL/REDUCE BID participation when OFI<0.
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import experiments.ofi_edge as oe
from engine import backtest as bt

SYM = "BTCUSDT"
DAYS = ["2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12", "2026-06-13", "2026-06-14", "2026-06-15"]
BAR = "5s"
HALF_SPREAD_BP = 0.5      # half the quoted bid-ask you capture per passive fill (BTC perp ~1bp spread)
MAKER_FEE_BP = -0.5       # maker REBATE (achievable at VIP / rebate venues); +ve = you pay
RHO = 0.10                # your participation: fraction of aggressor flow you passively absorb
INV_CAP = 50.0            # max |inventory| (base units), forces flat-ish book


def load_bars():
    import io, zipfile, urllib.request, ssl
    ctx = ssl.create_default_context()
    parts = []
    for d in DAYS:
        cf = oe.CACHE / f"mm_{SYM}_{d}_{BAR}.parquet"
        if cf.exists():
            parts.append(pd.read_parquet(cf)); continue
        url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{SYM}/{SYM}-aggTrades-{d}.zip"
        raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "M"}), timeout=90, context=ctx).read()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        with zf.open(zf.namelist()[0]) as fh:
            head = fh.read(200).decode("utf-8", "ignore")
        hdr = 0 if "price" in head.split("\n")[0] else None
        df = pd.read_csv(io.BytesIO(raw), compression="zip",
                         names=None if hdr == 0 else ["a", "price", "qty", "f", "l", "T", "m"], header=hdr)
        df.columns = [c.strip().lower() for c in df.columns]
        pcol = "price" if "price" in df.columns else df.columns[1]
        qcol = "quantity" if "quantity" in df.columns else ("qty" if "qty" in df.columns else df.columns[2])
        tcol = [c for c in df.columns if c in ("transact_time", "t", "time")][0]
        mcol = [c for c in df.columns if "maker" in c or c == "m"][0]
        px = pd.to_numeric(df[pcol]); q = pd.to_numeric(df[qcol])
        buyer_maker = df[mcol].astype(str).str.lower().isin(["true", "1"])
        # aggressor: buyer_maker True -> seller aggressor (SELL); False -> buyer aggressor (BUY)
        sell_agg = np.where(buyer_maker, q, 0.0)
        buy_agg = np.where(~buyer_maker, q, 0.0)
        ts = pd.to_datetime(pd.to_numeric(df[tcol]), unit="ms", utc=True)
        t = pd.DataFrame({"px": px.values, "sell_agg": sell_agg, "buy_agg": buy_agg}, index=ts)
        bars = pd.DataFrame({
            "mid": t["px"].resample(BAR).last(),
            "sell_agg": t["sell_agg"].resample(BAR).sum(),
            "buy_agg": t["buy_agg"].resample(BAR).sum(),
        }).dropna()
        bars.to_parquet(cf); parts.append(bars)
    b = pd.concat(parts).sort_index()
    return b[~b.index.duplicated()]


def simulate(bars, skew):
    """skew in [0,1]: fraction by which to CUT the adversely-selected side's participation
    based on rolling OFI sign. skew=0 -> symmetric MM."""
    mid = bars["mid"].values
    sa = bars["sell_agg"].values; ba = bars["buy_agg"].values
    ofi = pd.Series(ba - sa).rolling(12).mean().shift(1).fillna(0).values   # lagged buy-minus-sell pressure
    osd = np.std(ofi[ofi != 0]) + 1e-9
    n = len(bars); inv = 0.0
    spread_pnl = np.zeros(n); inv_pnl = np.zeros(n); fee_pnl = np.zeros(n)
    hs = HALF_SPREAD_BP / 1e4
    for i in range(n - 1):
        z = ofi[i] / osd
        bid_part = RHO * (1 - skew * max(0, -z) / 2)        # cut bid when sell pressure (z<0)
        ask_part = RHO * (1 - skew * max(0, z) / 2)         # cut ask when buy pressure (z>0)
        bid_part = max(0.0, min(RHO, bid_part)); ask_part = max(0.0, min(RHO, ask_part))
        bid_fill = sa[i] * bid_part                          # aggressor sells hit your bid -> you BUY (long)
        ask_fill = ba[i] * ask_part                          # aggressor buys hit your ask -> you SELL (short)
        # inventory cap: throttle the side that increases |inventory|
        if inv > INV_CAP:
            bid_fill = 0.0
        if inv < -INV_CAP:
            ask_fill = 0.0
        filled = bid_fill + ask_fill
        spread_pnl[i] = filled * hs * mid[i]                 # earn half-spread on each passive fill
        fee_pnl[i] = -filled * (MAKER_FEE_BP / 1e4) * mid[i] # rebate (MAKER_FEE_BP<0 => +ve)
        inv += (bid_fill - ask_fill)
        inv_pnl[i] = inv * (mid[i + 1] - mid[i])             # mark inventory to next mid (adverse selection)
    total = spread_pnl + fee_pnl + inv_pnl
    return dict(total=total, spread=spread_pnl, fee=fee_pnl, adverse=inv_pnl)


def report(name, res):
    t = res["total"]
    ppy = 365 * 24 * 3600 / 5                                # 5s bars per year
    mo = bt.metrics(t, ppy)
    print(f"{name:18} Sharpe {mo['sharpe_ann']:7.2f}  | spread {res['spread'].sum():10.1f}  "
          f"rebate {res['fee'].sum():9.1f}  advsel {res['adverse'].sum():11.1f}  NET {t.sum():11.1f}")
    return mo["sharpe_ann"], t


def main():
    print(f"loading {len(DAYS)}d {SYM} aggTrades -> {BAR} MM bars ...")
    bars = load_bars()
    n = len(bars); cut = int(n * 0.6)
    print(f"  {n} {BAR} bars; OOS = last {n-cut}\n")
    print(f"params: half_spread={HALF_SPREAD_BP}bp, maker_fee={MAKER_FEE_BP}bp (neg=rebate), participation={RHO}, inv_cap={INV_CAP}")
    print(f"{'strategy':18} {'Sharpe':>9}  | components ($ per traded notional, full sample)")
    sym0, t0 = report("symmetric MM", simulate(bars, skew=0.0))
    sym1, t1 = report("OFI-skew MM", simulate(bars, skew=1.0))
    # OOS comparison
    for lbl, t in [("symmetric", t0), ("OFI-skew", t1)]:
        ppy = 365 * 24 * 3600 / 5
        mo = bt.metrics(t[cut:], ppy)
        print(f"  OOS {lbl:10}: Sharpe {mo['sharpe_ann']:7.2f}  net {t[cut:].sum():.1f}")
    print("\nREAD: does OFI-skew beat symmetric MM (higher net / less adverse selection)?")
    print("If yes -> OFI adds real value to market-making (execution-grade edge, direction confirmed).")
    print("CAVEAT: stylized fill model (no queue position, latency, cancel/replace, real BBO); tests OFI's")
    print("value-ADD to MM, not deployable live PnL. Maker rebate assumed (VIP/rebate venue).")


if __name__ == "__main__":
    main()
