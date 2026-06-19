"""Quarterly futures calendar basis (cash-and-carry) — a STRUCTURALLY INDEPENDENT
edge from funding: a dated future MUST converge to spot at delivery, so
long-spot/short-future held to expiry earns exactly the entry basis, regardless
of the price path. Tested across all expired Binance USDT-M quarterlies.

net daily PnL (delta-neutral long spot / short future) = spot_ret - fut_ret.
Cumulative over the contract life ~= entry basis (deterministic convergence),
minus a one-off round-trip cost on two legs.
"""
from __future__ import annotations
import sys, time, warnings, pathlib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import fetch_binance as fb
from engine import backtest as bt

# expired + live USDT-M quarterly delivery codes (YYMMDD = delivery date)
CODES = ["250328", "250627", "250926", "251226", "260327", "260626"]
ASSETS = ["BTCUSDT", "ETHUSDT"]
COST_RT_BP = 30.0     # one-off round trip, both legs (taker-ish, conservative)


def contract(asset, code):
    """Daily delta-neutral carry PnL series for one dated contract, full life."""
    deliv = pd.Timestamp("20" + code, tz="UTC")
    start_ms = int((deliv - pd.Timedelta(days=210)).timestamp() * 1000)
    end_ms = int((deliv + pd.Timedelta(days=1)).timestamp() * 1000)
    fut = fb.klines(f"{asset}_{code}", "1d", start_ms, end_ms, futures=True)
    spot = fb.klines(asset, "1d", start_ms, end_ms, futures=False)
    if fut is None or spot is None or len(fut) < 60:
        return None
    df = pd.DataFrame(index=fut.index)
    df["fut"] = fut["close"]
    df["spot"] = spot["close"].reindex(df.index).ffill()
    df = df.dropna()
    if len(df) < 60:
        return None
    df = df[df.index <= deliv]
    # trim frozen post-settlement tail: the kline endpoint repeats the last
    # traded future price after delivery while spot keeps moving -> drop the
    # trailing run where fut is constant, exit at the last LIVE trading bar.
    fut_const_tail = (df["fut"] == df["fut"].iloc[-1])[::-1].cummin()[::-1]
    if fut_const_tail.sum() > 1:
        df = df.loc[~fut_const_tail | (df.index == df.index[fut_const_tail.argmax()])]
    if len(df) < 60:
        return None
    entry_basis = df["fut"].iloc[0] / df["spot"].iloc[0] - 1
    days = (df.index[-1] - df.index[0]).days or 1
    fut_ret = df["fut"].pct_change().fillna(0).values
    spot_ret = df["spot"].pct_change().fillna(0).values
    net = spot_ret - fut_ret                       # long spot, short future
    net[0] -= COST_RT_BP / 1e4                      # one-off entry+exit cost
    cum = np.prod(1 + net) - 1
    return dict(name=f"{asset[:-4]}_{code}", entry_basis=entry_basis, days=days,
                cum=cum, ann=cum * 365 / days, net=pd.Series(net, index=df.index),
                path_sharpe=net.mean() / net.std() * np.sqrt(365) if net.std() else np.nan,
                terminal_basis=df["fut"].iloc[-1] / df["spot"].iloc[-1] - 1)


def main():
    rows = []
    for a in ASSETS:
        for code in CODES:
            r = contract(a, code)
            if r:
                rows.append(r)
    print("=== Calendar basis cash-and-carry, per contract (long spot / short dated future, held to expiry) ===")
    print(f"{'contract':12} {'entry_basis%':>12} {'days':>5} {'realized%':>10} {'annualized%':>12} {'pathSharpe':>11} {'termBasis%':>11}")
    for r in rows:
        print(f"{r['name']:12} {r['entry_basis']*100:12.3f} {r['days']:5} {r['cum']*100:10.3f} "
              f"{r['ann']*100:12.2f} {r['path_sharpe']:11.2f} {r['terminal_basis']*100:11.4f}")

    # always-on rolling carry: realized return is deterministic per contract;
    # the honest aggregate is the average annualized across contracts (each held to expiry).
    anns = np.array([r["ann"] for r in rows])
    real = np.array([r["cum"] for r in rows])
    print(f"\nmean entry basis = {np.mean([r['entry_basis'] for r in rows])*100:.3f}%  "
          f"(contango>0); contracts in contango: {sum(r['entry_basis']>0 for r in rows)}/{len(rows)}")
    print(f"mean realized per contract = {real.mean()*100:.3f}%  | mean annualized = {anns.mean()*100:.2f}%")
    print(f"realized win rate (cum>0) = {(real>0).mean():.2f}  | worst contract = {real.min()*100:.2f}%")
    # CONDITIONAL: only enter contracts whose entry basis clears the round-trip cost
    elig = [r for r in rows if r["entry_basis"] > COST_RT_BP / 1e4]
    if elig:
        er = np.array([r["cum"] for r in elig]); ea = np.array([r["ann"] for r in elig])
        print(f"\nCONDITIONAL (enter only if entry_basis>{COST_RT_BP:.0f}bp): {len(elig)}/{len(rows)} contracts, "
              f"mean realized={er.mean()*100:.3f}%, mean ann={ea.mean()*100:.2f}%, win={ (er>0).mean():.2f}, worst={er.min()*100:.2f}%")
    print("\nNOTE: terminal basis ~0 confirms convergence (the deterministic anchor). "
          "Realized ~ entry basis - cost. Path mark-to-market can be negative mid-life; PnL is locked only at delivery.")


if __name__ == "__main__":
    main()
