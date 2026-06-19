"""Deployable delta-neutral funding carry, hardened from the battery finding.

Lesson from run_battery: chasing sign(funding) bar-by-bar dies on turnover
(funding>0 only ~47% of 8h bars -> constant flips -> ~55%/yr fees). Funding is
highly persistent, so harvest it with an EMA + hysteresis dead-band: enter when
the smoothed funding clears a threshold, hold through the zero-crossing noise,
flip only when the opposite threshold is breached.

Position convention (delta-neutral, two legs):
  sig=+1 -> short perp / long spot  (receive funding when funding>0)
  sig=-1 -> long perp / short spot  (receive funding when funding<0)
Per-period net = sig*(funding + spot_ret - perp_ret) - cost*turnover.
"""
from __future__ import annotations
import sys, time, json, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import fetch_binance as fb
from engine import backtest as bt

UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
            "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT"]
START = "2022-01-01"
PPY = 365 * 3
TRAIN_FRAC = 0.60


def _ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


def panel(symbols, start, end):
    out = {}
    for s in symbols:
        perp = fb.klines(s, "8h", start, end, futures=True)
        spot = fb.klines(s, "8h", start, end, futures=False)
        fr = fb.funding_rate(s, start, end)
        if any(v is None for v in (perp, spot, fr)) or len(perp) < 500:
            continue
        df = pd.DataFrame(index=perp.index)
        df["perp_ret"] = perp["close"].pct_change()
        df["spot_ret"] = spot["close"].reindex(df.index).ffill().pct_change()
        df["funding"] = fr["fundingRate"].reindex(df.index).fillna(0.0)
        out[s] = df.dropna()
    return out


def hysteresis_signal(funding: np.ndarray, ema_span: int, enter_thr: float) -> np.ndarray:
    """Smoothed-funding state machine. Lagged so signal at t uses funding<t."""
    ema = pd.Series(funding).ewm(span=ema_span, adjust=False).mean().shift(1).fillna(0).values
    sig = np.zeros(len(funding))
    state = 0
    for i in range(len(funding)):
        e = ema[i]
        if e > enter_thr:
            state = 1                       # persistent positive funding -> short perp
        elif e < -enter_thr:
            state = -1
        elif abs(e) < enter_thr / 3:        # dead-band -> flatten only when clearly gone
            state = 0
        sig[i] = state
    return sig


def carry_net(df, sig, cost_bps):
    f = df["funding"].values
    pr = df["perp_ret"].values
    sr = df["spot_ret"].values
    turn = 2 * np.abs(np.diff(sig, prepend=0.0))      # two legs
    return sig * (f + sr - pr) - (cost_bps / 1e4) * turn


def evaluate(net, position=None):
    n = len(net)
    tr, te = bt.oos_split(n, TRAIN_FRAC)
    mi = bt.metrics(net[tr], PPY, position[tr] if position is not None else None)
    mo = bt.metrics(net[te], PPY, position[te] if position is not None else None)
    return mi, mo, bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])


def main():
    end = int(time.time() * 1000)
    P = panel(UNIVERSE, _ms(START), end)
    print(f"loaded {len(P)} coins\n")

    # tune ema/threshold on IS only (cheap grid), pick best IS Sharpe of equal-wt portfolio
    grids = [(span, thr) for span in (3, 9, 21) for thr in (0.0, 0.5e-4, 1e-4, 2e-4)]
    best = None
    for span, thr in grids:
        nets = {}
        for s, df in P.items():
            sig = hysteresis_signal(df["funding"].values, span, thr)
            nets[s] = pd.Series(carry_net(df, sig, 10.0), index=df.index)
        port = pd.DataFrame(nets).fillna(0).mean(axis=1).values
        mi, mo, _ = evaluate(port)
        if best is None or mi["sharpe_ann"] > best[2]["sharpe_ann"]:
            best = (span, thr, mi, mo)
    span, thr = best[0], best[1]
    print(f"IS-selected params: ema_span={span}, enter_thr={thr*1e4:.1f}bp  "
          f"(IS Sharpe={best[2]['sharpe_ann']:.2f})\n")

    rows = []
    for cost in (5.0, 10.0):
        nets = {}
        for s, df in P.items():
            sig = hysteresis_signal(df["funding"].values, span, thr)
            net = carry_net(df, sig, cost)
            nets[s] = pd.Series(net, index=df.index)
            mi, mo, psr = evaluate(net, sig)
            if cost == 10.0:
                rows.append((s, mo, psr, mi["sharpe_ann"]))
        port = pd.DataFrame(nets).fillna(0)
        pv = port.mean(axis=1).values
        # turnover-aware portfolio position proxy = mean abs of signals
        sigs = pd.DataFrame({s: pd.Series(hysteresis_signal(P[s]["funding"].values, span, thr),
                                          index=P[s].index) for s in P}).fillna(0)
        pos = sigs.abs().mean(axis=1).values
        mi, mo, psr = evaluate(pv, pos)
        # static always-short baseline at this cost
        snets = {}
        for s, df in P.items():
            snets[s] = pd.Series(carry_net(df, np.ones(len(df)), cost), index=df.index)
        sp = pd.DataFrame(snets).fillna(0).mean(axis=1).values
        _, smo, spsr = evaluate(sp)
        print(f"=== cost={cost:.0f}bp/leg ===")
        print(f"  HYSTERESIS portfolio  OOS Sharpe={mo['sharpe_ann']:6.2f}  ret={mo['ret_ann']*100:6.2f}%  "
              f"maxDD={mo['maxdd']*100:6.2f}%  turn={mo['turnover']:.3f}  PSR={psr:.3f}")
        print(f"  STATIC short  portfolio OOS Sharpe={smo['sharpe_ann']:6.2f}  ret={smo['ret_ann']*100:6.2f}%  "
              f"maxDD={smo['maxdd']*100:6.2f}%  PSR={spsr:.3f}")

    print("\n--- per-coin HYSTERESIS carry, OOS, cost=10bp/leg ---")
    print(f"{'coin':10} {'IS_Shrp':>8} {'OOS_Shrp':>9} {'OOS_ret%':>9} {'maxDD%':>8} {'turn':>6} {'PSR':>6}")
    for s, mo, psr, isq in sorted(rows, key=lambda x: -x[1]["sharpe_ann"]):
        print(f"{s:10} {isq:8.2f} {mo['sharpe_ann']:9.2f} {mo['ret_ann']*100:9.2f} "
              f"{mo['maxdd']*100:8.2f} {mo['turnover']:6.3f} {psr:6.3f}")

    out = pathlib.Path(__file__).resolve().parent.parent / "reports" / "carry_result.json"
    out.write_text(json.dumps(dict(span=span, thr=thr,
                   per_coin={s: mo for s, mo, _, _ in rows}), indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
