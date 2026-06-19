"""Edge battery on real Binance public data, with chronological OOS + cost + DSR.

Edges tested:
  H1  funding_carry   delta-neutral spot/perp carry (structural risk premium)
  H2  ts_momentum     time-series momentum, sign(trailing return)   (Moskowitz 2012)
  H3  xs_momentum     cross-sectional momentum, long-top/short-bottom
  H4  st_reversal     short-term reversal on 8h bars (mean reversion)

Every strategy's reported numbers are OUT-OF-SAMPLE (last 40% of history, never
used to choose anything). DSR deflates the best Sharpe by the number of trials.
"""
from __future__ import annotations
import sys, time, json, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import fetch_binance as fb
from engine import backtest as bt
from engine import stats as st

UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
            "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT"]
START = "2022-01-01"
COST_BPS = 5.0            # per-leg taker-ish cost; stressed to 10 in report
TRAIN_FRAC = 0.60         # first 60% in-sample, last 40% strictly OOS
PPY_8H = 365 * 3
PPY_D = 365


def _ms(date_str: str) -> int:
    return int(pd.Timestamp(date_str, tz="UTC").timestamp() * 1000)


def load_panels(symbols, start, end):
    """Return dict symbol -> 8h DataFrame[spot_close, perp_close, funding]."""
    out = {}
    for s in symbols:
        try:
            perp = fb.klines(s, "8h", start, end, futures=True)
            spot = fb.klines(s, "8h", start, end, futures=False)
            fr = fb.funding_rate(s, start, end)
        except Exception as e:
            print(f"  skip {s}: {e}")
            continue
        if perp is None or spot is None or fr is None or len(perp) < 500:
            print(f"  skip {s}: insufficient data")
            continue
        df = pd.DataFrame(index=perp.index)
        df["perp_close"] = perp["close"]
        df["spot_close"] = spot["close"].reindex(df.index).ffill()
        # funding paid at funding timestamps -> align onto 8h grid
        df["funding"] = fr["fundingRate"].reindex(df.index).fillna(0.0)
        df = df.dropna()
        if len(df) > 500:
            out[s] = df
            print(f"  {s}: {len(df)} 8h bars  {df.index[0].date()}..{df.index[-1].date()}"
                  f"  mean|fund|={df.funding.abs().mean()*1e4:.2f}bp")
    return out


# ---------------- strategies (return per-period net on full history) -------------
def funding_carry(df, cost_bps):
    f = df["funding"].values
    perp_ret = df["perp_close"].pct_change().fillna(0).values
    spot_ret = df["spot_close"].pct_change().fillna(0).values
    sig = np.sign(np.concatenate([[0], f[:-1]]))         # use funding_{t-1}: no look-ahead
    turn = 2 * np.abs(np.diff(sig, prepend=0.0))         # two legs flip
    net = sig * (f + spot_ret - perp_ret) - (cost_bps / 1e4) * turn
    return net, sig


def carry_always_short(df, cost_bps):
    """Naive baseline: always short perp / long spot (harvest only when fund>0)."""
    f = df["funding"].values
    perp_ret = df["perp_close"].pct_change().fillna(0).values
    spot_ret = df["spot_close"].pct_change().fillna(0).values
    sig = np.ones(len(f))
    net = sig * (f + spot_ret - perp_ret)                # static -> ~no turnover cost
    return net, sig


def ts_momentum(close, lookback, cost_bps):
    r = close.pct_change().fillna(0).values
    trail = pd.Series(close).pct_change(lookback).shift(1).fillna(0).values
    sig = np.sign(trail)
    turn = np.abs(np.diff(sig, prepend=0.0))
    net = sig * r - (cost_bps / 1e4) * turn
    return net, sig


def st_reversal(close, cost_bps):
    r = close.pct_change().fillna(0).values
    sig = -np.sign(np.concatenate([[0], r[:-1]]))        # fade last bar
    turn = np.abs(np.diff(sig, prepend=0.0))
    net = sig * r - (cost_bps / 1e4) * turn
    return net, sig


def xs_momentum(panel_close: pd.DataFrame, lookback, cost_bps):
    """Daily cross-sectional momentum: long top third / short bottom third."""
    rets = panel_close.pct_change().fillna(0)
    trail = panel_close.pct_change(lookback).shift(1)
    pos = pd.DataFrame(0.0, index=panel_close.index, columns=panel_close.columns)
    for t in range(len(panel_close)):
        row = trail.iloc[t]
        valid = row.dropna()
        if len(valid) < 6:
            continue
        k = max(1, len(valid) // 3)
        winners = valid.nlargest(k).index
        losers = valid.nsmallest(k).index
        pos.iloc[t, pos.columns.get_indexer(winners)] = 1.0 / k
        pos.iloc[t, pos.columns.get_indexer(losers)] = -1.0 / k
    turn = pos.diff().abs().sum(axis=1).fillna(0).values
    net = (pos * rets).sum(axis=1).values - (cost_bps / 1e4) * turn
    return net, pos.abs().sum(axis=1).values


# ---------------- runner ---------------------------------------------------------
def evaluate(name, net, ppy, position=None):
    n = len(net)
    tr, te = bt.oos_split(n, TRAIN_FRAC)
    m_is = bt.metrics(net[tr], ppy, position[tr] if position is not None else None)
    m_oos = bt.metrics(net[te], ppy, position[te] if position is not None else None)
    p_oos = bt.psr(m_oos["sr_pp"], m_oos["n"], m_oos["skew"], m_oos["kurt"])
    return dict(name=name, ppy=ppy, IS_sharpe=m_is["sharpe_ann"],
                OOS=m_oos, OOS_psr=p_oos)


def main():
    end = int(time.time() * 1000)
    print("Loading panels ...")
    panels = load_panels(UNIVERSE, _ms(START), end)
    if not panels:
        print("NO DATA"); return
    results = []

    # H1 funding carry per symbol + equal-weight portfolio
    carry_nets = {}
    for s, df in panels.items():
        net, sig = funding_carry(df, COST_BPS)
        carry_nets[s] = pd.Series(net, index=df.index)
        results.append(evaluate(f"H1_carry/{s}", net, PPY_8H, sig))
    # equal-weight carry portfolio (align on common index)
    cn = pd.DataFrame(carry_nets).dropna(how="all").fillna(0)
    port = cn.mean(axis=1).values
    results.append(evaluate("H1_carry/PORTFOLIO", port, PPY_8H))
    # naive always-short carry baseline (BTC)
    nb, _ = carry_always_short(panels["BTCUSDT"], COST_BPS)
    results.append(evaluate("H1_carry_naive/BTCUSDT", nb, PPY_8H))

    # H2 TS momentum (daily close from perp) per symbol, lookbacks
    daily = {s: panels[s]["perp_close"].resample("1D").last().dropna()
             for s in panels}
    for s, c in daily.items():
        for L in (7, 14, 30):
            net, sig = ts_momentum(c, L, COST_BPS)
            results.append(evaluate(f"H2_tsmom{L}/{s}", net, PPY_D, sig))

    # H3 cross-sectional momentum (daily panel)
    pc = pd.DataFrame({s: daily[s] for s in daily}).dropna(how="all")
    for L in (7, 14, 30):
        net, pos = xs_momentum(pc, L, COST_BPS)
        results.append(evaluate(f"H3_xsmom{L}", net, PPY_D, pos))

    # H4 short-term reversal on 8h
    for s, df in panels.items():
        net, sig = st_reversal(df["perp_close"], COST_BPS)
        results.append(evaluate(f"H4_reversal/{s}", net, PPY_8H, sig))

    # ---- deflate within each sampling-frequency class (per-period SR is
    #      frequency-dependent; mixing 8h and daily Sharpes is invalid) ----
    sr_star = {}
    for ppy in {r["ppy"] for r in results}:
        pool = [r["OOS"]["sr_pp"] for r in results
                if r["ppy"] == ppy and np.isfinite(r["OOS"]["sr_pp"])]
        sr_star[ppy] = bt.dsr_benchmark(pool)
    for r in results:
        sr = r["OOS"]["sr_pp"]
        r["DSR"] = bt.psr(sr, r["OOS"]["n"], r["OOS"]["skew"], r["OOS"]["kurt"],
                          sr_benchmark=sr_star[r["ppy"]]) if np.isfinite(sr) else np.nan

    results.sort(key=lambda r: (r["OOS"]["sharpe_ann"]
                                if np.isfinite(r["OOS"]["sharpe_ann"]) else -9))
    out = dict(generated=pd.Timestamp.utcnow().isoformat(), start=START,
               cost_bps=COST_BPS, train_frac=TRAIN_FRAC, n_trials=len(results),
               sr_star_pp={str(k): v for k, v in sr_star.items()}, results=results)
    rep = pathlib.Path(__file__).resolve().parent.parent / "reports" / "battery_result.json"
    rep.write_text(json.dumps(out, indent=2, default=float))

    print("\n=== OOS leaderboard (deflated within frequency class) ===  trials=%d" % len(results))
    print("   SR*_pp by ppy: " + ", ".join(f"{k:.0f}->{v:.3f}" for k, v in sr_star.items()))
    print(f"{'strategy':28} {'OOS_Sharpe':>10} {'OOS_ret%':>9} {'maxDD%':>7} {'turn':>6} {'PSR':>6} {'DSR':>6}")
    for r in results[::-1][:25]:
        o = r["OOS"]
        print(f"{r['name']:28} {o['sharpe_ann']:10.2f} {o['ret_ann']*100:9.1f} "
              f"{o['maxdd']*100:7.1f} {o['turnover']:6.3f} "
              f"{r['OOS_psr']:6.2f} {r['DSR']:6.2f}")
    print(f"\nwrote {rep}")


if __name__ == "__main__":
    main()
