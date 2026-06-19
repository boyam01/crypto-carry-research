"""Can a math/risk transform rescue the fragile time-series momentum?

Raw sign-momentum had OOS Sharpe ~1 but -33%~-75% drawdowns. Test two
documented transforms, OOS + per-coin DSR within this enhancement family:
  RAW      pos = sign(trailing L-day return)
  VOLSCALE pos = RAW * target_vol / trailing_vol   (Barroso & Santa-Clara 2015)
  HURST    pos = RAW only when rolling DFA Hurst > 0.5 (persistent regime), else flat
"""
from __future__ import annotations
import sys, time, warnings, pathlib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import fetch_binance as fb
from engine import backtest as bt
from engine import stats as st

UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
            "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT"]
START = "2022-01-01"
L = 30                # momentum lookback (best from battery)
PPY = 365
COST = 5.0
TRAIN = 0.60


def rolling_hurst(ret: pd.Series, win: int = 150) -> np.ndarray:
    out = np.full(len(ret), np.nan)
    v = ret.values
    for i in range(win, len(v)):
        out[i] = st.hurst_dfa(v[i - win:i])
    return out


def signals(close: pd.Series):
    ret = close.pct_change().fillna(0)
    mom = np.sign(close.pct_change(L).shift(1).fillna(0).values)
    vol = ret.rolling(30).std().shift(1).bfill().values
    tgt = np.nanmedian(vol[vol > 0])
    H = pd.Series(rolling_hurst(ret)).shift(1).values
    raw = mom
    volscale = np.clip(mom * tgt / np.where(vol > 0, vol, np.nan), -3, 3)
    volscale = np.nan_to_num(volscale)
    hurst = np.where(H > 0.5, mom, 0.0)
    hurst = np.nan_to_num(hurst)
    return ret.values, dict(RAW=raw, VOLSCALE=volscale, HURST=hurst)


def bt_one(ret, pos, cost):
    net = bt.run(ret, pos, cost)
    n = len(net)
    tr, te = bt.oos_split(n, TRAIN)
    mo = bt.metrics(net[te], PPY, pos[te])
    p = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
    return net[te], mo, p


def main():
    end = int(time.time() * 1000)
    start = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    daily = {}
    for s in UNIVERSE:
        k = fb.klines(s, "1d", start, end, futures=True)
        if k is not None and len(k) > 400:
            daily[s] = k["close"]
    print(f"loaded {len(daily)} coins, daily\n")

    variants = ["RAW", "VOLSCALE", "HURST"]
    per_coin = {v: [] for v in variants}
    oos_nets = {v: {} for v in variants}
    for s, c in daily.items():
        ret, sigs = signals(c)
        for v in variants:
            net_oos, mo, p = bt_one(ret, sigs[v], COST)
            per_coin[v].append((s, mo, p))
            oos_nets[v][s] = pd.Series(net_oos, index=c.index[-len(net_oos):])

    for v in variants:
        print(f"=== {v} ===  (per-coin OOS, cost={COST:.0f}bp)")
        sr_pool = [mo["sr_pp"] for _, mo, _ in per_coin[v] if np.isfinite(mo["sr_pp"])]
        sr_star = bt.dsr_benchmark(sr_pool)
        port = pd.DataFrame(oos_nets[v]).fillna(0).mean(axis=1).values
        pm = bt.metrics(port, PPY)
        pp = bt.psr(pm["sr_pp"], pm["n"], pm["skew"], pm["kurt"])
        pdsr = bt.psr(pm["sr_pp"], pm["n"], pm["skew"], pm["kurt"], sr_benchmark=sr_star)
        for s, mo, p in sorted(per_coin[v], key=lambda x: -x[1]["sharpe_ann"]):
            print(f"   {s:9} Sharpe={mo['sharpe_ann']:6.2f} ret={mo['ret_ann']*100:7.1f}% "
                  f"maxDD={mo['maxdd']*100:7.1f}% turn={mo['turnover']:.3f} PSR={p:.2f}")
        print(f"   >> PORTFOLIO Sharpe={pm['sharpe_ann']:.2f} ret={pm['ret_ann']*100:.1f}% "
              f"maxDD={pm['maxdd']*100:.1f}% PSR={pp:.3f}  DSR(vs SR*={sr_star:.3f})={pdsr:.3f}\n")


if __name__ == "__main__":
    main()
