"""Two more math-paper edges, OOS + cost + PSR.

A) xsec_funding_carry — cross-sectional funding dispersion. Each 8h bar, rank
   coins by smoothed funding; go long-carry (short perp/long spot) on the
   highest-funding coins, short-carry on the lowest. Dollar-neutral & funding-
   sign-agnostic, so it harvests dispersion even when aggregate funding ~0.

B) cointegration pairs — Engle-Granger: regress log px_y on log px_x, test the
   residual for stationarity (Dickey-Fuller t-stat) and OU half-life on IS, then
   trade the z-scored spread with IS-fixed beta/params, evaluated OOS.
"""
from __future__ import annotations
import sys, time, itertools, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import fetch_binance as fb
from engine import backtest as bt
from engine import stats as st

UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
            "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT"]
START = "2022-01-01"
TRAIN_FRAC = 0.60


def _ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


# ---------------- A) cross-sectional funding carry ----------------
def load_carry_panel(end):
    f, sr, pr = {}, {}, {}
    for s in UNIVERSE:
        perp = fb.klines(s, "8h", _ms(START), end, futures=True)
        spot = fb.klines(s, "8h", _ms(START), end, futures=False)
        fr = fb.funding_rate(s, _ms(START), end)
        if any(v is None for v in (perp, spot, fr)) or len(perp) < 500:
            continue
        f[s] = fr["fundingRate"].reindex(perp.index).fillna(0.0)
        pr[s] = perp["close"].pct_change()
        sr[s] = spot["close"].reindex(perp.index).ffill().pct_change()
    idx = None
    for s in f:
        idx = f[s].index if idx is None else idx.intersection(f[s].index)
    F = pd.DataFrame({s: f[s].reindex(idx) for s in f}).dropna()
    SR = pd.DataFrame({s: sr[s].reindex(F.index) for s in f}).fillna(0)
    PR = pd.DataFrame({s: pr[s].reindex(F.index) for s in f}).fillna(0)
    return F, SR, PR


def xsec_funding_carry(F, SR, PR, ema_span, cost_bps):
    ema = F.ewm(span=ema_span, adjust=False).mean().shift(1).fillna(0)
    # demeaned cross-sectional rank weights, gross exposure normalised to 1
    r = ema.rank(axis=1)
    w = r.sub(r.mean(axis=1), axis=0)
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    carry_unit = F + SR - PR                  # return of one long-carry unit per coin
    gross = (w * carry_unit).sum(axis=1)
    turn = 2 * w.diff().abs().sum(axis=1).fillna(0)
    net = (gross - (cost_bps / 1e4) * turn).values
    pos = w.abs().sum(axis=1).values
    return net, pos


# ---------------- B) cointegration pairs ----------------
def df_tstat(resid):
    """Dickey-Fuller t-stat on residual (H0: unit root). More negative => more
    stationary. Δr_t = a + ρ r_{t-1} + e ; t-stat of ρ."""
    r = np.asarray(resid, float)
    rl = r[:-1]
    dr = np.diff(r)
    X = np.column_stack([np.ones_like(rl), rl])
    beta, *_ = np.linalg.lstsq(X, dr, rcond=None)
    resid_fit = dr - X @ beta
    s2 = resid_fit @ resid_fit / (len(dr) - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return beta[1] / se


def trade_pair(py_is, px_is, py_oos, px_oos, z_enter=2.0, z_exit=0.5, cost_bps=10.0):
    ly_is, lx_is = np.log(py_is), np.log(px_is)
    beta = np.polyfit(lx_is, ly_is, 1)        # [slope, intercept]
    resid_is = ly_is - (beta[0] * lx_is + beta[1])
    mu, sd = resid_is.mean(), resid_is.std()
    ly_o, lx_o = np.log(py_oos), np.log(px_oos)
    resid_o = ly_o - (beta[0] * lx_o + beta[1])
    z = (resid_o - mu) / sd
    # spread position: short spread when z>enter (resid high => y rich), exit near 0
    pos = np.zeros(len(z))
    state = 0
    for i in range(len(z)):
        if state == 0:
            if z[i] > z_enter:
                state = -1
            elif z[i] < -z_enter:
                state = 1
        elif state == -1 and z[i] < z_exit:
            state = 0
        elif state == 1 and z[i] > -z_exit:
            state = 0
        pos[i] = state
    # spread return ~ d(resid); position decided on prior bar
    spread_ret = np.diff(resid_o, prepend=resid_o[0])
    sig = np.concatenate([[0], pos[:-1]])
    turn = np.abs(np.diff(sig, prepend=0.0))
    net = sig * spread_ret - (cost_bps / 1e4) * turn
    return net, sig, df_tstat(resid_is), st.ou_half_life(resid_is)


def main():
    end = int(time.time() * 1000)

    print("=== A) cross-sectional funding-dispersion carry ===")
    F, SR, PR = load_carry_panel(end)
    print(f"   panel {F.shape[0]} 8h bars x {F.shape[1]} coins")
    best = None
    for span in (3, 9, 21, 48):
        net, pos = xsec_funding_carry(F, SR, PR, span, 10.0)
        tr, te = bt.oos_split(len(net), TRAIN_FRAC)
        mi = bt.metrics(net[tr], 1095, pos[tr])
        if best is None or mi["sharpe_ann"] > best[1]["sharpe_ann"]:
            best = (span, mi)
    span = best[0]
    net, pos = xsec_funding_carry(F, SR, PR, span, 10.0)
    tr, te = bt.oos_split(len(net), TRAIN_FRAC)
    mo = bt.metrics(net[te], 1095, pos[te])
    psr = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
    print(f"   IS-selected ema_span={span}")
    print(f"   OOS @10bp: Sharpe={mo['sharpe_ann']:.2f}  ret={mo['ret_ann']*100:.2f}%  "
          f"maxDD={mo['maxdd']*100:.2f}%  turn={mo['turnover']:.3f}  PSR={psr:.3f}")

    print("\n=== B) cointegration pair stat-arb (daily) ===")
    closes = {}
    for s in UNIVERSE:
        k = fb.klines(s, "1d", _ms(START), end, futures=True)
        if k is not None and len(k) > 300:
            closes[s] = k["close"]
    C = pd.DataFrame(closes).dropna()
    cut = int(len(C) * TRAIN_FRAC)
    survivors = []
    all_nets = []
    for a, b in itertools.combinations(C.columns, 2):
        py, px = C[a].values, C[b].values
        net, sig, dft, hl = trade_pair(py[:cut], px[:cut], py[cut:], px[cut:])
        # IS stationarity gate: DF t < -2.9 (5% crit) and sane OU half-life
        if dft < -2.9 and 2 < hl < 120:
            mo = bt.metrics(net, 365, sig)
            p = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
            survivors.append((f"{a[:-4]}~{b[:-4]}", dft, hl, mo, p))
            all_nets.append(pd.Series(net, index=C.index[cut:]))
    print(f"   {len(survivors)} of {len(list(itertools.combinations(C.columns,2)))} pairs passed IS cointegration gate")
    survivors.sort(key=lambda x: -x[3]["sharpe_ann"])
    print(f"   {'pair':16} {'DF_t':>6} {'OU_hl_d':>7} {'OOS_Shrp':>9} {'OOS_ret%':>9} {'maxDD%':>8} {'PSR':>6}")
    for name, dft, hl, mo, p in survivors[:12]:
        print(f"   {name:16} {dft:6.2f} {hl:7.1f} {mo['sharpe_ann']:9.2f} "
              f"{mo['ret_ann']*100:9.2f} {mo['maxdd']*100:8.2f} {p:6.3f}")
    if all_nets:
        port = pd.concat(all_nets, axis=1).fillna(0).mean(axis=1).values
        mo = bt.metrics(port, 365)
        p = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
        # deflate by number of pairs actually traded
        sr_star = bt.dsr_benchmark([s[3]["sr_pp"] for s in survivors])
        dsr = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"], sr_benchmark=sr_star)
        print(f"\n   EQUAL-WT PAIR PORTFOLIO OOS: Sharpe={mo['sharpe_ann']:.2f}  "
              f"ret={mo['ret_ann']*100:.2f}%  maxDD={mo['maxdd']*100:.2f}%  PSR={p:.3f}  DSR={dsr:.3f}")


if __name__ == "__main__":
    main()
