"""Diagnose WHY the combined alpha has +OOS Rank IC but negative L/S Sharpe.
Resolves the only remaining 'hidden 肉' questions:
  1. Decile monotonicity: is mean fwd-return monotone in alpha-rank, or do the
     tradable extremes invert? (non-monotone => untradeable as L/S)
  2. GROSS vs NET: is the L/S book negative even at zero cost (=> signal problem)
     or only after cost (=> turnover/fee problem, fixable with maker/low-turnover)?
  3. Long-only top-quintile vs equal-weight market (monetize as a tilt?)
  4. maker-fee (1bp) + weekly rebalance (lowest-turnover monetization)
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import experiments.alpha_miner as am
import experiments.alpha_bigU as ab
from engine import backtest as bt


def main():
    rng = np.random.default_rng(7)
    feat, fwd_log, fwd_simple, P, coins, idx, breadth = ab.build_panel_big()
    T = feat["ret"].shape[0]; tr, te = bt.oos_split(T, ab.TRAIN_FRAC); feats = list(feat.keys())
    pool, seen = [], set()
    for _ in range(ab.K_SEARCH):
        t = am.rand_tree(feats, rng, int(rng.integers(2, 5))); k = am.to_str(t)
        if k in seen:
            continue
        seen.add(k)
        try:
            a = am.ev(t, feat)
            if not np.isfinite(a).any():
                continue
            ic, icir = am.rank_ic(a, fwd_log, tr)
            if abs(icir) > 0.05:
                pool.append((abs(icir), np.sign(ic) if ic else 1.0, a))
        except Exception:
            continue
    pool.sort(key=lambda x: -x[0]); pool = pool[:300]
    Z = np.stack([am._zscore_rows(np.nan_to_num(s * a)) for _, s, a in pool[:60]])
    icw = np.array([w for w, _, _ in pool[:60]])[:, None, None]
    A = (Z * icw).sum(0) / icw.sum()                       # combined alpha
    r = fwd_simple
    oic, _ = am.rank_ic(A, fwd_log, te)
    print(f"combined OOS Rank IC = {oic:.4f}\n")

    # 1. decile monotonicity (OOS): mean next-day return per alpha-quintile
    Q = 5
    qret = np.full((te.stop - te.start, Q), np.nan)
    for i, t in enumerate(range(te.start, te.stop)):
        row, rr = A[t], r[t]; m = np.isfinite(row) & np.isfinite(rr)
        if m.sum() >= Q * 2:
            order = np.argsort(row[m]); rrm = rr[m][order]
            for q in range(Q):
                seg = rrm[q * len(rrm) // Q:(q + 1) * len(rrm) // Q]
                qret[i, q] = seg.mean()
    qmean = np.nanmean(qret, 0) * 1e4
    print("OOS mean next-day return by alpha quintile (bp):")
    print("  Q1(low)..Q5(high): " + "  ".join(f"{x:6.1f}" for x in qmean))
    print(f"  monotone-increasing? {'YES' if np.all(np.diff(qmean) > -3) else 'NO'};  "
          f"Q5-Q1 spread = {qmean[-1]-qmean[0]:.1f}bp/day\n")

    # 2. GROSS vs NET L/S (rank weights, lag0)
    w = ab.rank_weights(A)
    for cost, lbl in [(0.0, "GROSS"), (1.0, "maker 1bp"), (5.0, "taker 5bp")]:
        net = ab.book_from_weights(w, r, cost, lag=0)[te]; mo = bt.metrics(net, 365)
        print(f"  rank L/S {lbl:10}: OOS Sharpe {mo['sharpe_ann']:6.2f}  ret {mo['ret_ann']*100:7.2f}%/yr")
    # weekly rebalance (lowest turnover), gross + maker
    ww = w.copy()
    for t in range(1, T):
        if t % 7 != 0:
            ww[t] = ww[t - 1]
    for cost, lbl in [(0.0, "GROSS"), (1.0, "maker 1bp")]:
        net = ab.book_from_weights(ww, r, cost, lag=0)[te]; mo = bt.metrics(net, 365)
        print(f"  weekly L/S {lbl:10}: OOS Sharpe {mo['sharpe_ann']:6.2f}  ret {mo['ret_ann']*100:7.2f}%/yr")

    # 3. long-only top quintile vs equal-weight market (excess)
    n = A.shape[1]
    long_w = np.zeros_like(A)
    for t in range(T):
        row = A[t]; m = np.isfinite(row)
        if m.sum() >= 10:
            idxm = np.where(m)[0]; top = idxm[np.argsort(row[m])][-max(1, m.sum() // 5):]
            long_w[t, top] = 1.0 / len(top)
    mkt = np.zeros_like(A)
    for t in range(T):
        m = np.isfinite(A[t]) & np.isfinite(r[t])
        if m.sum() > 0:
            mkt[t, np.where(m)[0]] = 1.0 / m.sum()
    lo = ab.book_from_weights(long_w, r, 5.0, lag=0)[te]
    mk = np.nansum(np.vstack([np.zeros((1, n)), mkt[:-1]]) * r, 1)[te]
    excess = lo - mk
    mo = bt.metrics(excess, 365)
    print(f"\n  long top-quintile MINUS equal-wt market (excess): OOS Sharpe {mo['sharpe_ann']:.2f}  ret {mo['ret_ann']*100:.2f}%/yr")
    print("\nINTERPRETATION:")
    print("  - decile non-monotone or Q5-Q1<=0  => IC is mid-rank noise, untradeable as L/S")
    print("  - GROSS L/S<=0                      => signal genuinely has no tradable spread (not a cost problem)")
    print("  - GROSS>0 but net<0                 => cost/turnover problem (maker/weekly may rescue)")


if __name__ == "__main__":
    main()
