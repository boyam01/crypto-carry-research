"""Non-linear monetization: trade the QUINTILE SHAPE, not a linear L/S.

The combined alpha's quintile profile is non-monotone (Q1 best, middle worst) —
a linear L/S bets on monotonicity and necessarily loses on a U-shape. Here we:
  1. show the quintile profile in IS-A, IS-B, OOS (is the SHAPE stable?)
  2. fit quintile weights = demeaned IS mean-return-per-quintile (IS only)
  3. apply that fixed shape OOS -> dollar-neutral book (longs IS-good quintiles,
     shorts IS-bad), gross + net, deflated.
If the non-monotone shape is stable IS->OOS this monetizes; if it regime-flips, dead.
"""
from __future__ import annotations
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import experiments.alpha_miner as am
import experiments.alpha_bigU as ab
from engine import backtest as bt

K = 2500
Q = 5


def combined_alpha():
    rng = np.random.default_rng(7)
    feat, fwd_log, fwd_simple, P, coins, idx, breadth = ab.build_panel_big()
    T = feat["ret"].shape[0]; cut = int(T * 0.60)
    isA = slice(0, cut // 2); isB = slice(cut // 2, cut); te = slice(cut, T)
    feats = list(feat.keys())
    stable, seen = [], set()
    for _ in range(K):
        t = am.rand_tree(feats, rng, int(rng.integers(2, 5))); k = am.to_str(t)
        if k in seen:
            continue
        seen.add(k)
        try:
            a = am.ev(t, feat)
            if not np.isfinite(a).any():
                continue
            icA, irA = am.rank_ic(a, fwd_log, isA); icB, irB = am.rank_ic(a, fwd_log, isB)
        except Exception:
            continue
        if icA * icB > 0 and abs(irA) > 0.08 and abs(irB) > 0.08:
            stable.append((min(abs(irA), abs(irB)), np.sign(icA), a))
    stable.sort(key=lambda x: -x[0])
    Z = np.stack([am._zscore_rows(np.nan_to_num(s * a)) for w, s, a in stable[:60]])
    return Z.mean(0), fwd_simple, slice(0, cut // 2), slice(cut // 2, cut), slice(cut, T), T


def quintile_profile(A, r, sl):
    prof = np.full((sl.stop - sl.start, Q), np.nan)
    for i, t in enumerate(range(sl.start, sl.stop)):
        row, rr = A[t], r[t]; m = np.isfinite(row) & np.isfinite(rr)
        if m.sum() >= Q * 2:
            order = np.argsort(row[m]); rrm = rr[m][order]
            for q in range(Q):
                prof[i, q] = rrm[q * len(rrm) // Q:(q + 1) * len(rrm) // Q].mean()
    return np.nanmean(prof, 0)


def shape_book(A, r, shape, sl, cost_bps):
    """assign each name to its quintile daily; weight = shape[quintile]."""
    w = np.zeros_like(A)
    for t in range(A.shape[0]):
        row = A[t]; m = np.isfinite(row); n = m.sum()
        if n >= Q * 2:
            idxm = np.where(m)[0]; order = idxm[np.argsort(row[m])]
            for q in range(Q):
                seg = order[q * n // Q:(q + 1) * n // Q]
                if len(seg):
                    w[t, seg] = shape[q] / len(seg)
    g = np.nansum(np.abs(w), axis=1, keepdims=True)
    w = w / (g + 1e-9)
    return ab.book_from_weights(w, r, cost_bps, lag=0)[sl]


def main():
    A, r, isA, isB, te, T = combined_alpha()
    pA, pB, pO = (quintile_profile(A, r, s) * 1e4 for s in (isA, isB, te))
    print("quintile mean next-day return (bp), by period:")
    print("        " + "".join(f"  Q{q+1}" + " " * 4 for q in range(Q)))
    for lbl, p in [("IS-A", pA), ("IS-B", pB), ("OOS ", pO)]:
        print(f"  {lbl}: " + "  ".join(f"{x:6.1f}" for x in p))
    # is the shape stable? correlation of IS-full shape vs OOS shape
    pIS = (pA + pB) / 2
    shp = pIS - pIS.mean()
    shape_corr = np.corrcoef(pIS, pO)[0, 1]
    print(f"\nIS quintile-shape vs OOS shape correlation = {shape_corr:.3f}")
    print(f"  (>0 and high => the non-monotone shape is STABLE and tradable; <0 => it regime-flips)")

    print(f"\nfit shape on IS (demeaned IS quintile returns): {np.round(shp,4)}")
    for c, lbl in [(0.0, "GROSS"), (1.0, "maker1bp"), (5.0, "taker5bp")]:
        net = shape_book(A, r, shp, te, c); mo = bt.metrics(net, 365)
        p = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
        print(f"  shape-book {lbl:9}: OOS Sharpe {mo['sharpe_ann']:6.2f}  ret {mo['ret_ann']*100:7.2f}%/yr  PSR {p:.3f}")
    netg = shape_book(A, r, shp, te, 0.0); mg = bt.metrics(netg, 365)
    nett = shape_book(A, r, shp, te, 5.0); mt = bt.metrics(nett, 365)
    pt = bt.psr(mt["sr_pp"], mt["n"], mt["skew"], mt["kurt"])
    survives = mg["sharpe_ann"] > 0.3 and shape_corr > 0.3 and pt > 0.90
    print(f"\nVERDICT: {'LEAD — stable non-monotone shape monetizes (verify next)' if survives else 'DEAD — shape regime-flips or gross<=0; the non-linear book does not rescue it either'}")


if __name__ == "__main__":
    main()
