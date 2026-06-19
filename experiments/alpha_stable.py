"""Stability-selected ensemble — the principled fix for the diagnosed failure.

Diagnosis (alpha_diag): IS-IC-max selection overfits -> combined alpha sign-FLIPS
OOS (Q1>Q5), gross L/S Sharpe -0.53. Root cause = selecting formulas on IS noise.
Fix: split IS into two halves (IS-A, IS-B); keep ONLY formulas whose IC has the
SAME sign and decent magnitude in BOTH halves (walk-forward stability). Combine
those, judge OOS by quintile monotonicity + GROSS-then-net L/S Sharpe.

If stability-selected formulas transfer to OOS -> a real combinable edge.
If they also flip -> the signal is genuinely non-stationary, definitively no edge.
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import experiments.alpha_miner as am
import experiments.alpha_bigU as ab
from engine import backtest as bt

K = 2500


def main():
    rng = np.random.default_rng(7)
    feat, fwd_log, fwd_simple, P, coins, idx, breadth = ab.build_panel_big()
    T = feat["ret"].shape[0]
    cut = int(T * 0.60)
    isA = slice(0, cut // 2); isB = slice(cut // 2, cut); te = slice(cut, T)
    feats = list(feat.keys())
    print(f"BIG panel {T}d x {len(coins)} coins. IS-A={isA.stop}d IS-B={isB.stop-isB.start}d OOS={T-cut}d. searching {K}...")

    stable, allf, seen = [], [], set()
    for _ in range(K):
        t = am.rand_tree(feats, rng, int(rng.integers(2, 5))); k = am.to_str(t)
        if k in seen:
            continue
        seen.add(k)
        try:
            a = am.ev(t, feat)
            if not np.isfinite(a).any():
                continue
            icA, irA = am.rank_ic(a, fwd_log, isA)
            icB, irB = am.rank_ic(a, fwd_log, isB)
        except Exception:
            continue
        allf.append((icA, icB))
        # STABILITY: same sign in both IS halves + both ICIR meaningful
        if icA * icB > 0 and abs(irA) > 0.08 and abs(irB) > 0.08:
            s = np.sign(icA)
            stable.append((min(abs(irA), abs(irB)), s, a))
    stable.sort(key=lambda x: -x[0])
    allf = np.array(allf)
    cross = np.corrcoef(allf[:, 0], allf[:, 1])[0, 1] if len(allf) > 5 else float("nan")
    print(f"searched {len(allf)} formulas; IS-A vs IS-B IC correlation across formulas = {cross:.3f}")
    print(f"  (high => IC is stable/learnable; ~0 => IS IC is pure noise that won't transfer)")
    print(f"stability-selected (same sign both halves, |ICIR|>0.08 each): {len(stable)} formulas")

    if len(stable) < 3:
        print("\nToo few stable formulas to combine -> the formulas have NO IS-persistent signal.")
        print("VERDICT: DEAD — even within-IS the formula edges do not persist across sub-periods.")
        return

    Z = np.stack([am._zscore_rows(np.nan_to_num(s * a)) for w, s, a in stable[:60]])
    A = Z.mean(0)
    oicA = am.rank_ic(A, fwd_log, isA)[0]; oicB = am.rank_ic(A, fwd_log, isB)[0]
    oic = am.rank_ic(A, fwd_log, te)[0]
    print(f"\ncombined (stable) Rank IC: IS-A {oicA:+.4f}  IS-B {oicB:+.4f}  OOS {oic:+.4f}")

    # quintile monotonicity OOS
    Q = 5; r = fwd_simple
    qr = np.full((te.stop - te.start, Q), np.nan)
    for i, t in enumerate(range(te.start, te.stop)):
        row, rr = A[t], r[t]; m = np.isfinite(row) & np.isfinite(rr)
        if m.sum() >= Q * 2:
            order = np.argsort(row[m]); rrm = rr[m][order]
            for q in range(Q):
                qr[i, q] = rrm[q * len(rrm) // Q:(q + 1) * len(rrm) // Q].mean()
    qm = np.nanmean(qr, 0) * 1e4
    print("OOS quintile next-day ret (bp): " + "  ".join(f"{x:6.1f}" for x in qm) + f"   Q5-Q1={qm[-1]-qm[0]:+.1f}bp")

    w = ab.rank_weights(A)
    for c, lbl in [(0.0, "GROSS"), (1.0, "maker1bp"), (5.0, "taker5bp")]:
        mo = bt.metrics(ab.book_from_weights(w, r, c, lag=0)[te], 365)
        print(f"  stable L/S {lbl:9}: OOS Sharpe {mo['sharpe_ann']:6.2f}  ret {mo['ret_ann']*100:7.2f}%/yr")
    grossmo = bt.metrics(ab.book_from_weights(w, r, 0.0, lag=0)[te], 365)
    print(f"\nVERDICT: {'LEAD — stable selection transfers + monotone + gross>0; verify next' if (grossmo['sharpe_ann']>0.3 and qm[-1]-qm[0]>3) else 'DEAD — stability selection still does not produce a monotone, gross-positive OOS book'}")


if __name__ == "__main__":
    main()
