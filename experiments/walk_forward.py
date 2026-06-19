"""Walk-forward alpha combination — the textbook cure for the diagnosed
non-stationarity (IS->OOS sign flip, shape corr -0.507).

Instead of one fixed 60/40 split (which a 2-year-stale fit guarantees to fail),
re-fit the COMBINATION on a TRAILING window every month and trade the next month,
so the model tracks the drifting relationship. Formula pool fixed on the first
year only (no look-ahead); ridge weights re-estimated monthly on trailing data.

Causal throughout: at rebalance t, fit ridge on [t-LOOKBACK, t) (forward returns
known through t-1), predict scores for [t, t+STEP), trade earning ret[s->s+1].
Chained walk-forward OOS = day 252..end. Judged gross + net + vs buy&hold market.
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import experiments.alpha_miner as am
import experiments.alpha_bigU as ab
from engine import backtest as bt

K = 2500
POOLN = 80
LOOKBACK = 252
STEP = 21


def main():
    rng = np.random.default_rng(7)
    feat, fwd_log, fwd_simple, P, coins, idx, breadth = ab.build_panel_big()
    T = feat["ret"].shape[0]; N = len(coins); feats = list(feat.keys())
    sel = slice(0, LOOKBACK)                                   # pool selection window (year 1 only)
    print(f"BIG panel {T}d x {N} coins. pool selected on first {LOOKBACK}d; walk-forward {LOOKBACK}..{T} "
          f"(lookback {LOOKBACK}d, step {STEP}d).")

    pool, seen = [], set()
    for _ in range(K):
        t = am.rand_tree(feats, rng, int(rng.integers(2, 5))); k = am.to_str(t)
        if k in seen:
            continue
        seen.add(k)
        try:
            a = am.ev(t, feat)
            if not np.isfinite(a).any():
                continue
            ic, icir = am.rank_ic(a, fwd_log, sel)
            if abs(icir) > 0.05:
                pool.append((abs(icir), am._zscore_rows(np.nan_to_num(a))))
        except Exception:
            continue
    pool.sort(key=lambda x: -x[0]); pool = pool[:POOLN]
    AL = np.stack([a for _, a in pool])                       # (P, T, N) fixed alpha library
    print(f"fixed pool: {len(pool)} formulas (selected on first {LOOKBACK}d only)")

    pred = np.full((T, N), np.nan)                            # walk-forward predicted scores
    for t0 in range(LOOKBACK, T, STEP):
        tr0 = max(0, t0 - LOOKBACK)
        # design over trailing window rows (s in [tr0, t0)), last usable fwd at t0-1
        Xtr, ytr = [], []
        for s in range(tr0, t0):
            m = np.isfinite(fwd_log[s]) & np.all(np.isfinite(AL[:, s, :]), axis=0)
            if m.sum() > 4:
                Xtr.append(AL[:, s, m].T); ytr.append(fwd_log[s, m])
        if not Xtr:
            continue
        X = np.vstack(Xtr); y = np.concatenate(ytr)
        rr = Ridge(alpha=20.0).fit(X, y)
        for s in range(t0, min(t0 + STEP, T)):                # predict next STEP days
            m = np.all(np.isfinite(AL[:, s, :]), axis=0)
            if m.sum() > 4:
                pred[s, m] = rr.predict(AL[:, s, m].T)

    wf = slice(LOOKBACK, T)
    oic, _ = am.rank_ic(pred, fwd_log, wf)
    print(f"\nwalk-forward OOS Rank IC = {oic:.4f}")
    # quintile profile (walk-forward)
    Q = 5; qr = np.full((T - LOOKBACK, Q), np.nan)
    for i, t in enumerate(range(LOOKBACK, T)):
        row, rr2 = pred[t], fwd_simple[t]; m = np.isfinite(row) & np.isfinite(rr2)
        if m.sum() >= Q * 2:
            order = np.argsort(row[m]); rrm = rr2[m][order]
            for q in range(Q):
                qr[i, q] = rrm[q * len(rrm) // Q:(q + 1) * len(rrm) // Q].mean()
    qm = np.nanmean(qr, 0) * 1e4
    print("walk-forward quintile next-day ret (bp): " + "  ".join(f"{x:6.1f}" for x in qm) + f"   Q5-Q1={qm[-1]-qm[0]:+.1f}")

    w = ab.rank_weights(pred)
    for c, lbl in [(0.0, "GROSS"), (1.0, "maker1bp"), (5.0, "taker5bp")]:
        net = ab.book_from_weights(w, fwd_simple, c, lag=0)[wf]; mo = bt.metrics(net, 365)
        p = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
        print(f"  walk-fwd L/S {lbl:9}: Sharpe {mo['sharpe_ann']:6.2f}  ret {mo['ret_ann']*100:7.2f}%/yr  maxDD {mo['maxdd']*100:6.1f}%  PSR {p:.3f}")
    g = bt.metrics(ab.book_from_weights(w, fwd_simple, 0.0, lag=0)[wf], 365)
    nt = bt.metrics(ab.book_from_weights(w, fwd_simple, 5.0, lag=0)[wf], 365)
    pt = bt.psr(nt["sr_pp"], nt["n"], nt["skew"], nt["kurt"])
    survives = g["sharpe_ann"] > 0.5 and nt["sharpe_ann"] > 0.3 and pt > 0.90
    print(f"\nVERDICT: {'LEAD — walk-forward tracks the drift; tradeable net edge (verify hard next)' if survives else 'DEAD — even monthly walk-forward re-fitting cannot track the non-stationarity'}")


if __name__ == "__main__":
    main()
