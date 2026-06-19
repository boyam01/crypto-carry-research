"""Alpha COMBINATION layer — test the user's "排列組合肯定有肉" hypothesis.

Single formulas died (alpha_miner: best OOS L/S Sharpe 0.43, deflated DEAD). But a
PORTFOLIO of decorrelated weak alphas can beat any single one. We:
  1. search a pool of formulas, keep the top by IS ICIR (with their arrays)
  2. sign-align each by IS IC; greedily pick a DECORRELATED subset (IS only)
  3. combine 4 ways (equal-wt, IS-IC-wt, greedy-decorrelated equal-wt, IS ridge)
  4. judge each COMBINED book OOS (cost + PSR), deflate over the few combo configs

Honesty: selection + weights are IS-fit; OOS is untouched; deflation is over the
handful of combination configs tried (NOT 4120 — the search produced the pool).
"""
from __future__ import annotations
import sys, time, pathlib, json
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import experiments.alpha_miner as am
from engine import backtest as bt

K_SEARCH = 2500
POOL = 400          # keep top-N by IS |ICIR| as the combination pool
SEED = 7


def main():
    rng = np.random.default_rng(SEED)
    feat, fwd_log, fwd_simple, coins, idx = am.build_panel()
    T = feat["ret"].shape[0]; tr, te = bt.oos_split(T, am.TRAIN_FRAC)
    feats = list(feat.keys())
    print(f"panel {T}d x {len(coins)} coins; searching {K_SEARCH} formulas for the pool...")

    pool = []   # (is_icir, is_ic, alpha_array)
    seen = set()
    for _ in range(K_SEARCH):
        t = am.rand_tree(feats, rng, int(rng.integers(2, 5)))
        key = am.to_str(t)
        if key in seen:
            continue
        seen.add(key)
        try:
            a = am.ev(t, feat)
            if not np.isfinite(a).any():
                continue
            ic, icir = am.rank_ic(a, fwd_log, tr)
            if abs(icir) > 0.05:
                pool.append((icir, ic, a))
        except Exception:
            continue
    pool.sort(key=lambda x: -abs(x[0]))
    pool = pool[:POOL]
    print(f"pool: {len(pool)} formulas with |IS ICIR|>0.05 (top {POOL} kept)")

    # sign-align each alpha so IS IC > 0, then per-day z-score for combination
    aligned = []
    for icir, ic, a in pool:
        s = np.sign(ic) if ic != 0 else 1.0
        aligned.append((abs(icir), am._zscore_rows(np.nan_to_num(s * a))))
    aligned.sort(key=lambda x: -x[0])

    # greedy decorrelated subset on IS alpha values
    def is_flat(z):
        return np.nan_to_num(z[tr]).ravel()
    selected = [aligned[0][1]]
    sel_flat = [is_flat(aligned[0][1])]
    for w, z in aligned[1:]:
        f = is_flat(z)
        if max(abs(np.corrcoef(f, sf)[0, 1]) for sf in sel_flat) < 0.5:
            selected.append(z); sel_flat.append(f)
        if len(selected) >= 25:
            break
    print(f"greedy-decorrelated subset: {len(selected)} alphas (pairwise |IS corr|<0.5)")

    combos = {}
    Z_all = np.stack([z for _, z in aligned[:50]])          # top-50 aligned
    combos["equal_top50"] = Z_all.mean(0)
    icw = np.array([w for w, _ in aligned[:50]])[:, None, None]
    combos["icwt_top50"] = (Z_all * icw).sum(0) / icw.sum()
    combos["decorrelated_eqwt"] = np.mean(np.stack(selected), 0)

    # IS ridge: forward_ret ~ alphas, fit IS, predict OOS
    from sklearn.linear_model import Ridge
    Zr = np.stack(selected)                                  # (m, T, N)
    m, _, N = Zr.shape
    X = Zr.reshape(m, T * N).T                               # (T*N, m)
    y = fwd_log.reshape(T * N)
    msk = np.isfinite(X).all(1) & np.isfinite(y)
    is_rows = np.zeros(T * N, bool); is_rows[:tr.stop * N] = True
    rr = Ridge(alpha=10.0).fit(X[msk & is_rows], y[msk & is_rows])
    pred = np.full(T * N, np.nan); pred[msk] = rr.predict(X[msk])
    combos["is_ridge"] = pred.reshape(T, N)

    # evaluate each COMBINED book OOS
    print(f"\n{'combo':22} {'OOS_IC':>7} {'OOS_Sharpe':>11} {'OOS_ret%':>9} {'maxDD%':>8} {'PSR':>6}")
    res = {}
    sr_pps = []
    for name, alpha in combos.items():
        oic, _ = am.rank_ic(alpha, fwd_log, te)
        net = am.ls_net(alpha, fwd_simple, te); mo = bt.metrics(net, 365)
        p = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
        res[name] = dict(oos_ic=oic, oos_sharpe=mo["sharpe_ann"], oos_ret=mo["ret_ann"],
                         maxdd=mo["maxdd"], psr=p, sr_pp=mo["sr_pp"])
        if np.isfinite(mo["sr_pp"]):
            sr_pps.append(mo["sr_pp"])
        print(f"{name:22} {oic:7.4f} {mo['sharpe_ann']:11.2f} {mo['ret_ann']*100:9.2f} {mo['maxdd']*100:8.2f} {p:6.3f}")

    # deflate over the combo configs tried (+ a few implicit knobs)
    n_configs = len(combos) + 3            # +pool-size/threshold/topN knobs implicitly tried
    sr_star = bt.dsr_benchmark(sr_pps + [0.0] * 3) if len(sr_pps) > 1 else 0.0
    best = max(res.items(), key=lambda kv: kv[1]["oos_sharpe"] if np.isfinite(kv[1]["oos_sharpe"]) else -9)
    bdsr = bt.psr(best[1]["sr_pp"], (te.stop - te.start), 0, 3, sr_benchmark=sr_star)
    survives = best[1]["oos_sharpe"] > 0 and best[1]["psr"] > 0.95 and bdsr > 0.95
    print(f"\nbest combo = {best[0]}  OOS Sharpe {best[1]['oos_sharpe']:.2f}  PSR {best[1]['psr']:.3f}")
    print(f"deflated over ~{n_configs} combo configs: DSR* per-period={sr_star:.4f}, best deflated PSR={bdsr:.3f}")
    print(f"VERDICT: {'SURVIVES — combination yields a real OOS edge the singles lacked' if survives else 'DEAD — combining the formulas does not beat deflation either'}")

    out = pathlib.Path(__file__).resolve().parent.parent / "reports" / "alpha_ensemble_result.json"
    out.write_text(json.dumps(dict(pool=len(pool), selected=len(selected), sr_star=sr_star,
                                   best=best[0], survives=bool(survives), combos=res), indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
