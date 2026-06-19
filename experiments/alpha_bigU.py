"""Test the lead from alpha_ensemble: the combined alpha has POSITIVE OOS IC (~0.08)
but NEGATIVE L/S Sharpe on 14 coins. Hypothesis: the signal is real but the
universe is too thin to monetize. Expand to ~50 crypto perps (ragged/NaN-tolerant
panel) and try multiple monetization schemes on the combined alpha.

Discipline unchanged: search+combine on IS (first 60%), judge OOS (last 40%),
cost-charged, deflate over the schemes tried. A positive OOS IC that finally
turns into a deflation-surviving OOS Sharpe = real edge; if not, the signal is
genuinely unmonetizable here.
"""
from __future__ import annotations
import sys, time, pathlib, json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import experiments.alpha_miner as am
from engine import fetch_binance as fb
from engine import backtest as bt

BIG = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","LTCUSDT",
"LINKUSDT","AVAXUSDT","TRXUSDT","DOTUSDT","ATOMUSDT","NEARUSDT","MATICUSDT","UNIUSDT",
"ETCUSDT","BCHUSDT","FILUSDT","XLMUSDT","ALGOUSDT","SANDUSDT","MANAUSDT","AXSUSDT",
"FTMUSDT","THETAUSDT","EGLDUSDT","VETUSDT","AAVEUSDT","EOSUSDT","XTZUSDT","ICPUSDT",
"APEUSDT","GALAUSDT","CHZUSDT","ENJUSDT","ZILUSDT","HBARUSDT","GRTUSDT","1INCHUSDT",
"CRVUSDT","COMPUSDT","SNXUSDT","YFIUSDT","DASHUSDT","ZECUSDT","WAVESUSDT","KSMUSDT",
"RUNEUSDT","SUSHIUSDT"]
K_SEARCH = 1500
TRAIN_FRAC = 0.60


def build_panel_big():
    end = int(time.time() * 1000); start = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() * 1000)
    kl, fr = {}, {}
    for s in BIG:
        try:
            k = fb.klines(s, "1d", start, end, futures=True)
            f = fb.funding_rate(s, start, end)
        except Exception:
            continue
        if k is not None and len(k) > 400:
            kl[s] = k; fr[s] = f
    coins = list(kl.keys())
    idx = None                                            # UNION of all indices (ragged)
    for s in coins:
        idx = kl[s].index if idx is None else idx.union(kl[s].index)
    idx = idx.sort_values()

    def M(col):
        return pd.DataFrame({s: kl[s][col].reindex(idx) for s in coins}).values.astype(float)
    P, H, L, O = M("close"), M("high"), M("low"), M("open")
    V, QV, TB = M("volume"), M("qvol"), M("tbbav")
    fund = pd.DataFrame({s: (fr[s]["fundingRate"].resample("1D").sum() if fr[s] is not None
                             else kl[s]["close"] * 0).reindex(idx).fillna(0) for s in coins}).values
    with np.errstate(all="ignore"):
        ret = np.vstack([np.full((1, P.shape[1]), np.nan), np.diff(np.log(P), axis=0)])
        ret5 = np.vstack([np.full((5, P.shape[1]), np.nan), np.log(P[5:] / P[:-5])])
        feat = {
            "ret": am._zscore_rows(ret), "ret5": am._zscore_rows(ret5),
            "hl": am._zscore_rows((H - L) / P), "co": am._zscore_rows((P - O) / O),
            "vol": am._zscore_rows(np.log(V + 1)), "ofi": am._zscore_rows((2 * TB - V) / (V + 1e-9)),
            "vwapdev": am._zscore_rows(P / ((H + L + P) / 3) - 1), "fund": am._zscore_rows(fund),
            "rng": am._zscore_rows((H - L) / (np.abs(P - O) + 1e-9)),
        }
        fwd_log = np.vstack([np.diff(np.log(P), axis=0), np.full((1, P.shape[1]), np.nan)])
        fwd_simple = np.vstack([P[1:] / P[:-1] - 1, np.full((1, P.shape[1]), np.nan)])
    # avg names per day (cross-section breadth)
    breadth = np.isfinite(P).sum(1)
    return feat, fwd_log, fwd_simple, P, coins, idx, breadth


def book_from_weights(w, r, cost_bps, lag=0):
    """net[t] = w_lagged[t] * r[t] - cost*turnover.  r[t]=return t->t+1.
    lag=0: decide w at close t, hold to t+1 (matches the IC, look-ahead-free).
    lag=1: +1 conservative execution day (w from t-1). turnover on the rebalance."""
    w = np.nan_to_num(w)
    if lag:
        w = np.vstack([np.zeros((lag, w.shape[1])), w[:-lag]])
    wprev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    turn = np.abs(w - wprev).sum(axis=1)
    return np.nansum(w * r, axis=1) - cost_bps / 1e4 * turn


def rank_weights(a):
    w = np.zeros_like(a)
    for t in range(a.shape[0]):
        row = a[t]; m = np.isfinite(row)
        if m.sum() > 6:
            rk = rankdata(row[m]) - (m.sum() + 1) / 2
            w[t, m] = rk / (np.abs(rk).sum() + 1e-9)
    return w


def decile_weights(a, frac=0.2):
    w = np.zeros_like(a)
    for t in range(a.shape[0]):
        row = a[t]; m = np.isfinite(row); n = m.sum()
        if n >= 10:
            k = max(1, int(n * frac)); idxm = np.where(m)[0]; order = idxm[np.argsort(row[m])]
            w[t, order[-k:]] = 0.5 / k; w[t, order[:k]] = -0.5 / k
    return w


def main():
    rng = np.random.default_rng(7)
    feat, fwd_log, fwd_simple, P, coins, idx, breadth = build_panel_big()
    T = feat["ret"].shape[0]; tr, te = bt.oos_split(T, TRAIN_FRAC)
    feats = list(feat.keys())
    print(f"BIG panel {T}d x {len(coins)} coins; OOS avg breadth={breadth[te].mean():.0f} names/day "
          f"(vs 14 before); searching {K_SEARCH} formulas...")

    pool, seen = [], set()
    for _ in range(K_SEARCH):
        t = am.rand_tree(feats, rng, int(rng.integers(2, 5))); key = am.to_str(t)
        if key in seen:
            continue
        seen.add(key)
        try:
            a = am.ev(t, feat)
            if not np.isfinite(a).any():
                continue
            ic, icir = am.rank_ic(a, fwd_log, tr)
            if abs(icir) > 0.05:
                pool.append((abs(icir), np.sign(ic) if ic else 1.0, a))
        except Exception:
            continue
    pool.sort(key=lambda x: -x[0])
    pool = pool[:300]
    print(f"pool {len(pool)} formulas |IS ICIR|>0.05")
    Z = np.stack([am._zscore_rows(np.nan_to_num(s * a)) for _, s, a in pool[:60]])
    icw = np.array([w for w, _, _ in pool[:60]])[:, None, None]
    combined = (Z * icw).sum(0) / icw.sum()                # IS-IC-weighted combined alpha

    oic, _ = am.rank_ic(combined, fwd_log, te)
    print(f"combined alpha OOS Rank IC = {oic:.4f}  (positive => real signal; question is monetization)\n")

    # inverse-vol scaling for one scheme
    vol20 = np.full_like(P, np.nan)
    lr = np.vstack([np.zeros((1, P.shape[1])), np.diff(np.log(P), axis=0)])
    for t in range(20, T):
        vol20[t] = np.nanstd(lr[t - 20:t], axis=0)
    schemes = {
        "rank_LS": rank_weights(combined),
        "decile_20pct": decile_weights(combined, 0.2),
        "icwt": combined / (np.nansum(np.abs(combined), axis=1, keepdims=True) + 1e-9),
        "rank_volscaled": rank_weights(combined) / (vol20 + 1e-9) ,
    }
    # 3-day hold version of the best-breadth scheme (cut turnover)
    rw = rank_weights(combined); rw3 = rw.copy()
    for t in range(1, T):
        if t % 3 != 0:
            rw3[t] = rw3[t - 1]
    schemes["rank_hold3d"] = rw3

    print(f"{'scheme':16} {'OOS_Sharpe':>11} {'OOS_ret%':>9} {'maxDD%':>8} {'turn':>7} {'PSR':>6}")
    res, sr_pps = {}, []
    for name, w in schemes.items():
        if name == "rank_volscaled":                       # renormalize gross to 1
            g = np.nansum(np.abs(w), axis=1, keepdims=True); w = w / (g + 1e-9)
        net0 = book_from_weights(w, fwd_simple, 5.0, lag=0)[te]
        net1 = book_from_weights(w, fwd_simple, 5.0, lag=1)[te]
        mo = bt.metrics(net0, 365); mo1 = bt.metrics(net1, 365)
        p = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
        res[name] = dict(sharpe=mo["sharpe_ann"], sharpe_lag1=mo1["sharpe_ann"], ret=mo["ret_ann"],
                         maxdd=mo["maxdd"], turn=float(np.abs(np.diff(np.nan_to_num(w[te]),axis=0,prepend=0)).sum(1).mean()),
                         psr=p, sr_pp=mo["sr_pp"])
        if np.isfinite(mo["sr_pp"]):
            sr_pps.append(mo["sr_pp"])
        print(f"{name:16} {mo['sharpe_ann']:11.2f} {mo['ret_ann']*100:9.2f} {mo['maxdd']*100:8.2f} {res[name]['turn']:7.3f} {p:6.3f}   (lag1 Sharpe {mo1['sharpe_ann']:.2f})")

    sr_star = bt.dsr_benchmark(sr_pps + [0.0, 0.0]) if len(sr_pps) > 1 else 0.0
    best = max(res.items(), key=lambda kv: kv[1]["sharpe"] if np.isfinite(kv[1]["sharpe"]) else -9)
    bdsr = bt.psr(best[1]["sr_pp"], te.stop - te.start, 0, 3, sr_benchmark=sr_star)
    survives = best[1]["sharpe"] > 0 and best[1]["psr"] > 0.95 and bdsr > 0.95
    print(f"\ncombined OOS IC={oic:.4f}; best scheme={best[0]} OOS Sharpe={best[1]['sharpe']:.2f} PSR={best[1]['psr']:.3f}")
    print(f"deflated (DSR* per-period={sr_star:.4f}) best deflated PSR={bdsr:.3f}")
    print(f"VERDICT: {'SURVIVES — bigger universe monetizes the signal' if survives else 'DEAD — signal real (OOS IC>0) but still not a deflation-surviving tradable book'}")
    out = pathlib.Path(__file__).resolve().parent.parent / "reports" / "alpha_bigU_result.json"
    out.write_text(json.dumps(dict(coins=len(coins), oos_breadth=float(breadth[te].mean()),
        combined_oos_ic=oic, sr_star=sr_star, best=best[0], survives=bool(survives), schemes=res), indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
