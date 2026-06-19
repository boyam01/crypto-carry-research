"""Formulaic-alpha search (WorldQuant-101 / genetic-programming style).

The operationalization of "apply all formulas": SEARCH the formula space instead
of hand-picking. Each alpha is a tree over the operator library from the research
doc (arithmetic, protected unary, ts_*, cross-sectional rank/neutralize) on a
panel of per-day-standardized features. Fitness = cross-sectional Rank IC.

Discipline (GP is the MOST overfit-prone method — deflation is everything):
- features are per-day z-scored => stationary, comparable, no price-level artifacts
- evolve/select ONLY on IS (first 60%); judge on OOS (last 40%)
- the real test is OOS *tradable L/S Sharpe*, not IC (a formula can have
  significant OOS IC yet lose money — seen in v1: IC +0.10, Sharpe -1.38)
- DEFLATE the best OOS Sharpe over the number of formulas searched (DSR, Bailey-LdP)
"""
from __future__ import annotations
import sys, time, pathlib, json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import fetch_binance as fb
from engine import backtest as bt

UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
            "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT", "TRXUSDT", "DOTUSDT",
            "ATOMUSDT", "NEARUSDT"]
START = "2022-01-01"
TRAIN_FRAC = 0.60
K = 4000


def _zscore_rows(x):
    mu = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, keepdims=True)
    return np.clip((x - mu) / (sd + 1e-9), -4, 4)


def build_panel():
    end = int(time.time() * 1000); start = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    raw = {}
    for s in UNIVERSE:
        k = fb.klines(s, "1d", start, end, futures=True)
        fr = fb.funding_rate(s, start, end)
        if k is None or len(k) < 600:
            continue
        raw[s] = (k, fr)
    idx = None
    for s in raw:
        idx = raw[s][0].index if idx is None else idx.intersection(raw[s][0].index)
    coins = list(raw.keys())

    def M(col, fut_fund=False):
        if fut_fund:
            return pd.DataFrame({s: (raw[s][1]["fundingRate"].resample("1D").sum()
                                     if raw[s][1] is not None else raw[s][0]["close"] * 0)
                                 .reindex(idx).fillna(0) for s in coins}).values.astype(float)
        return pd.DataFrame({s: raw[s][0][col].reindex(idx) for s in coins}).values.astype(float)

    P, H, L, O = M("close"), M("high"), M("low"), M("open")
    V, QV, TB = M("volume"), M("qvol"), M("tbbav")
    ret = np.vstack([np.zeros(P.shape[1]), np.diff(np.log(P), axis=0)])
    # base features -> per-day z-scored (stationary & cross-sectionally comparable)
    feat = {
        "ret": _zscore_rows(ret),
        "ret5": _zscore_rows(np.vstack([np.zeros((5, P.shape[1])), np.log(P[5:] / P[:-5])])),
        "hl": _zscore_rows((H - L) / P),
        "co": _zscore_rows((P - O) / O),
        "vol": _zscore_rows(np.log(V + 1)),
        "ofi": _zscore_rows((2 * TB - V) / (V + 1e-9)),
        "vwapdev": _zscore_rows(P / ((H + L + P) / 3) - 1),
        "fund": _zscore_rows(M(None, True)),
        "rng": _zscore_rows((H - L) / (np.abs(P - O) + 1e-9)),
    }
    fwd_log = np.vstack([np.diff(np.log(P), axis=0), np.zeros(P.shape[1])])
    fwd_simple = np.vstack([P[1:] / P[:-1] - 1, np.zeros(P.shape[1])])
    return feat, fwd_log, fwd_simple, coins, idx


# ---------------- vectorized causal operators ----------------
def _rollmean(x, d):
    cs = np.cumsum(np.nan_to_num(x), axis=0)
    out = np.full_like(x, np.nan)
    out[d - 1:] = (cs[d - 1:] - np.vstack([np.zeros((1, x.shape[1])), cs[:-d]])) / d
    return out

def _rollstd(x, d):
    m = _rollmean(x, d); m2 = _rollmean(x * x, d)
    return np.sqrt(np.clip(m2 - m * m, 0, None))

def _delay(x, d):
    return np.vstack([np.full((d, x.shape[1]), np.nan), x[:-d]])

def _rank(x):
    out = np.full_like(x, np.nan)
    for t in range(x.shape[0]):
        row = x[t]; m = np.isfinite(row)
        if m.sum() > 1:
            out[t, m] = rankdata(row[m]) / m.sum() - 0.5
    return out

OPS = {
    "add": (2, lambda a, b: a + b), "sub": (2, lambda a, b: a - b),
    "mul": (2, lambda a, b: a * b),
    "div": (2, lambda a, b: np.where(np.abs(b) > 1e-9, a / np.where(np.abs(b) > 1e-9, b, 1), 1.0)),
    "neg": (1, lambda a: -a), "abs": (1, np.abs), "sign": (1, np.sign),
    "sqrt": (1, lambda a: np.sqrt(np.abs(a))),
    "rank": (1, _rank), "neut": (1, lambda a: a - np.nanmean(a, axis=1, keepdims=True)),
    "tsmean3": (1, lambda a: _rollmean(a, 3)), "tsmean10": (1, lambda a: _rollmean(a, 10)),
    "tsstd10": (1, lambda a: _rollstd(a, 10)),
    "tsdelta5": (1, lambda a: a - _delay(a, 5)), "delay1": (1, lambda a: _delay(a, 1)),
}
UNARY = [k for k, v in OPS.items() if v[0] == 1]
BINARY = [k for k, v in OPS.items() if v[0] == 2]


def rand_tree(feats, rng, depth):
    if depth <= 0 or (rng.random() < 0.3 and depth < 3):
        return ("feat", feats[rng.integers(len(feats))])
    if rng.random() < 0.5:
        return (UNARY[rng.integers(len(UNARY))], rand_tree(feats, rng, depth - 1))
    return (BINARY[rng.integers(len(BINARY))], rand_tree(feats, rng, depth - 1), rand_tree(feats, rng, depth - 1))

def nodes(t): return 1 if t[0] == "feat" else 1 + sum(nodes(c) for c in t[1:])
def ev(t, F): return F[t[1]] if t[0] == "feat" else OPS[t[0]][1](*[ev(c, F) for c in t[1:]])
def to_str(t): return t[1] if t[0] == "feat" else f"{t[0]}({','.join(to_str(c) for c in t[1:])})"

def rank_ic(alpha, fwd, sl):
    a, f = alpha[sl], fwd[sl]; ics = []
    for t in range(a.shape[0]):
        ar, fr = a[t], f[t]; m = np.isfinite(ar) & np.isfinite(fr)
        if m.sum() > 4 and np.std(ar[m]) > 0:
            ics.append(np.corrcoef(rankdata(ar[m]), rankdata(fr[m]))[0, 1])
    ics = np.array(ics)
    return (float(np.nanmean(ics)), float(np.nanmean(ics) / (np.nanstd(ics) + 1e-9))) if len(ics) > 20 else (0.0, 0.0)

def ls_net(alpha, fwd_simple, sl, cost_bps=5.0):
    a, r = alpha[sl], fwd_simple[sl]
    w = np.zeros_like(a)
    for t in range(a.shape[0]):
        row = a[t]; m = np.isfinite(row)
        if m.sum() > 4:
            rk = rankdata(row[m]) - (m.sum() + 1) / 2
            w[t, m] = rk / (np.abs(rk).sum() + 1e-9)
    wprev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    turn = np.abs(w - wprev).sum(axis=1)
    return np.nansum(wprev * r, axis=1) - cost_bps / 1e4 * turn

def mutate(t, feats, rng):
    if rng.random() < 0.3 or t[0] == "feat":
        return rand_tree(feats, rng, int(rng.integers(1, 4)))
    return (t[0],) + tuple(mutate(c, feats, rng) if rng.random() < 0.4 else c for c in t[1:])


def main():
    rng = np.random.default_rng(12345)
    feat, fwd_log, fwd_simple, coins, idx = build_panel()
    T = feat["ret"].shape[0]; tr, te = bt.oos_split(T, TRAIN_FRAC)
    feats = list(feat.keys())
    print(f"panel {T}d x {len(coins)} coins {idx[0].date()}..{idx[-1].date()}  (features per-day z-scored)")

    pop = []
    for _ in range(K):
        t = rand_tree(feats, rng, int(rng.integers(2, 5)))
        try:
            a = ev(t, feat)
            if not np.isfinite(a).any():
                continue
            _, icir = rank_ic(a, fwd_log, tr)
            pop.append((abs(icir) - 0.002 * nodes(t), icir, t, a))
        except Exception:
            continue
    pop.sort(key=lambda x: -x[0])
    elite = pop[:40]
    for _ in range(3):                                   # 3 generations of mutation (IS only)
        kids = []
        for _, _, t, _ in elite:
            mt = mutate(t, feats, rng)
            try:
                a = ev(mt, feat); _, icir = rank_ic(a, fwd_log, tr)
                kids.append((abs(icir) - 0.002 * nodes(mt), icir, mt, a))
            except Exception:
                pass
        elite = sorted(elite + kids, key=lambda x: -x[0])[:40]
    K_total = K + 3 * 40

    # OOS evaluation: tradable L/S Sharpe is the verdict; IC for reference
    rows = []
    for sc, icir, t, a in elite:
        oic, oicir = rank_ic(a, fwd_log, te)
        net = ls_net(a, fwd_simple, te); mo = bt.metrics(net, 365)
        rows.append(dict(is_icir=icir, oos_ic=oic, oos_icir=oicir,
                         oos_sharpe=mo["sharpe_ann"], sr_pp=mo["sr_pp"], formula=to_str(t)))
    rows.sort(key=lambda r: -(r["oos_sharpe"] if np.isfinite(r["oos_sharpe"]) else -9))

    # DSR deflation over the whole search
    sr_pps = [r["sr_pp"] for r in rows if np.isfinite(r["sr_pp"])]
    sr_star = bt.dsr_benchmark(sr_pps) if len(sr_pps) > 2 else 0.0
    best = rows[0]
    n_oos = te.stop - te.start
    best["dsr"] = bt.psr(best["sr_pp"], n_oos, 0, 3, sr_benchmark=sr_star) if np.isfinite(best["sr_pp"]) else 0.0
    survives = np.isfinite(best["oos_sharpe"]) and best["oos_sharpe"] > 0 and best["dsr"] > 0.95

    print(f"\nsearched {K_total} formulas. Top by OOS tradable L/S Sharpe:")
    print(f"{'OOS_Shrp':>8} {'OOS_IC':>7} {'IS_ICIR':>8} {'formula'}")
    for r in rows[:10]:
        print(f"{r['oos_sharpe']:8.2f} {r['oos_ic']:7.4f} {r['is_icir']:8.3f}  {r['formula'][:66]}")
    print(f"\nDSR* (per-period) over {K_total} formulas = {sr_star:.4f};  best OOS sr_pp={best['sr_pp']:.4f}  deflated PSR={best['dsr']:.3f}")
    print(f"VERDICT: {'SURVIVES — real OOS-tradable formulaic alpha' if survives else 'DEAD — best searched formula does not beat search-noise deflation as a tradable L/S book'}")

    out = pathlib.Path(__file__).resolve().parent.parent / "reports" / "alpha_miner_result.json"
    out.write_text(json.dumps(dict(K=K_total, sr_star=sr_star, survives=bool(survives),
                                   top=rows[:10]), indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
