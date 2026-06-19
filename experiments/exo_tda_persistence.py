"""Exotic-math candidate: tda_persistence  (family = topological-data-analysis)

METHOD: Topological Data Analysis / persistent homology as a crash / regime
detector, after Gidea & Katz (2018) "Topological Data Analysis of Financial
Time Series: Landscapes of Crashes" (Physica A). Their finding: the L^p norm of
the persistence LANDSCAPE built from a sliding window of a few index returns
RISES sharply in the run-up to a crash (dot-com, 2008). The topology of the
multivariate-return point cloud becomes "rougher" (more / longer-lived loops
and clusters) before a regime break.

WHAT I BUILT (all persistence math in numpy; NO gudhi/ripser/persim):
  * Point cloud at day t: a sliding window of W daily log-return VECTORS from a
    handful of coins (BTC,ETH,SOL,BNB by default). Each of the W days is one
    point in R^Ncoin (Gidea-Katz use the multivariate index-return cloud).
    Optionally Takens delay-embedding of a single series is supported.
  * H0 persistence via UNION-FIND over the Vietoris-Rips filtration: sort all
    C(W,2) pairwise Euclidean distances ascending; sweep epsilon; each edge that
    merges two distinct components KILLS one H0 class whose "death" = that edge
    length (birth=0 for all points). Exact single-linkage / 0-dim persistence.
      -> features: total H0 persistence (sum of deaths), L2 norm of the H0
         persistence diagram, and the L2 norm of the H0 persistence LANDSCAPE
         (Bubenik 2015), computed exactly in numpy.
  * H1 PROXY (flagged): a true H1 needs the boundary-matrix reduction (not done
    without gudhi). As a documented stand-in I use the LARGEST-GAP / "loop scale"
    statistic of the Vietoris-Rips MST: the longest edge in the minimum spanning
    tree (the scale at which the cloud is still disconnected) and the death of
    the longest-lived H0 bar are correlated with how "stretched"/multi-modal the
    cloud is. This is a PROXY for topological roughness, NOT genuine H1. I report
    it but do not lean on it. (Implementing exact H1 in pure numpy via the
    standard column reduction is possible but error-prone; I chose the honest
    H0-exact + H1-proxy route and SAY SO.)

DUAL FRAMING (per governance): the persistence-norm is a REGIME/RISK detector,
not obviously a return predictor. We test BOTH:
  (A) STANDALONE alpha: go short BTC (or short the equal-weight book) when the
      persistence-norm z-score is high (>thr); flat / long otherwise.
  (B) RISK-OFF OVERLAY: gate a simple base book (long-BTC; and an equal-weight
      long book) -> de-risk (cut exposure to g_low) when persistence-norm is
      high. Compare base-vs-gated OOS Sharpe and maxDD.

GOVERNANCE baked in:
  * NO LOOK-AHEAD: the window ending at day t uses returns r_{t-W+1..t}; the
    persistence feature computed from it is shifted >=1 bar before it sizes any
    position decided at t and held over t+1. z-score uses a causal trailing mean/
    std (expanding or rolling, shift 1).
  * COST: 5bp/leg on |Δposition| (overlay base book is low-turnover; standalone
    short is bounded).
  * CHRONOLOGICAL OOS: ALL knobs (W, coin set, embedding, landscape level, norm
    p, z-window, thresholds, gate level) tuned on first 60%; metrics on last 40%.
  * DEFLATE: every variant tried is counted; within-family DSR via
    bt.dsr_benchmark; OOS PSR reported.
  * close-to-close daily maxDD is an ILLUSION (no intrabar gap/liq) -> flagged.
"""
from __future__ import annotations
import sys, time, json, pathlib, itertools
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import fetch_binance as fb
from engine import backtest as bt

CACHE = ROOT / "data" / "cache"
INTERVAL = "1d"
PPY = 365
TRAIN_FRAC = 0.60
COST_BPS = 5.0
START = "2022-01-01"
END = "2026-06-18"
ALL_COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
             "ADAUSDT", "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT"]


def _ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
def load_panel(coins):
    end = _ms(END)
    start = _ms(START)
    closes = {}
    for s in coins:
        k = fb.klines(s, INTERVAL, start, end, futures=True)
        if k is None or len(k) < 500:
            continue
        closes[s] = k["close"].astype(float)
    px = pd.DataFrame(closes).dropna()
    logret = np.log(px).diff().dropna()
    return px, logret


# ---------------------------------------------------------------------------
# PERSISTENT HOMOLOGY  (pure numpy)
# ---------------------------------------------------------------------------
def _pairwise_dist(X):
    """Euclidean pairwise distance matrix for point cloud X (m points x d)."""
    sq = np.sum(X * X, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def h0_persistence(X):
    """Exact 0-dim persistence (single-linkage) via union-find over the
    Vietoris-Rips filtration of point cloud X.

    Returns dict of features:
      deaths           : sorted array of H0 bar deaths (births all 0).
      total_pers       : sum of finite deaths (total H0 persistence).
      l2_diag          : sqrt(sum deaths^2)  (L2 norm of the H0 diagram).
      max_death        : longest H0 bar (== longest MST edge).
      land_l1, land_l2 : L1 / L2 norm of the H0 persistence LANDSCAPE.
    The most-persistent (infinite) component is dropped (one component lives
    forever); the remaining m-1 deaths are the MST edge lengths (Kruskal).
    """
    m = X.shape[0]
    if m < 3:
        return None
    D = _pairwise_dist(X)
    iu, ju = np.triu_indices(m, k=1)
    w = D[iu, ju]
    order = np.argsort(w, kind="mergesort")
    iu, ju, w = iu[order], ju[order], w[order]
    parent = np.arange(m)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    deaths = []
    ncomp = m
    for a, b, dist in zip(iu, ju, w):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            deaths.append(dist)          # one H0 class dies here
            ncomp -= 1
            if ncomp == 1:
                break
    deaths = np.sort(np.asarray(deaths, float))
    total = float(deaths.sum())
    l2_diag = float(np.sqrt(np.sum(deaths ** 2)))
    max_death = float(deaths[-1]) if len(deaths) else 0.0
    land_l1, land_l2 = _h0_landscape_norms(deaths)
    return dict(deaths=deaths, total_pers=total, l2_diag=l2_diag,
                max_death=max_death, land_l1=land_l1, land_l2=land_l2)


def _h0_landscape_norms(deaths):
    """L1/L2 norm of the persistence LANDSCAPE for an H0 diagram (Bubenik 2015).

    For a bar (birth=0, death=d) the tent function peaks at (d/2, d/2). The
    k-th landscape lambda_k(x) is the k-th largest tent value at x. The L^p norm
    of the landscape ||lambda||_p uses ALL layers. For H0 with births=0 the
    landscape norm has a clean closed form per layer; we compute it numerically
    on a fine grid for correctness (deaths are few, so this is cheap & exact).
    """
    if len(deaths) == 0:
        return 0.0, 0.0
    dmax = deaths[-1]
    if dmax <= 0:
        return 0.0, 0.0
    grid = np.linspace(0.0, dmax, 256)
    # tent for each bar (0,d): t(x)=min(x, d-x) for x in [0,d], else 0
    # shape (nbars, ngrid)
    d = deaths[:, None]
    x = grid[None, :]
    tent = np.minimum(x, d - x)
    tent = np.where((x >= 0) & (x <= d), tent, 0.0)
    tent = np.maximum(tent, 0.0)
    # landscape = sorted (desc) over bars at each grid point, summed over layers
    # ||lambda||_p^p = integral over x of sum_k lambda_k(x)^p
    #               = integral over x of sum_bars tent(x)^p  (since p-norm of a
    #                 vector equals sum over its entries; layers are just sorted
    #                 entries -> sum over entries is layer-order-invariant)
    l1 = float(np.trapz(np.sum(tent, axis=0), grid))
    l2 = float(np.sqrt(np.trapz(np.sum(tent ** 2, axis=0), grid)))
    return l1, l2


def h1_proxy(X):
    """H1 PROXY (NOT real H1 - flagged). Topological 'roughness' / loop-scale
    statistics of the Vietoris-Rips complex that DON'T need boundary reduction:
      mst_gap   : (max MST edge) - (median MST edge); a big gap => cloud is
                  multi-modal / stretched (proxy for a persistent loop scale).
      mean_knn  : mean nearest-neighbour distance (cloud spread).
    Real H1 (loops) would require the standard column reduction of the boundary
    matrix; I did NOT implement that without gudhi. Reported as a proxy only.
    """
    m = X.shape[0]
    if m < 3:
        return dict(mst_gap=0.0, mean_knn=0.0)
    D = _pairwise_dist(X)
    h0 = h0_persistence(X)
    deaths = h0["deaths"]
    mst_gap = float(deaths[-1] - np.median(deaths)) if len(deaths) else 0.0
    Dn = D.copy()
    np.fill_diagonal(Dn, np.inf)
    mean_knn = float(np.mean(np.min(Dn, axis=1)))
    return dict(mst_gap=mst_gap, mean_knn=mean_knn)


def delay_embed(series, dim, tau):
    """Takens delay embedding of a 1-D series into R^dim with lag tau.
    Row t = [s_t, s_{t-tau}, ..., s_{t-(dim-1)tau}]. Returns (rows, idx)."""
    n = len(series)
    span = (dim - 1) * tau
    rows = np.empty((n - span, dim))
    for j in range(dim):
        rows[:, j] = series[span - j * tau: n - j * tau]
    return rows


# ---------------------------------------------------------------------------
# FEATURE: daily persistence-norm time series  (causal)
# ---------------------------------------------------------------------------
def persistence_series(logret, coins, W, feature="land_l2", embed=None):
    """Compute a daily persistence-feature series.

    embed=None  -> point cloud = the last W daily return-VECTORS over `coins`
                   (W points in R^len(coins)); Gidea-Katz multivariate cloud.
    embed=(d,tau) -> point cloud = Takens delay-embedding of the EQUAL-WEIGHT
                   market log-return over the last W days into R^d.
    The window ending at index t uses ret rows [t-W+1 .. t] (all <= t), so the
    feature at t is point-in-time. Caller shifts >=1 before trading.
    Returns a pd.Series indexed like logret (NaN until first full window).
    """
    R = logret[coins].values
    idx = logret.index
    n = len(R)
    vals = np.full(n, np.nan)
    if embed is None:
        for t in range(W - 1, n):
            X = R[t - W + 1: t + 1, :]          # W x ncoin
            # standardize each coin within window so scale differences don't
            # dominate the topology (per-window z; uses only window data)
            mu = X.mean(axis=0)
            sd = X.std(axis=0)
            sd[sd == 0] = 1.0
            Xs = (X - mu) / sd
            h = h0_persistence(Xs)
            if h is not None:
                vals[t] = h[feature]
    else:
        d, tau = embed
        mkt = R.mean(axis=1)                     # equal-weight market log-ret
        for t in range(W - 1, n):
            seg = mkt[t - W + 1: t + 1]
            emb = delay_embed(seg, d, tau)
            if emb.shape[0] < 3:
                continue
            mu = emb.mean(axis=0); sd = emb.std(axis=0); sd[sd == 0] = 1.0
            h = h0_persistence((emb - mu) / sd)
            if h is not None:
                vals[t] = h[feature]
    return pd.Series(vals, index=idx)


def causal_z(series, zwin):
    """Causal z-score: (x - trailing_mean)/trailing_std using ONLY past values.
    Trailing window of length zwin (rolling). The returned z at t uses values
    up to and including t; caller shifts it >=1 before trading."""
    m = series.rolling(zwin, min_periods=max(10, zwin // 3)).mean()
    s = series.rolling(zwin, min_periods=max(10, zwin // 3)).std()
    return (series - m) / s.replace(0, np.nan)


# ---------------------------------------------------------------------------
# BACKTEST helpers
# ---------------------------------------------------------------------------
def evaluate(net, position=None):
    net = np.asarray(net, float)
    n = len(net)
    tr, te = bt.oos_split(n, TRAIN_FRAC)
    pos = None if position is None else np.asarray(position, float)
    mi = bt.metrics(net[tr], PPY, None if pos is None else pos[tr])
    mo = bt.metrics(net[te], PPY, None if pos is None else pos[te])
    p = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
    return mi, mo, p


def main():
    t0 = time.time()
    px, logret = load_panel(ALL_COINS)
    print(f"panel: {logret.shape[0]} daily bars x {logret.shape[1]} coins "
          f"{str(logret.index[0])[:10]} -> {str(logret.index[-1])[:10]}")
    print(f"coins: {list(logret.columns)}\n")

    # base assets
    btc_ret = logret["BTCUSDT"].values
    ew_ret = logret.mean(axis=1).values          # equal-weight long book ("carry-ish" proxy)

    n_variants = 0
    sr_trials = []          # per-period IS Sharpes for DSR deflation
    diag = {}               # store diagnostics

    # =====================================================================
    # PHASE 1 (IS-ONLY tuning): pick the persistence configuration whose norm
    # best ANTI-correlates with next-day BTC return on the TRAIN slice (a
    # crash detector should be high BEFORE down days). We ONLY look at the
    # first 60% here. This is the knob-selection; counted as variants.
    # =====================================================================
    n = logret.shape[0]
    tr, te = bt.oos_split(n, TRAIN_FRAC)

    coin_sets = {
        "BES": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "BESB": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
        "BIG6": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"],
    }
    W_grid = [20, 30, 40, 60]
    feat_grid = ["land_l2", "l2_diag", "total_pers", "max_death"]
    embeds = {"multivar": None, "takens_d3": (3, 1), "takens_d5": (5, 1)}

    best = None
    cache_feat = {}          # cache computed persistence series by config key
    for cs_name, cs in coin_sets.items():
        for emb_name, emb in embeds.items():
            for W in W_grid:
                for feat in feat_grid:
                    n_variants += 1
                    key = (cs_name, emb_name, W, feat)
                    pser = persistence_series(logret, cs, W, feature=feat, embed=emb)
                    cache_feat[key] = pser
                    # IS-only diagnostic: corr of feature(t-1) with btc_ret(t)
                    f1 = pser.shift(1)
                    valid = f1.notna() & np.isfinite(btc_ret)
                    f1v = f1.values
                    # restrict to TRAIN slice
                    mask = np.zeros(n, bool); mask[tr] = True
                    mask &= valid.values
                    if mask.sum() < 60:
                        continue
                    c = np.corrcoef(f1v[mask], btc_ret[mask])[0, 1]
                    # crash detector: want NEGATIVE corr (high norm -> low fwd ret)
                    score = -c
                    if best is None or score > best["score"]:
                        best = dict(key=key, score=score, corr=float(c),
                                    cs=cs, emb=emb, W=W, feat=feat,
                                    cs_name=cs_name, emb_name=emb_name)

    print("=== PHASE 1: IS persistence-config selection (first 60% only) ===")
    print(f"variants scanned for config: {n_variants}")
    print(f"IS-best config: coins={best['cs_name']} embed={best['emb_name']} "
          f"W={best['W']} feat={best['feat']} | IS corr(feat_t-1, btc_ret_t)="
          f"{best['corr']:+.3f} (want negative)\n")
    diag["phase1_best"] = dict(coins=best["cs_name"], embed=best["emb_name"],
                               W=best["W"], feat=best["feat"], is_corr=best["corr"])

    pser = cache_feat[best["key"]]
    # also store a couple of correlations for honesty (top configs)
    feat_norm = pser

    # =====================================================================
    # PHASE 2: build z-scored signal (causal) and test STANDALONE + OVERLAY.
    # Tune z-window & thresholds & gate on IS only; report OOS.
    # =====================================================================
    z_wins = [60, 120, 250]
    thr_grid = [0.5, 1.0, 1.5, 2.0]
    gate_lo_grid = [0.0, 0.25, 0.5]

    # ---- (A) STANDALONE: short BTC when norm-z high, else flat (or long) ----
    print("=== PHASE 2A: STANDALONE alpha (short BTC when persistence-z high) ===")
    stand_best = None
    for zwin in z_wins:
        z = causal_z(feat_norm, zwin).shift(1)        # signal decided at t-1
        zv = z.values
        for thr in thr_grid:
            for base_long in (False, True):
                n_variants += 1
                # position: -1 when z>thr (risk-off/short); base_long? +1 : 0 otherwise
                pos = np.where(np.isfinite(zv) & (zv > thr), -1.0,
                               1.0 if base_long else 0.0)
                pos = np.where(np.isfinite(zv), pos, 0.0)
                net = bt.run(btc_ret, pos, COST_BPS)
                mi, mo, _ = evaluate(net, pos)
                sr_trials.append(mi["sr_pp"])
                if mi["n"] > 50 and (stand_best is None or
                                     mi["sharpe_ann"] > stand_best["is_sharpe"]):
                    stand_best = dict(zwin=zwin, thr=thr, base_long=base_long,
                                      is_sharpe=mi["sharpe_ann"])
    # OOS for standalone best
    z = causal_z(feat_norm, stand_best["zwin"]).shift(1)
    zv = z.values
    pos = np.where(np.isfinite(zv) & (zv > stand_best["thr"]), -1.0,
                   1.0 if stand_best["base_long"] else 0.0)
    pos = np.where(np.isfinite(zv), pos, 0.0)
    net = bt.run(btc_ret, pos, COST_BPS)
    mi_s, mo_s, psr_s = evaluate(net, pos)
    print(f"IS-selected standalone: zwin={stand_best['zwin']} thr={stand_best['thr']} "
          f"base_long={stand_best['base_long']} (IS Sharpe={stand_best['is_sharpe']:.2f})")
    print(f"  OOS Sharpe={mo_s['sharpe_ann']:.2f} ret={mo_s['ret_ann']*100:.2f}% "
          f"maxDD={mo_s['maxdd']*100:.2f}% turn={mo_s['turnover']:.4f} PSR={psr_s:.3f}\n")

    # =====================================================================
    # PHASE 2B: RISK-OFF OVERLAY on base books (long-BTC and equal-weight long)
    # base position = +1 always; gated position = g_low when z>thr else 1.
    # Compare base vs gated OOS Sharpe & maxDD.
    # =====================================================================
    print("=== PHASE 2B: RISK-OFF OVERLAY (de-risk base book when persistence-z high) ===")

    def overlay_eval(asset_ret, zser, zwin, thr, g_low, cost):
        z = causal_z(zser, zwin).shift(1).values
        gate = np.where(np.isfinite(z) & (z > thr), g_low, 1.0)
        gate = np.where(np.isfinite(z), gate, 1.0)
        base_pos = np.ones_like(asset_ret)
        base_net = bt.run(asset_ret, base_pos, cost)
        gated_net = bt.run(asset_ret, gate, cost)
        return base_net, gated_net, gate

    overlay_results = {}
    for book_name, book_ret in [("longBTC", btc_ret), ("EWlong", ew_ret)]:
        ov_best = None
        for zwin in z_wins:
            for thr in thr_grid:
                for g_low in gate_lo_grid:
                    n_variants += 1
                    bnet, gnet, gate = overlay_eval(book_ret, feat_norm, zwin, thr,
                                                    g_low, COST_BPS)
                    mi_b, _, _ = evaluate(bnet)
                    mi_g, _, _ = evaluate(gnet, gate)
                    sr_trials.append(mi_g["sr_pp"])
                    # IS objective: gated IS Sharpe (must also be defined)
                    if mi_g["n"] > 50 and (ov_best is None or
                                           mi_g["sharpe_ann"] > ov_best["is_sharpe"]):
                        ov_best = dict(zwin=zwin, thr=thr, g_low=g_low,
                                       is_sharpe=mi_g["sharpe_ann"])
        # OOS at selected knobs
        bnet, gnet, gate = overlay_eval(book_ret, feat_norm, ov_best["zwin"],
                                        ov_best["thr"], ov_best["g_low"], COST_BPS)
        _, mo_b, psr_b = evaluate(bnet)
        _, mo_g, psr_g = evaluate(gnet, gate)
        overlay_results[book_name] = dict(
            sel=ov_best, base_oos=mo_b, gated_oos=mo_g, psr_base=psr_b, psr_gated=psr_g)
        print(f"[{book_name}] IS-sel zwin={ov_best['zwin']} thr={ov_best['thr']} "
              f"g_low={ov_best['g_low']} (IS gated Sharpe={ov_best['is_sharpe']:.2f})")
        print(f"   BASE  OOS Sharpe={mo_b['sharpe_ann']:6.2f} ret={mo_b['ret_ann']*100:7.2f}% "
              f"maxDD={mo_b['maxdd']*100:7.2f}% PSR={psr_b:.3f}")
        print(f"   GATED OOS Sharpe={mo_g['sharpe_ann']:6.2f} ret={mo_g['ret_ann']*100:7.2f}% "
              f"maxDD={mo_g['maxdd']*100:7.2f}% turn={mo_g['turnover']:.4f} PSR={psr_g:.3f}")
        d_sh = mo_g["sharpe_ann"] - mo_b["sharpe_ann"]
        d_dd = mo_g["maxdd"] - mo_b["maxdd"]
        print(f"   DELTA Sharpe={d_sh:+.2f}  DELTA maxDD={d_dd*100:+.2f}pp "
              f"({'IMPROVES' if d_sh > 0.1 or d_dd > 0.03 else 'no clear help'})\n")

    # =====================================================================
    # H1 PROXY diagnostic (reported, not traded on its own merit)
    # =====================================================================
    print("=== H1 PROXY diagnostic (NOT real H1; reported for context) ===")
    cs = best["cs"]
    mst_gap = np.full(n, np.nan)
    R = logret[cs].values
    Wb = best["W"]
    for t in range(Wb - 1, n):
        X = R[t - Wb + 1: t + 1, :]
        mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1.0
        h = h1_proxy((X - mu) / sd)
        mst_gap[t] = h["mst_gap"]
    mst_gap_s = pd.Series(mst_gap, index=logret.index)
    f1 = mst_gap_s.shift(1)
    mask = np.zeros(n, bool); mask[te] = True
    mask &= f1.notna().values & np.isfinite(btc_ret)
    h1_corr_oos = float(np.corrcoef(f1.values[mask], btc_ret[mask])[0, 1]) if mask.sum() > 30 else np.nan
    print(f"  OOS corr(mst_gap_t-1, btc_ret_t) = {h1_corr_oos:+.3f} "
          f"(proxy only; want negative for risk-off)\n")
    diag["h1_proxy_oos_corr"] = h1_corr_oos

    # =====================================================================
    # DEFLATION
    # =====================================================================
    sr_star = bt.dsr_benchmark(sr_trials)
    # pick the headline overlay (longBTC) and standalone for verdict
    ov = overlay_results["longBTC"]
    ov_ew = overlay_results["EWlong"]
    print(f"=== DEFLATION ===")
    print(f"total variants tried (config + signal + overlay): {n_variants}")
    print(f"DSR within-family SR* (per-period, {len(sr_trials)} trial Sharpes) = {sr_star:.5f}")
    print(f"  standalone OOS SR_pp = {mo_s['sr_pp']:.5f}  -> "
          f"{'BEATS' if mo_s['sr_pp'] > sr_star else 'FAILS'} deflated bar")
    print(f"  overlay(longBTC) gated OOS SR_pp = {ov['gated_oos']['sr_pp']:.5f}\n")

    # =====================================================================
    # VERDICT
    # =====================================================================
    # Standalone path
    stand_edge = (psr_s >= 0.95 and mo_s["sr_pp"] > sr_star and
                  mo_s["sharpe_ann"] > 0 and mo_s["ret_ann"] > 0)
    stand_marg = (0.80 <= psr_s < 0.95 and mo_s["ret_ann"] > 0)

    # Overlay path: improvement must be MATERIAL & robust (Sharpe up >=0.2 OR
    # maxDD cut >=5pp) on the longBTC book AND PSR of gated >= 0.95 for EDGE.
    def overlay_improves(o, need_psr):
        d_sh = o["gated_oos"]["sharpe_ann"] - o["base_oos"]["sharpe_ann"]
        d_dd = o["gated_oos"]["maxdd"] - o["base_oos"]["maxdd"]
        material = (d_sh >= 0.20) or (d_dd >= 0.05)
        return material and (o["psr_gated"] >= need_psr) and o["gated_oos"]["ret_ann"] > 0

    overlay_edge = overlay_improves(ov, 0.95) or overlay_improves(ov_ew, 0.95)
    overlay_marg = (overlay_improves(ov, 0.80) or overlay_improves(ov_ew, 0.80))

    if stand_edge or overlay_edge:
        verdict = "EDGE"
    elif stand_marg or overlay_marg:
        verdict = "MARGINAL"
    else:
        verdict = "DEAD"

    # role
    d_sh_btc = ov["gated_oos"]["sharpe_ann"] - ov["base_oos"]["sharpe_ann"]
    d_sh_ew = ov_ew["gated_oos"]["sharpe_ann"] - ov_ew["base_oos"]["sharpe_ann"]
    overlay_helps = (overlay_improves(ov, 0.80) or overlay_improves(ov_ew, 0.80))
    stand_ok = stand_edge or stand_marg
    if stand_ok and overlay_helps:
        role = "both"
    elif overlay_helps:
        role = "risk-overlay"
    elif stand_ok:
        role = "standalone-alpha"
    else:
        role = "none"

    notes = (
        f"Gidea-Katz persistence on a sliding window of {best['cs_name']} daily "
        f"return vectors (embed={best['emb_name']}, W={best['W']}, feat={best['feat']}). "
        f"H0 persistence is EXACT (union-find / single-linkage MST); landscape L2 "
        f"norm computed in numpy; H1 is a flagged PROXY (mst-gap), not real H1 "
        f"(no gudhi). STANDALONE short-when-high: OOS Sharpe={mo_s['sharpe_ann']:.2f}, "
        f"PSR={psr_s:.3f}, ret={mo_s['ret_ann']*100:.1f}%/yr -> "
        f"{'edge' if stand_edge else 'marginal' if stand_marg else 'no standalone alpha'}. "
        f"OVERLAY on long-BTC: base Sharpe={ov['base_oos']['sharpe_ann']:.2f} -> "
        f"gated {ov['gated_oos']['sharpe_ann']:.2f} (Delta={d_sh_btc:+.2f}, "
        f"maxDD {ov['base_oos']['maxdd']*100:.1f}% -> {ov['gated_oos']['maxdd']*100:.1f}%); "
        f"OVERLAY on EW-long: Delta Sharpe={d_sh_ew:+.2f}. "
        f"IS feature-config corr(feat,fwd-ret)={best['corr']:+.3f}. "
        f"{n_variants} variants tried; within-family DSR SR*={sr_star:.4f} vs "
        f"standalone OOS SR_pp={mo_s['sr_pp']:.4f}. "
        f"H1-proxy OOS corr={h1_corr_oos:+.3f}. "
        f"PRIOR holds: directional persistence-spike trading is overfit-prone; the "
        f"honest read is whether it helps as a risk overlay. close-to-close daily "
        f"maxDD is an illusion (no intrabar gap/liq)."
    )
    print("=== VERDICT:", verdict, f"(role={role}) ===")
    print(notes)

    # market_neutral: standalone short-BTC is directional; EW overlay is long-only
    out = dict(
        key="tda_persistence",
        method="Topological Data Analysis / persistent homology (Gidea-Katz 2018)",
        family="topological-data-analysis",
        file="experiments/exo_tda_persistence.py",
        implemented=True,
        libs_implemented=("H0 persistence via numpy union-find over Vietoris-Rips "
                          "filtration (exact single-linkage MST); persistence "
                          "landscape L1/L2 norm in numpy (Bubenik); H1 is a "
                          "documented PROXY (mst-gap), NOT real H1 (no gudhi/ripser)."),
        verdict=verdict,
        role=role,
        market_neutral=False,
        universe=(f"{logret.shape[1]} USDT-perps daily {START}..{END}; "
                  f"point-cloud coins={best['cs_name']}"),
        n_obs=int(mo_s["n"]),
        method_detail=dict(
            coins=best["cs_name"], embed=best["emb_name"], window=best["W"],
            feature=best["feat"], is_feature_corr=best["corr"]),
        # standalone
        oos_sharpe=float(mo_s["sharpe_ann"]),
        oos_ret_ann_pct=float(mo_s["ret_ann"] * 100),
        psr=float(psr_s),
        dsr=float(sr_star),
        maxdd_pct=float(mo_s["maxdd"] * 100),
        turnover=float(mo_s["turnover"]),
        cost_bps=COST_BPS,
        # overlay
        overlay_base_sharpe=float(ov["base_oos"]["sharpe_ann"]),
        overlay_gated_sharpe=float(ov["gated_oos"]["sharpe_ann"]),
        overlay_detail=dict(
            longBTC=dict(base_sharpe=float(ov["base_oos"]["sharpe_ann"]),
                         gated_sharpe=float(ov["gated_oos"]["sharpe_ann"]),
                         base_maxdd_pct=float(ov["base_oos"]["maxdd"] * 100),
                         gated_maxdd_pct=float(ov["gated_oos"]["maxdd"] * 100),
                         psr_gated=float(ov["psr_gated"]),
                         sel=ov["sel"]),
            EWlong=dict(base_sharpe=float(ov_ew["base_oos"]["sharpe_ann"]),
                        gated_sharpe=float(ov_ew["gated_oos"]["sharpe_ann"]),
                        base_maxdd_pct=float(ov_ew["base_oos"]["maxdd"] * 100),
                        gated_maxdd_pct=float(ov_ew["gated_oos"]["maxdd"] * 100),
                        psr_gated=float(ov_ew["psr_gated"]),
                        sel=ov_ew["sel"])),
        n_variants_tried=int(n_variants),
        diagnostics=diag,
        notes=notes,
        data_caveats=("Daily close-to-close USDT-perp futures; maxDD is "
                      "close-to-close (no intrabar gap/liquidation -> optimistic). "
                      "H1 is a proxy, not genuine 1-dim homology. Per-window "
                      "z-standardization of the point cloud uses only in-window "
                      "data (causal). Feature shifted >=1 bar before trading."),
        runtime_sec=round(time.time() - t0, 1),
    )
    rp = ROOT / "reports" / "exo_tda_persistence.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {rp}  (runtime {out['runtime_sec']}s)")
    return out


if __name__ == "__main__":
    main()
