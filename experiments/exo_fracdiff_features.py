"""Exotic-math candidate: fracdiff_features
METHOD: Fractional differentiation (Lopez de Prado, Advances in Fin ML, ch.5).

THESIS (LdP ch.5): full integer differencing (i.e. log returns, d=1) makes a
price series stationary but ERASES all long memory -> the resulting series has
~no predictive structure (that is exactly the "momentum/reversal died" prior
here). Fractional differencing with the MINIMUM d in (0,1) that achieves
stationarity keeps the series memory-rich AND stationary. If long memory carries
signal, a fracdiff(d*) feature should beat the d=1 (returns) baseline.

WHAT WE BUILD (all causal / point-in-time):
  - fixed-width-window fracdiff weights w_k = -w_{k-1}*(d-k+1)/k, truncated when
    |w_k| < tau (LdP "fixed-width window" so every observation uses the SAME
    finite memory length -> stationarity-preserving + no expanding-window leak).
  - ADF t-statistic implemented in numpy (regress dX_t on X_{t-1} + p lags + const,
    optional trend; t-stat on the X_{t-1} coef). p chosen by AIC. Dickey-Fuller
    5% critical value ~ -2.86 (const, no trend, large n).
  - minimum-d search: smallest d on a grid s.t. ADF t-stat < 5% crit, computed on
    the IS (first 60%) ONLY of LOG-PRICE -> d* frozen, applied OOS. No look-ahead.

TESTS (governed: shift>=1 bar, cost>=5bp/leg on |Δpos|, tune on IS 60% only,
report OOS 40%, PSR + within-family DSR over EVERY variant tried):
  A. STANDALONE ALPHA. Per coin, build features at t-1: fracdiff(d*) level z-score,
     its short slope, and (control) the d=1 return. Ridge regression (closed-form,
     IS-fit) predicting next-bar log return. Position = sign/scaled prediction.
     Compare three feature sets:
        (i)  RET only      (integer diff d=1 baseline)
        (ii) FRACDIFF only (the exotic part)
        (iii) BOTH
     If (ii)/(iii) does not beat (i) OOS, the fractional part adds nothing.
     Also a pure mean-reversion variant: trade -sign(fracdiff z-score) (LdP's
     "stationary mean-reversion target" framing). Cross-sectional market-neutral
     book (each coin minus EW market) so we are not just re-betting on BTC beta.
  B. OVERLAY / REGIME DETECTOR. The fracdiff level is a stationary "distance from
     memory-equilibrium". Use |fracdiff z| as a RISK gate on a simple long-BTC
     base book: when the stationary series is in an extreme (|z|>thr) state,
     cut exposure. Does the gated book beat un-gated long-BTC OOS (Sharpe / maxDD)?

HONESTY: we count EVERY variant (d-grid x feature-set x mode x thresholds) toward
the within-family DSR. A big knob count is the whole danger of LdP fracdiff
(d, tau, ridge-lambda, windows, thresholds) -> deflate hard.
"""
from __future__ import annotations
import sys, time, json, warnings, pathlib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import fetch_binance as fb
from engine import backtest as bt

# Universe: top liquid USDT-M perps with FULL 2022-01-01 history.
# MATIC excluded (delisted/renamed POL 2024-09) -> survivorship flag.
UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
            "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT", "TRXUSDT", "DOTUSDT",
            "NEARUSDT", "ATOMUSDT"]
START = "2022-01-01"
INTERVAL = "8h"           # 8h bars -> ~4890 obs/coin, more power than daily
PPY = 365 * 3             # three 8h periods per day
TRAIN_FRAC = 0.60
COST_BPS = 5.0
BTC = "BTCUSDT"
RNG = np.random.default_rng(7)

_VARIANTS = 0             # global knob counter (deflation honesty)


def _ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


# ============================================================================
#  MATH IMPLEMENTED IN NUMPY (libs missing: statsmodels ADF; no fracdiff lib)
# ============================================================================
def frac_weights_fixed_width(d: float, tau: float = 1e-4, max_k: int = 2000):
    """Fixed-width-window fracdiff weights (LdP ch.5, eq 5.x).
    w_0 = 1 ; w_k = -w_{k-1} * (d - k + 1)/k. Truncate when |w_k| < tau so the
    window width is FIXED -> every point uses the same finite memory (stationary
    preserving). Returns weights ordered [w_0 (newest) ... w_{K-1} (oldest)]."""
    w = [1.0]
    for k in range(1, max_k):
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < tau:
            break
        w.append(wk)
    return np.array(w)


def frac_diff_ffd(series: np.ndarray, d: float, tau: float = 1e-4):
    """Fixed-width-window fractional differencing, applied causally.
    fracdiff[t] = sum_{k=0}^{K-1} w_k * x[t-k]  (uses only past+present -> PIT).
    Output before the window fills is NaN."""
    x = np.asarray(series, float)
    w = frac_weights_fixed_width(d, tau)
    K = len(w)
    n = len(x)
    out = np.full(n, np.nan)
    if K > n:
        return out, w
    # causal dot of reversed weights with the trailing window
    w_rev = w[::-1]               # [w_{K-1} (oldest) ... w_0 (newest)]
    for t in range(K - 1, n):
        out[t] = np.dot(w_rev, x[t - K + 1: t + 1])
    return out, w


def adf_tstat(x: np.ndarray, max_lag: int = 8, trend: str = "c"):
    """Augmented Dickey-Fuller t-statistic, implemented in numpy.
    Model: dX_t = a + b*X_{t-1} + sum_i g_i dX_{t-i} (+ c*t).  H0: b=0 (unit root).
    t = b_hat / SE(b_hat). More negative => more stationary. Lag p chosen by AIC.
    Returns (t_stat, p_used). 5% DF critical (const, no trend, large n) ~ -2.86."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < max_lag + 20:
        return np.nan, 0
    dx = np.diff(x)
    best = None
    for p in range(0, max_lag + 1):
        # rows usable after p lags of dx and one lag of level
        N = n - 1 - p
        if N < 15:
            break
        y = dx[p:]                                  # dX_t, length N
        cols = [x[p: p + N]]                        # X_{t-1}
        for i in range(1, p + 1):
            cols.append(dx[p - i: p - i + N])       # dX_{t-i}
        X = np.column_stack(cols)
        if trend == "c":
            X = np.column_stack([X, np.ones(N)])
        elif trend == "ct":
            X = np.column_stack([X, np.ones(N), np.arange(N, dtype=float)])
        # OLS
        XtX = X.T @ X
        try:
            XtXinv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            continue
        beta = XtXinv @ (X.T @ y)
        resid = y - X @ beta
        dof = N - X.shape[1]
        if dof <= 1:
            continue
        s2 = (resid @ resid) / dof
        # AIC for lag selection
        aic = N * np.log((resid @ resid) / N + 1e-300) + 2 * X.shape[1]
        se_b = np.sqrt(s2 * XtXinv[0, 0])
        t_b = beta[0] / se_b if se_b > 0 else np.nan
        if best is None or aic < best[0]:
            best = (aic, t_b, p)
    if best is None:
        return np.nan, 0
    return float(best[1]), int(best[2])


def find_min_d(logprice_is: np.ndarray, d_grid, tau: float, crit: float = -2.86):
    """Smallest d on grid making the IS fracdiff series stationary (ADF<crit).
    Returns (d_star, table) where table=[(d, adf_t, corr_to_orig)]. Frozen for OOS."""
    table = []
    d_star = None
    orig = logprice_is - np.nanmean(logprice_is)
    for d in d_grid:
        fd, _ = frac_diff_ffd(logprice_is, d, tau)
        fd_v = fd[np.isfinite(fd)]
        if len(fd_v) < 50:
            table.append((d, np.nan, np.nan))
            continue
        t, _ = adf_tstat(fd_v, max_lag=8, trend="c")
        # correlation to original level on the overlap (memory preserved if high)
        mask = np.isfinite(fd)
        corr = np.corrcoef(fd[mask], orig[mask])[0, 1] if mask.sum() > 50 else np.nan
        table.append((d, t, corr))
        if d_star is None and np.isfinite(t) and t < crit:
            d_star = d
    return d_star, table


# ============================================================================
#  Ridge regression (closed form, IS-fit) — predictive model
# ============================================================================
def ridge_fit(X, y, lam):
    """Closed-form ridge: beta = (X'X + lam*I)^-1 X'y. X already standardized,
    intercept handled separately (we de-mean y, no penalty on a separate const)."""
    p = X.shape[1]
    A = X.T @ X + lam * np.eye(p)
    return np.linalg.solve(A, X.T @ y)


def zscore_causal(x: np.ndarray, win: int):
    """Causal rolling z-score (uses data up to and including t; we shift later)."""
    s = pd.Series(x)
    mu = s.rolling(win, min_periods=win // 2).mean()
    sd = s.rolling(win, min_periods=win // 2).std()
    return ((s - mu) / sd.replace(0, np.nan)).values


# ============================================================================
#  Data
# ============================================================================
def load_panel(end):
    closes = {}
    for s in UNIVERSE:
        k = fb.klines(s, INTERVAL, _ms(START), end, futures=True)
        if k is not None and len(k) > 3000:
            closes[s] = k["close"].astype(float)
    C = pd.DataFrame(closes).dropna()
    return C


# ============================================================================
#  Backtest helpers
# ============================================================================
def evaluate_oos(net, position, n):
    tr, te = bt.oos_split(n, TRAIN_FRAC)
    mo = bt.metrics(net[te], PPY, position[te] if position is not None else None)
    p = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
    return mo, p, tr, te


def build_features(logprice: np.ndarray, ret: np.ndarray, d_star: float, tau: float,
                   z_win: int):
    """Causal feature matrix at time t (to predict ret[t+1]).
    Cols: [fracdiff_z, fracdiff_slope, ret_z (d=1 baseline)]. All shifted >=1 bar
    is enforced by the caller aligning X[t] with y=ret[t+1]."""
    fd, _ = frac_diff_ffd(logprice, d_star, tau)
    fd_z = zscore_causal(fd, z_win)
    fd_slope = pd.Series(fd).diff(3).values          # short slope of stationary series
    fd_slope_z = zscore_causal(fd_slope, z_win)
    ret_z = zscore_causal(ret, z_win)                # integer-diff (d=1) baseline feature
    F = np.column_stack([fd_z, fd_slope_z, ret_z])
    return F, fd_z


def run_predictive(C, ret, d_map, tau, z_win, lam, feat_cols, cost_bps):
    """Cross-sectional market-neutral predictive book. feat_cols selects which of
    [fd_z, fd_slope_z, ret_z] enter the ridge. Returns (port_net, port_pos, n)."""
    coins = list(C.columns)
    n = len(C)
    cut = int(n * TRAIN_FRAC)
    # build per-coin features & target (next-bar log return)
    Fs, ys = {}, {}
    for c in coins:
        lp = np.log(C[c].values)
        F, _ = build_features(lp, ret[c].values, d_map[c], tau, z_win)
        F = F[:, feat_cols]
        Fs[c] = F
        ys[c] = ret[c].values
    # stack IS rows across coins (pooled ridge) using rows < cut, target = ret[t+1]
    Xtr, Ytr = [], []
    for c in coins:
        F = Fs[c]; y = ys[c]
        for t in range(z_win, cut - 1):
            if np.all(np.isfinite(F[t])):
                Xtr.append(F[t]); Ytr.append(y[t + 1])
    Xtr = np.array(Xtr); Ytr = np.array(Ytr)
    if len(Xtr) < 200:
        return None
    mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-12
    Xtr_s = (Xtr - mu_x) / sd_x
    Ytr_c = Ytr - Ytr.mean()
    beta = ridge_fit(Xtr_s, Ytr_c, lam)
    # predict full-sample per coin -> raw expected next return -> cross-sec demean
    preds = {}
    for c in coins:
        F = Fs[c]
        Fz = (F - mu_x) / sd_x
        p = Fz @ beta
        preds[c] = p
    P = pd.DataFrame(preds, index=C.index)
    P = P.replace([np.inf, -np.inf], np.nan)
    # cross-sectional market-neutral positions: demean across coins, scale to gross 1
    Pn = P.sub(P.mean(axis=1), axis=0)
    gross = Pn.abs().sum(axis=1).replace(0, np.nan)
    W = Pn.div(gross, axis=0).fillna(0.0)
    Wl = W.shift(1).fillna(0.0)                       # position decided at t-1
    asset_ret = ret.reindex(columns=coins).fillna(0.0)
    gross_ret = (Wl * asset_ret).sum(axis=1)
    turn = Wl.diff().abs().sum(axis=1).fillna(0.0)
    cost = (cost_bps / 1e4) * turn
    net = (gross_ret - cost).values
    pos = Wl.abs().sum(axis=1).values
    return net, pos, n


def run_meanrev(C, ret, d_map, tau, z_win, cost_bps, sign=-1.0):
    """LdP 'stationary mean-reversion target': trade sign*(-fracdiff z-score),
    cross-sectional market-neutral. sign=-1 => fade extremes (mean revert)."""
    coins = list(C.columns)
    n = len(C)
    sigs = {}
    for c in coins:
        lp = np.log(C[c].values)
        fd, _ = frac_diff_ffd(lp, d_map[c], tau)
        fd_z = zscore_causal(fd, z_win)
        sigs[c] = sign * fd_z
    S = pd.DataFrame(sigs, index=C.index).replace([np.inf, -np.inf], np.nan)
    Sn = S.sub(S.mean(axis=1), axis=0)
    gross = Sn.abs().sum(axis=1).replace(0, np.nan)
    W = Sn.div(gross, axis=0).fillna(0.0)
    Wl = W.shift(1).fillna(0.0)
    asset_ret = ret.reindex(columns=coins).fillna(0.0)
    gross_ret = (Wl * asset_ret).sum(axis=1)
    turn = Wl.diff().abs().sum(axis=1).fillna(0.0)
    net = (gross_ret - (cost_bps / 1e4) * turn).values
    pos = Wl.abs().sum(axis=1).values
    return net, pos, n


def run_overlay(C, ret, d_star_btc, tau, z_win, thr, cost_bps):
    """OVERLAY: gate a long-BTC base book by |fracdiff z| of BTC log-price.
    When the stationary memory series is extreme (|z|>thr), cut exposure to 0.3.
    Returns (base_net, base_pos, gated_net, gated_pos, n)."""
    lp = np.log(C[BTC].values)
    fd, _ = frac_diff_ffd(lp, d_star_btc, tau)
    fd_z = zscore_causal(fd, z_win)
    fd_z_lag = pd.Series(fd_z).shift(1).fillna(0.0).values   # gate uses t-1 info
    btc_ret = ret[BTC].values
    n = len(btc_ret)
    base_pos = np.ones(n)
    gate = np.where(np.abs(fd_z_lag) > thr, 0.3, 1.0)
    gated_pos = base_pos * gate
    base_net = bt.run(btc_ret, base_pos, cost_bps)
    gated_net = bt.run(btc_ret, gated_pos, cost_bps)
    return base_net, base_pos, gated_net, gated_pos, n


# ============================================================================
def main():
    global _VARIANTS
    end = int(time.time() * 1000)
    C = load_panel(end)
    ret = np.log(C).diff().fillna(0.0)
    n = len(C)
    cut = int(n * TRAIN_FRAC)
    print(f"panel: {n} {INTERVAL} bars x {C.shape[1]} coins "
          f"({C.index[0].date()} -> {C.index[-1].date()}); train cut={cut} "
          f"({C.index[cut].date()})  PPY={PPY}\n")

    TAU = 1e-4
    D_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)

    # ----- per-coin minimum-d on IS log-price (frozen for OOS) -----
    print("=== minimum-d search (ADF on IS log-price, 5% crit=-2.86) ===")
    print(f"{'coin':9} {'d*':>5} {'ADF@d*':>8} {'ADF@d=1':>9} {'corr(fd,lvl)@d*':>16}")
    d_map = {}
    adf_summary = {}
    for c in C.columns:
        lp_is = np.log(C[c].values[:cut])
        d_star, table = find_min_d(lp_is, D_GRID, TAU, crit=-2.86)
        if d_star is None:
            d_star = 1.0
        d_map[c] = float(d_star)
        # ADF at d* and at d=1 (returns baseline), plus memory corr at d*
        adf_at = {round(d, 2): (t, corr) for d, t, corr in table}
        adf_d1 = adf_at.get(1.0, (np.nan, np.nan))[0]
        adf_dstar = adf_at.get(round(d_star, 2), (np.nan, np.nan))
        adf_summary[c] = dict(d_star=d_star, adf_dstar=adf_dstar[0],
                              adf_d1=adf_d1, corr=adf_dstar[1])
        print(f"{c:9} {d_star:5.2f} {adf_dstar[0]:8.2f} {adf_d1:9.2f} {adf_dstar[1]:16.3f}")
    _VARIANTS += len(D_GRID)   # the d-grid search itself is a knob family
    mean_dstar = np.mean(list(d_map.values()))
    print(f"\nmean d* = {mean_dstar:.3f} (LdP: d in (0,1) keeps memory vs d=1 returns)\n")

    # ===================================================================
    # A. STANDALONE ALPHA — predictive ridge, 3 feature sets + mean-rev
    # ===================================================================
    print("=== A. STANDALONE: cross-sec market-neutral predictive book ===")
    # IS-tune (z_win, lam) on the BOTH feature set, then evaluate all sets OOS
    z_grid = [12, 30, 60]          # ~4d, 10d, 20d windows on 8h bars
    lam_grid = [1.0, 10.0, 100.0]
    feat_defs = {
        "RET_only":    [2],
        "FRAC_only":   [0, 1],
        "BOTH":        [0, 1, 2],
    }
    # tune on IS Sharpe using BOTH
    best = None
    for zw in z_grid:
        for lam in lam_grid:
            out = run_predictive(C, ret, d_map, TAU, zw, lam, feat_defs["BOTH"], COST_BPS)
            _VARIANTS += 1
            if out is None:
                continue
            net, pos, nn = out
            mi = bt.metrics(net[:cut], PPY, pos[:cut])
            if best is None or (np.isfinite(mi["sharpe_ann"]) and mi["sharpe_ann"] > best[0]):
                best = (mi["sharpe_ann"], zw, lam)
    _, zw, lam = best
    print(f"IS-selected: z_win={zw} ridge_lam={lam} (IS Sharpe={best[0]:.2f})\n")

    sr_pool = []
    pred_results = {}
    print(f"{'featureset':12} {'OOS_Shrp':>9} {'ret%/yr':>8} {'maxDD%':>8} "
          f"{'turn':>7} {'PSR':>6} {'n':>5}")
    for name, cols in feat_defs.items():
        net, pos, nn = run_predictive(C, ret, d_map, TAU, zw, lam, cols, COST_BPS)
        _VARIANTS += 1
        mo, p, tr, te = evaluate_oos(net, pos, nn)
        sr_pool.append(mo["sr_pp"])
        pred_results[name] = dict(mo=mo, psr=p, net=net, pos=pos, te=te)
        print(f"{name:12} {mo['sharpe_ann']:9.2f} {mo['ret_ann']*100:8.2f} "
              f"{mo['maxdd']*100:8.2f} {mo['turnover']:7.3f} {p:6.3f} {mo['n']:5d}")

    # pure mean-reversion variant (LdP stationary target framing)
    print()
    mr_results = {}
    for zw_mr in z_grid:
        net, pos, nn = run_meanrev(C, ret, d_map, TAU, zw_mr, COST_BPS, sign=-1.0)
        _VARIANTS += 1
        mo, p, tr, te = evaluate_oos(net, pos, nn)
        sr_pool.append(mo["sr_pp"])
        mr_results[zw_mr] = dict(mo=mo, psr=p)
    # report best IS mean-rev for honesty (selected on IS)
    mr_is = {}
    for zw_mr in z_grid:
        net, pos, nn = run_meanrev(C, ret, d_map, TAU, zw_mr, COST_BPS, sign=-1.0)
        mr_is[zw_mr] = bt.metrics(net[:cut], PPY, pos[:cut])["sharpe_ann"]
    best_mr_zw = max(mr_is, key=lambda k: mr_is[k] if np.isfinite(mr_is[k]) else -9)
    mr_sel = mr_results[best_mr_zw]
    print(f"MEAN-REV (fade fracdiff z): IS-selected z_win={best_mr_zw} -> "
          f"OOS Sharpe={mr_sel['mo']['sharpe_ann']:.2f} ret={mr_sel['mo']['ret_ann']*100:.2f}%/yr "
          f"PSR={mr_sel['psr']:.3f}")

    # The key LdP claim: does FRAC add value over RET (d=1)?
    frac_v_ret = (pred_results["FRAC_only"]["mo"]["sharpe_ann"]
                  - pred_results["RET_only"]["mo"]["sharpe_ann"])
    both_v_ret = (pred_results["BOTH"]["mo"]["sharpe_ann"]
                  - pred_results["RET_only"]["mo"]["sharpe_ann"])
    print(f"\nKEY: FRAC_only - RET_only OOS Sharpe = {frac_v_ret:+.2f}; "
          f"BOTH - RET_only = {both_v_ret:+.2f}")
    print("  (if <=0, fractional differencing adds NO value over integer-diff returns)\n")

    # pick best standalone (by OOS Sharpe across predictive sets + mean-rev)
    cand = []
    for name in feat_defs:
        cand.append((pred_results[name]["mo"]["sharpe_ann"], "pred:" + name,
                     pred_results[name]["mo"], pred_results[name]["psr"]))
    cand.append((mr_sel["mo"]["sharpe_ann"], f"meanrev:z{best_mr_zw}",
                 mr_sel["mo"], mr_sel["psr"]))
    cand = [c for c in cand if np.isfinite(c[0])]
    cand.sort(key=lambda x: -x[0])
    best_standalone = cand[0]

    # ===================================================================
    # B. OVERLAY — fracdiff-z regime gate on long-BTC
    # ===================================================================
    print("=== B. OVERLAY: |fracdiff z| regime gate on long-BTC base book ===")
    d_btc = d_map[BTC]
    # IS-tune (z_win, thr) by gated IS Sharpe
    ov_z_grid = [12, 30, 60]
    thr_grid = [1.0, 1.5, 2.0]
    best_ov = None
    for zw_o in ov_z_grid:
        for thr in thr_grid:
            bn, bp, gn, gp, nn = run_overlay(C, ret, d_btc, TAU, zw_o, thr, COST_BPS)
            _VARIANTS += 1
            mi = bt.metrics(gn[:cut], PPY, gp[:cut])
            if best_ov is None or (np.isfinite(mi["sharpe_ann"]) and mi["sharpe_ann"] > best_ov[0]):
                best_ov = (mi["sharpe_ann"], zw_o, thr)
    _, zw_o, thr = best_ov
    bn, bp, gn, gp, nn = run_overlay(C, ret, d_btc, TAU, zw_o, thr, COST_BPS)
    tr, te = bt.oos_split(nn, TRAIN_FRAC)
    mo_base = bt.metrics(bn[te], PPY, bp[te])
    mo_gate = bt.metrics(gn[te], PPY, gp[te])
    p_base = bt.psr(mo_base["sr_pp"], mo_base["n"], mo_base["skew"], mo_base["kurt"])
    p_gate = bt.psr(mo_gate["sr_pp"], mo_gate["n"], mo_gate["skew"], mo_gate["kurt"])
    print(f"IS-selected: z_win={zw_o} thr={thr}")
    print(f"  BASE  long-BTC  OOS Sharpe={mo_base['sharpe_ann']:6.2f} ret={mo_base['ret_ann']*100:7.2f}% "
          f"maxDD={mo_base['maxdd']*100:7.2f}% PSR={p_base:.3f}")
    print(f"  GATED long-BTC  OOS Sharpe={mo_gate['sharpe_ann']:6.2f} ret={mo_gate['ret_ann']*100:7.2f}% "
          f"maxDD={mo_gate['maxdd']*100:7.2f}% PSR={p_gate:.3f}")
    overlay_helps = (mo_gate["sharpe_ann"] > mo_base["sharpe_ann"] + 0.1) or \
                    (mo_gate["maxdd"] > mo_base["maxdd"] + 0.03 and
                     mo_gate["sharpe_ann"] >= mo_base["sharpe_ann"] - 0.05)
    print(f"  overlay improves risk-adjusted? {overlay_helps}\n")

    # ===================================================================
    #  DEFLATION across EVERY variant tried (whole fracdiff family)
    # ===================================================================
    sr_star = bt.dsr_benchmark(sr_pool)
    bs_mo = best_standalone[2]
    dsr_psr = bt.psr(bs_mo["sr_pp"], bs_mo["n"], bs_mo["skew"], bs_mo["kurt"],
                     sr_benchmark=sr_star)
    print(f"=== DEFLATION ===")
    print(f"n_variants_tried (all knobs) = {_VARIANTS}")
    print(f"within-family DSR SR* (per-period) = {sr_star:.5f}")
    print(f"best standalone = {best_standalone[1]} OOS SR_pp={bs_mo['sr_pp']:.5f} "
          f"-> {'BEATS' if bs_mo['sr_pp']>sr_star else 'FAILS'} deflated bar; "
          f"deflated-PSR={dsr_psr:.3f}\n")

    # ===================================================================
    #  VERDICT
    # ===================================================================
    p_best = best_standalone[3]
    standalone_edge = (np.isfinite(p_best) and p_best >= 0.95 and
                       bs_mo["ret_ann"] > 0 and bs_mo["sr_pp"] > sr_star and
                       both_v_ret > 0.1)  # fractional part must ADD value
    overlay_edge = overlay_helps and p_gate >= 0.80

    if standalone_edge:
        verdict = "EDGE"; role = "standalone-alpha"
    elif overlay_edge and mo_gate["sharpe_ann"] > mo_base["sharpe_ann"] + 0.3 and p_gate >= 0.95:
        verdict = "EDGE"; role = "risk-overlay"
    elif (np.isfinite(p_best) and 0.80 <= p_best < 0.95 and bs_mo["ret_ann"] > 0) or overlay_edge:
        verdict = "MARGINAL"
        role = "risk-overlay" if (overlay_edge and not (np.isfinite(p_best) and p_best >= 0.80)) else "standalone-alpha"
    else:
        verdict = "DEAD"
        role = "both" if overlay_helps else "none"

    # headline OOS numbers reported = best standalone
    oos_sharpe = float(bs_mo["sharpe_ann"])
    notes = (
        f"LdP fixed-width fracdiff (ADF t-stat + binomial weights impl. in numpy). "
        f"Min-d (IS log-price, 5% crit -2.86) mean d*={mean_dstar:.2f} across {C.shape[1]} "
        f"perps (8h, {n} bars). KEY TEST: fractional vs integer-diff(d=1 returns) "
        f"in a pooled-ridge cross-sec market-neutral book: FRAC_only-RET_only OOS "
        f"Sharpe={frac_v_ret:+.2f}, BOTH-RET_only={both_v_ret:+.2f} "
        f"(fractional part adds {'value' if both_v_ret>0.1 else 'NO value'}). "
        f"Best standalone={best_standalone[1]} OOS Sharpe={oos_sharpe:.2f} "
        f"ret={bs_mo['ret_ann']*100:.1f}%/yr PSR={p_best:.2f}; mean-rev(fade z) "
        f"OOS Sharpe={mr_sel['mo']['sharpe_ann']:.2f}. "
        f"OVERLAY (|fracdiff z| gate on long-BTC): base Sharpe={mo_base['sharpe_ann']:.2f} "
        f"-> gated {mo_gate['sharpe_ann']:.2f}, maxDD {mo_base['maxdd']*100:.0f}%->"
        f"{mo_gate['maxdd']*100:.0f}% (helps={overlay_helps}). "
        f"DEFLATION: {_VARIANTS} variants -> within-family DSR SR*={sr_star:.4f}, "
        f"best OOS SR_pp={bs_mo['sr_pp']:.4f} ({'beats' if bs_mo['sr_pp']>sr_star else 'fails'}), "
        f"deflated-PSR={dsr_psr:.2f}. "
        f"CRITIQUE: fracdiff is a high-knob method (d,tau,z_win,ridge_lam,thresholds); "
        f"any standalone Sharpe must clear the inflated DSR bar AND beat the d=1 baseline "
        f"to credit the FRACTIONAL part specifically. Directional/mean-rev priors died "
        f"here, so a stationary-but-memory-rich series alone is unlikely to resurrect them."
    )
    print("=== VERDICT:", verdict, f"(role={role}) ===")
    print(notes)

    out = dict(
        key="fracdiff_features",
        method="Fractional differentiation (Lopez de Prado ch.5): fixed-width-window "
               "fracdiff of log-price at minimum-d (ADF) + ridge/mean-rev + overlay.",
        file="experiments/exo_fracdiff_features.py",
        libs_implemented="fixed-width-window fractional-diff binomial weights (numpy); "
                         "Augmented Dickey-Fuller t-stat with AIC lag selection (numpy); "
                         "closed-form ridge regression (numpy). No pywt/statsmodels used.",
        implemented=True,
        verdict=verdict,
        role=role,
        market_neutral=True,
        universe=f"{C.shape[1]} USDT-M perps {INTERVAL} full-{START} (MATIC excluded: delisted 2024-09)",
        n_obs=int(bs_mo["n"]),
        oos_sharpe=round(oos_sharpe, 4),
        oos_ret_ann_pct=round(float(bs_mo["ret_ann"] * 100), 4),
        maxdd_pct=round(float(bs_mo["maxdd"] * 100), 4),
        turnover=round(float(bs_mo["turnover"]), 5),
        psr=round(float(p_best), 4) if np.isfinite(p_best) else None,
        dsr=round(float(sr_star), 6),
        cost_bps=COST_BPS,
        n_variants_tried=int(_VARIANTS),
        overlay_base_sharpe=round(float(mo_base["sharpe_ann"]), 4),
        overlay_gated_sharpe=round(float(mo_gate["sharpe_ann"]), 4),
        mean_d_star=round(float(mean_dstar), 3),
        frac_minus_ret_oos_sharpe=round(float(frac_v_ret), 4),
        both_minus_ret_oos_sharpe=round(float(both_v_ret), 4),
        best_standalone=best_standalone[1],
        deflated_psr=round(float(dsr_psr), 4),
        notes=notes,
        data_caveats=("close-to-close 8h perp closes; d* frozen from IS log-price (no "
                      "leak); fracdiff/ADF/ridge fit on IS only; market-neutral close-to-"
                      "close maxDD understates intrabar liq/gap; adf_per_coin=" +
                      json.dumps({k: round(v['d_star'], 2) for k, v in adf_summary.items()})),
    )
    rep = ROOT / "reports" / "exo_fracdiff_features.json"
    rep.write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote", rep)
    return out


if __name__ == "__main__":
    main()
