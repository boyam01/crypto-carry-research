"""Candidate: rmt_ou_statarb  (family = exotic-math / stat-arb)

Random Matrix Theory (Marchenko-Pastur) + Ornstein-Uhlenbeck mean-reversion,
the FACTOR-RESIDUAL version of Avellaneda-Lee (2010) "Statistical Arbitrage in
the U.S. Equities Market".  The COIN-PAIR cointegration version already died on
this machine (cand_pairs_kalman.py: OOS Sharpe 0.16, PSR 0.58, DSR 0.03).  This
asks a genuinely different question: does mean-reversion of the IDIOSYNCRATIC
residual left after projecting out RMT-cleaned statistical eigen-factors survive
where raw cointegration pairs did not?

METHOD (per rebalance date t, using ONLY returns strictly before t)
-------------------------------------------------------------------
1. Take a trailing window W of daily simple returns for N coins -> R (W x N).
2. Standardise each column (z-score within the window) -> Y.  Correlation matrix
   C = Y'Y / W.  (Avellaneda-Lee use the correlation, not covariance, matrix.)
3. RMT / Marchenko-Pastur clean.  For a W x N panel of iid noise the eigenvalue
   spectrum of C lies in [lambda_-, lambda_+] with
       lambda_+- = (1 +- sqrt(N/W))^2   (q = N/W; sigma^2 = 1 for a corr matrix).
   Eigenvalues ABOVE lambda_+ carry signal; the rest is noise.  We KEEP the
   signal eigenvectors (>= lambda_+, plus we always treat the top "market" mode
   as signal) and treat them as the statistical risk factors.
4. Eigenportfolios.  For each signal eigenvector v_k, the factor return on a day
   is the return of the eigenportfolio  Q_k = v_k / sigma_i  (Avellaneda-Lee weight
   each name by 1/vol so the factor is a return, not a z-score combination).  We
   regress each coin's window return series on these factor returns (OLS, no
   intercept beyond de-meaning) -> betas + idiosyncratic residual series e_i(tau).
5. OU on the cumulative residual.  X_i = cumsum(e_i) is modelled as a discrete
   OU / AR(1):  X(tau+1) = a + b X(tau) + zeta.  kappa = -log(b)*252 (speed),
   m = a/(1-b) (equilibrium), sigma_eq = std of residual of the AR(1) fit
   / sqrt(1-b^2).  The s-score is the dimensionless deviation
       s_i = (X_i(last) - m_i) / sigma_eq_i      (Avellaneda-Lee eq. for s-score)
   (we also de-mean s by its cross-section each day = dollar-neutral tilt).
6. Trade.  Enter long the residual when s < -s_bo (cheap, expect mean-revert up),
   short when s > +s_bo; close when |s| < s_close.  Market-neutral: positions are
   de-meaned across the cross-section and gross-normalised to 1.  Only names with
   a fast enough OU (kappa above a floor => half-life below window) are eligible.

The traded asset for name i is the COIN itself but we hold a dollar-neutral book
(sum of signed weights ~ 0), so the systematic factor exposure is hedged by
construction (we are long cheap residuals / short rich residuals across coins).
This is the honest, tradeable analogue of "trade the residual": going long the
eigen-residual of coin i = long coin i and short its factor replication; the
de-meaned cross-section of coins approximates that hedge cheaply (no per-name
synthetic factor leg, which would be untradeable / expensive).

GOVERNANCE (hard rules — see header of the repo brief)
------------------------------------------------------
- NO LOOK-AHEAD: the s-score on date t is built from returns in (t-W, t-1].  The
  resulting target weights are SHIFTED 1 bar (decided on close t, earn coin
  returns over t->t+1).  Every rolling/RMT/OU quantity is causal.
- COST: 10 bp/leg on |Δweight| summed across the book (many-leg, daily reform of
  a ~N-name book => use the higher illiquid-leaning cost; also report 5 bp).
- CHRONOLOGICAL OOS: ALL knobs (window W, signal-factor rule, s thresholds,
  rebalance frequency, kappa floor) are scanned on the FIRST 60% of dates only;
  the single best-IS config is reported on the LAST 40% (untouched).
- DEFLATE: we count EVERY variant tried (n_variants_tried) and report the within
  -family DSR benchmark (bt.dsr_benchmark over all configs' IS per-period Sharpe)
  and the OOS PSR of the chosen config.  Exotic methods have many knobs => this
  matters a lot.
- maxDD here is close-to-close & market-neutral => an ILLUSION (no intrabar liq /
  gap).  Flagged; gross vs net both shown.
- DUAL FRAMING: also test the s-score dispersion as a RISK OVERLAY that gates a
  simple long-BTC base book (high cross-sectional |s| dispersion = factor / noise
  regime).  Report overlay_base_sharpe vs overlay_gated_sharpe.

LIBS IMPLEMENTED IN NUMPY (none available off the shelf): Marchenko-Pastur edge
+ eigen-clean of the correlation matrix; eigenportfolio construction; per-name
OLS factor regression -> residual; discrete OU/AR(1) fit -> s-score.  (All from
the cited Avellaneda-Lee 2010 equations; numpy only.)
"""
from __future__ import annotations
import sys, time, json, itertools, warnings, pathlib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import fetch_binance as fb
from engine import backtest as bt

# Liquid USDT-M perps with real 2022+ daily history (same curated set as the
# cointegration experiment, for a like-for-like comparison; ~20+ names).
UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
    "ATOMUSDT", "UNIUSDT", "ETCUSDT", "BCHUSDT", "FILUSDT", "TRXUSDT",
    "NEARUSDT", "AAVEUSDT", "EOSUSDT", "XLMUSDT", "ALGOUSDT", "SANDUSDT",
    "MANAUSDT", "AXSUSDT", "FTMUSDT", "THETAUSDT", "EGLDUSDT", "VETUSDT",
    "ICPUSDT",
]
START = "2022-01-01"
TRAIN_FRAC = 0.60
PPY = 365
COST_BP_LEG = 10.0          # many-leg daily book -> illiquid-leaning cost
COST_BP_LEG_LO = 5.0        # also report at the 5bp floor


def _ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


# ------------------------------- data --------------------------------
def load_returns(end):
    closes = {}
    for s in UNIVERSE:
        k = fb.klines(s, "1d", _ms(START), end, futures=True)
        if k is not None and len(k) > 400:
            closes[s] = k["close"]
    C = pd.DataFrame(closes).sort_index()
    C = C.dropna(axis=1, thresh=int(0.97 * len(C)))     # near-complete history only
    C = C.dropna()
    ret = C.pct_change().dropna()                        # simple daily returns
    return C, ret


# ---------------------- RMT / Marchenko-Pastur clean -----------------
def mp_signal_eigs(corr, q, always_keep_top=1):
    """Eigen-decompose a correlation matrix; return the indices/vectors whose
    eigenvalue exceeds the Marchenko-Pastur upper edge lambda_+ = (1+sqrt(q))^2
    with q = N/W.  These are the 'signal' factors; the rest is RMT noise.

    Always keep at least the top `always_keep_top` modes (the market mode is the
    dominant signal even if W is short).  Returns (vals_sig, vecs_sig, lam_plus).
    """
    vals, vecs = np.linalg.eigh(corr)               # ascending
    order = np.argsort(vals)[::-1]                   # descending
    vals, vecs = vals[order], vecs[:, order]
    lam_plus = (1.0 + np.sqrt(q)) ** 2
    keep = vals > lam_plus
    nkeep = int(keep.sum())
    if nkeep < always_keep_top:
        nkeep = always_keep_top
    nkeep = max(1, min(nkeep, len(vals) - 1))       # keep at least 1, leave noise space
    return vals[:nkeep], vecs[:, :nkeep], lam_plus


# ----------------------- OU / AR(1) on residual ----------------------
def ou_sscore(resid, dt_per_year=252):
    """Fit discrete OU / AR(1) to the CUMULATIVE residual and return the s-score.

    X(tau) = cumsum(resid).  AR(1): X_{t} = a + b X_{t-1} + zeta.
    kappa = -log(b) * dt_per_year (speed of mean reversion),
    m = a/(1-b) (equilibrium level of X),
    sigma_eq = sqrt( var(zeta) / (1 - b^2) )  (equilibrium std of X),
    s = (X_last - m) / sigma_eq.

    Returns (s, kappa, half_life_days).  s=nan if not mean-reverting (b>=1 or b<=0).
    """
    X = np.cumsum(resid)
    if len(X) < 20:
        return np.nan, np.nan, np.nan
    x0 = X[:-1]
    x1 = X[1:]
    # OLS  x1 = a + b x0
    A = np.column_stack([np.ones_like(x0), x0])
    coef, *_ = np.linalg.lstsq(A, x1, rcond=None)
    a, b = coef
    if not (0.0 < b < 1.0):                # not mean-reverting on this window
        return np.nan, np.nan, np.nan
    zeta = x1 - (a + b * x0)
    var_z = zeta @ zeta / max(1, len(zeta) - 2)
    kappa = -np.log(b) * dt_per_year
    m = a / (1.0 - b)
    var_eq = var_z / (1.0 - b * b)
    if var_eq <= 0:
        return np.nan, np.nan, np.nan
    sigma_eq = np.sqrt(var_eq)
    s = (X[-1] - m) / sigma_eq
    half_life = np.log(2) / (-np.log(b)) if b < 1 else np.inf   # in window-bars
    return float(s), float(kappa), float(half_life)


# ------------------- compute s-scores for every name at one date -----
def sscores_at(retwin):
    """retwin: (W x N) trailing return window (rows = days, cols = names), the
    LAST row being the most recent day < rebalance date.  Returns the per-name
    s-score array and the cross-sectional |s| dispersion (for the overlay).

    Causal: this only ever sees the trailing window passed by the caller.
    """
    W, N = retwin.shape
    q = N / W
    sig = retwin.std(axis=0)
    sig[sig == 0] = np.nan
    Y = (retwin - retwin.mean(axis=0)) / sig            # standardise columns
    Y = np.nan_to_num(Y)
    corr = (Y.T @ Y) / W
    vals_s, vecs_s, lam_plus = mp_signal_eigs(corr, q, always_keep_top=1)
    # eigenportfolio factor returns: weight name i by v_k[i]/sigma_i (Avellaneda-Lee)
    # factor_return_k(tau) = sum_i (v_k[i]/sigma_i) * retwin[tau, i]
    inv_sig = 1.0 / np.where(np.isfinite(sig), sig, np.inf)
    Qw = vecs_s * inv_sig[:, None]                       # (N x K) eigenportfolio weights
    F = retwin @ Qw                                      # (W x K) factor return series
    # regress each name's return series on the K factors (with intercept) -> resid
    G = np.column_stack([np.ones(W), F])                 # (W x (K+1))
    GtG_inv = np.linalg.pinv(G.T @ G)
    B = GtG_inv @ (G.T @ retwin)                         # ((K+1) x N) coefs
    resid = retwin - G @ B                               # (W x N) idiosyncratic resid
    s = np.full(N, np.nan)
    kap = np.full(N, np.nan)
    hl = np.full(N, np.nan)
    for i in range(N):
        si, ki, hi = ou_sscore(resid[:, i])
        s[i], kap[i], hl[i] = si, ki, hi
    return s, kap, hl, int(vecs_s.shape[1]), float(lam_plus)


# --------------------------- the strategy ----------------------------
# PERFORMANCE: the eigen-clean + OU fit (sscores_at) is the only step that
# depends on `window`; it is INDEPENDENT of s_bo/s_close/kappa_floor/rebal.  We
# therefore precompute s/kappa/halflife/dispersion at EVERY date once per window
# and cache it, so the (s_bo x s_close x kappa_floor x rebal) grid becomes a
# cheap state-machine over the cached arrays.  Same numbers, ~100x faster.
_SSCACHE: dict[int, tuple] = {}


def precompute_sscores(ret, window):
    """Return (S, KAP, HL, SDISP) each (T x N) / (T,), where row t holds the
    s-score etc. computed from the trailing window (t-window .. t-1) -> strictly
    causal.  Rows before window are nan.  Cached per window."""
    if window in _SSCACHE:
        return _SSCACHE[window]
    T, N = ret.shape
    S = np.full((T, N), np.nan)
    KAP = np.full((T, N), np.nan)
    HL = np.full((T, N), np.nan)
    SDISP = np.full(T, np.nan)
    for t in range(window + 5, T):
        retwin = ret[t - window:t, :]              # rows (t-window .. t-1): < t
        s, kap, hl, _, _ = sscores_at(retwin)
        S[t] = s
        KAP[t] = kap
        HL[t] = hl
        SDISP[t] = np.nanstd(s)
    _SSCACHE[window] = (S, KAP, HL, SDISP)
    return _SSCACHE[window]


def run_config(ret, dates, window, s_bo, s_close, kappa_floor, rebal,
               cost_bps, idx_slice):
    """Run the RMT-OU residual mean-reversion.  Consumes the per-window cached
    s-scores; applies the (s_bo, s_close, kappa_floor, rebal) state machine.

    ret: (T x N) ndarray of returns.  window: trailing window length (days).
    s_bo: entry |s| threshold; s_close: exit |s| threshold.
    kappa_floor: minimum OU speed (1/yr) for a name to be tradeable.
    rebal: recompute / re-target every `rebal` bars (hold target in between).
    Returns dict(net, gross, pos(T x N target weight), turn, sdisp(per-bar)).
    All quantities causal; positions SHIFTED 1 bar before earning returns.
    """
    T, N = ret.shape
    S, KAP, HL, SDISP = precompute_sscores(ret, window)
    pos = np.zeros((T, N))                 # target weights decided AT bar t (info < t)
    sdisp = np.full(T, np.nan)             # cross-sectional |s| dispersion at t
    held = np.zeros(N)                     # current state-machine target (+1/-1/0)
    last_compute = -10**9
    start_t = max(window + 5, idx_slice.start if idx_slice.start else window + 5)
    for t in range(start_t, T):
        if (t - last_compute) >= rebal or last_compute < 0:
            s = S[t]; kap = KAP[t]
            sdisp[t] = SDISP[t]
            elig = np.isfinite(s) & np.isfinite(kap) & (kap >= kappa_floor)
            # vectorised state machine on the new s
            new = held.copy()
            new[~elig] = 0.0
            flat = (held == 0.0) & elig
            new[flat & (s < -s_bo)] = 1.0          # residual low -> long
            new[flat & (s > s_bo)] = -1.0          # residual high -> short
            inpos = (held != 0.0) & elig
            new[inpos & (np.abs(s) < s_close)] = 0.0   # close near equilibrium
            # (names in position with |s|>=s_close keep held value, already copied)
            # market-neutral + gross-normalise to 1
            raw = new.copy()
            if np.any(raw != 0):
                raw = raw - raw.mean()              # dollar-neutral
                g = np.abs(raw).sum()
                w = raw / g if g > 0 else raw
            else:
                w = raw
            held = new
            pos[t] = w
            last_compute = t
        else:
            sdisp[t] = SDISP[last_compute]
            pos[t] = pos[last_compute]              # hold target between rebalances

    # SHIFT >=1 bar: weights decided at t earn coin returns over t->t+1
    sig_w = np.zeros_like(pos)
    sig_w[1:] = pos[:-1]
    gross_step = (sig_w * ret).sum(axis=1)          # portfolio gross return per bar
    dpos = np.abs(np.diff(sig_w, axis=0, prepend=np.zeros((1, N)))).sum(axis=1)
    cost = (cost_bps / 1e4) * dpos
    net = gross_step - cost
    turn = dpos
    return dict(net=net, gross=gross_step, pos=sig_w, turn=turn, sdisp=sdisp)


def evaluate(net, pos_sumabs, sl):
    seg = net[sl]
    m = bt.metrics(seg, PPY)
    p = bt.psr(m["sr_pp"], m["n"], m["skew"], m["kurt"])
    return m, p


def main():
    t0 = time.time()
    end = int(time.time() * 1000)
    C, retdf = load_returns(end)
    ret = retdf.values
    dates = retdf.index
    T, N = ret.shape
    cut = int(T * TRAIN_FRAC)
    is_sl = slice(0, cut)
    oos_sl = slice(cut, T)
    print(f"Loaded {N} coins x {T} daily return bars "
          f"({dates[0].date()} -> {dates[-1].date()})  IS=0..{cut} OOS={cut}..{T}  "
          f"({time.time()-t0:.1f}s)")
    print(f"Universe: {', '.join(c[:-4] for c in C.columns)}")

    # ----- knob grid (ALL tuned on IS only) -----
    WINDOWS   = [40, 60, 90, 120]
    S_BO      = [1.0, 1.25, 1.5, 2.0]
    S_CLOSE   = [0.25, 0.5, 0.75]
    KAPPA_FL  = [4.0, 8.5, 12.0]        # 8.5/yr ~ Avellaneda-Lee floor (half-life<~30d)
    REBAL     = [1, 3, 5]              # daily / 3d / weekly

    grid = list(itertools.product(WINDOWS, S_BO, S_CLOSE, KAPPA_FL, REBAL))
    print(f"\nScanning {len(grid)} configs on IS (60%) only "
          f"[window x s_bo x s_close x kappa_floor x rebal]...")

    is_results = []        # (cfg, is_sharpe, is_sr_pp, is_psr)
    family_sr_pp = []      # IS per-period Sharpe of EVERY config (DSR family)
    for (w, sbo, scl, kf, rb) in grid:
        if scl >= sbo:                     # require close threshold < entry threshold
            continue
        r = run_config(ret, dates, w, sbo, scl, kf, rb, COST_BP_LEG, is_sl)
        m, p = evaluate(r["net"], None, is_sl)
        if not np.isfinite(m["sr_pp"]):
            continue
        family_sr_pp.append(m["sr_pp"])
        is_results.append(dict(cfg=(w, sbo, scl, kf, rb), m=m, psr=p,
                               turn=float(np.nanmean(r["turn"][is_sl]))))

    n_variants = len(is_results)
    print(f"  {n_variants} valid configs evaluated on IS.")
    if not is_results:
        out = dict(candidate="rmt_ou_statarb", verdict="ERROR",
                   note="No valid configs on IS.")
        (ROOT / "reports" / "exo_rmt_ou_statarb.json").write_text(json.dumps(out, indent=2, default=float))
        return out

    # pick best-IS config by IS annualised Sharpe (selection uses IS ONLY)
    is_results.sort(key=lambda d: -d["m"]["sharpe_ann"])
    best = is_results[0]
    w, sbo, scl, kf, rb = best["cfg"]
    print(f"\nBest-IS config: window={w} s_bo={sbo} s_close={scl} kappa_floor={kf} "
          f"rebal={rb}  | IS Sharpe={best['m']['sharpe_ann']:.2f} "
          f"IS ret={best['m']['ret_ann']*100:.1f}% turn={best['turn']:.2f}")
    print("  IS top-5 configs:")
    for d in is_results[:5]:
        print(f"    {d['cfg']}  IS_Shrp={d['m']['sharpe_ann']:6.2f} "
              f"ret%={d['m']['ret_ann']*100:6.1f} turn={d['turn']:.2f} PSR={d['psr']:.2f}")

    # ----- OOS evaluation of the SINGLE chosen config (10bp and 5bp) -----
    r_oos = run_config(ret, dates, w, sbo, scl, kf, rb, COST_BP_LEG, oos_sl)
    mO, pO = evaluate(r_oos["net"], None, oos_sl)
    gO = bt.metrics(r_oos["gross"][oos_sl], PPY)            # gross OOS for honesty
    turnO = float(np.nanmean(r_oos["turn"][oos_sl]))

    r_oos5 = run_config(ret, dates, w, sbo, scl, kf, rb, COST_BP_LEG_LO, oos_sl)
    mO5, pO5 = evaluate(r_oos5["net"], None, oos_sl)

    # ----- DSR deflation across the family of ALL configs tried -----
    sr_star = bt.dsr_benchmark(family_sr_pp)
    dsrO = bt.psr(mO["sr_pp"], mO["n"], mO["skew"], mO["kurt"], sr_benchmark=sr_star)

    print("\n" + "=" * 78)
    print("STANDALONE OOS (market-neutral RMT-OU residual mean-reversion)")
    print("=" * 78)
    print(f"  OOS_bars     = {mO['n']}")
    print(f"  Sharpe_ann   = {mO['sharpe_ann']:.3f}   (gross {gO['sharpe_ann']:.3f})")
    print(f"  ret_ann      = {mO['ret_ann']*100:.2f}%")
    print(f"  vol_ann      = {mO['vol_ann']*100:.2f}%")
    print(f"  maxDD        = {mO['maxdd']*100:.2f}%   (close-to-close, market-neutral => ILLUSION; no intrabar liq/gap)")
    print(f"  turnover     = {turnO:.3f}  (sum|Δw|/bar across book)")
    print(f"  PSR (10bp)   = {pO:.3f}    | PSR (5bp) = {pO5:.3f}  (Sharpe {mO5['sharpe_ann']:.2f})")
    print(f"  DSR SR*      = {sr_star:.4f} per-period  (family of {len(family_sr_pp)} configs)")
    print(f"  DSR (PSR vs SR*) = {dsrO:.3f}")

    # ===================== OVERLAY (dual framing) =====================
    # The cross-sectional |s| dispersion is a factor/idiosyncratic-regime gauge.
    # Hypothesis: when residual dispersion is HIGH, idiosyncratic risk dominates
    # (good for mean-reversion / risky for trend) -> de-risk a long-BTC base book.
    # Base book = 100% long BTC.  Overlay: scale BTC exposure by a gate in [0,1]
    # based on the trailing percentile of sdisp (computed causally, IS-calibrated
    # threshold).  Compare base vs gated OOS Sharpe.
    btc_col = list(C.columns).index("BTCUSDT")
    btc_ret = ret[:, btc_col]
    # build sdisp series from a fixed mid config window (use chosen w, daily rebal
    # so we have a dispersion reading every day)
    r_disp = run_config(ret, dates, w, sbo, scl, kf, 1, COST_BP_LEG, slice(window_floor := 0, T))
    sdisp = r_disp["sdisp"]
    sdisp_s = pd.Series(sdisp, index=dates)
    # causal gate: trailing 252d percentile rank of dispersion (high disp -> low gate)
    roll_rank = sdisp_s.rolling(252, min_periods=60).apply(
        lambda v: (v[-1] >= v).mean(), raw=True)
    # IS-calibrate a single threshold: de-risk (gate=0.3) when disp in top 30% IS,
    # else full (gate=1.0).  Threshold chosen on IS only.
    is_rank = roll_rank.values[is_sl]
    thr = np.nanpercentile(is_rank[np.isfinite(is_rank)], 70) if np.isfinite(is_rank).any() else 0.7
    gate = np.where(roll_rank.values >= thr, 0.30, 1.0)
    gate = np.nan_to_num(gate, nan=1.0)
    gate_sig = np.concatenate([[1.0], gate[:-1]])          # shift 1 bar
    base_pos = np.ones(T)
    base_net = bt.run(btc_ret, base_pos, COST_BP_LEG)      # base: buy-hold BTC (cost on entry)
    gated_pos = gate_sig
    gated_net = bt.run(btc_ret, gated_pos, COST_BP_LEG)
    mB = bt.metrics(base_net[oos_sl], PPY, base_pos[oos_sl])
    mG = bt.metrics(gated_net[oos_sl], PPY, gated_pos[oos_sl])
    print("\n" + "=" * 78)
    print("OVERLAY (s-dispersion regime gate on a long-BTC base book), OOS")
    print("=" * 78)
    print(f"  base  long-BTC : Sharpe={mB['sharpe_ann']:.3f} ret={mB['ret_ann']*100:.1f}% maxDD={mB['maxdd']*100:.1f}%")
    print(f"  gated by |s|disp: Sharpe={mG['sharpe_ann']:.3f} ret={mG['ret_ann']*100:.1f}% maxDD={mG['maxdd']*100:.1f}%")
    print(f"  overlay improves Sharpe: {mG['sharpe_ann'] - mB['sharpe_ann']:+.3f}")

    # ----------------------------- VERDICT -----------------------------
    overlay_improves = (mG["sharpe_ann"] > mB["sharpe_ann"] + 0.10) and (mG["maxdd"] > mB["maxdd"])
    standalone_edge = (pO >= 0.95) and (dsrO >= 0.95) and (mO["ret_ann"] > 0) and (turnO < 5.0)
    standalone_marg = (pO >= 0.80) and (mO["ret_ann"] > 0) and (turnO < 5.0)

    if standalone_edge:
        role = "standalone-alpha"
        verdict = "EDGE"
        note = (f"RMT-OU residual mean-reversion EDGE: OOS Sharpe={mO['sharpe_ann']:.2f}, "
                f"PSR={pO:.2f}, DSR={dsrO:.2f} (SR*={sr_star:.3f}), net {COST_BP_LEG}bp/leg, "
                f"turnover {turnO:.2f}/bar. Survives where coint pairs (DSR 0.03) did not.")
    elif standalone_marg and overlay_improves:
        role = "both"
        verdict = "MARGINAL"
        note = (f"RMT-OU residual: standalone OOS positive but fragile (PSR={pO:.2f}, "
                f"DSR={dsrO:.2f}<0.95, SR*={sr_star:.3f}); AND |s|-dispersion overlay lifts "
                f"long-BTC Sharpe {mB['sharpe_ann']:.2f}->{mG['sharpe_ann']:.2f}.")
    elif standalone_marg:
        role = "standalone-alpha"
        verdict = "MARGINAL"
        note = (f"RMT-OU residual mean-reversion: OOS positive (PSR={pO:.2f}) but fails the "
                f"0.95 DSR bar (DSR={dsrO:.2f}, SR*={sr_star:.3f}); fragile / multiple-testing inflated.")
    elif overlay_improves:
        role = "risk-overlay"
        verdict = "MARGINAL"
        note = (f"RMT-OU residual has NO standalone alpha (PSR={pO:.2f}) but the |s|-dispersion "
                f"REGIME gate improves a long-BTC book OOS: Sharpe {mB['sharpe_ann']:.2f}->"
                f"{mG['sharpe_ann']:.2f}, maxDD {mB['maxdd']*100:.0f}%->{mG['maxdd']*100:.0f}%.")
    else:
        role = "none"
        verdict = "DEAD"
        note = (f"RMT-OU residual mean-reversion dies OOS: Sharpe={mO['sharpe_ann']:.2f}, "
                f"PSR={pO:.2f}, DSR={dsrO:.2f} (SR*={sr_star:.3f}); overlay no help "
                f"({mB['sharpe_ann']:.2f}->{mG['sharpe_ann']:.2f}). RMT eigen-factors do not "
                f"rescue stat-arb where cointegration pairs already failed.")

    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict}  (role={role})")
    print(note)
    print("=" * 78)

    out = dict(
        candidate="rmt_ou_statarb", family="exotic-math/stat-arb",
        method="RMT (Marchenko-Pastur) eigen-clean + eigenportfolio factor regression "
               "+ OU/AR(1) s-score residual mean-reversion (Avellaneda-Lee 2010, factor-residual)",
        libs_implemented="numpy: Marchenko-Pastur upper-edge eigen-clean of correlation matrix; "
                         "eigenportfolio (1/vol-weighted eigenvectors) factor returns; per-name OLS "
                         "factor regression -> idiosyncratic residual; discrete OU/AR(1) fit -> s-score.",
        universe=", ".join(c[:-4] for c in C.columns),
        universe_n=int(N), n_obs=int(T),
        start=str(dates[0].date()), end=str(dates[-1].date()),
        is_oos_cut=int(cut), train_frac=TRAIN_FRAC,
        cost_bp_leg=COST_BP_LEG,
        best_config=dict(window=int(w), s_bo=float(sbo), s_close=float(scl),
                         kappa_floor=float(kf), rebal=int(rb)),
        n_variants_tried=int(len(grid)), n_variants_valid=int(n_variants),
        is_best=dict(sharpe=float(best["m"]["sharpe_ann"]),
                     ret_ann=float(best["m"]["ret_ann"]), turn=float(best["turn"])),
        standalone=dict(
            oos_n=int(mO["n"]),
            oos_sharpe=float(mO["sharpe_ann"]),
            oos_sharpe_gross=float(gO["sharpe_ann"]),
            oos_sharpe_5bp=float(mO5["sharpe_ann"]),
            oos_ret_ann=float(mO["ret_ann"]),
            oos_vol_ann=float(mO["vol_ann"]),
            oos_maxdd=float(mO["maxdd"]),
            oos_psr=float(pO),
            oos_psr_5bp=float(pO5),
            oos_dsr=float(dsrO),
            dsr_sr_star=float(sr_star),
            turnover=float(turnO),
        ),
        overlay=dict(
            base_sharpe=float(mB["sharpe_ann"]),
            gated_sharpe=float(mG["sharpe_ann"]),
            base_maxdd=float(mB["maxdd"]),
            gated_maxdd=float(mG["maxdd"]),
            base_ret_ann=float(mB["ret_ann"]),
            gated_ret_ann=float(mG["ret_ann"]),
            gate_threshold=float(thr),
        ),
        role=role, verdict=verdict, note=note,
        maxdd_illusion_flag=True,
        governance=dict(
            no_lookahead="s-score from returns in (t-W, t-1]; weights shifted 1 bar.",
            cost="10bp/leg on |Δw| across book (5bp also reported).",
            chronological_oos="all knobs scanned on first 60%; last 40% reported.",
            deflate=f"{len(grid)} variants in grid; within-family DSR SR*={sr_star:.4f}.",
            maxdd="close-to-close market-neutral maxDD flagged as illusion.",
        ),
    )
    rp = ROOT / "reports" / "exo_rmt_ou_statarb.json"
    rp.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {rp}  ({time.time()-t0:.1f}s total)")
    return out


if __name__ == "__main__":
    main()
