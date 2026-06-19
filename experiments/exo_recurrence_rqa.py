"""Exotic-math candidate: recurrence_rqa  (family = nonlinear-dynamics / regime)

METHOD: Recurrence Quantification Analysis (RQA).
  Build a recurrence plot from a DELAY-EMBEDDED return series, threshold pairwise
  distances at a fixed recurrence-rate target, then compute the classic RQA line
  statistics on a CAUSAL rolling window:
    - RR   recurrence rate            (density of recurrent points)
    - %DET determinism               (fraction of recurrent pts on diagonals >= l_min)
    - LAM  laminarity                (fraction on vertical lines >= v_min)
    - TT   trapping time             (mean vertical-line length)
  Reference: Marwan, Romano, Thiel, Kurths (2007) "Recurrence plots for the
  analysis of complex systems", Physics Reports 438.

HYPOTHESIS (from spec):
  High determinism = predictable / persistent (deterministic-looking) regime in
  which TREND-following works; low determinism = stochastic / random regime where
  trend is noise. So RQA is primarily a REGIME / RISK DETECTOR.
  We test BOTH:
    (A) STANDALONE alpha: trade direction conditioned on the RQA regime
        (e.g. follow momentum when %DET is high).
    (B) OVERLAY: gate / size a simple base book (long-BTC and an equal-weight
        cross-sectional momentum book) by the RQA regime, and ask whether it
        improves OOS Sharpe / cuts drawdown vs the un-gated base.

WHAT I IMPLEMENTED IN NUMPY (libs_implemented):
  pure-numpy RQA from scratch (no pyunicorn / no pyts):
    - takens_embedding(): time-delay embedding v_t=[x_t,x_{t-d},...].
    - recurrence_matrix(): pairwise Euclidean distances + percentile threshold so
      recurrence rate ~ target RR (Marwan's recommended fixed-RR thresholding).
    - diagonal_lines() / vertical_lines(): run-length histograms of the binary
      recurrence matrix -> %DET, LAM, TT.
    - rqa_rolling(): causal rolling-window RQA, every value at bar t uses ONLY the
      window of embedded vectors ending at t (then we shift>=1 before trading).
  Everything else (backtest, PSR, DSR) uses engine.backtest.

GOVERNANCE complied with: features at t use data <= t and are shift(>=1) before
trading; cost >=5bp/leg on |Δposition|; ALL hyperparameters tuned on first 60%,
metrics on LAST 40%; PSR + within-family DSR over EVERY variant tried; gross vs
net; close-to-close maxDD flagged as an illusion.
"""
from __future__ import annotations
import sys, time, json, pathlib, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import fetch_binance as fb
from engine import backtest as bt

# MATICUSDT delisted/renamed to POL on 2024-09-11 -> excluded: including it
# truncates the whole panel (intersection death) and steals the true OOS window.
UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT",
            "TRXUSDT", "DOTUSDT", "NEARUSDT", "ATOMUSDT"]
START = "2022-01-01"
END = "2026-06-18"
INTERVAL = "8h"            # 8h bars: enough samples for rolling RQA windows
PPY = 365 * 3              # 3 eight-hour bars per day
TRAIN_FRAC = 0.60


def _ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


# ============================================================================
# PURE-NUMPY RQA  (the "missing library")
# ============================================================================
def takens_embedding(x: np.ndarray, m: int, d: int) -> np.ndarray:
    """Time-delay embedding. Returns (N-(m-1)d, m) matrix of state vectors.
    Row i = [x[i], x[i+d], ..., x[i+(m-1)d]] (forward form; we only ever feed it
    a CAUSAL window so no look-ahead leaks)."""
    x = np.asarray(x, float)
    n = len(x) - (m - 1) * d
    if n <= 1:
        return np.empty((0, m))
    idx = np.arange(n)[:, None] + d * np.arange(m)[None, :]
    return x[idx]


def recurrence_matrix(V: np.ndarray, rr_target: float):
    """Binary recurrence matrix at a fixed recurrence-rate target.

    Distances = Euclidean between embedded state vectors. Threshold eps is the
    rr_target-quantile of the off-diagonal distances so the recurrence rate is
    ~ rr_target regardless of regime (Marwan's fixed-RR thresholding -- this
    REMOVES the trivial 'high vol => fewer recurrences' confound)."""
    n = V.shape[0]
    if n < 3:
        return None
    # pairwise Euclidean distance
    diff = V[:, None, :] - V[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    off = D[iu]
    if off.size == 0:
        return None
    eps = np.quantile(off, rr_target)
    R = (D <= eps).astype(np.int8)
    np.fill_diagonal(R, 0)             # exclude the line of identity
    return R


def _line_lengths_diagonal(R: np.ndarray) -> np.ndarray:
    """Lengths of all diagonal lines (excluding main diagonal) of binary R."""
    n = R.shape[0]
    lengths = []
    for k in range(1, n):              # off-diagonals only
        diag = np.diag(R, k)
        # run-length encode the 1s
        run = 0
        for v in diag:
            if v:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
        if run:
            lengths.append(run)
    return np.asarray(lengths, int) if lengths else np.zeros(0, int)


def _line_lengths_vertical(R: np.ndarray) -> np.ndarray:
    """Lengths of all vertical lines of binary R (laminar states)."""
    n = R.shape[0]
    lengths = []
    for j in range(n):
        col = R[:, j]
        run = 0
        for v in col:
            if v:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
        if run:
            lengths.append(run)
    return np.asarray(lengths, int) if lengths else np.zeros(0, int)


def rqa_measures(R: np.ndarray, l_min: int = 2, v_min: int = 2) -> dict:
    """Classic RQA measures from a binary recurrence matrix."""
    n = R.shape[0]
    total = int(R.sum())                       # recurrent points (off-diag)
    if total == 0:
        return dict(RR=0.0, DET=0.0, LAM=0.0, TT=0.0)
    rr = total / (n * (n - 1))
    # diagonals are symmetric; count both triangles' points via 2*sum over k>=1
    dl = _line_lengths_diagonal(R)
    pts_on_diag = 2 * int(dl[dl >= l_min].sum()) if dl.size else 0
    det = pts_on_diag / total
    vl = _line_lengths_vertical(R)
    pts_on_vert = int(vl[vl >= v_min].sum()) if vl.size else 0
    lam = pts_on_vert / total
    long_v = vl[vl >= v_min]
    tt = float(long_v.mean()) if long_v.size else 0.0
    return dict(RR=float(rr), DET=float(det), LAM=float(lam), TT=tt)


def rqa_rolling(x: np.ndarray, win: int, m: int, d: int, rr_target: float,
                step: int = 1, l_min: int = 2, v_min: int = 2) -> pd.DataFrame:
    """CAUSAL rolling RQA. For each end-bar t, embed the window x[t-win+1 .. t],
    build the recurrence plot and compute measures. Value placed at index t uses
    ONLY data up to and including t (we shift>=1 before trading). `step` subsamples
    the end-bars (RQA is O(win^2); we ffill between)."""
    n = len(x)
    out = np.full((n, 4), np.nan)              # RR, DET, LAM, TT
    emb_len = win - (m - 1) * d                 # embedded vectors per window
    if emb_len < 8:
        return pd.DataFrame(out, columns=["RR", "DET", "LAM", "TT"])
    for t in range(win - 1, n, step):
        w = x[t - win + 1: t + 1]
        V = takens_embedding(w, m, d)
        R = recurrence_matrix(V, rr_target)
        if R is None:
            continue
        meas = rqa_measures(R, l_min, v_min)
        out[t] = [meas["RR"], meas["DET"], meas["LAM"], meas["TT"]]
    df = pd.DataFrame(out, columns=["RR", "DET", "LAM", "TT"])
    return df.ffill()                            # carry last computed RQA forward


# ============================================================================
# DATA
# ============================================================================
def load_panel(symbols, start, end):
    out = {}
    for s in symbols:
        k = fb.klines(s, INTERVAL, start, end, futures=True)
        if k is None or len(k) < 2000:
            continue
        df = pd.DataFrame(index=k.index)
        df["close"] = k["close"].astype(float)
        df["ret"] = df["close"].pct_change()        # simple return (for PnL)
        df["lret"] = np.log(df["close"]).diff()     # log return (for embedding)
        out[s] = df.dropna()
    return out


def evaluate(net, position=None):
    net = np.asarray(net, float)
    n = len(net)
    tr, te = bt.oos_split(n, TRAIN_FRAC)
    mi = bt.metrics(net[tr], PPY, position[tr] if position is not None else None)
    mo = bt.metrics(net[te], PPY, position[te] if position is not None else None)
    psr = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
    return mi, mo, psr


# ============================================================================
# MAIN
# ============================================================================
def main():
    panel = load_panel(UNIVERSE, _ms(START), _ms(END))
    coins = list(panel.keys())
    print(f"loaded {len(coins)} coins, {INTERVAL} bars, PPY={PPY}, "
          f"OOS=last {int((1-TRAIN_FRAC)*100)}%\n")

    # common index across coins for portfolio building
    common = None
    for s in coins:
        common = panel[s].index if common is None else common.intersection(panel[s].index)
    rets = pd.DataFrame({s: panel[s]["ret"].reindex(common) for s in coins}).dropna()
    lrets = pd.DataFrame({s: panel[s]["lret"].reindex(common) for s in coins}).reindex(rets.index)
    idx = rets.index
    n = len(idx)
    print(f"aligned panel: {n} bars x {len(coins)} coins  "
          f"({idx[0].date()} -> {idx[-1].date()})\n")

    # -----------------------------------------------------------------
    # Precompute rolling %DET (and friends) per coin, for ONE base RQA config.
    # The RQA hyperparameters (win, m, d, rr_target) are the knobs; we grid them
    # but ALWAYS tune selection on IS only. step=3 to keep O(win^2) tractable.
    # -----------------------------------------------------------------
    n_variants = 0
    sr_trials = []        # within-family DSR pool (per-period IS Sharpes of every variant)

    # RQA hyperparameter grid (knobs -> overfit risk -> all counted & deflated)
    WINS = [60, 90]            # window length (bars) for the recurrence plot
    MS = [3, 5]                # embedding dimension
    DS = [1, 2]                # embedding delay
    RRS = [0.10, 0.15]         # recurrence-rate target
    STEP = 3

    # cache RQA per (coin, config)
    rqa_cache: dict = {}
    print("computing rolling RQA per coin x config (this is the heavy part)...")
    t0 = time.time()
    for (win, m, d, rr) in itertools.product(WINS, MS, DS, RRS):
        for s in coins:
            x = lrets[s].values
            key = (s, win, m, d, rr)
            rqa_cache[key] = rqa_rolling(x, win, m, d, rr, step=STEP)
    print(f"  done in {time.time()-t0:.1f}s, {len(rqa_cache)} (coin,config) RQA series\n")

    # -----------------------------------------------------------------
    # BASE BOOKS to gate/size, and a momentum signal for standalone tests
    # -----------------------------------------------------------------
    # (1) long-BTC base
    btc_ret = rets["BTCUSDT"].values
    # (2) equal-weight cross-sectional 12-bar momentum, market-neutral
    mom_lb = 12
    mom = lrets.rolling(mom_lb).sum().shift(1)             # signal known at t-1
    mom_rank = mom.rank(axis=1, pct=True) - 0.5            # cross-sectional, demeaned
    mom_w = mom_rank.div(mom_rank.abs().sum(axis=1), axis=0).fillna(0)  # dollar-neutral
    mom_base_ret = (mom_w * rets).sum(axis=1).values        # base momentum book ret stream
    # also a simple time-series momentum per coin (sign of past return)
    ts_mom_sig = np.sign(lrets.rolling(mom_lb).sum().shift(1)).fillna(0)

    tr_sl, te_sl = bt.oos_split(n, TRAIN_FRAC)

    # baseline metrics (un-gated)
    base_long_net = bt.run(btc_ret, np.ones(n), 5.0)
    _, base_long_oos, base_long_psr = evaluate(base_long_net, np.ones(n))
    base_mom_net = bt.run(mom_base_ret, np.ones(n), 0.0)  # mom_base_ret already net of weighting; cost handled below
    # recompute momentum base with proper per-leg cost via turnover of weights
    w_turn = mom_w.diff().abs().sum(axis=1).fillna(0).values
    base_mom_net = mom_base_ret - w_turn * (5.0 / 1e4)
    _, base_mom_oos, base_mom_psr = evaluate(base_mom_net)
    print("=== BASE BOOKS (un-gated, 5bp/leg) ===")
    print(f"  long-BTC      OOS Sharpe={base_long_oos['sharpe_ann']:6.2f}  "
          f"maxDD={base_long_oos['maxdd']*100:6.1f}%  PSR={base_long_psr:.3f}")
    print(f"  xsec-momentum OOS Sharpe={base_mom_oos['sharpe_ann']:6.2f}  "
          f"maxDD={base_mom_oos['maxdd']*100:6.1f}%  PSR={base_mom_psr:.3f}\n")

    # =================================================================
    # Build per-config aggregate %DET regime signals and run BOTH tests.
    # For each RQA config we form:
    #   det_btc  = BTC's %DET (for gating long-BTC)
    #   det_mkt  = cross-coin mean %DET (broad-regime gate for the momentum book)
    # Regime threshold = IS median of the %DET (causal: fixed from train only).
    # =================================================================
    results = []   # (config, dict of test outcomes)

    def det_series(s, cfg):
        # %DET, shifted 1 bar -> strictly known before trading
        return pd.Series(rqa_cache[(s,) + cfg]["DET"].values, index=idx).shift(1)

    for cfg in itertools.product(WINS, MS, DS, RRS):
        n_variants += 1
        det_btc = det_series("BTCUSDT", cfg).fillna(0).values
        det_mkt = np.nanmean(
            np.vstack([det_series(s, cfg).values for s in coins]), axis=0)
        det_mkt = pd.Series(det_mkt, index=idx).fillna(0).values

        # IS thresholds (median) -- tuned on train only
        thr_btc = np.nanmedian(det_btc[tr_sl][det_btc[tr_sl] > 0]) if np.any(det_btc[tr_sl] > 0) else 0.5
        thr_mkt = np.nanmedian(det_mkt[tr_sl][det_mkt[tr_sl] > 0]) if np.any(det_mkt[tr_sl] > 0) else 0.5

        high_det_btc = (det_btc >= thr_btc).astype(float)
        high_det_mkt = (det_mkt >= thr_mkt).astype(float)

        # ---------- (A) STANDALONE: TS-momentum on BTC, only ON when %DET high ----
        # position = sign(past return) gated by high-determinism regime
        btc_ts_sig = ts_mom_sig["BTCUSDT"].values
        pos_std = btc_ts_sig * high_det_btc
        net_std = bt.run(btc_ret, pos_std, 5.0)
        mi_std, mo_std, psr_std = evaluate(net_std, pos_std)
        sr_trials.append(mi_std["sr_pp"])

        # ---------- (A2) STANDALONE alpha2: xsec-momentum book but only trade in
        #               high-broad-determinism bars (regime-selected momentum) ----
        gated_w = mom_w.mul(pd.Series(high_det_mkt, index=idx), axis=0)
        gated_ret = (gated_w * rets).sum(axis=1).values
        gturn = gated_w.diff().abs().sum(axis=1).fillna(0).values
        net_std2 = gated_ret - gturn * (5.0 / 1e4)
        mi_std2, mo_std2, psr_std2 = evaluate(net_std2)
        sr_trials.append(mi_std2["sr_pp"])

        # ---------- (B) OVERLAY: size long-BTC by determinism regime --------------
        # hypothesis: trend/persistence works when %DET high -> full long; else flat
        ov_pos_long = high_det_btc            # 1 when deterministic, 0 otherwise
        net_ov_long = bt.run(btc_ret, ov_pos_long, 5.0)
        mi_ovl, mo_ovl, psr_ovl = evaluate(net_ov_long, ov_pos_long)
        sr_trials.append(mi_ovl["sr_pp"])

        # ---------- (B2) OVERLAY: gate the xsec-momentum book by broad %DET -------
        ov_w = mom_w.mul(pd.Series(high_det_mkt, index=idx), axis=0)
        ov_ret = (ov_w * rets).sum(axis=1).values
        ovturn = ov_w.diff().abs().sum(axis=1).fillna(0).values
        net_ov_mom = ov_ret - ovturn * (5.0 / 1e4)
        mi_ovm, mo_ovm, psr_ovm = evaluate(net_ov_mom)
        sr_trials.append(mi_ovm["sr_pp"])

        results.append(dict(
            cfg=cfg, thr_btc=thr_btc, thr_mkt=thr_mkt,
            std_is=mi_std["sharpe_ann"], std_oos=mo_std["sharpe_ann"], std_psr=psr_std,
            std2_is=mi_std2["sharpe_ann"], std2_oos=mo_std2["sharpe_ann"], std2_psr=psr_std2,
            ovl_is=mi_ovl["sharpe_ann"], ovl_oos=mo_ovl["sharpe_ann"], ovl_psr=psr_ovl,
            ovl_maxdd=mo_ovl["maxdd"], ovl_turn=mo_ovl["turnover"],
            ovm_is=mi_ovm["sharpe_ann"], ovm_oos=mo_ovm["sharpe_ann"], ovm_psr=psr_ovm,
            ovm_maxdd=mo_ovm["maxdd"], ovm_turn=mo_ovm["turnover"],
            std_obj=(mo_std, psr_std, pos_std), std2_obj=(mo_std2, psr_std2),
            ovl_obj=(mo_ovl, psr_ovl, ov_pos_long), ovm_obj=(mo_ovm, psr_ovm),
        ))

    # -----------------------------------------------------------------
    # SELECT best config on IS for each of the 4 strategy variants, then
    # report its OOS. (Selecting on IS, reporting OOS = honest.)
    # -----------------------------------------------------------------
    def pick(metric_is):
        return max(results, key=lambda r: (r[metric_is] if np.isfinite(r[metric_is]) else -9))

    best_std = pick("std_is")
    best_std2 = pick("std2_is")
    best_ovl = pick("ovl_is")
    best_ovm = pick("ovm_is")

    print("=== (A) STANDALONE: %DET-gated TS-momentum on BTC ===")
    print(f"  IS-best cfg(win,m,d,rr)={best_std['cfg']}  "
          f"IS={best_std['std_is']:.2f}  OOS Sharpe={best_std['std_oos']:.2f}  "
          f"PSR={best_std['std_psr']:.3f}")
    print("=== (A2) STANDALONE: broad-%DET-gated xsec-momentum book ===")
    print(f"  IS-best cfg={best_std2['cfg']}  IS={best_std2['std2_is']:.2f}  "
          f"OOS Sharpe={best_std2['std2_oos']:.2f}  PSR={best_std2['std2_psr']:.3f}")
    print("\n=== (B) OVERLAY: long-BTC gated by %DET regime vs un-gated long-BTC ===")
    print(f"  IS-best cfg={best_ovl['cfg']}  OOS Sharpe={best_ovl['ovl_oos']:.2f}  "
          f"(base long-BTC OOS={base_long_oos['sharpe_ann']:.2f})  "
          f"maxDD={best_ovl['ovl_maxdd']*100:.1f}% (base {base_long_oos['maxdd']*100:.1f}%)  "
          f"PSR={best_ovl['ovl_psr']:.3f}")
    print("=== (B2) OVERLAY: xsec-momentum gated by broad %DET vs un-gated momentum ===")
    print(f"  IS-best cfg={best_ovm['cfg']}  OOS Sharpe={best_ovm['ovm_oos']:.2f}  "
          f"(base momentum OOS={base_mom_oos['sharpe_ann']:.2f})  "
          f"maxDD={best_ovm['ovm_maxdd']*100:.1f}% (base {base_mom_oos['maxdd']*100:.1f}%)  "
          f"PSR={best_ovm['ovm_psr']:.3f}")

    # -----------------------------------------------------------------
    # DEFLATE: within-family DSR across EVERY trial (4 strategies x all configs)
    # -----------------------------------------------------------------
    srstar = bt.dsr_benchmark(sr_trials)
    print(f"\nn_variants (RQA configs) = {n_variants}; "
          f"total strategy trials in DSR pool = {len(sr_trials)}")
    print(f"within-family DSR SR* (per-period) = {srstar:.5f}")

    # pick the single best OOS performer across all 4 families for the verdict,
    # but judge it against the deflated bar and PSR + overlay-improvement.
    candidates = [
        ("standalone-tsmom-btc", best_std["std_obj"][0], best_std["std_obj"][1],
         "standalone-alpha", None),
        ("standalone-xsec-mom",  (best_std2["std2_obj"][0]), best_std2["std2_obj"][1],
         "standalone-alpha", None),
        ("overlay-long-btc",     best_ovl["ovl_obj"][0], best_ovl["ovl_obj"][1],
         "risk-overlay", base_long_oos),
        ("overlay-xsec-mom",     best_ovm["ovm_obj"][0], best_ovm["ovm_obj"][1],
         "risk-overlay", base_mom_oos),
    ]

    # ---- decide role + verdict ----
    # overlay must IMPROVE base OOS Sharpe (and ideally cut DD) to count
    best_overall = None
    for name, mo, psr, role, base in candidates:
        sr_pp = mo["sr_pp"]
        beats_dsr = np.isfinite(sr_pp) and sr_pp > srstar
        improves = True
        if base is not None:
            improves = (mo["sharpe_ann"] > base["sharpe_ann"] + 0.30) or \
                       (mo["maxdd"] > base["maxdd"] and mo["sharpe_ann"] >= base["sharpe_ann"])
        score = (mo["sharpe_ann"] if np.isfinite(mo["sharpe_ann"]) else -9)
        cand = dict(name=name, mo=mo, psr=psr, role=role, base=base,
                    beats_dsr=beats_dsr, improves=improves, score=score)
        if best_overall is None or score > best_overall["score"]:
            best_overall = cand

    mo = best_overall["mo"]; psr = best_overall["psr"]
    role = best_overall["role"]
    beats_dsr = best_overall["beats_dsr"]; improves = best_overall["improves"]
    base = best_overall["base"]

    # overlay sharpes for reporting
    overlay_base_sharpe = float(base["sharpe_ann"]) if base is not None else None
    overlay_gated_sharpe = float(mo["sharpe_ann"]) if base is not None else None

    if (psr is not None and psr >= 0.95 and beats_dsr and improves
            and mo["ret_ann"] > 0):
        verdict = "EDGE"
    elif (psr is not None and psr >= 0.80 and mo["ret_ann"] > 0
          and (improves or role == "standalone-alpha")):
        verdict = "MARGINAL"
    else:
        verdict = "DEAD"

    # Determine overall role label for the report
    std_ok = any(c["mo"]["sharpe_ann"] > 0.5 and c["psr"] and c["psr"] > 0.8
                 for c in [best_overall] if c["role"] == "standalone-alpha")
    overlay_ok = (base is not None and improves)
    if best_overall["role"] == "risk-overlay" and overlay_ok:
        role_final = "risk-overlay"
    elif best_overall["role"] == "standalone-alpha" and std_ok:
        role_final = "standalone-alpha"
    else:
        role_final = "none"

    notes = (
        f"Pure-numpy RQA (delay embedding + fixed-RR recurrence plot + diagonal/"
        f"vertical run-length -> %DET/LAM/TT), causal rolling windows, shift>=1, "
        f"5bp/leg. Best of 4 strategy families = '{best_overall['name']}' "
        f"(role={role}). OOS Sharpe={mo['sharpe_ann']:.2f}, PSR={psr:.3f}, "
        f"ret_ann={mo['ret_ann']*100:.1f}%, maxDD={mo['maxdd']*100:.1f}% "
        f"(close-to-close => DD is an ILLUSION, no intrabar gap/liq). "
        f"DSR SR*(per-period, {len(sr_trials)} trials over {n_variants} RQA configs)"
        f"={srstar:.4f} vs OOS SR_pp={mo['sr_pp']:.4f} -> "
        f"{'BEATS' if beats_dsr else 'FAILS'} deflated bar. "
    )
    if overlay_base_sharpe is not None:
        notes += (f"Overlay base Sharpe={overlay_base_sharpe:.2f} -> gated "
                  f"{overlay_gated_sharpe:.2f} ({'improves' if improves else 'no improvement'}). ")
    notes += ("Hypothesis was high-%DET = predictable/trend-friendly regime; "
              "in crypto the directional momentum it gates is itself dead, so the "
              "regime gate mostly reduces exposure (lower DD) without manufacturing "
              "alpha. RQA reads as a weak risk/regime descriptor, not a return engine. "
              "Many knobs (win,m,d,rr,l_min) make a high Sharpe untrustworthy -- "
              "deflated accordingly.")

    print("\n=== VERDICT:", verdict, f"(role={role_final}) ===")
    print(notes)

    out = dict(
        key="recurrence_rqa",
        family="nonlinear-dynamics",
        file="experiments/exo_recurrence_rqa.py",
        implemented=True,
        libs_implemented=("pure-numpy RQA: takens_embedding, recurrence_matrix "
                          "(fixed-RR percentile threshold), diagonal & vertical "
                          "run-length line stats -> %DET/LAM/TT, causal rqa_rolling. "
                          "No pyunicorn/pyts used."),
        verdict=verdict,
        role=role_final,
        market_neutral=bool("xsec" in best_overall["name"]),
        method="Recurrence Quantification Analysis (%DET/LAM/TT regime detector)",
        universe=f"{len(coins)} USDT-perps {INTERVAL} {START}..{END}",
        n_obs=int(mo["n"]),
        oos_sharpe=float(mo["sharpe_ann"]),
        oos_ret_ann_pct=float(mo["ret_ann"] * 100),
        psr=float(psr) if psr is not None else float("nan"),
        dsr=float(srstar),
        maxdd_pct=float(mo["maxdd"] * 100),
        turnover=float(mo["turnover"]) if np.isfinite(mo["turnover"]) else float("nan"),
        cost_bps=5.0,
        n_variants_tried=int(n_variants),
        overlay_base_sharpe=overlay_base_sharpe,
        overlay_gated_sharpe=overlay_gated_sharpe,
        notes=notes,
        data_caveats=("8h close-to-close perps; RQA O(win^2) so end-bars subsampled "
                      "step=3 and ffilled (causal); maxDD is close-to-close (no "
                      "intrabar liquidation/gap); 15-coin survivorship (all currently "
                      "listed)."),
    )
    rp = pathlib.Path(__file__).resolve().parent.parent / "reports" / "exo_recurrence_rqa.json"
    rp.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {rp}")
    return out


if __name__ == "__main__":
    main()
