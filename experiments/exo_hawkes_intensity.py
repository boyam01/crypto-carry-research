"""Exotic-math candidate: hawkes_intensity  (family = self-exciting point process)

METHOD: Hawkes self-exciting point process on a RECONSTRUCTED event series.

----------------------------------------------------------------------------
WHAT IS IMPLEMENTED IN NUMPY (libs_implemented):
  - Exponential-kernel Hawkes conditional intensity recursion
        lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta (t - t_i))
    via the standard O(N) recursion  A_k = exp(-beta dt) (A_{k-1} + 1).
  - Discrete-time / "binned" Hawkes log-likelihood for an exponential kernel
    and its MLE fit by scipy.optimize.minimize (L-BFGS-B over log-params),
    with a method-of-moments initial guess and a pure-MoM fallback.
  - Branching ratio  n = alpha / beta  (criticality / cascade strength).
  - Causal, point-in-time ROLLING Hawkes refit: at each bar the model is
    estimated ONLY on a trailing window of past events, so the conditional
    intensity / branching-ratio feature used at bar t uses data < t only.
  All of scipy.optimize / scipy.stats are allowed libs; the Hawkes math itself
  (recursion, NLL, MoM, branching ratio) is hand-written numpy here.
----------------------------------------------------------------------------

EVENT SERIES (reconstructed from cached full-history 8h futures klines):
  The `trades` column = number of trades per 8h bar (full 2022->now history,
  the long-sample input the spec asks for; OI history is only ~21d so it is
  NOT the primary input). We define EVENTS two complementary ways and test
  both as the point process whose self-excitation we measure:
    (A) "activity events": a bar is an event when its trade-count exceeds a
        CAUSAL trailing-median * kappa (a burst of trading intensity).
    (B) "large-move events": a bar is an event when |return| exceeds a causal
        trailing volatility * kappa (a jump). Jumps cluster (vol clustering)
        which is exactly Hawkes self-excitation.
  For the LIKELIHOOD fit we treat each event's position (bar index) as the
  event time t_i on the integer 8h grid and fit mu, alpha, beta by MLE on a
  trailing window. The conditional intensity ratio
        R_t = lambda(t) / mu      (how far above baseline we are)
  and the branching ratio n_t are the regime features.

HYPOTHESIS (from spec): when conditional intensity / branching ratio spikes
  (self-exciting cascade in progress) short-horizon behavior is predictable:
  CONTINUATION during the cascade, REVERSION after it. We test sign-of-recent-
  move * cascade-gate as a directional signal, AND the post-cascade fade.

DUAL FRAMING (spec-mandated):
  (1) STANDALONE alpha: trade the continuation/fade signal directly.
  (2) OVERLAY / risk detector: use the cascade flag to GATE a simple base book
      (long-BTC, and an equal-weight long basket). Cascades = elevated tail
      risk; de-risk (cut exposure) while branching ratio is critical. Report
      base vs gated OOS Sharpe and maxDD.

GOVERNANCE:
  - All features causal: trailing thresholds, rolling Hawkes refit on PAST
    events only, signal shifted +1 bar (decide on close t, hold from t+1).
  - cost >= 5bp/leg on |Delta position| (8h bars; directional single leg).
  - chronological OOS: every hyperparameter (event def, kappa, window, refit
    cadence, intensity threshold, horizon, continuation-vs-fade) tuned on the
    first 60% of the timeline; metrics reported on the LAST 40% only.
  - DEFLATE: count EVERY variant tried; report within-family DSR benchmark and
    a DSR-adjusted PSR.  Hawkes has many knobs -> be skeptical.
  - PRIOR: directional momentum/reversal died here before. A Hawkes-gated
    directional bet is momentum/reversal in disguise -> expect standalone to be
    weak; the honest hope is the OVERLAY risk-detector framing.
"""
from __future__ import annotations
import sys, json, pathlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import fetch_binance as fb
from engine import backtest as bt

CACHE = ROOT / "data" / "cache"
PPY = 365 * 3                      # 8h bars per year (3 per day)
TRAIN_FRAC = 0.60
COST_BPS = 5.0                     # taker, one directional leg
COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
START = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
END = int(datetime.now(timezone.utc).timestamp() * 1000)


# ----------------------------------------------------------------------------
# Hawkes exponential-kernel math (hand-implemented numpy)
# ----------------------------------------------------------------------------
def hawkes_recursion(event_times: np.ndarray, beta: float) -> np.ndarray:
    """A_k = sum_{i<k} exp(-beta (t_k - t_i)) via O(N) recursion.
    Returns the per-event excitation-sum (excluding self)."""
    et = np.asarray(event_times, float)
    A = np.zeros(len(et))
    for k in range(1, len(et)):
        dt = et[k] - et[k - 1]
        A[k] = np.exp(-beta * dt) * (A[k - 1] + 1.0)
    return A


def hawkes_nll(params_log: np.ndarray, event_times: np.ndarray, T: float) -> float:
    """Negative log-likelihood of an exponential-kernel Hawkes process observed
    on [0, T] with event times et. params_log = log(mu), log(alpha), log(beta).
    Standard Ozaki (1979) recursive likelihood:
        logL = sum_k log(mu + alpha*A_k)  - mu*T
               - (alpha/beta) sum_k (1 - exp(-beta (T - t_k)))
    """
    mu, alpha, beta = np.exp(params_log)
    et = np.asarray(event_times, float)
    if len(et) < 2 or mu <= 0 or alpha <= 0 or beta <= 0:
        return 1e12
    A = hawkes_recursion(et, beta)
    lam = mu + alpha * A
    if np.any(lam <= 0):
        return 1e12
    term_sum = np.sum(np.log(lam))
    comp = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - et)))
    nll = -(term_sum - comp)
    if not np.isfinite(nll):
        return 1e12
    return nll


def hawkes_mom(event_times: np.ndarray, T: float):
    """Method-of-moments init: mu from unconditional rate discounted by
    branching, beta from the autocorr decay timescale of inter-event spacing,
    alpha = n*beta. Crude but a robust starting point / fallback."""
    et = np.asarray(event_times, float)
    N = len(et)
    if N < 5 or T <= 0:
        return None
    rate = N / T                                    # unconditional intensity
    iet = np.diff(et)                               # inter-event times
    mean_iet = iet.mean() if len(iet) else 1.0
    # branching-ratio proxy via Fano factor of counts binned at ~mean spacing
    # (var/mean of counts; >1 => clustered). Map to n in [0, 0.9].
    binw = max(1.0, mean_iet * 4)
    nb = int(np.ceil(T / binw))
    counts, _ = np.histogram(et, bins=nb, range=(0, T))
    fano = counts.var() / counts.mean() if counts.mean() > 0 else 1.0
    n = float(np.clip(1.0 - 1.0 / max(fano, 1.0), 0.0, 0.9))   # Fano=1 -> n=0
    beta = 1.0 / max(mean_iet, 1e-6)
    alpha = n * beta
    mu = rate * (1.0 - n)                            # E[lambda] = mu/(1-n)
    mu = max(mu, 1e-6); alpha = max(alpha, 1e-6); beta = max(beta, 1e-6)
    return mu, alpha, beta, n


def hawkes_fit(event_times: np.ndarray, T: float):
    """MLE fit (L-BFGS-B over log-params) with MoM init + MoM fallback.
    Returns dict(mu, alpha, beta, n, lam_last) or None."""
    et = np.asarray(event_times, float)
    if len(et) < 6:
        return None
    mom = hawkes_mom(et, T)
    if mom is None:
        return None
    mu0, alpha0, beta0, n0 = mom
    x0 = np.log([mu0, alpha0, beta0])
    try:
        res = minimize(hawkes_nll, x0, args=(et, T), method="L-BFGS-B",
                       options=dict(maxiter=80, ftol=1e-7))
        mu, alpha, beta = np.exp(res.x)
        ok = res.success and np.isfinite(res.fun)
    except Exception:
        ok = False
    if not ok or alpha / beta >= 1.0 or alpha / beta <= 0:
        # fall back to MoM (stationary, sub-critical by construction)
        mu, alpha, beta, _ = mom
    n = alpha / beta
    # conditional intensity at the right edge T given all events
    A = hawkes_recursion(et, beta)
    # excitation contributed at time T from each past event:
    excite_T = alpha * np.sum(np.exp(-beta * (T - et)))
    lam_last = mu + excite_T
    return dict(mu=float(mu), alpha=float(alpha), beta=float(beta),
                n=float(min(n, 0.999)), lam_last=float(lam_last),
                ratio=float(lam_last / mu) if mu > 0 else np.nan)


# ----------------------------------------------------------------------------
# Data + event reconstruction (causal)
# ----------------------------------------------------------------------------
def load_coin(sym: str) -> pd.DataFrame | None:
    k = fb.klines(sym, "8h", START, END, futures=True)
    if k is None or len(k) < 500:
        return None
    df = pd.DataFrame(index=k.index)
    df["close"] = k["close"]
    df["ret"] = k["close"].pct_change()
    df["trades"] = k["trades"].astype(float)
    return df.dropna(subset=["ret", "trades"])


def build_events(df: pd.DataFrame, mode: str, kappa: float, win: int) -> np.ndarray:
    """Boolean event flag per bar, CAUSAL (uses only past bars for thresholds).
    mode='activity': trades > trailing_median(win) * kappa
    mode='move'    : |ret|  > trailing_std(win)   * kappa
    """
    n = len(df)
    flag = np.zeros(n, dtype=bool)
    if mode == "activity":
        x = df["trades"].values
        s = pd.Series(x)
        thr = s.shift(1).rolling(win, min_periods=win // 2).median().values * kappa
        flag = (x > thr) & np.isfinite(thr)
    elif mode == "move":
        r = df["ret"].values
        s = pd.Series(np.abs(r))
        thr = s.shift(1).rolling(win, min_periods=win // 2).std().values * kappa
        # std of |ret| ~ scale; threshold on |ret|
        thr2 = pd.Series(r).shift(1).rolling(win, min_periods=win // 2).std().values * kappa
        flag = (np.abs(r) > thr2) & np.isfinite(thr2)
    return flag


def rolling_hawkes_feature(event_flag: np.ndarray, fit_win: int, stride: int):
    """At each bar t, fit a Hawkes process on events strictly BEFORE t inside a
    trailing window of `fit_win` bars; refit every `stride` bars (carry forward
    between refits). Returns arrays (ratio_t, n_t) aligned to bars, point-in-time.
    """
    n = len(event_flag)
    ratio = np.full(n, np.nan)
    nbr = np.full(n, np.nan)
    last_fit = None
    for t in range(n):
        # refit on a cadence, using only events in [t-fit_win, t)  (strictly past)
        if t >= 50 and (last_fit is None or (t % stride == 0)):
            lo = max(0, t - fit_win)
            idx = np.where(event_flag[lo:t])[0].astype(float)   # event positions rel to lo
            T = float(t - lo)                                   # window length in bars
            if len(idx) >= 6:
                fit = hawkes_fit(idx, T)
                if fit is not None:
                    # recompute conditional intensity at the CURRENT edge using
                    # the fitted params and the SAME past events (point-in-time)
                    last_fit = fit
        if last_fit is not None:
            # update conditional intensity ratio at bar t using past events only
            lo = max(0, t - fit_win)
            idx = np.where(event_flag[lo:t])[0].astype(float)
            if len(idx) >= 1:
                beta = last_fit["beta"]; alpha = last_fit["alpha"]; mu = last_fit["mu"]
                Tcur = float(t - lo)
                excite = alpha * np.sum(np.exp(-beta * (Tcur - idx)))
                lam = mu + excite
                ratio[t] = lam / mu if mu > 0 else np.nan
            nbr[t] = last_fit["n"]
    return ratio, nbr


# ----------------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------------
def standalone_position(df, ratio, nbr, ratio_thr, mode_dir, horizon):
    """Directional standalone signal.
    cascade_on = ratio >= ratio_thr (intensity well above baseline).
    mode_dir='cont' : during cascade, ride sign of recent return for `horizon`.
    mode_dir='fade' : during cascade, fade sign of recent return for `horizon`.
    Position decided at bar t from info<=t; caller shifts +1.
    """
    r = df["ret"].values
    n = len(df)
    cascade = (ratio >= ratio_thr) & np.isfinite(ratio)
    sgn = np.sign(r)
    raw = np.zeros(n)
    base = sgn if mode_dir == "cont" else -sgn
    hold = 0
    cur = 0.0
    for t in range(n):
        if cascade[t]:
            cur = base[t]
            hold = horizon
        if hold > 0:
            raw[t] = cur
            hold -= 1
        else:
            raw[t] = 0.0
    return raw


def shift_hold(pos):
    return np.concatenate([[0.0], pos[:-1]])


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def eval_net(net, pos):
    nv = np.asarray(net, float)
    n = len(nv)
    tr, te = bt.oos_split(n, TRAIN_FRAC)
    pv = np.asarray(pos, float) if pos is not None else None
    mi = bt.metrics(nv[tr], PPY, pv[tr] if pv is not None else None)
    mo = bt.metrics(nv[te], PPY, pv[te] if pv is not None else None)
    p = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
    return mi, mo, p


def main():
    print("Loading coins + reconstructing events (8h, full history)...")
    frames = {}
    for sym in COINS:
        f = load_coin(sym)
        if f is not None:
            frames[sym] = f
            print(f"  {sym:9} {len(f)} 8h bars  {f.index[0]} -> {f.index[-1]}")
    if not frames:
        _write(dict(key="hawkes_intensity", implemented=True, verdict="ERROR",
                    notes="no klines frames built"))
        return

    # ----- precompute rolling Hawkes features per (coin, event-mode, knobs) -----
    # knob grids (count EVERY variant for deflation)
    event_modes = ["activity", "move"]
    kappa_grid = [1.5, 2.0, 3.0]
    fitwin_grid = [120, 240]            # ~40d / ~80d trailing windows
    stride = 3                          # refit cadence (bars); held fixed
    evwin = 90                          # trailing window for event thresholding
    ratio_thr_grid = [2.0, 3.0, 5.0]
    horizon_grid = [1, 2, 3]
    dir_grid = ["cont", "fade"]

    # cache features keyed by (mode,kappa,fitwin)
    feat_cache = {}
    for mode in event_modes:
        for kappa in kappa_grid:
            for fw in fitwin_grid:
                per_coin = {}
                for sym, df in frames.items():
                    flag = build_events(df, mode, kappa, evwin)
                    ratio, nbr = rolling_hawkes_feature(flag, fw, stride)
                    per_coin[sym] = (ratio, nbr, flag)
                feat_cache[(mode, kappa, fw)] = per_coin
    # report event counts for transparency
    print("\nEvent counts (full sample):")
    for (mode, kappa, fw), pc in feat_cache.items():
        if fw == fitwin_grid[0]:
            tot = sum(int(v[2].sum()) for v in pc.values())
            print(f"  mode={mode:8} kappa={kappa}  total events={tot}")

    # ===================================================================
    # PART 1 — STANDALONE alpha (pooled equal-weight across coins)
    # ===================================================================
    print("\n=== PART 1: STANDALONE directional alpha (pooled) ===")
    trials_sr = []     # per-period IS Sharpes for DSR deflation
    best = None
    n_variants = 0
    for mode in event_modes:
        for kappa in kappa_grid:
            for fw in fitwin_grid:
                pc = feat_cache[(mode, kappa, fw)]
                for rthr in ratio_thr_grid:
                    for hz in horizon_grid:
                        for dr in dir_grid:
                            n_variants += 1
                            nets, poss = [], []
                            for sym, df in frames.items():
                                ratio, nbr, flag = pc[sym]
                                pos = standalone_position(df, ratio, nbr, rthr, dr, hz)
                                ph = shift_hold(pos)
                                net = bt.run(df["ret"].values, ph, COST_BPS)
                                nets.append(pd.Series(net, index=df.index))
                                poss.append(pd.Series(np.abs(ph), index=df.index))
                            netdf = pd.concat(nets, axis=1)
                            posdf = pd.concat(poss, axis=1)
                            port_net = netdf.mean(axis=1).values
                            port_pos = posdf.mean(axis=1).values
                            mi, mo, p = eval_net(port_net, port_pos)
                            trials_sr.append(mi["sr_pp"])
                            active = np.mean(port_pos > 0)
                            if (np.isfinite(mi["sharpe_ann"]) and active > 0.02
                                    and (best is None or mi["sharpe_ann"] > best["is"])):
                                best = dict(mode=mode, kappa=kappa, fw=fw, rthr=rthr,
                                            hz=hz, dr=dr, **{"is": mi["sharpe_ann"]})
    print(f"standalone variants tried: {n_variants}")
    if best is None:
        print("no standalone variant active enough; standalone DEAD")
        standalone = dict(oos_sharpe=float("nan"), psr=float("nan"), verdict="DEAD")
    else:
        pc = feat_cache[(best["mode"], best["kappa"], best["fw"])]
        nets, poss = [], []
        for sym, df in frames.items():
            ratio, nbr, flag = pc[sym]
            pos = standalone_position(df, ratio, nbr, best["rthr"], best["dr"], best["hz"])
            ph = shift_hold(pos)
            net = bt.run(df["ret"].values, ph, COST_BPS)
            nets.append(pd.Series(net, index=df.index))
            poss.append(pd.Series(np.abs(ph), index=df.index))
        netdf = pd.concat(nets, axis=1); posdf = pd.concat(poss, axis=1)
        port_net = netdf.mean(axis=1).values; port_pos = posdf.mean(axis=1).values
        mi, mo, p = eval_net(port_net, port_pos)
        sr_star = bt.dsr_benchmark(trials_sr)
        dsr = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"], sr_benchmark=sr_star)
        print(f"IS-selected: {best}")
        print(f"  OOS Sharpe={mo['sharpe_ann']:.2f} ret_ann={mo['ret_ann']*100:.2f}% "
              f"maxDD={mo['maxdd']*100:.1f}% turn={mo['turnover']:.3f}")
        print(f"  PSR={p:.3f}  SR*(pp)={sr_star:.4f}  DSR-PSR={dsr:.3f}")
        standalone = dict(oos_sharpe=float(mo["sharpe_ann"]),
                          oos_ret_ann_pct=float(mo["ret_ann"] * 100),
                          psr=float(p), dsr=float(dsr), sr_star=float(sr_star),
                          maxdd_pct=float(mo["maxdd"] * 100),
                          turnover=float(mo["turnover"]), params=best)

    # ===================================================================
    # PART 2 — OVERLAY / risk detector on a base book
    # ===================================================================
    # Base books:
    #   (a) long-BTC (always +1)
    #   (b) equal-weight long basket of all coins (always +1 each)
    # Overlay: when branching ratio n_t (or intensity ratio) is CRITICAL, cut
    # exposure to `risk_off_w` (de-risk during self-exciting cascades).
    print("\n=== PART 2: OVERLAY risk-detector (gate a long base book) ===")
    btc = frames["BTCUSDT"]
    # equal-weight basket returns (rebalanced each bar, always long)
    ew_ret = pd.concat([frames[s]["ret"] for s in frames], axis=1).mean(axis=1)

    # use branching ratio n as the regime gauge; threshold tuned IS on percentile
    nthr_pct_grid = [0.80, 0.90, 0.95]
    riskoff_grid = [0.0, 0.5]
    gauge_grid = ["n", "ratio"]
    ov_event_modes = ["activity", "move"]
    ov_kappa_grid = [2.0, 3.0]
    ov_fw_grid = [120, 240]

    ov_trials = []
    ov_best = None
    ov_variants = 0

    def overlay_series(base_ret_series, gauge_vec, thr, risk_off_w):
        """Gate a long base book: weight=1 normally, risk_off_w when gauge>=thr.
        Weight decided at t from info<=t; shift +1 for holding."""
        g = np.asarray(gauge_vec, float)
        on = (g >= thr) & np.isfinite(g)
        w = np.where(on, risk_off_w, 1.0)
        w_held = shift_hold(w)                     # weight applied from t+1
        r = base_ret_series.values
        net = bt.run(r, w_held, COST_BPS)
        return net, w_held

    for base_name, base_ret in [("BTC", btc["ret"]), ("EW", ew_ret)]:
        for gauge in gauge_grid:
            for mode in ov_event_modes:
                for kappa in ov_kappa_grid:
                    for fw in ov_fw_grid:
                        # build the gauge on BTC's clock for BTC base,
                        # and pooled-mean gauge for EW base
                        pc = feat_cache.get((mode, kappa, fw))
                        if pc is None:
                            # fw not in part1 cache for some combos -> compute
                            pc = {}
                            for sym, df in frames.items():
                                flag = build_events(df, mode, kappa, evwin)
                                ratio, nbr = rolling_hawkes_feature(flag, fw, stride)
                                pc[sym] = (ratio, nbr, flag)
                            feat_cache[(mode, kappa, fw)] = pc
                        if base_name == "BTC":
                            ratio, nbr, _ = pc["BTCUSDT"]
                            gvec = nbr if gauge == "n" else ratio
                            gser = pd.Series(gvec, index=btc.index).reindex(base_ret.index).values
                        else:
                            # pooled mean gauge across coins on common index
                            gmat = []
                            for sym, df in frames.items():
                                ratio, nbr, _ = pc[sym]
                                gv = nbr if gauge == "n" else ratio
                                gmat.append(pd.Series(gv, index=df.index))
                            gser = pd.concat(gmat, axis=1).mean(axis=1).reindex(base_ret.index).values
                        # threshold = IS percentile of the gauge (tuned on train only)
                        n_all = len(base_ret)
                        tr, te = bt.oos_split(n_all, TRAIN_FRAC)
                        g_tr = gser[tr]
                        g_tr = g_tr[np.isfinite(g_tr)]
                        if len(g_tr) < 50:
                            continue
                        for pct in nthr_pct_grid:
                            thr = np.quantile(g_tr, pct)
                            for roff in riskoff_grid:
                                ov_variants += 1
                                net, w_held = overlay_series(base_ret, gser, thr, roff)
                                mi, mo, p = eval_net(net, np.abs(w_held))
                                ov_trials.append(mi["sr_pp"])
                                # selection: improve IS Sharpe over un-gated base
                                base_net = bt.run(base_ret.values, np.ones(n_all), COST_BPS)
                                bmi, bmo, _ = eval_net(base_net, np.ones(n_all))
                                improve = mi["sharpe_ann"] - bmi["sharpe_ann"]
                                if (np.isfinite(mi["sharpe_ann"]) and
                                        (ov_best is None or improve > ov_best["improve_is"])):
                                    ov_best = dict(base=base_name, gauge=gauge, mode=mode,
                                                   kappa=kappa, fw=fw, pct=pct, roff=roff,
                                                   thr=float(thr), improve_is=float(improve))
    print(f"overlay variants tried: {ov_variants}")

    overlay_base_sharpe = float("nan")
    overlay_gated_sharpe = float("nan")
    overlay = dict()
    if ov_best is not None:
        base_ret = btc["ret"] if ov_best["base"] == "BTC" else ew_ret
        n_all = len(base_ret)
        pc = feat_cache[(ov_best["mode"], ov_best["kappa"], ov_best["fw"])]
        if ov_best["base"] == "BTC":
            ratio, nbr, _ = pc["BTCUSDT"]
            gvec = nbr if ov_best["gauge"] == "n" else ratio
            gser = pd.Series(gvec, index=btc.index).reindex(base_ret.index).values
        else:
            gmat = []
            for sym, df in frames.items():
                ratio, nbr, _ = pc[sym]
                gv = nbr if ov_best["gauge"] == "n" else ratio
                gmat.append(pd.Series(gv, index=df.index))
            gser = pd.concat(gmat, axis=1).mean(axis=1).reindex(base_ret.index).values
        # threshold from TRAIN percentile (already point-in-time)
        tr, te = bt.oos_split(n_all, TRAIN_FRAC)
        g_tr = gser[tr]; g_tr = g_tr[np.isfinite(g_tr)]
        thr = np.quantile(g_tr, ov_best["pct"])
        net, w_held = overlay_series(base_ret, gser, thr, ov_best["roff"])
        gmi, gmo, gp = eval_net(net, np.abs(w_held))
        base_net = bt.run(base_ret.values, np.ones(n_all), COST_BPS)
        bmi, bmo, bp = eval_net(base_net, np.ones(n_all))
        sr_star_ov = bt.dsr_benchmark(ov_trials)
        gdsr = bt.psr(gmo["sr_pp"], gmo["n"], gmo["skew"], gmo["kurt"], sr_benchmark=sr_star_ov)
        overlay_base_sharpe = float(bmo["sharpe_ann"])
        overlay_gated_sharpe = float(gmo["sharpe_ann"])
        print(f"IS-selected overlay: {ov_best}")
        print(f"  base={ov_best['base']}  OOS base Sharpe={bmo['sharpe_ann']:.3f} "
              f"maxDD={bmo['maxdd']*100:.1f}%")
        print(f"  GATED OOS Sharpe={gmo['sharpe_ann']:.3f} maxDD={gmo['maxdd']*100:.1f}% "
              f"turn={gmo['turnover']:.3f}")
        print(f"  delta Sharpe (gated-base)={gmo['sharpe_ann']-bmo['sharpe_ann']:+.3f}  "
              f"PSR(gated)={gp:.3f} SR*={sr_star_ov:.4f} DSR-PSR={gdsr:.3f}")
        # fraction of time risk-off in OOS
        roff_frac_oos = float(np.mean(np.abs(w_held[te]) < 1.0))
        overlay = dict(base=ov_best["base"], gauge=ov_best["gauge"],
                       base_sharpe=overlay_base_sharpe, gated_sharpe=overlay_gated_sharpe,
                       base_maxdd_pct=float(bmo["maxdd"] * 100),
                       gated_maxdd_pct=float(gmo["maxdd"] * 100),
                       delta_sharpe=float(gmo["sharpe_ann"] - bmo["sharpe_ann"]),
                       gated_psr=float(gp), gated_dsr=float(gdsr),
                       sr_star=float(sr_star_ov), roff_frac_oos=roff_frac_oos,
                       params=ov_best, turnover=float(gmo["turnover"]))

    # ===================================================================
    # VERDICT
    # ===================================================================
    total_variants = n_variants + ov_variants
    sa_ok = (np.isfinite(standalone.get("psr", np.nan)) and
             standalone.get("oos_sharpe", -9) > 0 and
             standalone.get("psr", 0) >= 0.95 and
             standalone.get("dsr", 0) >= 0.95)
    # overlay "clearly improving": positive delta Sharpe AND gated PSR strong
    # AND drawdown not worse, deflation-survived
    ov_delta = overlay.get("delta_sharpe", -9) if overlay else -9
    ov_improves = (overlay and ov_delta > 0.10 and
                   overlay.get("gated_dsr", 0) >= 0.95 and
                   overlay.get("gated_maxdd_pct", -100) >= overlay.get("base_maxdd_pct", -1e9))

    role = "none"
    if sa_ok and ov_improves:
        role = "both"; verdict = "EDGE"
    elif sa_ok:
        role = "standalone-alpha"; verdict = "EDGE"
    elif ov_improves:
        role = "risk-overlay"; verdict = "EDGE"
    else:
        # marginal if anything is positive-but-fragile
        sa_marg = (standalone.get("oos_sharpe", -9) > 0 and
                   0.80 <= standalone.get("psr", 0) < 0.95)
        ov_marg = (overlay and ov_delta > 0 and
                   0.80 <= overlay.get("gated_psr", 0) < 0.95)
        if sa_marg or ov_marg:
            verdict = "MARGINAL"
            role = "risk-overlay" if (ov_marg and not sa_marg) else (
                   "standalone-alpha" if sa_marg and not ov_marg else "both")
        else:
            verdict = "DEAD"; role = "none"

    notes = (
        f"Hawkes exp-kernel MLE (Ozaki recursion + MoM init/fallback) on events "
        f"reconstructed from 8h trade-count bursts & |ret| jumps, full 2022->2026 "
        f"history, {len(frames)} coins. Rolling causal refit (window/stride), "
        f"intensity-ratio & branching-ratio n as regime gauges. "
        f"STANDALONE best OOS Sharpe={standalone.get('oos_sharpe', float('nan')):.2f} "
        f"PSR={standalone.get('psr', float('nan')):.2f} "
        f"DSR-PSR={standalone.get('dsr', float('nan')):.2f}. "
        f"OVERLAY best: base Sharpe={overlay_base_sharpe:.2f} -> gated "
        f"Sharpe={overlay_gated_sharpe:.2f} (delta={ov_delta:+.2f}), "
        f"base maxDD={overlay.get('base_maxdd_pct', float('nan')):.1f}% -> gated "
        f"maxDD={overlay.get('gated_maxdd_pct', float('nan')):.1f}%. "
        f"Total variants tried (deflation count)={total_variants}. "
        f"PRIOR: directional momentum/reversal died here; Hawkes-gated directional "
        f"is that in disguise -> standalone expected weak. close-to-close delta on "
        f"single perp leg; maxDD is close-to-close (no intrabar liq/gap) so the "
        f"overlay drawdown improvement is an UPPER bound on real risk relief. "
        f"role={role}, verdict={verdict}."
    )
    print("\n=== VERDICT ===")
    print(f"role={role}  verdict={verdict}")
    print(notes)

    out = dict(
        key="hawkes_intensity",
        file="experiments/exo_hawkes_intensity.py",
        implemented=True,
        method="Hawkes self-exciting point process (exp kernel, MLE Ozaki recursion "
               "+ method-of-moments) on reconstructed trade-count/jump event series; "
               "intensity-ratio & branching-ratio n as cascade/regime gauges.",
        libs_implemented="hawkes_recursion (O(N) exp-kernel excitation), hawkes_nll "
               "(Ozaki recursive log-likelihood), hawkes_mom (Fano-factor method-of-"
               "moments init/fallback), hawkes_fit (L-BFGS-B MLE over log-params), "
               "rolling causal point-in-time refit. (scipy.optimize/stats are allowed "
               "libs; all Hawkes math hand-written in numpy.)",
        verdict=verdict,
        role=role,
        universe="+".join(frames.keys()),
        n_obs=int(min(len(f) for f in frames.values())),
        oos_sharpe=float(standalone.get("oos_sharpe", float("nan"))),
        oos_ret_ann_pct=float(standalone.get("oos_ret_ann_pct", float("nan"))),
        psr=float(standalone.get("psr", float("nan"))),
        dsr=float(standalone.get("dsr", float("nan"))),
        maxdd_pct=float(standalone.get("maxdd_pct", float("nan"))),
        turnover=float(standalone.get("turnover", float("nan"))),
        cost_bps=COST_BPS,
        market_neutral=False,
        n_variants_tried=int(total_variants),
        overlay_base_sharpe=overlay_base_sharpe,
        overlay_gated_sharpe=overlay_gated_sharpe,
        standalone=standalone,
        overlay=overlay,
        method_short="hawkes_intensity",
        data_caveats="events reconstructed from 8h 'trades' count bursts (causal "
               "trailing-median*kappa) and |ret| jumps (causal trailing-std*kappa); "
               "OI history only ~21d so NOT used as primary (per spec). Single perp "
               "leg, directional; close-to-close maxDD (no intrabar liq/gap) -> "
               "drawdown relief is an upper bound. All thresholds causal, signal "
               "shifted +1 bar, hyperparams tuned on first 60% only.",
        notes=notes,
    )
    _write(out)
    return out


def _write(d):
    p = ROOT / "reports" / "exo_hawkes_intensity.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2, default=float))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
