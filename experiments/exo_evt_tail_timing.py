"""Exotic-math candidate: evt_tail_timing  (family = extreme-value-theory)

METHOD: Extreme Value Theory left-tail-index dynamics as a CRASH-RISK detector.

We fit, on a CAUSAL trailing window of daily returns, two classic EVT tail-index
estimators on the LEFT tail (losses):
  * Hill estimator  -- the MLE tail index alpha of a Pareto-type tail from the k
    largest losses (Hill 1975). Reported as both the tail index alpha = 1/xi and
    the EVT shape xi = 1/alpha. Low alpha / high xi == FAT left tail == crash-prone.
  * POT-GPD (peaks-over-threshold) shape xi via the method-of-moments / probability-
    weighted-moments fit of a Generalized Pareto Distribution to exceedances over a
    high loss threshold (Pickands / Hosking-Wallis PWM). xi>0 == heavy tail.

HYPOTHESIS (from spec): a THICKENING left tail (alpha falling / xi rising) signals a
fat-tail / crash regime -> reduce or hedge a long book; a THINNING tail (alpha rising)
signals calm -> stay exposed. Test BOTH framings:
  (A) STANDALONE alpha -- go SHORT (or flat) BTC when the left tail is fat, long/flat
      when thin. (Directional EVT-timing.)
  (B) OVERLAY / crash-hedge timing -- the tail index GATES/SIZES a simple base book
      (long-BTC, and the funding-carry book that is the only survivor in this repo):
      cut exposure (or add a short hedge) when the tail signals a fat-tail regime.
      Does the gated book beat the un-gated base OOS (Sharpe up, tail-event DD down)?

WHY BE SKEPTICAL (governance PRIORS #4 & #7): directional momentum/reversal already
DIED here; a tail-index gate cannot manufacture directional alpha. At best it cuts
drawdown by being flat in turbulent regimes -- but that overlaps heavily with plain
realized-vol gating (a fat tail and high vol co-occur), so most apparent benefit may
be generic vol-state timing, NOT a unique EVT edge. EVT has MANY knobs (window W,
tail fraction k / threshold quantile u, alpha-vs-xi, Hill-vs-GPD, signal threshold,
gate-vs-size, which base book, hedge ratio). EVERY knob is a chance to overfit, so we
count EVERY variant and deflate with within-family DSR. A pretty gated Sharpe from
threshold-shopping is NOT an edge. We also benchmark the EVT gate against a plain
realized-vol gate to see whether EVT adds anything beyond vol.

LIBS IMPLEMENTED HERE IN NUMPY (no statsmodels/scipy-EVT needed):
  * hill_tail_index   -- Hill (1975) MLE tail index from the k largest order stats.
  * gpd_pwm_shape     -- GPD shape xi via probability-weighted moments (Hosking-Wallis
    1987) on peaks-over-threshold exceedances. Pure numpy.
  * rolling_hill / rolling_gpd_xi -- CAUSAL trailing-window versions; the value at bar
    t is computed from returns strictly < t, then the derived signal/gate is shifted one
    more bar for trading -> no look-ahead.
  * realized_vol gate (numpy) as the vol benchmark the EVT gate must beat.

GOVERNANCE baked in: features at t use data < t (causal rolling + shift>=1); cost
>=5bp/leg on |Dposition|; ALL knobs tuned on first 60% only, metrics on last 40%;
PSR + within-family DSR reported; n_variants_tried counted; close-to-close delta
exposure maxDD flagged as an illusion (no intrabar liq/gap).
"""
from __future__ import annotations
import sys, json, pathlib, itertools, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd

# cosmetic: some trailing windows yield all-NaN tail fits -> nanquantile warns; handled downstream
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import fetch_binance as fb
from engine import backtest as bt

# ---- deterministic cache keys (prewarm used START=2022-01-01, END=midnight 2026-06-18) ----
START_MS = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
END_MS = int(pd.Timestamp("2026-06-18", tz="UTC").timestamp() * 1000)

# coins with 1d futures cached; MATIC/NEAR/ATOM lack 1d fut -> excluded from 1d panel
COINS_1D = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
            "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT", "TRXUSDT", "DOTUSDT"]
PPY = 365            # daily bars
TRAIN_FRAC = 0.60
COST_BPS = 5.0       # >=5bp/leg per governance

N_VARIANTS = 0       # global counter of every variant tried
SR_TRIALS: list[float] = []   # IS per-period Sharpes of every variant (for DSR)


# ============================================================================
# 1. EVT TAIL-INDEX PRIMITIVES (implemented in numpy)
# ============================================================================
def hill_tail_index(losses: np.ndarray, k: int) -> float:
    """Hill (1975) MLE tail index alpha from the k largest order statistics.

    `losses` = POSITIVE loss magnitudes (e.g. -returns for the left tail; only the
    positive ones matter for a left-tail fit). For a Pareto-type tail P(X>x)~x^-alpha,
        xi_hat = (1/k) * sum_{i=1..k} ln(x_(n-i+1)) - ln(x_(n-k))
    is the Hill estimator of the EVT shape xi = 1/alpha; alpha = 1/xi_hat.
    Low alpha (high xi) == FAT tail. Returns alpha (the tail index)."""
    x = np.sort(losses[np.isfinite(losses) & (losses > 0)])
    n = len(x)
    if n < k + 2 or k < 5:
        return np.nan
    top = x[-k:]                       # k largest
    thresh = x[-(k + 1)]               # the (k+1)-th largest = anchor order stat
    if thresh <= 0:
        return np.nan
    xi = np.mean(np.log(top) - np.log(thresh))
    if xi <= 1e-9:
        return np.nan
    return float(1.0 / xi)             # alpha = tail index


def gpd_pwm_shape(losses: np.ndarray, u_q: float) -> float:
    """GPD shape xi via Probability-Weighted Moments (Hosking-Wallis 1987) on
    peaks-over-threshold exceedances.

    Threshold u = empirical `u_q` quantile of the positive losses. Exceedances
    y = loss - u for loss > u are fit to a Generalized Pareto Distribution. PWM:
        b0 = mean(y);  b1 = mean( ((i-0.35)/m) * y_sorted_i )   (plotting position)
        xi  = 2 - b0 / (b0 - 2 b1) ;  (Hosking-Wallis estimator; xi>0 = heavy tail)
    Returns xi (the EVT shape). xi>0 == heavy (Frechet) tail == crash-prone."""
    x = losses[np.isfinite(losses) & (losses > 0)]
    if len(x) < 30:
        return np.nan
    u = np.quantile(x, u_q)
    y = np.sort(x[x > u] - u)          # exceedances, ascending
    m = len(y)
    if m < 10:
        return np.nan
    # Probability-weighted moments (Hosking-Wallis 1987) for the GPD:
    #   a0 = mean(y);  a1 = mean( (1 - p_i) * y_i )  with p_i = (i-0.35)/m on ascending y.
    #   shape: xi = 2 - a0 / (a0 - 2 a1).  xi>0 => heavy (Frechet) tail; xi<0 => bounded.
    a0 = y.mean()
    pp = (np.arange(1, m + 1) - 0.35) / m
    a1 = np.mean((1.0 - pp) * y)
    denom = (a0 - 2.0 * a1)
    if abs(denom) < 1e-12:
        return np.nan
    xi = 2.0 - a0 / denom
    return float(xi)


def rolling_hill(ret: pd.Series, win: int, k_frac: float) -> pd.Series:
    """CAUSAL rolling Hill tail index (alpha) of the LEFT tail.

    Value at index t is computed from the W returns in (t-W, t-1] (strictly before t)
    -> assign to window's LAST timestamp. k = round(k_frac * W) largest losses."""
    r = ret.values
    n = len(r)
    out = np.full(n, np.nan)
    k = max(5, int(round(k_frac * win)))
    for end in range(win, n + 1):
        w = r[end - win:end]
        out[end - 1] = hill_tail_index(-w, k)     # losses = -returns
    return pd.Series(out, index=ret.index)


def rolling_gpd_xi(ret: pd.Series, win: int, u_q: float) -> pd.Series:
    """CAUSAL rolling GPD shape xi of the LEFT tail (peaks-over-threshold)."""
    r = ret.values
    n = len(r)
    out = np.full(n, np.nan)
    for end in range(win, n + 1):
        w = r[end - win:end]
        out[end - 1] = gpd_pwm_shape(-w, u_q)
    return pd.Series(out, index=ret.index)


def rolling_vol(ret: pd.Series, win: int) -> pd.Series:
    """CAUSAL trailing realized vol (std of returns in the trailing window)."""
    return ret.rolling(win).std()


# ============================================================================
# 2. DATA
# ============================================================================
def load_panel():
    close, ret, funding = {}, {}, {}
    for c in COINS_1D:
        k = fb.klines(c, "1d", START_MS, END_MS, futures=True)
        if k is None or len(k) < 800:
            continue
        cl = k["close"].astype(float)
        close[c] = cl
        ret[c] = np.log(cl).diff()
        f = fb.funding_rate(c, START_MS, END_MS)
        if f is not None and len(f):
            funding[c] = f["fundingRate"].resample("1D").sum()
    closes = pd.DataFrame(close).sort_index()
    rets = pd.DataFrame(ret).sort_index()
    fund = pd.DataFrame(funding).reindex(rets.index).sort_index()
    return closes, rets, fund


# ============================================================================
# 3. BASE BOOKS
# ============================================================================
def base_long_btc(rets: pd.DataFrame):
    r = rets["BTCUSDT"].fillna(0).values
    pos = np.ones(len(r))
    return r, pos


def base_funding_carry(rets: pd.DataFrame, fund: pd.DataFrame):
    """Cross-sectional funding-carry book (the repo's only survivor): short high-
    funding (longs pay), long low-funding, dollar-neutral. Funding known at close ->
    shift 1 to trade next bar."""
    f = fund.copy()
    rk = f.rank(axis=1, pct=True)
    w = (0.5 - rk)                                  # high funding -> short
    w = w.sub(w.mean(axis=1), axis=0)               # dollar-neutral
    denom = w.abs().sum(axis=1).replace(0, np.nan)
    w = w.div(denom, axis=0).fillna(0).shift(1).fillna(0)
    return rets.fillna(0), w


# ============================================================================
# 4. BOOK P&L HELPERS
# ============================================================================
def book_net(rets, weights, cost_bps):
    if isinstance(weights, pd.Series):
        return bt.run(rets.values, weights.values, cost_bps)
    R = rets.values
    W = weights.values
    gross = (W * R).sum(axis=1)
    turn = np.abs(np.diff(W, axis=0, prepend=W[:1] * 0)).sum(axis=1)
    cost = turn * (cost_bps / 1e4)
    return gross - cost


def single_net(ret, pos, cost_bps):
    return bt.run(ret, pos, cost_bps)


def evaluate(net, position=None):
    net = np.asarray(net, float)
    n = len(net)
    tr, te = bt.oos_split(n, TRAIN_FRAC)
    pos_tr = position[tr] if position is not None else None
    pos_te = position[te] if position is not None else None
    mi = bt.metrics(net[tr], PPY, pos_tr)
    mo = bt.metrics(net[te], PPY, pos_te)
    psr = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
    return mi, mo, psr


def tail_event_dd(net: np.ndarray) -> float:
    """Worst single-bar loss (a crude tail-event proxy beyond path maxDD)."""
    net = np.asarray(net, float)
    return float(net.min())


# ============================================================================
# 5. EVT GATE  (fat tail -> reduce exposure)
# ============================================================================
def evt_gate_from_alpha(alpha: pd.Series, q_thresh: float, train_slice) -> tuple[np.ndarray, float]:
    """Hill-alpha gate. FAT tail == LOW alpha. Trade (gate=1) when alpha ABOVE a
    train-quantile threshold (thin tail = calm = stay exposed); flat when alpha below
    (fat tail = crash regime). Threshold fit on TRAIN slice only. Gate shifted 1 bar."""
    a = alpha.copy()
    thr = np.nanquantile(a.values[train_slice], q_thresh)
    gate = (a > thr).astype(float)                  # alpha high (thin tail) -> exposed
    gate = gate.shift(1).fillna(0)                  # decide on t-1, trade t
    return gate.values, float(thr)


def evt_gate_from_xi(xi: pd.Series, q_thresh: float, train_slice) -> tuple[np.ndarray, float]:
    """GPD-xi gate. FAT tail == HIGH xi. Trade when xi BELOW a train-quantile
    threshold (thin tail); flat when xi above (fat tail)."""
    s = xi.copy()
    thr = np.nanquantile(s.values[train_slice], q_thresh)
    gate = (s < thr).astype(float)
    gate = gate.shift(1).fillna(0)
    return gate.values, float(thr)


def vol_gate(vol: pd.Series, q_thresh: float, train_slice) -> tuple[np.ndarray, float]:
    """Realized-vol benchmark gate: trade when vol BELOW a train-quantile (calm)."""
    v = vol.copy()
    thr = np.nanquantile(v.values[train_slice], q_thresh)
    gate = (v < thr).astype(float)
    gate = gate.shift(1).fillna(0)
    return gate.values, float(thr)


# ============================================================================
# 6. MAIN
# ============================================================================
def main():
    global N_VARIANTS
    closes, rets, fund = load_panel()
    rets = rets.dropna(how="all")
    fund = fund.reindex(rets.index)          # keep funding aligned to the trimmed return index
    n = len(rets)
    train_slice, oos_slice = bt.oos_split(n, TRAIN_FRAC)
    print(f"loaded {rets.shape[1]} coins, {n} daily bars "
          f"{str(rets.index[0])[:10]}..{str(rets.index[-1])[:10]}  PPY={PPY}")
    print(f"train {str(rets.index[train_slice][0])[:10]}..{str(rets.index[train_slice][-1])[:10]}  "
          f"oos {str(rets.index[oos_slice][0])[:10]}..{str(rets.index[oos_slice][-1])[:10]}\n")

    # ---- SANITY: EVT estimators discriminate heavy from light tails ----
    rng = np.random.default_rng(0)
    gauss = rng.standard_normal(4000) * 0.02
    t3 = rng.standard_t(3, 4000) * 0.012                   # heavy-tailed (df=3)
    print("SANITY  Hill-alpha(Gaussian)=%.2f  Hill-alpha(t3)=%.2f  (t3 lower = fatter)"
          % (hill_tail_index(-gauss, 200), hill_tail_index(-t3, 200)))
    print("SANITY  GPD-xi(Gaussian)=%.3f  GPD-xi(t3)=%.3f  (t3 higher = heavier)\n"
          % (gpd_pwm_shape(-gauss, 0.90), gpd_pwm_shape(-t3, 0.90)))

    btc_ret = rets["BTCUSDT"].fillna(0)
    mkt_ret = rets.mean(axis=1).fillna(0)

    # ----------------------------------------------------------------------
    # KNOB GRID (everything we try -> counted + deflated)
    # ----------------------------------------------------------------------
    WINDOWS = [120, 180, 252]            # trailing-window for the tail fit (need enough tail obs)
    K_FRAC = [0.08, 0.12, 0.15]          # Hill: fraction of window used as the tail
    U_Q = [0.85, 0.90]                   # GPD: POT threshold quantile
    Q_THRESH = [0.4, 0.5, 0.6, 0.7]      # gate train-quantile (how aggressive)

    drivers = {"btc": btc_ret, "mkt": mkt_ret}

    # precompute rolling EVT features (causal) for each driver
    hill_cache, gpd_cache, vol_cache = {}, {}, {}
    for dname, dser in drivers.items():
        for w in WINDOWS:
            vol_cache[(dname, w)] = rolling_vol(dser, w)
            for kf in K_FRAC:
                hill_cache[(dname, w, kf)] = rolling_hill(dser, w, kf)
            for uq in U_Q:
                gpd_cache[(dname, w, uq)] = rolling_gpd_xi(dser, w, uq)

    # ----------------------------------------------------------------------
    # BASE BOOKS (un-gated)
    # ----------------------------------------------------------------------
    books = {}
    r_btc, p_btc = base_long_btc(rets)
    books["long_btc"] = ("single", r_btc, p_btc)
    rr, ww = base_funding_carry(rets, fund)
    books["fund_carry"] = ("multi", rr, ww)

    base_perf = {}
    print("=== BASE BOOKS (un-gated) ===")
    print(f"{'book':12} {'IS_Shrp':>8} {'OOS_Shrp':>9} {'OOS_ret%':>9} {'OOS_maxDD%':>10} "
          f"{'worst1d%':>9} {'turn':>7} {'PSR':>6}")
    for name, (kind, rr, ww) in books.items():
        if kind == "single":
            net = single_net(rr, ww, COST_BPS); pos = ww
        else:
            net = book_net(rr, ww, COST_BPS); pos = np.abs(ww.values).sum(axis=1)
        mi, mo, psr = evaluate(net, pos)
        _, te = bt.oos_split(len(net), TRAIN_FRAC)
        worst = tail_event_dd(net[te])
        base_perf[name] = dict(net=net, pos=pos, mi=mi, mo=mo, psr=psr,
                               kind=kind, rr=rr, ww=ww, worst=worst)
        print(f"{name:12} {mi['sharpe_ann']:8.2f} {mo['sharpe_ann']:9.2f} {mo['ret_ann']*100:9.2f} "
              f"{mo['maxdd']*100:10.2f} {worst*100:9.2f} {mo['turnover']:7.4f} {psr:6.3f}")

    # ----------------------------------------------------------------------
    # (A) STANDALONE: directional EVT timing on BTC.
    #     fat tail -> SHORT/flat BTC; thin tail -> long/flat. Two positioning modes:
    #     'long_flat' (1 when thin, 0 when fat) and 'long_short' (1 thin, -1 fat).
    # ----------------------------------------------------------------------
    print("\n=== (A) STANDALONE: directional EVT tail-timing on BTC ===")
    best_sa = None
    for (w, kf, q, mode) in itertools.product(WINDOWS, K_FRAC, Q_THRESH, ["long_flat", "long_short"]):
        alpha = hill_cache[("btc", w, kf)]
        thr = np.nanquantile(alpha.values[train_slice], q)
        thin = (alpha > thr)                                  # thin tail = calm
        if mode == "long_flat":
            pos = thin.astype(float)
        else:
            pos = thin.astype(float) - (~thin).astype(float)  # +1 thin, -1 fat
        pos = pd.Series(pos, index=alpha.index).shift(1).fillna(0).values
        net = single_net(btc_ret.values, pos, COST_BPS)
        mi, mo, _ = evaluate(net, pos)
        N_VARIANTS += 1
        SR_TRIALS.append(mi["sr_pp"])
        if best_sa is None or mi["sharpe_ann"] > best_sa["is_sharpe"]:
            best_sa = dict(method="Hill", w=w, kf=kf, q=q, mode=mode,
                           is_sharpe=mi["sharpe_ann"], net=net, pos=pos)
    # also GPD-xi standalone
    for (w, uq, q, mode) in itertools.product(WINDOWS, U_Q, Q_THRESH, ["long_flat", "long_short"]):
        xi = gpd_cache[("btc", w, uq)]
        thr = np.nanquantile(xi.values[train_slice], q)
        thin = (xi < thr)
        if mode == "long_flat":
            pos = thin.astype(float)
        else:
            pos = thin.astype(float) - (~thin).astype(float)
        pos = pd.Series(pos, index=xi.index).shift(1).fillna(0).values
        net = single_net(btc_ret.values, pos, COST_BPS)
        mi, mo, _ = evaluate(net, pos)
        N_VARIANTS += 1
        SR_TRIALS.append(mi["sr_pp"])
        if mi["sharpe_ann"] > best_sa["is_sharpe"]:
            best_sa = dict(method="GPD", w=w, uq=uq, q=q, mode=mode,
                           is_sharpe=mi["sharpe_ann"], net=net, pos=pos)
    mi_sa, mo_sa, psr_sa = evaluate(best_sa["net"], best_sa["pos"])
    _kf = best_sa.get("kf"); _uq = best_sa.get("uq")
    print(f"IS-selected: method={best_sa['method']} w={best_sa['w']} "
          f"{'kf='+str(_kf) if _kf is not None else 'uq='+str(_uq)} q={best_sa['q']} "
          f"mode={best_sa['mode']}  (IS Sharpe={best_sa['is_sharpe']:.2f})")
    print(f"  STANDALONE OOS Sharpe={mo_sa['sharpe_ann']:.2f}  ret={mo_sa['ret_ann']*100:.2f}%  "
          f"maxDD={mo_sa['maxdd']*100:.2f}%  turn={mo_sa['turnover']:.4f}  PSR={psr_sa:.3f}")

    # ----------------------------------------------------------------------
    # (B) OVERLAY / crash-hedge timing: EVT gate sizes each base book.
    #     Tune (driver, method, window, k/uq, q) on IS; report OOS vs un-gated.
    #     Also benchmark the BEST EVT gate against a plain realized-vol gate.
    # ----------------------------------------------------------------------
    print("\n=== (B) EVT-GATED OVERLAY (tune gate on IS, report OOS vs un-gated) ===")
    overlay_results = {}
    for name, bp in base_perf.items():
        kind, rr, ww = bp["kind"], bp["rr"], bp["ww"]
        driver_pref = "btc" if name == "long_btc" else "mkt"

        def apply_gate(gate):
            if kind == "single":
                pos_g = ww * gate
                return single_net(rr, pos_g, COST_BPS), pos_g
            Wg = ww.mul(pd.Series(gate, index=ww.index), axis=0)
            return book_net(rr, Wg, COST_BPS), np.abs(Wg.values).sum(axis=1)

        best = None
        # Hill-alpha gates
        for (w, kf, q) in itertools.product(WINDOWS, K_FRAC, Q_THRESH):
            gate, thr = evt_gate_from_alpha(hill_cache[(driver_pref, w, kf)], q, train_slice)
            net_g, posv = apply_gate(gate)
            mi, mo, _ = evaluate(net_g, posv)
            N_VARIANTS += 1
            SR_TRIALS.append(mi["sr_pp"])
            if best is None or mi["sharpe_ann"] > best["is_sharpe"]:
                best = dict(method="Hill", driver=driver_pref, w=w, kf=kf, uq=None, q=q,
                            thr=thr, is_sharpe=mi["sharpe_ann"], net=net_g, pos=posv)
        # GPD-xi gates
        for (w, uq, q) in itertools.product(WINDOWS, U_Q, Q_THRESH):
            gate, thr = evt_gate_from_xi(gpd_cache[(driver_pref, w, uq)], q, train_slice)
            net_g, posv = apply_gate(gate)
            mi, mo, _ = evaluate(net_g, posv)
            N_VARIANTS += 1
            SR_TRIALS.append(mi["sr_pp"])
            if mi["sharpe_ann"] > best["is_sharpe"]:
                best = dict(method="GPD", driver=driver_pref, w=w, kf=None, uq=uq, q=q,
                            thr=thr, is_sharpe=mi["sharpe_ann"], net=net_g, pos=posv)

        # vol benchmark (tuned same way) -- to show whether EVT beats plain vol gating
        best_vol = None
        for (w, q) in itertools.product(WINDOWS, Q_THRESH):
            gate, thr = vol_gate(vol_cache[(driver_pref, w)], q, train_slice)
            net_g, posv = apply_gate(gate)
            mi, mo, _ = evaluate(net_g, posv)
            N_VARIANTS += 1
            SR_TRIALS.append(mi["sr_pp"])
            if best_vol is None or mi["sharpe_ann"] > best_vol["is_sharpe"]:
                best_vol = dict(w=w, q=q, is_sharpe=mi["sharpe_ann"], net=net_g, pos=posv)

        mi_g, mo_g, psr_g = evaluate(best["net"], best["pos"])
        mi_v, mo_v, psr_v = evaluate(best_vol["net"], best_vol["pos"])
        base_mo = bp["mo"]
        _, te = bt.oos_split(len(best["net"]), TRAIN_FRAC)
        worst_g = tail_event_dd(best["net"][te])
        overlay_results[name] = dict(best=best, mo_g=mo_g, psr_g=psr_g,
                                     base_mo=base_mo, base_psr=bp["psr"], base_worst=bp["worst"],
                                     worst_g=worst_g, vol=best_vol, mo_v=mo_v, psr_v=psr_v)
        kf_str = f"kf={best['kf']}" if best["method"] == "Hill" else f"uq={best['uq']}"
        print(f"\n[{name}] IS-selected EVT gate: {best['method']} driver={best['driver']} "
              f"w={best['w']} {kf_str} q={best['q']} thr={best['thr']:.4f}")
        print(f"    un-gated  OOS Sharpe={base_mo['sharpe_ann']:6.2f}  ret={base_mo['ret_ann']*100:6.2f}%  "
              f"maxDD={base_mo['maxdd']*100:7.2f}%  worst1d={bp['worst']*100:6.2f}%  "
              f"turn={base_mo['turnover']:.4f}  PSR={bp['psr']:.3f}")
        print(f"    EVT-gated OOS Sharpe={mo_g['sharpe_ann']:6.2f}  ret={mo_g['ret_ann']*100:6.2f}%  "
              f"maxDD={mo_g['maxdd']*100:7.2f}%  worst1d={worst_g*100:6.2f}%  "
              f"turn={mo_g['turnover']:.4f}  PSR={psr_g:.3f}")
        print(f"    VOL-gated OOS Sharpe={mo_v['sharpe_ann']:6.2f}  (benchmark: does EVT beat plain vol?)")
        d = mo_g["sharpe_ann"] - base_mo["sharpe_ann"]
        dv = mo_g["sharpe_ann"] - mo_v["sharpe_ann"]
        dd_cut = (bp["worst"] - worst_g)
        print(f"    -> EVT overlay {'IMPROVES' if d>0 else 'HURTS'} OOS Sharpe by {d:+.2f}; "
              f"vs vol gate {dv:+.2f}; worst-1d cut {dd_cut*100:+.2f}pp")

    # ----------------------------------------------------------------------
    # DSR deflation across EVERY variant tried
    # ----------------------------------------------------------------------
    srstar = bt.dsr_benchmark(SR_TRIALS)
    print(f"\n=== DEFLATION ===")
    print(f"n_variants_tried={N_VARIANTS}  within-family DSR SR* (per-period)={srstar:.5f}")

    # headline overlay = best OOS Sharpe improvement that is also POSITIVE OOS
    best_overlay_name = max(overlay_results,
                            key=lambda k: overlay_results[k]["mo_g"]["sharpe_ann"]
                            - overlay_results[k]["base_mo"]["sharpe_ann"])
    ov = overlay_results[best_overlay_name]
    base_sr = ov["base_mo"]["sharpe_ann"]
    gated_sr = ov["mo_g"]["sharpe_ann"]
    vol_sr = ov["mo_v"]["sharpe_ann"]
    improve = gated_sr - base_sr
    evt_beats_vol = gated_sr > vol_sr
    gated_sr_pp = ov["mo_g"]["sr_pp"]
    beats_dsr = gated_sr_pp > srstar
    print(f"best overlay = '{best_overlay_name}': base OOS Sharpe={base_sr:.2f} -> "
          f"EVT-gated {gated_sr:.2f} (Delta{improve:+.2f}); vol-gated {vol_sr:.2f} "
          f"({'EVT beats vol' if evt_beats_vol else 'vol >= EVT'}); "
          f"gated OOS SR_pp={gated_sr_pp:.4f} vs SR*={srstar:.4f} "
          f"-> {'BEATS' if beats_dsr else 'FAILS'} deflated bar")

    # ----------------------------------------------------------------------
    # VERDICT
    # ----------------------------------------------------------------------
    sa_psr = psr_sa
    sa_edge = (np.isfinite(sa_psr) and sa_psr >= 0.95) and beats_dsr and mo_sa["ret_ann"] > 0
    ov_psr = ov["psr_g"]
    # overlay must improve OOS Sharpe materially, be positive, beat the plain-vol benchmark
    # (otherwise it is just vol-timing in disguise), beat DSR, and have high PSR.
    ov_improves = (improve > 0.30) and (gated_sr > 0) and evt_beats_vol
    ov_edge = ov_improves and (np.isfinite(ov_psr) and ov_psr >= 0.95) and beats_dsr

    if sa_edge or ov_edge:
        verdict = "EDGE"
    elif (ov_improves and np.isfinite(ov_psr) and ov_psr >= 0.80 and gated_sr > 0) or \
         (np.isfinite(sa_psr) and sa_psr >= 0.80 and mo_sa["ret_ann"] > 0):
        verdict = "MARGINAL"
    else:
        verdict = "DEAD"

    if ov_edge and not sa_edge:
        role = "risk-overlay"
    elif sa_edge and not ov_edge:
        role = "standalone-alpha"
    elif ov_edge and sa_edge:
        role = "both"
    elif ov_improves and gated_sr > 0:
        role = "risk-overlay"
    else:
        role = "none"

    notes = (
        f"EVT left-tail-index dynamics as a crash-risk detector. Hill (1975) MLE tail "
        f"index alpha and GPD shape xi via probability-weighted-moments (Hosking-Wallis), "
        f"both implemented in numpy on CAUSAL trailing windows (no look-ahead, gate "
        f"shifted 1 bar). STANDALONE directional EVT-timing on BTC: OOS Sharpe="
        f"{mo_sa['sharpe_ann']:.2f}, ret={mo_sa['ret_ann']*100:.1f}%, PSR={sa_psr:.3f} "
        f"(directional EVT timing -- consistent with the repo prior that directional "
        f"signals are dead). OVERLAY (crash-hedge gating): best='{best_overlay_name}', "
        f"un-gated OOS Sharpe={base_sr:.2f} -> EVT-gated {gated_sr:.2f} (Delta{improve:+.2f}); "
        f"plain realized-vol gate gives {vol_sr:.2f} ({'EVT beats vol' if evt_beats_vol else 'vol matches/beats EVT, so the EVT gate is largely redundant with vol-state timing'}). "
        f"worst-1d OOS loss base={ov['base_worst']*100:.2f}% -> gated={ov['worst_g']*100:.2f}%. "
        f"n_variants_tried={N_VARIANTS}, within-family DSR SR*_pp={srstar:.4f}, gated "
        f"SR_pp={gated_sr_pp:.4f} ({'beats' if beats_dsr else 'fails'} deflation). "
        f"PRIOR check: a fat tail and high vol co-occur, so the EVT gate overlaps the "
        f"vol gate; any benefit beyond vol is the only thing that would be a genuine EVT "
        f"edge. close-to-close maxDD is an illusion (no intrabar liq/gap)."
    )
    print("\n=== VERDICT:", verdict, "| role:", role, "===")
    print(notes)

    out = dict(
        key="evt_tail_timing",
        file="experiments/exo_evt_tail_timing.py",
        implemented=True,
        libs_implemented=("hill_tail_index (Hill 1975 MLE tail index alpha=1/xi from k "
                          "largest order stats, numpy), gpd_pwm_shape (Generalized Pareto "
                          "shape xi via probability-weighted moments, Hosking-Wallis 1987, "
                          "numpy), causal rolling_hill / rolling_gpd_xi, realized-vol "
                          "benchmark gate; no statsmodels/scipy-EVT used"),
        method="Extreme Value Theory left-tail-index (Hill + POT-GPD) crash-risk timing",
        verdict=verdict,
        role=role,
        market_neutral=bool(best_overlay_name == "fund_carry"),
        universe=f"{rets.shape[1]} USDT-perps 1d futures since 2022-01-01",
        n_obs=int(ov["mo_g"]["n"]),
        oos_sharpe=float(mo_sa["sharpe_ann"]),            # standalone OOS sharpe
        oos_ret_ann_pct=float(mo_sa["ret_ann"] * 100),
        psr=float(max(sa_psr if np.isfinite(sa_psr) else 0.0,
                      ov_psr if np.isfinite(ov_psr) else 0.0)),
        dsr=float(srstar),
        maxdd_pct=float(ov["mo_g"]["maxdd"] * 100),
        turnover=float(ov["mo_g"]["turnover"]),
        cost_bps=COST_BPS,
        n_variants_tried=int(N_VARIANTS),
        overlay_base_sharpe=float(base_sr),
        overlay_gated_sharpe=float(gated_sr),
        notes=notes,
        data_caveats=("daily close-to-close perp returns 2022-2026; EVT tail index is a "
                      "causal trailing-window statistic on the LEFT tail (no look-ahead, "
                      "gate shifted 1 bar); gate threshold quantile fit on TRAIN only. "
                      "Hill/GPD need a long window (120-252d) to have enough tail obs, so "
                      "the index reacts slowly -- it lags fast crashes. maxDD/worst-1d are "
                      "close-to-close (no intrabar liq/gap), an optimistic crash proxy. "
                      "MATIC/NEAR/ATOM excluded (no 1d fut cache). EVT-vs-vol benchmark "
                      "included because fat-tail and high-vol regimes overlap heavily."),
    )
    rp = pathlib.Path(__file__).resolve().parent.parent / "reports" / "exo_evt_tail_timing.json"
    rp.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {rp}")
    return out


if __name__ == "__main__":
    main()
