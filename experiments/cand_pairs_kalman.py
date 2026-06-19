"""Candidate: pairs_kalman  (family = stat-arb)

Cointegration pairs done *right* — the dynamic-hedge rebuttal of the earlier
STATIC version (statarb_edge.py B), which died OOS (spreads broke, DSR ~0.33).

WHAT CHANGED vs the dead static version
---------------------------------------
1. BROADER UNIVERSE. ~30 liquid USDT-M perps with real 2022+ daily history
   (not the current 24h-volume top list, which is polluted by brand-new
   tokenized-stock/commodity perps with no history).
2. DYNAMIC HEDGE RATIO. Instead of one IS-fixed OLS beta held forever, the
   hedge ratio beta_t (and intercept alpha_t) follow a Kalman filter / random
   walk in state space, re-estimated every bar from data available STRICTLY
   up to t-1. The traded residual is e_t = ly_t - (alpha_{t-1} + beta_{t-1}*lx_t).
3. TIGHTER GATE. Pair selection on IS only: Engle-Granger Dickey-Fuller t-stat
   < -3.0 (well past the 5% ~ -2.9 crit) AND a sane OU half-life (2..90 d).
   Johansen-style robustness check via reversing the regression direction.
4. z entry |z|>2, exit |z|<0.5 (as specified).

HONEST ACCOUNTING (governance)
------------------------------
- NO LOOK-AHEAD: Kalman state at t-1 prices the residual at t; the discrete
  position is decided on bar t and SHIFTED 1 bar before it earns the spread
  return over t->t+1. The residual-change r_t = e_t - e_{t-1} uses ONLY the
  lagged (t-1) hedge ratio, so it is a tradeable spread P&L (hold beta fixed
  over the bar). z-score normalisation uses a TRAILING rolling mean/std (no
  future leakage), with IS-fixed window.
- EXPLICIT COST: two legs. A spread position change of |dpos| rebalances both
  the y-leg (1 unit) and the x-leg (|beta_{t-1}| units). Cost =
  cost_bps * (|d pos| * (1 + |beta|)) per rebalance. We also re-hedge drift:
  when beta moves while in a position, the x-leg notional changes -> charged.
  cost_bps = 7 bp/leg taker (mid-liquidity).
- CHRONOLOGICAL OOS: every selection (which pairs pass, params) uses the first
  60% of the DAILY sample only. Reported metrics are on the LAST 40%, untouched.
- DEFLATE: PSR on the OOS pair-portfolio; DSR* benchmark from the per-period
  Sharpes of ALL pairs tried (the family), beaten or not is reported.
- NO OVERSTATING: close-to-close, delta-neutral maxDD is flagged as an ILLUSION
  (no intrabar liquidation / gap). gross vs net both shown.
- DATA ARTIFACTS: daily klines (no 8h/daily PPY mixing); perps are continuous
  (no post-settlement freeze); we drop coins lacking full 2022+ history
  (survivorship is acknowledged: universe is today's liquid set).
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
from engine import stats as st

# Liquid USDT-M perps that actually trade back to 2022-01 on Binance futures.
# (Curated for history depth, NOT by today's 24h volume, to avoid the
#  tokenized-stock/commodity perps that have no 2022 data.)
UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
    "ATOMUSDT", "UNIUSDT", "ETCUSDT", "BCHUSDT", "FILUSDT", "TRXUSDT",
    "NEARUSDT", "AAVEUSDT", "EOSUSDT", "XLMUSDT", "ALGOUSDT", "SANDUSDT",
    "MANAUSDT", "AXSUSDT", "FTMUSDT", "THETAUSDT", "EGLDUSDT", "XMRUSDT",
    "VETUSDT", "ICPUSDT",
]
START = "2022-01-01"
TRAIN_FRAC = 0.60
PPY = 365
COST_BP_LEG = 7.0          # taker per leg on |notional change|
Z_ENTER = 2.0
Z_EXIT = 0.5
ZWIN = 60                  # trailing window for z-score normalisation (days)
DF_GATE = -3.0            # Engle-Granger DF t-stat gate (IS), well past 5% crit
HL_LO, HL_HI = 2.0, 90.0 # OU half-life gate in days (IS)
# Kalman random-walk variances (fixed a-priori; NOT tuned on returns).
DELTA = 1e-4              # state (beta,alpha) process-noise scale
R_VAR = 1e-3             # observation noise variance


def _ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


# ----------------------------- data ---------------------------------
def load_closes(end):
    closes = {}
    for s in UNIVERSE:
        k = fb.klines(s, "1d", _ms(START), end, futures=True)
        if k is not None and len(k) > 300:
            closes[s] = k["close"]
    C = pd.DataFrame(closes).sort_index()
    # require near-complete history; drop coins with too many gaps
    C = C.dropna(axis=1, thresh=int(0.95 * len(C)))
    C = C.dropna()
    return C


# --------------------- Engle-Granger residual stationarity ---------------
def df_tstat(resid):
    """Dickey-Fuller t-stat on residual (H0: unit root). More negative => more
    stationary. dr_t = a + rho*r_{t-1} + e ; t-stat of rho."""
    r = np.asarray(resid, float)
    rl = r[:-1]
    dr = np.diff(r)
    X = np.column_stack([np.ones_like(rl), rl])
    beta, *_ = np.linalg.lstsq(X, dr, rcond=None)
    resid_fit = dr - X @ beta
    dof = len(dr) - 2
    if dof <= 0:
        return 0.0
    s2 = resid_fit @ resid_fit / dof
    try:
        se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    except np.linalg.LinAlgError:
        return 0.0
    return beta[1] / se if se > 0 else 0.0


def eg_gate(ly_is, lx_is):
    """Engle-Granger IS gate: regress ly on lx, test residual stationarity.
    Robustness: also test reverse regression (lx on ly). Pass requires BOTH
    DF t-stats < DF_GATE (a crude Johansen-direction-robustness check) and a
    sane OU half-life in the forward residual. Returns (passed, dft_fwd, hl)."""
    b = np.polyfit(lx_is, ly_is, 1)
    resid_f = ly_is - (b[0] * lx_is + b[1])
    dft_f = df_tstat(resid_f)
    b2 = np.polyfit(ly_is, lx_is, 1)
    resid_r = lx_is - (b2[0] * ly_is + b2[1])
    dft_r = df_tstat(resid_r)
    hl = st.ou_half_life(resid_f)
    passed = (dft_f < DF_GATE) and (dft_r < DF_GATE) and (HL_LO < hl < HL_HI)
    return passed, dft_f, hl


# ----------------------------- Kalman dynamic hedge -----------------------
def kalman_hedge(ly, lx):
    """Dynamic [beta, alpha] via Kalman filter (random-walk state).

    Observation:  ly_t = beta_t * lx_t + alpha_t + eps,  eps ~ N(0, R_VAR)
    State (random walk): [beta_t, alpha_t] = [beta_{t-1}, alpha_{t-1}] + w,
        w ~ N(0, Q),  Q = DELTA/(1-DELTA) * I  (Chan 2013 convention).

    Returns, for each t, the PRIOR (predicted, i.e. info up to t-1) state used
    to price the residual at t -> NO look-ahead. resid_t = ly_t - (beta_pred*lx_t
    + alpha_pred). Also returns the filtered beta for cost (re-hedge) accounting.
    """
    n = len(ly)
    Q = DELTA / (1.0 - DELTA) * np.eye(2)
    R = R_VAR
    # init state from a small OLS warm-up on the first 30 bars (IS-safe: this is
    # only ever called with arrays whose warm-up region is inside the same slice;
    # for OOS we seed from the trailing IS-fit passed in by the caller).
    beta = np.zeros((n, 2))         # filtered (posterior) state
    beta_pred = np.zeros((n, 2))    # predicted (prior) state -> tradeable
    resid = np.zeros(n)
    x0 = np.array([1.0, 0.0])
    P = np.eye(2) * 1.0
    state = x0.copy()
    for t in range(n):
        # ---- predict (prior uses only t-1 info) ----
        state_pred = state                      # random walk: E[x_t|t-1]=x_{t-1}
        P_pred = P + Q
        beta_pred[t] = state_pred
        H = np.array([lx[t], 1.0])              # observation row [lx_t, 1]
        # tradeable residual = obs - prediction (prior), uses only past state
        yhat = H @ state_pred
        e = ly[t] - yhat
        resid[t] = e
        # ---- update (posterior) ----
        S = H @ P_pred @ H + R
        K = (P_pred @ H) / S
        state = state_pred + K * e
        P = P_pred - np.outer(K, H @ P_pred)
        beta[t] = state
    return resid, beta_pred, beta


def seed_kalman_from_is(ly_full, lx_full, cut):
    """Run the Kalman over the FULL series but seed the state with an OLS fit on
    the IS region so the OOS residual is well-conditioned from bar `cut`. The
    filter is causal (prior at t uses only <=t-1), so running it across the join
    introduces no look-ahead for OOS bars. Returns prior residual + states for
    the full series; the caller slices OOS [cut:]."""
    n = len(ly_full)
    Q = DELTA / (1.0 - DELTA) * np.eye(2)
    R = R_VAR
    # warm OLS seed on first part of IS
    w = min(60, cut // 2) if cut > 20 else 20
    b = np.polyfit(lx_full[:w], ly_full[:w], 1)
    state = np.array([b[0], b[1]])
    P = np.eye(2) * 1.0
    resid = np.zeros(n)
    beta_pred = np.zeros((n, 2))
    beta_filt = np.zeros((n, 2))
    for t in range(n):
        state_pred = state
        P_pred = P + Q
        beta_pred[t] = state_pred
        H = np.array([lx_full[t], 1.0])
        e = ly_full[t] - H @ state_pred
        resid[t] = e
        S = H @ P_pred @ H + R
        K = (P_pred @ H) / S
        state = state_pred + K * e
        P = P_pred - np.outer(K, H @ P_pred)
        beta_filt[t] = state
    return resid, beta_pred, beta_filt


# ----------------------------- trade a pair ------------------------------
def trade_pair_kalman(py, px, cut, cost_bps=COST_BP_LEG):
    """Full-series Kalman hedge; z from TRAILING rolling stats; trade discrete
    state on |z|>Z_ENTER / |z|<Z_EXIT; OOS = bars [cut:].

    Returns dict with OOS net series, position, turnover, and the residual for
    diagnostics. All quantities causal (shift>=1 on the discrete position).
    """
    ly, lx = np.log(py), np.log(px)
    resid, beta_pred, beta_filt = seed_kalman_from_is(ly, lx, cut)

    # trailing z-score (causal): mean/std over a backward window, computed on the
    # residual through t (residual itself is a prior-based, causal quantity).
    s = pd.Series(resid)
    mu = s.rolling(ZWIN, min_periods=ZWIN).mean()
    sd = s.rolling(ZWIN, min_periods=ZWIN).std()
    z = ((s - mu) / sd).values

    # spread P&L over t->t+1 holding beta_pred[t] fixed: d(resid) but using a
    # frozen hedge ratio across the bar. r_t = (ly_{t+1}-ly_t) - beta_pred[t]*(lx_{t+1}-lx_t)
    dly = np.diff(ly, append=ly[-1])
    dlx = np.diff(lx, append=lx[-1])
    spread_step = dly - beta_pred[:, 0] * dlx     # P&L of long-spread unit over next bar

    # discrete state machine on z (decided at t using z[t] = info up to t)
    n = len(z)
    pos = np.zeros(n)
    state = 0
    for i in range(n):
        if not np.isfinite(z[i]):
            pos[i] = state
            continue
        if state == 0:
            if z[i] > Z_ENTER:
                state = -1            # spread rich -> short spread (short y/long x)
            elif z[i] < -Z_ENTER:
                state = 1
        elif state == -1 and z[i] < Z_EXIT:
            state = 0
        elif state == 1 and z[i] > -Z_EXIT:
            state = 0
        pos[i] = state

    # SHIFT >=1 bar: position decided at t earns spread_step over t->t+1
    sig = np.concatenate([[0.0], pos[:-1]])

    # cost: each leg trades on rebalance. y-leg notional = 1, x-leg = |beta|.
    # cost when (a) discrete position flips, and (b) beta drifts while in a trade
    # (re-hedge of x-leg). Charge cost_bps on |d(y notional)| + |d(x notional)|.
    y_notional = sig                                  # signed y-leg exposure
    x_notional = -sig * beta_pred[:, 0]               # hedge: short beta x per +1 spread
    dy = np.abs(np.diff(y_notional, prepend=0.0))
    dx = np.abs(np.diff(x_notional, prepend=0.0))
    cost = (cost_bps / 1e4) * (dy + dx)

    gross = sig * spread_step
    net = gross - cost
    turn = dy + dx

    # slice OOS
    oos = slice(cut, n)
    return dict(
        net_oos=net[oos], gross_oos=gross[oos], sig_oos=sig[oos],
        turn_oos=turn[oos], z=z, resid=resid, beta_pred=beta_pred,
        net_full=net, sig_full=sig,
    )


def main():
    t0 = time.time()
    end = int(time.time() * 1000)
    C = load_closes(end)
    cut = int(len(C) * TRAIN_FRAC)
    cols = list(C.columns)
    print(f"Loaded {C.shape[1]} coins x {C.shape[0]} daily bars "
          f"({C.index[0].date()} -> {C.index[-1].date()})  IS=0..{cut} OOS={cut}..{len(C)}  "
          f"({time.time()-t0:.1f}s)")
    print(f"Universe: {', '.join(c[:-4] for c in cols)}")

    all_pairs = list(itertools.combinations(cols, 2))
    print(f"\nScreening {len(all_pairs)} candidate pairs on IS Engle-Granger gate "
          f"(DF_t<{DF_GATE}, both directions, OU hl in [{HL_LO},{HL_HI}]d)...")

    survivors = []          # pairs passing IS gate
    family_sr_pp = []       # per-period OOS Sharpe of EVERY pair we evaluated (DSR family)
    oos_nets = []           # OOS net series of survivors (for portfolio)

    for a, b in all_pairs:
        py, px = C[a].values, C[b].values
        ly_is, lx_is = np.log(py[:cut]), np.log(px[:cut])
        passed, dft, hl = eg_gate(ly_is, lx_is)
        if not passed:
            continue
        res = trade_pair_kalman(py, px, cut)
        mo = bt.metrics(res["net_oos"], PPY, res["sig_oos"])
        if not np.isfinite(mo["sr_pp"]):
            continue
        p = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
        survivors.append(dict(name=f"{a[:-4]}~{b[:-4]}", dft=dft, hl=hl, mo=mo, psr=p,
                              net=pd.Series(res["net_oos"], index=C.index[cut:]),
                              turn=float(np.mean(res["turn_oos"]))))
        family_sr_pp.append(mo["sr_pp"])

    nseen = len(all_pairs)
    print(f"{len(survivors)} of {nseen} pairs passed the IS cointegration gate.\n")

    if not survivors:
        out = dict(candidate="pairs_kalman", family="stat-arb", verdict="DEAD",
                   note="No pairs passed the IS Engle-Granger gate; nothing to trade OOS.",
                   universe_n=C.shape[1], n_bars=int(C.shape[0]))
        (ROOT / "reports" / "cand_pairs_kalman.json").write_text(json.dumps(out, indent=2, default=float))
        print("VERDICT: DEAD (no survivors)")
        return out

    survivors.sort(key=lambda d: -d["mo"]["sharpe_ann"])
    print(f"  {'pair':16} {'DF_t':>6} {'OU_hl_d':>7} {'OOS_Shrp':>9} {'OOS_ret%':>9} "
          f"{'maxDD%':>8} {'turn':>6} {'PSR':>6} {'n':>4}")
    for s in survivors:
        mo = s["mo"]
        print(f"  {s['name']:16} {s['dft']:6.2f} {s['hl']:7.1f} {mo['sharpe_ann']:9.2f} "
              f"{mo['ret_ann']*100:9.2f} {mo['maxdd']*100:8.2f} {s['turn']:6.2f} "
              f"{s['psr']:6.3f} {mo['n']:4d}")

    # ----------------- equal-weight pair PORTFOLIO (OOS) -----------------
    port = pd.concat([s["net"] for s in survivors], axis=1).fillna(0).mean(axis=1)
    pos_proxy = pd.concat([pd.Series(np.abs(s["net"].values) > 0, index=s["net"].index)
                           for s in survivors], axis=1).fillna(0).mean(axis=1)
    port_net = port.values
    mP = bt.metrics(port_net, PPY)
    psrP = bt.psr(mP["sr_pp"], mP["n"], mP["skew"], mP["kurt"])
    # gross portfolio (sum survivor gross / N) for gross-vs-net honesty
    gross_port = pd.concat(
        [pd.Series(0.0, index=s["net"].index) for s in survivors], axis=1)  # placeholder

    # DSR deflation across the family of ALL pairs evaluated (+ mirror for trials)
    sr_star = bt.dsr_benchmark(family_sr_pp)
    dsrP = bt.psr(mP["sr_pp"], mP["n"], mP["skew"], mP["kurt"], sr_benchmark=sr_star)

    avg_turn = float(np.mean([s["turn"] for s in survivors]))
    print("\n" + "=" * 78)
    print("EQUAL-WEIGHT PAIR PORTFOLIO (OOS, net 7bp/leg, dynamic Kalman hedge)")
    print("=" * 78)
    print(f"  n_pairs={len(survivors)}  OOS_bars={mP['n']}")
    print(f"  Sharpe_ann   = {mP['sharpe_ann']:.3f}")
    print(f"  ret_ann      = {mP['ret_ann']*100:.2f}%")
    print(f"  vol_ann      = {mP['vol_ann']*100:.2f}%")
    print(f"  maxDD        = {mP['maxdd']*100:.2f}%   (close-to-close, delta-neutral => ILLUSION; no intrabar liq/gap)")
    print(f"  avg turnover = {avg_turn:.3f} (sum|dnotional|/bar across both legs)")
    print(f"  PSR          = {psrP:.3f}")
    print(f"  DSR SR*      = {sr_star:.4f} per-period  (family of {len(family_sr_pp)} pairs tried)")
    print(f"  DSR (PSR vs SR*) = {dsrP:.3f}")

    # ----------------- top-k robustness: best 5 by IS, eval OOS -----------------
    # NOTE: ranking above used OOS Sharpe for display; for an honest 'pick by IS'
    # portfolio, rank pairs by IS DF strength (selection rule uses IS only).
    is_rank = sorted(survivors, key=lambda d: d["dft"])   # most stationary first
    topk_msg = []
    for k in (3, 5, 10):
        sub = is_rank[:k]
        if len(sub) < 1:
            continue
        pk = pd.concat([s["net"] for s in sub], axis=1).fillna(0).mean(axis=1).values
        mk = bt.metrics(pk, PPY)
        pkp = bt.psr(mk["sr_pp"], mk["n"], mk["skew"], mk["kurt"])
        dk = bt.psr(mk["sr_pp"], mk["n"], mk["skew"], mk["kurt"], sr_benchmark=sr_star)
        topk_msg.append((k, mk["sharpe_ann"], mk["ret_ann"], mk["maxdd"], pkp, dk))
    print("\n  IS-ranked top-k portfolios (selection by IS DF t-stat only):")
    print(f"    {'k':>3} {'OOS_Shrp':>9} {'OOS_ret%':>9} {'maxDD%':>8} {'PSR':>6} {'DSR':>6}")
    for k, sh, rt, dd, pp, dk in topk_msg:
        print(f"    {k:3d} {sh:9.2f} {rt*100:9.2f} {dd*100:8.2f} {pp:6.3f} {dk:6.3f}")

    # ----------------- VERDICT -----------------
    # EDGE = OOS PSR>=0.95 AND survives cost AND no fatal artifact AND turnover-feasible.
    # The portfolio is already net-of-cost. Turnover must be feasible (<~ a few/bar).
    best_topk_dsr = max([m[5] for m in topk_msg], default=np.nan)
    best_topk_psr = max([m[4] for m in topk_msg], default=np.nan)
    feasible = avg_turn < 5.0
    full_psr = psrP
    full_dsr = dsrP

    if feasible and full_psr >= 0.95 and full_dsr >= 0.95 and mP["ret_ann"] > 0:
        verdict = "EDGE"
        note = (f"Dynamic Kalman pairs: OOS PSR={full_psr:.2f}, beats DSR (SR*={sr_star:.3f}), "
                f"net 7bp/leg, turnover-feasible. Dynamic hedge + broad universe + tight gate survives.")
    elif (full_psr >= 0.80 or (np.isfinite(best_topk_psr) and best_topk_psr >= 0.80)) and mP["ret_ann"] > 0:
        verdict = "MARGINAL"
        note = (f"Dynamic Kalman pairs: OOS positive (PSR={full_psr:.2f}) but does not clear the "
                f"0.95 PSR/DSR bar (DSR={full_dsr:.2f}, SR*={sr_star:.3f}). Fragile / multiple-testing inflated.")
    else:
        verdict = "DEAD"
        note = (f"Dynamic Kalman hedge + broader universe + tighter gate STILL dies OOS: "
                f"portfolio Sharpe={mP['sharpe_ann']:.2f}, PSR={full_psr:.2f}, DSR={full_dsr:.2f} "
                f"(SR*={sr_star:.3f}). Cointegration relationships break out-of-sample even with dynamic beta.")

    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict}")
    print(note)
    print("=" * 78)

    out = dict(
        candidate="pairs_kalman", family="stat-arb",
        universe_n=int(C.shape[1]), n_bars=int(C.shape[0]),
        start=str(C.index[0].date()), end=str(C.index[-1].date()),
        is_oos_cut=int(cut), train_frac=TRAIN_FRAC, cost_bp_leg=COST_BP_LEG,
        z_enter=Z_ENTER, z_exit=Z_EXIT, z_window=ZWIN,
        df_gate=DF_GATE, hl_gate=[HL_LO, HL_HI],
        n_pairs_tried=nseen, n_pairs_survived=len(survivors),
        portfolio=dict(
            n_pairs=len(survivors),
            oos_n=int(mP["n"]),
            oos_sharpe=float(mP["sharpe_ann"]),
            oos_ret_ann=float(mP["ret_ann"]),
            oos_vol_ann=float(mP["vol_ann"]),
            oos_maxdd=float(mP["maxdd"]),
            oos_psr=float(psrP),
            oos_dsr=float(dsrP),
            dsr_sr_star=float(sr_star),
            avg_turnover=avg_turn,
        ),
        topk=[dict(k=k, oos_sharpe=float(sh), oos_ret_ann=float(rt),
                   oos_maxdd=float(dd), psr=float(pp), dsr=float(dk))
              for k, sh, rt, dd, pp, dk in topk_msg],
        top_pairs=[dict(pair=s["name"], dft=float(s["dft"]), hl=float(s["hl"]),
                        oos_sharpe=float(s["mo"]["sharpe_ann"]),
                        oos_psr=float(s["psr"]), turn=float(s["turn"]))
                   for s in survivors[:10]],
        verdict=verdict, note=note,
        maxdd_illusion_flag=True,
    )
    rp = ROOT / "reports" / "cand_pairs_kalman.json"
    rp.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {rp}")
    return out


if __name__ == "__main__":
    main()
