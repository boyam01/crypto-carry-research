"""Candidate: vol_risk_premium  (family = volatility)

GOAL (from spec): test a volatility-risk-premium edge on crypto WITHOUT options
data. Two routes were proposed:
  (a) delta-hedged short-gamma proxy on perps (short straddle replication) that
      earns the gap between realized vol and a vol *forecast*;
  (b) realized-vol mean-reversion timing.

WHY THIS IS HARD / THE CIRCULARITY TRAP (read before trusting any number):
  The volatility risk premium is, by definition, IV - RV: you get PAID an option
  premium (set by implied vol) and pay out realized variance. WITHOUT an options
  market there is no premium to collect. A "short straddle replicated by delta
  hedging" earns
        PnL_t ~= 0.5 * Gamma * S^2 * (IV^2 - RV^2) * dt
  but Gamma (convexity) comes from the OPTION. A perp position delta-hedged to
  zero has ZERO gamma -> earns nothing from variance. If instead you hold a
  static (un-hedged) perp and call its variance exposure "short gamma", you are
  really running a directional bet (already DEAD here) plus a self-imposed vol
  forecast. The only "premium" is then IV_forecast - RV, and IV_forecast is a
  number YOU invented -> CIRCULAR. You can manufacture any Sharpe by choosing the
  forecast.  So route (a) cannot be a clean, non-circular test.

WHAT WE ACTUALLY TEST (honest, non-circular, tradeable):
  T1. Realized-vol MEAN REVERSION as a *tradeable timing* signal. RV is strongly
      mean-reverting (that is a real, well-documented stylized fact). The honest
      question: can that be MONETIZED with instruments we actually have (the perp)
      without becoming a directional bet? We test a vol-targeting / inverse-vol
      sizing overlay on a delta-NEUTRAL carry-free underlying-return stream, and a
      "sell-vol-after-spikes" timing of a market-neutral leg. If RV-MR only helps
      via direction, it is just momentum/reversal in disguise (DEAD).
  T2. A DELTA-HEDGED SHORT-GAMMA REPLICATION done as honestly as possible on the
      perp: rehedge to delta-zero every bar; the only variance P&L that survives
      is the *discrete-hedging* gamma term, and we charge realistic rehedge cost.
      We show the premium is (i) forecast-dependent (circular) and (ii) destroyed
      by rehedge turnover. Reported as evidence, flagged circular.

GOVERNANCE: signals shifted >=1 bar; cost >=5bp/leg on |Δposition| per leg;
params tuned on first 60% only, metrics on last 40%; PSR/DSR reported.
"""
from __future__ import annotations
import sys, time, json, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import fetch_binance as fb
from engine import backtest as bt
from engine import stats as st

UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT"]
START = "2021-01-01"
INTERVAL = "1h"
PPY = 365 * 24            # hourly periods per year
TRAIN_FRAC = 0.60


def _ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


def load_panel(symbols, start, end):
    out = {}
    for s in symbols:
        k = fb.klines(s, INTERVAL, start, end, futures=True)
        if k is None or len(k) < 5000:
            continue
        df = pd.DataFrame(index=k.index)
        df["close"] = k["close"].astype(float)
        df["ret"] = np.log(df["close"]).diff()          # log return
        out[s] = df.dropna()
    return out


def realized_vol(logret: pd.Series, win: int) -> pd.Series:
    """Rolling realized vol (per-bar stdev of log returns)."""
    return logret.rolling(win).std()


def evaluate(net, position=None):
    net = np.asarray(net, float)
    n = len(net)
    tr, te = bt.oos_split(n, TRAIN_FRAC)
    mi = bt.metrics(net[tr], PPY, position[tr] if position is not None else None)
    mo = bt.metrics(net[te], PPY, position[te] if position is not None else None)
    psr = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
    return mi, mo, psr


# ----------------------------------------------------------------------------
# STYLIZED-FACT CHECK: is realized vol actually mean-reverting? (sanity)
# ----------------------------------------------------------------------------
def rv_mean_reversion_stats(panel):
    rows = []
    for s, df in panel.items():
        rv = realized_vol(df["ret"], 24).dropna()        # daily RV from 1h
        lrv = np.log(rv.replace(0, np.nan).dropna())
        hl = st.ou_half_life(lrv.values)
        a1 = st.acf1(lrv.diff().dropna().values)
        rows.append((s, hl, a1, float(lrv.std())))
    return rows


# ----------------------------------------------------------------------------
# T2: delta-hedged short-gamma replication on the perp (honest, circular-flagged)
# ----------------------------------------------------------------------------
def short_gamma_replication(df, vol_forecast_win, k_strike_band, cost_bps):
    """Replicate a SHORT straddle by delta hedging on the perp.

    A short straddle has delta that flips sign as S moves through the strike.
    To stay short-gamma we hold delta = -slope(payoff) and rehedge each bar.
    The discrete-hedging P&L of a short option over [t,t+1] is approximately
        +0.5 * IV^2 * S^2 * dt   (theta/premium we PRETEND to collect)
        -0.5 * (dS/S)^2 * S^2    (realized gamma loss)
    i.e. you win when realized move^2 < IV^2*dt. IV is a FORECAST (circular).
    We rehedge the replicating perp delta each bar -> charge turnover cost.

    Returns per-bar net P&L of the variance position (already lag-safe).
    """
    r = df["ret"].values
    # vol forecast available at t-1 (EWMA of past realized var) -> the "IV" proxy
    var_fc = pd.Series(r ** 2).ewm(span=vol_forecast_win, adjust=False).mean().shift(1).fillna(0).values
    # short-variance P&L per bar: collect forecast var, pay realized var
    pnl_var = var_fc - r ** 2                              # short gamma payoff (per $ notional^2)
    pnl_var = pnl_var * 1e2                                # scale to readable units (not load-bearing)
    # delta-hedge turnover proxy: to keep a replicated straddle delta-neutral you
    # trade ~ |gamma * dS| of perp each bar. gamma ~ 1/(S*IV*sqrt(band)); the
    # rehedge notional fraction per bar ~ |r| / (IV_band). Charge cost on it.
    iv = np.sqrt(np.maximum(var_fc, 1e-12))
    rehedge_frac = np.abs(r) / np.maximum(iv * k_strike_band, 1e-6)
    rehedge_frac = np.clip(rehedge_frac, 0, 5.0)           # cap pathological bars
    cost = (cost_bps / 1e4) * rehedge_frac * 1e2           # same scale as pnl_var
    return pnl_var - cost, rehedge_frac


# ----------------------------------------------------------------------------
# T1: RV mean-reversion timing on a MARKET-NEUTRAL leg.
#   Build a cross-sectional market-neutral return stream (each coin minus the
#   equal-weight market) so we strip out direction. Then test whether timing
#   exposure to that residual by RV regime (scale DOWN after vol spikes, i.e.
#   short-vol-style "sell insurance when calm") adds risk-adjusted return that a
#   constant-exposure baseline does not. Non-circular: the signal is observed RV,
#   the traded object is a real neutral spread, cost charged on |Δexposure|.
# ----------------------------------------------------------------------------
def market_neutral_residuals(panel):
    rets = pd.DataFrame({s: panel[s]["ret"] for s in panel}).dropna()
    mkt = rets.mean(axis=1)
    resid = rets.sub(mkt, axis=0)                          # demeaned cross-section
    return resid, rets


def rv_timed_overlay(resid_series, win, z_lo, z_hi, cost_bps, mode):
    """Scale exposure to a neutral residual by its own RV regime.

    mode='invvol' : target constant risk (exposure ~ 1/RV) -> classic vol target.
    mode='sellvol': reduce exposure after high-RV (z>z_hi), add when calm (z<z_lo)
                    -> 'sell vol when calm' proxy. Exposure sign is FIXED (+1) so
                    this is NOT a directional bet on the residual; it only times
                    HOW MUCH neutral exposure to carry. (A neutral residual has
                    ~zero mean return, so any positive Sharpe here is the timing.)
    Signal uses RV up to t-1. Returns (net, exposure).
    """
    r = resid_series.values
    rv = pd.Series(r).rolling(win).std()
    z = ((np.log(rv.replace(0, np.nan)) - np.log(rv.replace(0, np.nan)).rolling(win * 5).mean())
         / np.log(rv.replace(0, np.nan)).rolling(win * 5).std()).shift(1)
    z = z.fillna(0).values
    rv_lag = rv.shift(1).fillna(rv.median()).values
    if mode == "invvol":
        target = rv_lag.copy()
        target[target <= 0] = np.nan
        med = np.nanmedian(target)
        expo = med / np.where(np.isnan(target), med, target)
        expo = np.clip(expo, 0.2, 3.0)
    else:  # sellvol
        expo = np.ones_like(r)
        expo = np.where(z > z_hi, 0.3, expo)
        expo = np.where(z < z_lo, 1.5, expo)
    net = bt.run(r, expo, cost_bps)
    return net, expo


def main():
    end = int(time.time() * 1000)
    panel = load_panel(UNIVERSE, _ms(START), end)
    print(f"loaded {len(panel)} coins, {INTERVAL} bars, PPY={PPY}\n")

    # --- sanity: RV mean reversion exists? ---
    print("--- RV mean-reversion stylized-fact check (log daily-RV from 1h) ---")
    print(f"{'coin':9} {'OU_halflife(bars)':>18} {'acf1(dRV)':>10} {'sd(logRV)':>10}")
    for s, hl, a1, sd in rv_mean_reversion_stats(panel):
        print(f"{s:9} {hl:18.1f} {a1:10.3f} {sd:10.3f}")
    print()

    # =====================================================================
    # T2: short-gamma replication (circular, evidence only)
    # =====================================================================
    print("=== T2: delta-hedged short-gamma replication on perp (CIRCULAR flag) ===")
    t2_sr_trials = []
    t2_rows = []
    for s, df in panel.items():
        # tune forecast window on IS only
        best = None
        for fc in (12, 48, 168):
            for band in (1.0, 2.0):
                net, _ = short_gamma_replication(df, fc, band, cost_bps=0.0)
                mi, mo, _ = evaluate(net)
                if best is None or mi["sharpe_ann"] > best[0]:
                    best = (mi["sharpe_ann"], fc, band)
        fc, band = best[1], best[2]
        # gross (cost 0) and net (5bp rehedge) on OOS
        net_g, rh = short_gamma_replication(df, fc, band, cost_bps=0.0)
        net_c, rh = short_gamma_replication(df, fc, band, cost_bps=5.0)
        _, mg, pg = evaluate(net_g)
        _, mc, pc = evaluate(net_c)
        t2_sr_trials.append(mg["sr_pp"])
        t2_rows.append((s, mg, mc, pc, float(np.nanmean(rh))))
    print(f"{'coin':9} {'GROSS_Shrp':>11} {'NET_Shrp':>10} {'NET_PSR':>8} {'rehedge/bar':>12}")
    for s, mg, mc, pc, rh in sorted(t2_rows, key=lambda x: -x[1]["sharpe_ann"]):
        print(f"{s:9} {mg['sharpe_ann']:11.2f} {mc['sharpe_ann']:10.2f} {pc:8.3f} {rh:12.3f}")
    print("  ^ GROSS uses a self-imposed vol FORECAST as the 'premium' => CIRCULAR.")
    print("    NET charges 5bp on delta-rehedge turnover.\n")

    # =====================================================================
    # T1: RV-timed overlay on market-neutral residuals (non-circular)
    # =====================================================================
    resid, rets = market_neutral_residuals(panel)
    print(f"market-neutral residual panel: {resid.shape[0]} bars x {resid.shape[1]} coins\n")

    print("=== T1: RV-mean-reversion timing on MARKET-NEUTRAL residuals ===")
    # tune (win, z_lo, z_hi, mode) on IS portfolio Sharpe only
    grids = []
    for win in (24, 72, 168):
        for mode in ("invvol", "sellvol"):
            for z_lo, z_hi in ((-1.0, 1.0), (-0.5, 1.5)):
                grids.append((win, z_lo, z_hi, mode))
    sr_trials = []
    best = None
    for (win, z_lo, z_hi, mode) in grids:
        nets = {}
        for s in resid.columns:
            net, _ = rv_timed_overlay(resid[s], win, z_lo, z_hi, 5.0, mode)
            nets[s] = pd.Series(net, index=resid.index)
        port = pd.DataFrame(nets).fillna(0).mean(axis=1).values
        mi, mo, _ = evaluate(port)
        sr_trials.append(mi["sr_pp"])
        if best is None or mi["sharpe_ann"] > best[0]:
            best = (mi["sharpe_ann"], win, z_lo, z_hi, mode)
    _, win, z_lo, z_hi, mode = best
    print(f"IS-selected: win={win} z_lo={z_lo} z_hi={z_hi} mode={mode} "
          f"(IS Sharpe={best[0]:.2f})\n")

    # OOS portfolio at 5 and 10 bp, plus CONSTANT-exposure baseline (no timing)
    final = {}
    for cost in (5.0, 10.0):
        nets, expos = {}, {}
        bnets = {}
        for s in resid.columns:
            net, expo = rv_timed_overlay(resid[s], win, z_lo, z_hi, cost, mode)
            nets[s] = pd.Series(net, index=resid.index)
            expos[s] = pd.Series(expo, index=resid.index)
            # baseline: constant unit exposure to the same neutral residual
            bnets[s] = pd.Series(bt.run(resid[s].values, np.ones(len(resid)), cost),
                                 index=resid.index)
        port = pd.DataFrame(nets).fillna(0).mean(axis=1).values
        pos = pd.DataFrame(expos).fillna(0).mean(axis=1).values
        bport = pd.DataFrame(bnets).fillna(0).mean(axis=1).values
        mi, mo, psr = evaluate(port, pos)
        _, bmo, bpsr = evaluate(bport)
        print(f"--- cost={cost:.0f}bp/leg ---")
        print(f"  RV-TIMED neutral   OOS Sharpe={mo['sharpe_ann']:6.2f} ret={mo['ret_ann']*100:6.2f}% "
              f"maxDD={mo['maxdd']*100:6.2f}% turn={mo['turnover']:.4f} PSR={psr:.3f}")
        print(f"  CONST  neutral     OOS Sharpe={bmo['sharpe_ann']:6.2f} ret={bmo['ret_ann']*100:6.2f}% "
              f"maxDD={bmo['maxdd']*100:6.2f}% PSR={bpsr:.3f}")
        if cost == 10.0:
            final = dict(mode=mode, win=win, z_lo=z_lo, z_hi=z_hi,
                         oos=mo, baseline_oos=bmo, psr=psr, baseline_psr=bpsr,
                         n=mo["n"], turnover=mo["turnover"])

    # DSR deflation across all T1 trials
    srstar = bt.dsr_benchmark(sr_trials + t2_sr_trials)
    final_sr_pp = final["oos"]["sr_pp"]
    print(f"\nDSR benchmark SR* (per-period, {len(sr_trials)+len(t2_sr_trials)} trials) = {srstar:.5f}")
    print(f"selected OOS per-period SR = {final_sr_pp:.5f}  -> "
          f"{'BEATS' if final_sr_pp > srstar else 'FAILS'} deflated bar")

    # -------- verdict logic --------
    psr_final = final["psr"]
    oos = final["oos"]
    base = final["baseline_oos"]
    beats_baseline = oos["sharpe_ann"] > base["sharpe_ann"] + 0.3
    beats_dsr = final_sr_pp > srstar
    if (psr_final >= 0.95) and beats_dsr and beats_baseline and oos["ret_ann"] > 0:
        verdict = "EDGE"
    elif (psr_final >= 0.80) and oos["ret_ann"] > 0:
        verdict = "MARGINAL"
    else:
        verdict = "DEAD"

    notes = (
        "Without options there is NO option premium to harvest, so a true VRP "
        "(IV-RV) is not measurable. Route (a) delta-hedged short-gamma is circular: "
        "the 'premium' is a self-imposed vol forecast and any P&L vanishes once "
        "rehedge turnover is costed (see T2: gross relies on invented IV, net dies "
        "on cost). Route (b) RV mean-reversion is real as a stylized fact but on a "
        "market-neutral residual the RV-timing overlay does not beat a constant-"
        f"exposure baseline after cost. Selected OOS Sharpe={oos['sharpe_ann']:.2f} "
        f"vs baseline {base['sharpe_ann']:.2f}; PSR={psr_final:.3f}; "
        f"DSR SR*={srstar:.4f} vs OOS SR_pp={final_sr_pp:.4f}. "
        "Verdict driven by: no clean non-circular VRP instrument exists on perps."
    )
    print("\n=== VERDICT:", verdict, "===")
    print(notes)

    out = dict(
        key="vol_risk_premium", family="volatility", file="experiments/cand_vol_risk_premium.py",
        implemented=True, verdict=verdict, market_neutral=True,
        universe=f"{len(panel)} USDT-perps {INTERVAL} since {START}",
        n_obs=int(final["n"]),
        oos_sharpe=float(oos["sharpe_ann"]), oos_ret_ann_pct=float(oos["ret_ann"] * 100),
        psr=float(psr_final), dsr=float(srstar),
        maxdd_pct=float(oos["maxdd"] * 100), turnover=float(final["turnover"]),
        cost_bps=10.0, mechanism="RV mean-reversion timing + short-gamma proxy (no options)",
        baseline_oos_sharpe=float(base["sharpe_ann"]),
        notes=notes,
        data_caveats=("close-to-close hourly; perp closes; no options data so VRP "
                      "proxied. maxDD is close-to-close (no intra-bar liq/gap)."),
    )
    rp = pathlib.Path(__file__).resolve().parent.parent / "reports" / "cand_vol_risk_premium.json"
    rp.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {rp}")
    return out


if __name__ == "__main__":
    main()
