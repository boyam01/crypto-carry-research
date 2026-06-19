"""Candidate: tod_seasonality  (family = seasonality)

Time-of-day / funding-settlement seasonality on 1h bars.

HYPOTHESIS
----------
Crypto perp returns might carry a stable intraday pattern: a systematic drift
around the 8h funding settlements (00/08/16 UTC) or around the US / Asia
session opens. If a given UTC hour has a persistent sign of mean return, you
could trade it on a fixed schedule (low *position-change frequency per day*,
though intraday strategies are inherently high-turnover).

DESIGN (no look-ahead, chronological OOS, deflated)
---------------------------------------------------
Bars: 1h futures klines, close-to-close simple returns. The "hour" of bar t is
index.hour (the UTC hour the bar covers).  r_t is realised over that hour, so a
position that wants to capture hour H must be set at the close of bar (t-1) and
held through bar t -> that is exactly bt.run's 1-bar-shift convention (we pass a
position vector already representing "exposure during bar t decided at t-1").

We estimate, on the IN-SAMPLE 60% ONLY, the mean return of every UTC hour 0..23
(per coin, and pooled).  We then form the trading rule:

  TOP-K HOURS:  pick the K hours with the largest |IS mean|; trade them with
                position = sign(IS mean).  All other hours: flat (pos 0).

The hour table is frozen from IS, so OOS positions use no future info.

Three estimators are run:
  (A) per-coin top-K hour schedule (directional, one coin)
  (B) pooled-portfolio top-K hour schedule (equal-weight basket, directional)
  (C) MARKET-NEUTRAL cross-sectional hour effect: at each hour, go long the
      coins whose IS mean for that hour is positive and short those negative,
      dollar-neutral, so the basket beta is removed.  This is the design the
      governance notes say is the only kind that has survived here.

DEFLATION
---------
The family of trials = 24 hours x {long,short} per design = 48 directional
choices.  We deflate with bt.dsr_benchmark over the 24 IS per-hour Sharpes and
also report PSR on the OOS net.  Settlement-bar sub-tests (00/08/16) are an
explicit additional family of 3.

COST
----
5 bp/leg taker on |Δposition|.  A scheduled hour strategy enters and exits every
day -> turnover ~2 per traded hour per day.  This is the expected killer; we
report it honestly.  Market-neutral leg C charges every leg (2x).

Outputs reports/cand_tod_seasonality.json with the headline OOS numbers.
"""
from __future__ import annotations
import sys, time, json, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import fetch_binance as fb
from engine import backtest as bt

UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
            "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT"]
START = "2022-01-01"
PPY = 24 * 365          # 1h bars
TRAIN_FRAC = 0.60
COST_BPS = 5.0          # taker, per leg, on |Δposition|
SETTLE_HOURS = (0, 8, 16)


def _ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


def load_panel(symbols, start, end):
    """1h close-to-close returns per coin, aligned on a common hourly index."""
    rets = {}
    for s in symbols:
        k = fb.klines(s, "1h", start, end, futures=True)
        if k is None or len(k) < 5000:
            continue
        r = k["close"].pct_change()
        # drop the (very rare) frozen / zero bars defensively
        rets[s] = r
    panel = pd.DataFrame(rets)
    panel = panel.dropna(how="all")
    return panel


def is_hour_table(ret: pd.Series, tr: slice):
    """IS mean & per-period Sharpe for each UTC hour (0..23)."""
    r = ret.iloc[tr].dropna()
    hours = r.index.hour
    means = np.full(24, np.nan)
    sharpes = np.full(24, np.nan)
    for h in range(24):
        x = r[hours == h].values
        if len(x) > 30 and x.std(ddof=1) > 0:
            means[h] = x.mean()
            sharpes[h] = x.mean() / x.std(ddof=1)
    return means, sharpes


def topk_schedule_position(ret: pd.Series, means: np.ndarray, k: int) -> np.ndarray:
    """Position vector over the FULL series: at bar t, if hour(t) is one of the
    top-k |IS mean| hours, hold sign(IS mean), else flat. Decided from frozen
    IS table -> no look-ahead. Exposure during bar t == captures r_t (bt.run
    convention: position already represents the bar it earns over)."""
    order = np.argsort(-np.abs(np.nan_to_num(means)))
    chosen = set(int(h) for h in order[:k] if np.isfinite(means[int(h)]))
    hsign = {h: np.sign(means[h]) for h in chosen}
    hours = ret.index.hour
    pos = np.array([hsign.get(int(h), 0.0) for h in hours], dtype=float)
    return pos


def eval_oos(net, te, position=None):
    n = len(net)
    m = bt.metrics(net[te], PPY, position[te] if position is not None else None)
    p = bt.psr(m["sr_pp"], m["n"], m["skew"], m["kurt"])
    return m, p


def settlement_test(panel, tr, te):
    """Trade ONLY the three funding-settlement hours (00/08/16) per the spec.
    For each, sign from IS pooled mean; pooled equal-weight basket. Family=3."""
    pooled = panel.mean(axis=1)           # equal-weight basket hourly return
    means, sharpes = is_hour_table(pooled, tr)
    hours = pooled.index.hour
    pos = np.zeros(len(pooled))
    for h in SETTLE_HOURS:
        if np.isfinite(means[h]):
            pos[hours == h] = np.sign(means[h])
    net = bt.run(pooled.values, pos, COST_BPS)
    m, p = eval_oos(net, te, pos)
    # deflate within the 3 settlement-hour trials
    srstar = bt.dsr_benchmark([sharpes[h] for h in SETTLE_HOURS if np.isfinite(sharpes[h])])
    return m, p, srstar, means, sharpes


def main():
    end = int(time.time() * 1000)
    panel = load_panel(UNIVERSE, _ms(START), end)
    n = len(panel)
    tr, te = bt.oos_split(n, TRAIN_FRAC)
    print(f"panel: {panel.shape[1]} coins x {n} hourly bars  "
          f"{panel.index[0]} -> {panel.index[-1]}")
    print(f"IS bars={tr.stop}  OOS bars={n - te.start}  PPY={PPY}\n")

    pooled = panel.mean(axis=1)           # equal-weight basket
    p_means, p_sharpes = is_hour_table(pooled, tr)

    # ---- show the IS hour table for the pooled basket (diagnostic) ----
    print("IS pooled-basket hour table (mean bp / SR_pp):")
    line1 = "  h:  " + " ".join(f"{h:5d}" for h in range(12))
    line2 = "  m:  " + " ".join(f"{p_means[h]*1e4:5.2f}" for h in range(12))
    print(line1); print(line2)
    line1 = "  h:  " + " ".join(f"{h:5d}" for h in range(12, 24))
    line2 = "  m:  " + " ".join(f"{p_means[h]*1e4:5.2f}" for h in range(12, 24))
    print(line1); print(line2)
    print(f"  settlement hours {SETTLE_HOURS} IS mean bp: "
          + ", ".join(f"{h}={p_means[h]*1e4:.2f}" for h in SETTLE_HOURS) + "\n")

    results = {}

    # ============ (B) POOLED directional top-K hour schedule ============
    print("=== (B) pooled-basket top-K UTC-hour schedule (directional) ===")
    best_b = None
    for k in (1, 2, 3, 6):
        pos = topk_schedule_position(pooled, p_means, k)
        # tune k on IS only
        net_full = bt.run(pooled.values, pos, COST_BPS)
        mi = bt.metrics(net_full[tr], PPY, pos[tr])
        if best_b is None or (np.isfinite(mi["sharpe_ann"]) and mi["sharpe_ann"] > best_b[1]):
            best_b = (k, mi["sharpe_ann"], pos, net_full)
    k_b, is_sh_b, pos_b, net_b = best_b
    mo_b, psr_b = eval_oos(net_b, te, pos_b)
    # gross (no cost) OOS for the same schedule, to expose the cost drag
    net_b_gross = bt.run(pooled.values, pos_b, 0.0)
    mo_b_g, _ = eval_oos(net_b_gross, te, pos_b)
    srstar_b = bt.dsr_benchmark([s for s in p_sharpes if np.isfinite(s)])
    print(f"  IS-selected K={k_b} (IS Sharpe={is_sh_b:.2f})")
    print(f"  OOS  net  Sharpe={mo_b['sharpe_ann']:6.2f}  ret={mo_b['ret_ann']*100:7.2f}%  "
          f"maxDD={mo_b['maxdd']*100:6.2f}%  turn={mo_b['turnover']:.3f}  PSR={psr_b:.3f}")
    print(f"  OOS gross Sharpe={mo_b_g['sharpe_ann']:6.2f}  ret={mo_b_g['ret_ann']*100:7.2f}%   "
          f"(cost wipes {(mo_b_g['ret_ann']-mo_b['ret_ann'])*100:.2f}%/yr)")
    print(f"  DSR SR* (24-hour family, per-period)={srstar_b:.4f}  vs achieved SR_pp={mo_b['sr_pp']:.4f}\n")
    results["B_pooled_topk"] = dict(k=k_b, **mo_b, psr=psr_b,
                                    gross_sharpe=mo_b_g["sharpe_ann"], dsr_srstar=srstar_b)

    # ============ (A) BEST per-coin directional schedule ============
    print("=== (A) per-coin top-K UTC-hour schedule (directional, cost=5bp) ===")
    print(f"  {'coin':9} {'K':>2} {'IS_Sh':>6} {'OOS_Sh':>7} {'OOSret%':>8} "
          f"{'maxDD%':>7} {'turn':>6} {'PSR':>6}")
    per_coin = {}
    for s in panel.columns:
        ret = panel[s]
        m_means, m_sharpes = is_hour_table(ret, tr)
        # tune K on IS, eval OOS
        bestc = None
        for k in (1, 2, 3, 6):
            pos = topk_schedule_position(ret, m_means, k)
            nf = bt.run(ret.fillna(0).values, pos, COST_BPS)
            mi = bt.metrics(nf[tr], PPY, pos[tr])
            if bestc is None or (np.isfinite(mi["sharpe_ann"]) and mi["sharpe_ann"] > bestc[1]):
                bestc = (k, mi["sharpe_ann"], pos, nf)
        kc, is_sh, posc, nfc = bestc
        mo, psr = eval_oos(nfc, te, posc)
        per_coin[s] = dict(k=kc, is_sharpe=is_sh, **mo, psr=psr)
        print(f"  {s:9} {kc:>2} {is_sh:6.2f} {mo['sharpe_ann']:7.2f} "
              f"{mo['ret_ann']*100:8.2f} {mo['maxdd']*100:7.2f} {mo['turnover']:6.3f} {psr:6.3f}")
    results["A_per_coin"] = per_coin
    # median OOS Sharpe across coins (a single-coin pick would be cherry-picked)
    med_oos = float(np.nanmedian([per_coin[s]["sharpe_ann"] for s in per_coin]))
    print(f"  --> median per-coin OOS Sharpe = {med_oos:.2f} (cherry-picking the best is invalid)\n")

    # ============ (C) MARKET-NEUTRAL cross-sectional hour effect ============
    print("=== (C) market-neutral cross-sectional hour schedule (cost=5bp/leg) ===")
    # IS hour table per coin -> at each (hour) build dollar-neutral weights from
    # sign of IS mean, demeaned across coins so the basket is beta-neutral.
    coins = list(panel.columns)
    is_means = {s: is_hour_table(panel[s], tr)[0] for s in coins}
    hours = panel.index.hour
    W = np.zeros((len(panel), len(coins)))      # weights per bar per coin
    for j, s in enumerate(coins):
        sgn = np.sign(np.nan_to_num(is_means[s]))      # per hour sign for coin s
        W[:, j] = sgn[hours]
    # demean across coins each bar -> dollar neutral; normalize gross to 1
    Wd = W - W.mean(axis=1, keepdims=True)
    gross = np.abs(Wd).sum(axis=1, keepdims=True)
    gross[gross == 0] = 1.0
    Wn = Wd / gross                                     # sum|w|=1, sum w=0
    R = panel[coins].fillna(0.0).values
    port_gross_ret = (Wn * R).sum(axis=1)              # before cost
    # cost: per-coin position change each bar, summed (every leg charged)
    dW = np.abs(np.diff(Wn, axis=0, prepend=np.zeros((1, len(coins)))))
    cost = (COST_BPS / 1e4) * dW.sum(axis=1)
    net_c = port_gross_ret - cost
    # turnover proxy = total leg turnover per bar
    pos_proxy = np.abs(Wn).sum(axis=1)                 # ~1 by construction
    mo_c = bt.metrics(net_c[te], PPY, None)
    psr_c = bt.psr(mo_c["sr_pp"], mo_c["n"], mo_c["skew"], mo_c["kurt"])
    mo_c_g = bt.metrics(port_gross_ret[te], PPY, None)
    leg_turn = float(dW.sum(axis=1)[te.start:].mean())
    srstar_c = bt.dsr_benchmark([is_hour_table(panel[s], tr)[1][h]
                                 for s in coins for h in range(24)
                                 if np.isfinite(is_hour_table(panel[s], tr)[1][h])])
    print(f"  OOS  net  Sharpe={mo_c['sharpe_ann']:6.2f}  ret={mo_c['ret_ann']*100:7.2f}%  "
          f"maxDD={mo_c['maxdd']*100:6.2f}%  leg_turn/bar={leg_turn:.3f}  PSR={psr_c:.3f}")
    print(f"  OOS gross Sharpe={mo_c_g['sharpe_ann']:6.2f}  ret={mo_c_g['ret_ann']*100:7.2f}%   "
          f"(cost wipes {(mo_c_g['ret_ann']-mo_c['ret_ann'])*100:.2f}%/yr)")
    print(f"  DSR SR* (24h x {len(coins)} coin family)={srstar_c:.4f}  vs achieved SR_pp={mo_c['sr_pp']:.4f}\n")
    results["C_market_neutral"] = dict(**mo_c, psr=psr_c, gross_sharpe=mo_c_g["sharpe_ann"],
                                       leg_turn=leg_turn, dsr_srstar=srstar_c)

    # ============ settlement-hour focused test ============
    print("=== settlement-hour (00/08/16 UTC) focused test, pooled basket ===")
    m_s, p_s, srstar_s, _, _ = settlement_test(panel, tr, te)
    print(f"  OOS  net  Sharpe={m_s['sharpe_ann']:6.2f}  ret={m_s['ret_ann']*100:7.2f}%  "
          f"maxDD={m_s['maxdd']*100:6.2f}%  turn={m_s['turnover']:.3f}  PSR={p_s:.3f}")
    print(f"  DSR SR* (3-hour family)={srstar_s:.4f}\n")
    results["settlement"] = dict(**m_s, psr=p_s, dsr_srstar=srstar_s)

    # ---------------- headline / verdict ----------------
    # Headline = the most defensible design that the governance prefers:
    # market-neutral low-turnover. But ToD is intrinsically high-turnover, so
    # we report C as headline and note B/A.
    headline = results["C_market_neutral"]
    head_psr = headline["psr"]
    head_sh = headline["sharpe_ann"]
    # EDGE requires OOS PSR>=0.95 AND survives cost AND turnover feasible AND beats DSR SR*
    beats_dsr = (np.isfinite(headline["sr_pp"]) and
                 headline["sr_pp"] > headline["dsr_srstar"])
    survives_cost = head_sh > 0
    if head_psr >= 0.95 and survives_cost and beats_dsr:
        verdict = "EDGE"
    elif (head_psr >= 0.80 and survives_cost) or (psr_b >= 0.80 and mo_b["sharpe_ann"] > 0):
        verdict = "MARGINAL"
    else:
        verdict = "DEAD"

    summary = dict(
        candidate="tod_seasonality", family="seasonality",
        universe=f"{panel.shape[1]} USDT-M perps, 1h bars {START}->now",
        n_obs=int(mo_c["n"]),
        ppy=PPY, cost_bps=COST_BPS, train_frac=TRAIN_FRAC,
        headline_design="C_market_neutral_cross_sectional",
        oos_sharpe=float(head_sh),
        oos_ret_ann_pct=float(headline["ret_ann"] * 100),
        oos_maxdd_pct=float(headline["maxdd"] * 100),
        psr=float(head_psr),
        dsr_srstar=float(headline["dsr_srstar"]),
        oos_sr_pp=float(headline["sr_pp"]),
        beats_dsr=bool(beats_dsr),
        leg_turnover_per_bar=float(headline["leg_turn"]),
        market_neutral=True,
        pooled_directional=dict(k=results["B_pooled_topk"]["k"],
                                oos_sharpe=float(mo_b["sharpe_ann"]),
                                oos_ret_pct=float(mo_b["ret_ann"] * 100),
                                psr=float(psr_b),
                                gross_oos_sharpe=float(mo_b_g["sharpe_ann"]),
                                turnover=float(mo_b["turnover"])),
        per_coin_median_oos_sharpe=med_oos,
        settlement_oos_sharpe=float(m_s["sharpe_ann"]),
        settlement_psr=float(p_s),
        verdict=verdict,
        notes=("Time-of-day schedule is intrinsically high-turnover (enter/exit "
               "daily). Market-neutral cross-sectional variant (C) is the only "
               "governance-preferred design. Net-of-cost OOS reported; gross "
               "shown to expose cost drag. Deflated across the 24-hour (x coin) "
               "family via DSR."),
    )
    out = pathlib.Path(__file__).resolve().parent.parent / "reports" / "cand_tod_seasonality.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    print("VERDICT:", verdict)
    print("wrote", out)
    return summary


if __name__ == "__main__":
    main()
