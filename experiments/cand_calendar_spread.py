"""Candidate: calendar_spread  (family = basis-carry)

Two structurally distinct tests on Binance USDT-M quarterly futures (BTC, ETH),
stitched across consecutive quarterly pairs from expired+live contracts.

SETUP. For every consecutive (near, far) quarterly pair that co-trades, define
  near_basis_t = near_fut_t / spot_t - 1     (annualized via days-to-near-expiry)
  far_basis_t  = far_fut_t  / spot_t - 1     (annualized via days-to-far-expiry)
  slope_t      = far_basis_t - near_basis_t  (term-structure steepness, raw)
The calendar spread leg is  long near / short far  (or the reverse), DELTA-NEUTRAL
in spot terms (both legs ~1 unit notional of the same coin, so spot exposure ~0).

TEST A  — Calendar-spread carry to the near roll.
  Term structure is usually upward (far basis > near basis) in contango. The
  calendar spread (short far / long near) should converge: as the near contract
  approaches its delivery its basis -> 0, while the far still carries basis, so
  the SPREAD (far_px - near_px) tends to *widen* in contango... we test the
  realized PnL of holding the dollar-neutral spread from co-listing to the near
  contract's last live bar, charging full taker cost on all legs at entry+exit.
  Direction chosen ONLY on the in-sample (first 60% of pairs, chronological);
  applied unchanged out-of-sample (last 40%).

TEST B  — Basis-momentum / Carry predictor (Koijen-Moskowitz-Pedersen-Vrugt 2018).
  Does the annualized term-structure slope predict next-period SPOT return?
  Cross-sectional + time-series: each rebalance go long the coin with the
  steeper (more positive) carry, short the flatter one -> market-neutral L/S
  across {BTC,ETH}. OOS + cost. Daily rebalance on dated-future-implied carry.

GOVERNANCE: signals shifted >=1 bar; >=5bp/leg (we use 7bp, multi-leg counted
per leg); chronological 60/40 OOS; PSR + DSR deflation; frozen post-settlement
future bars trimmed; close-to-close delta-neutral DD flagged as an illusion.
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

ASSETS = ["BTCUSDT", "ETHUSDT"]
# consecutive USDT-M quarterly delivery codes (YYMMDD). Ordered chronologically.
CODES = ["220325", "220624", "220930", "221230",
         "230331", "230630", "230929", "231229",
         "240329", "240628", "240927", "241227",
         "250328", "250627", "250926", "251226",
         "260327", "260626"]
COST_BP_LEG = 7.0          # taker per leg on |position change|
PPY = 365


def _ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


def load_future(asset, code):
    """Daily close of a dated future, frozen post-settlement tail trimmed,
    truncated at delivery. Returns Series indexed UTC (close-time aligned)."""
    deliv = pd.Timestamp("20" + code, tz="UTC")
    start_ms = _ms("2021-06-01")
    end_ms = int((deliv + pd.Timedelta(days=2)).timestamp() * 1000)
    df = fb.klines(f"{asset}_{code}", "1d", start_ms, end_ms, futures=True)
    if df is None or len(df) < 30:
        return None, deliv
    s = df["close"].copy()
    s = s[s.index <= deliv]
    if len(s) < 30:
        return None, deliv
    # trim frozen post-settlement run: endpoint repeats last traded px after
    # delivery. Drop the trailing constant run, keep its first bar as exit.
    const_tail = (s == s.iloc[-1])[::-1].cummin()[::-1]
    if const_tail.sum() > 1:
        first_frozen = const_tail.values.argmax()
        s = s.iloc[:first_frozen + 1]              # keep first frozen bar as exit
    return s, deliv


def load_spot(asset):
    df = fb.klines(asset, "1d", _ms("2021-06-01"), int(time.time() * 1000), futures=False)
    return df["close"] if df is not None else None


def build_pairs():
    """For each asset, build consecutive (near, far) co-trading panels.
    Returns list of dicts with aligned daily spot/near/far closes + meta."""
    pairs = []
    for asset in ASSETS:
        spot = load_spot(asset)
        if spot is None:
            continue
        futs = {}
        for code in CODES:
            s, deliv = load_future(asset, code)
            if s is not None and len(s) >= 30:
                futs[code] = (s, deliv)
        codes_avail = [c for c in CODES if c in futs]
        for i in range(len(codes_avail) - 1):
            nc, fc = codes_avail[i], codes_avail[i + 1]
            ns, nd = futs[nc]
            fs, fd = futs[fc]
            # co-trading window: both live, before near delivery
            idx = ns.index.intersection(fs.index)
            idx = idx[idx <= nd]
            if len(idx) < 20:
                continue
            df = pd.DataFrame(index=idx)
            df["spot"] = spot.reindex(idx).ffill()
            df["near"] = ns.reindex(idx)
            df["far"] = fs.reindex(idx)
            df = df.dropna()
            if len(df) < 20:
                continue
            dtn = np.array([(nd - t).days for t in df.index], float)
            dtf = np.array([(fd - t).days for t in df.index], float)
            df["dtn"] = np.maximum(dtn, 1.0)
            df["dtf"] = np.maximum(dtf, 1.0)
            df["near_basis"] = df["near"] / df["spot"] - 1.0
            df["far_basis"] = df["far"] / df["spot"] - 1.0
            # annualized carry of each leg (basis / time-to-expiry)
            df["near_carry"] = df["near_basis"] * 365.0 / df["dtn"]
            df["far_carry"] = df["far_basis"] * 365.0 / df["dtf"]
            df["slope_ann"] = df["far_carry"] - df["near_carry"]   # term-structure slope
            pairs.append(dict(asset=asset, near=nc, far=fc, deliv_near=nd,
                              start=df.index[0], df=df))
    return pairs


# ---------------- TEST A: calendar-spread carry to the near roll -------------
def test_A(pairs):
    """Dollar-neutral calendar spread held from co-listing to near's last bar.
    Position convention: dir=+1 -> long near / short far (1 unit each).
    Per-bar spread PnL = dir*(near_ret - far_ret); spot leg cancels (both same
    coin, equal notional) -> the position is delta-neutral by construction.
    Entry+exit cost = 2 legs * 2 (round trip) * COST_BP_LEG on the notional."""
    rows = []
    for p in pairs:
        df = p["df"]
        near_ret = df["near"].pct_change().fillna(0).values
        far_ret = df["far"].pct_change().fillna(0).values
        # entry slope decides nothing yet (direction fixed globally below); we
        # record the contango-conditioned realized spread return for +1 (long
        # near / short far). Report both raw and the sign-of-entry-slope variant.
        spread_ret = near_ret - far_ret                  # dir=+1
        # one round trip on 2 legs at entry and exit:
        rt_cost = 2 * 2 * COST_BP_LEG / 1e4
        cum_long_near = np.prod(1 + spread_ret) - 1 - rt_cost
        entry_slope = df["slope_ann"].iloc[0]
        days = max((df.index[-1] - df.index[0]).days, 1)
        rows.append(dict(name=f"{p['asset'][:-4]}_{p['near']}/{p['far']}",
                         asset=p["asset"], start=p["start"],
                         entry_slope_ann=entry_slope,
                         entry_near_basis=df["near_basis"].iloc[0],
                         entry_far_basis=df["far_basis"].iloc[0],
                         days=days,
                         cum_long_near=cum_long_near,
                         cum_short_near=-(np.prod(1 + spread_ret) - 1) - rt_cost,
                         spread_ret=pd.Series(spread_ret, index=df.index)))
    rows.sort(key=lambda r: r["start"])
    return rows


# ---------------- TEST B: carry / basis-momentum predictor ------------------
def daily_carry_panel(pairs):
    """Build a daily cross-asset carry panel using, for each asset/date, the
    NEAREST live quarterly's annualized basis as the carry signal, and SPOT
    daily return as the tradeable. Stitch across pairs (use 'near' leg)."""
    # collect, per asset, daily (carry, spot_ret) from the near legs of pairs,
    # dropping the last roll-week to avoid expiry microstructure.
    recs = {a: {} for a in ASSETS}
    spot_cache = {a: load_spot(a) for a in ASSETS}
    for p in pairs:
        a = p["asset"]; df = p["df"]
        sub = df[df["dtn"] >= 5]                     # avoid last 5d before expiry
        for t in sub.index:
            recs[a][t] = float(sub.loc[t, "near_carry"])
    # daily spot returns
    panels = {}
    for a in ASSETS:
        if not recs[a]:
            continue
        carry = pd.Series(recs[a]).sort_index()
        carry = carry[~carry.index.duplicated()]
        sp = spot_cache[a]
        sret = sp.pct_change().reindex(carry.index)
        panels[a] = pd.DataFrame({"carry": carry, "spot_ret": sret}).dropna()
    return panels


def test_B(panels):
    """Cross-sectional carry: long high-carry / short low-carry across BTC,ETH.
    Signal at t uses carry observed at t-1 (shift). Market-neutral (dollar
    weights sum to 0). Charge cost on |Δweight| per asset per leg.
    Also a time-series variant: each asset long if its carry>0 else short."""
    common = None
    for a, dfp in panels.items():
        common = dfp.index if common is None else common.intersection(dfp.index)
    if common is None or len(common) < 60:
        return None
    carry = pd.DataFrame({a: panels[a]["carry"].reindex(common) for a in panels})
    sret = pd.DataFrame({a: panels[a]["spot_ret"].reindex(common) for a in panels})
    carry, sret = carry.dropna(), sret.reindex(carry.index).dropna()
    carry = carry.reindex(sret.index)
    if len(sret) < 60:
        return None
    # ---- cross-sectional demeaned carry weights, lagged 1 bar ----
    cs = carry.sub(carry.mean(axis=1), axis=0)
    w = cs.div(cs.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)  # sum|w|=1, sum w=0
    w_lag = w.shift(1).fillna(0.0)
    gross = (w_lag * sret).sum(axis=1)
    dturn = w_lag.diff().abs().sum(axis=1).fillna(w_lag.abs().sum(axis=1))
    net_xs = gross - (COST_BP_LEG / 1e4) * dturn       # per-leg cost on weight changes
    # ---- time-series sign(carry) per asset, equal weight, lagged ----
    sgn = np.sign(carry).shift(1).fillna(0.0)
    g_ts = (sgn * sret).mean(axis=1)
    t_turn = sgn.diff().abs().mean(axis=1).fillna(0.0)
    net_ts = g_ts - (COST_BP_LEG / 1e4) * t_turn
    return dict(net_xs=net_xs.values, turn_xs=dturn.values,
                net_ts=net_ts.values, turn_ts=t_turn.values,
                index=sret.index)


def evaluate(net, ppy=PPY):
    net = np.asarray(net, float)
    n = len(net)
    tr, te = bt.oos_split(n, 0.60)
    mi = bt.metrics(net[tr], ppy)
    mo = bt.metrics(net[te], ppy)
    p = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
    return mi, mo, p


def main():
    t0 = time.time()
    pairs = build_pairs()
    print(f"Built {len(pairs)} consecutive quarterly pairs across {ASSETS} "
          f"({time.time()-t0:.1f}s)\n")
    for p in pairs:
        print(f"  {p['asset'][:-4]:4} {p['near']}/{p['far']}  "
              f"{p['start'].date()}->{p['deliv_near'].date()}  "
              f"({len(p['df'])} co-bars)")

    # ===================== TEST A =====================
    print("\n" + "=" * 78)
    print("TEST A — calendar-spread carry to near roll (dollar-neutral long near/short far)")
    print("=" * 78)
    A = test_A(pairs)
    print(f"{'pair':18} {'entry_slope_ann%':>16} {'near_b%':>8} {'far_b%':>8} "
          f"{'days':>5} {'cum_LN%':>9}")
    for r in A:
        print(f"{r['name']:18} {r['entry_slope_ann']*100:16.3f} "
              f"{r['entry_near_basis']*100:8.3f} {r['entry_far_basis']*100:8.3f} "
              f"{r['days']:5d} {r['cum_long_near']*100:9.3f}")
    # term-structure stylized fact:
    slopes = np.array([r["entry_slope_ann"] for r in A])
    print(f"\nterm structure: mean entry slope(far_carry-near_carry) = "
          f"{slopes.mean()*100:.3f}%/yr | upward(far>near): {(slopes>0).mean():.2f} "
          f"of {len(slopes)} pairs")

    # IS direction selection (chronological 60%): which sign of the spread paid?
    nA = len(A)
    cut = int(nA * 0.60)
    is_rows, oos_rows = A[:cut], A[cut:]
    is_ln = np.array([r["cum_long_near"] for r in is_rows])
    chosen_dir = 1 if is_ln.mean() >= 0 else -1     # +1 long near/short far
    print(f"\nIS (first {cut} pairs): mean long-near spread = {is_ln.mean()*100:.3f}% "
          f"-> chosen direction = {'LONG near / SHORT far' if chosen_dir>0 else 'SHORT near / LONG far'}")
    oos_real = np.array([chosen_dir * (r['cum_long_near'] if chosen_dir>0 else r['cum_short_near'])
                         if False else (r['cum_long_near'] if chosen_dir>0 else r['cum_short_near'])
                         for r in oos_rows])
    # realized per-pair OOS (already cost-charged, held to roll)
    oos_real = np.array([(r['cum_long_near'] if chosen_dir > 0 else r['cum_short_near'])
                         for r in oos_rows])
    days = np.array([r["days"] for r in oos_rows], float)
    oos_ann = oos_real * 365.0 / np.maximum(days, 1)
    print(f"OOS ({len(oos_rows)} pairs): mean realized={oos_real.mean()*100:.3f}%/pair "
          f"net cost | mean annualized={oos_ann.mean()*100:.2f}% | "
          f"win={ (oos_real>0).mean():.2f} | worst={oos_real.min()*100:.2f}%")
    # per-pair realized series Sharpe (treat each pair as one independent obs)
    if len(oos_real) >= 5 and oos_real.std(ddof=1) > 0:
        sr_pp_A = oos_real.mean() / oos_real.std(ddof=1)
        from scipy import stats as _st
        skA = float(_st.skew(oos_real)); kuA = float(_st.kurtosis(oos_real, fisher=False))
        psrA = bt.psr(sr_pp_A, len(oos_real), skA, kuA)
    else:
        sr_pp_A, psrA = np.nan, np.nan
    print(f"OOS per-pair SR (one obs/pair, n={len(oos_real)}): sr={sr_pp_A:.3f}  PSR={psrA:.3f}")

    # pooled daily spread series Sharpe (OOS pairs), cost amortized at roll only
    daily_nets_oos = []
    for r in oos_rows:
        sr = r["spread_ret"].values * chosen_dir
        sr = sr.copy(); sr[-1] -= 2 * 2 * COST_BP_LEG / 1e4   # round-trip cost at exit bar
        daily_nets_oos.append(sr)
    pooled = np.concatenate(daily_nets_oos) if daily_nets_oos else np.array([])
    if len(pooled) > 10:
        mA = bt.metrics(pooled, PPY)
        pA_daily = bt.psr(mA["sr_pp"], mA["n"], mA["skew"], mA["kurt"])
        print(f"OOS pooled DAILY spread: Sharpe={mA['sharpe_ann']:.2f} "
              f"ret_ann={mA['ret_ann']*100:.2f}% maxDD={mA['maxdd']*100:.2f}% "
              f"PSR={pA_daily:.3f}  (close-to-close DD = ILLUSION, no intrabar liq)")
    else:
        mA, pA_daily = {}, np.nan

    # ===================== TEST B =====================
    print("\n" + "=" * 78)
    print("TEST B — carry predictor: cross-sectional L/S + time-series sign(carry)")
    print("=" * 78)
    panels = daily_carry_panel(pairs)
    for a, dfp in panels.items():
        print(f"  {a}: {len(dfp)} daily carry obs  "
              f"carry mean={dfp['carry'].mean()*100:.2f}%/yr "
              f"std={dfp['carry'].std()*100:.2f}%")
    B = test_B(panels)
    resB = {}
    if B is not None:
        for tag, net, turn in [("XS_carry_LS", B["net_xs"], B["turn_xs"]),
                               ("TS_sign_carry", B["net_ts"], B["turn_ts"])]:
            mi, mo, p = evaluate(net)
            resB[tag] = dict(mi=mi, mo=mo, psr=p, turn=float(np.mean(turn)))
            print(f"\n  [{tag}] IS Sharpe={mi['sharpe_ann']:6.2f} | "
                  f"OOS Sharpe={mo['sharpe_ann']:6.2f} ret={mo['ret_ann']*100:6.2f}% "
                  f"maxDD={mo['maxdd']*100:6.2f}% turn={np.mean(turn):.3f} PSR={p:.3f} n={mo['n']}")
    else:
        print("  insufficient overlapping daily carry to run Test B")

    # ===================== DSR deflation across the family =====================
    fam_sr = []
    if np.isfinite(sr_pp_A):
        fam_sr.append(sr_pp_A)
    if B is not None:
        for tag in resB:
            if np.isfinite(resB[tag]["mo"]["sr_pp"]):
                fam_sr.append(resB[tag]["mo"]["sr_pp"])
    # pad family with the variants we effectively searched (2 dirs A, 2 tests B)
    sr_star = bt.dsr_benchmark(fam_sr + [-s for s in fam_sr]) if len(fam_sr) >= 1 else 0.0
    print(f"\nDSR family SR* (deflated bar to beat, per-period) = {sr_star:.4f}")

    # ===================== VERDICT =====================
    # Best OOS candidate
    cand = []
    if np.isfinite(psrA):
        cand.append(("A_calendar_spread", sr_pp_A, psrA,
                     oos_ann.mean() if len(oos_ann) else np.nan,
                     (mA.get("maxdd", np.nan) if mA else np.nan),
                     float(np.mean([r["days"] for r in oos_rows]) and 1.0)))
    if B is not None:
        for tag in resB:
            mo = resB[tag]["mo"]
            cand.append((tag, mo["sr_pp"], resB[tag]["psr"], mo["ret_ann"],
                         mo["maxdd"], resB[tag]["turn"]))
    best = max(cand, key=lambda c: (c[2] if np.isfinite(c[2]) else -1)) if cand else None

    out = dict(
        candidate="calendar_spread", family="basis-carry",
        assets=ASSETS, n_pairs=len(pairs), cost_bp_leg=COST_BP_LEG,
        test_A=dict(
            mean_entry_slope_ann=float(slopes.mean()),
            frac_upward=float((slopes > 0).mean()),
            chosen_dir=int(chosen_dir),
            oos_n_pairs=len(oos_rows),
            oos_mean_realized_per_pair=float(oos_real.mean()) if len(oos_real) else None,
            oos_mean_annualized=float(oos_ann.mean()) if len(oos_ann) else None,
            oos_win_rate=float((oos_real > 0).mean()) if len(oos_real) else None,
            oos_worst_pair=float(oos_real.min()) if len(oos_real) else None,
            oos_perpair_sr=float(sr_pp_A) if np.isfinite(sr_pp_A) else None,
            oos_perpair_psr=float(psrA) if np.isfinite(psrA) else None,
            oos_pooled_daily_sharpe=float(mA["sharpe_ann"]) if mA else None,
            oos_pooled_daily_psr=float(pA_daily) if np.isfinite(pA_daily) else None,
        ),
        test_B={tag: dict(
            oos_sharpe=float(resB[tag]["mo"]["sharpe_ann"]),
            oos_ret_ann=float(resB[tag]["mo"]["ret_ann"]),
            oos_maxdd=float(resB[tag]["mo"]["maxdd"]),
            oos_psr=float(resB[tag]["psr"]),
            turnover=float(resB[tag]["turn"]),
            n=int(resB[tag]["mo"]["n"]),
        ) for tag in resB} if B is not None else {},
        dsr_sr_star=float(sr_star),
    )

    # ---- decide verdict ----
    # Test A is a real, low-turnover, market-neutral carry IF realized OOS is
    # positive net cost with high win rate and the per-pair PSR clears 0.95 and
    # the deflated SR* is beaten. Daily-pooled PSR is the honest gate.
    psr_gate = max([x for x in [psrA, pA_daily] if np.isfinite(x)], default=np.nan)
    oos_ok = (len(oos_real) and oos_real.mean() > 0 and (oos_real > 0).mean() >= 0.6)
    a_sr = sr_pp_A if np.isfinite(sr_pp_A) else -1
    beats_dsr_A = a_sr > sr_star
    b_best_psr = max([resB[t]["psr"] for t in resB], default=np.nan) if B else np.nan

    verdict = "DEAD"
    note = ""
    if np.isfinite(pA_daily) and pA_daily >= 0.95 and oos_ok and beats_dsr_A:
        verdict = "EDGE"
        note = "Calendar-spread carry: OOS positive net 7bp/leg, daily PSR>=0.95, beats DSR."
    elif (np.isfinite(psrA) and psrA >= 0.80) or (np.isfinite(b_best_psr) and b_best_psr >= 0.80):
        verdict = "MARGINAL"
        note = ("Calendar spread realized positive but fragile (few independent pairs / "
                "daily PSR or DSR not cleared); carry predictor weak.")
    else:
        verdict = "DEAD"
        note = "Neither calendar-spread carry nor carry-momentum clears PSR/cost/DSR OOS."

    out["verdict"] = verdict
    out["note"] = note
    out["psr_gate"] = float(psr_gate) if np.isfinite(psr_gate) else None
    out["best_candidate"] = best[0] if best else None

    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict}")
    print(note)
    print("=" * 78)

    rp = ROOT / "reports" / "cand_calendar_spread.json"
    rp.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {rp}")
    return out


if __name__ == "__main__":
    main()
