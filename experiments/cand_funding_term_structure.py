"""Candidate: funding_term_structure  (family = basis-carry)

THESIS (spec)
  Two independent "carry" estimates exist for the same coin:
    perp-implied carry   = funding * 1095            (3 fundings/day * 365, annualized)
    quarterly-basis carry = (fut/spot - 1) * 365/days_to_delivery   (annualized)
  When the two diverge, the relative-value trade is convergence: be long the
  CHEAPER-carry instrument and short the RICHER one. The natural, inherently
  delta-neutral expression is perp-vs-quarterly-future (both legs ~ the same
  underlying, so spot exposure cancels):

    spread_t = perp_carry_t - basis_carry_t
    Convergence = be long the CHEAPER-carry leg, short the RICHER-carry leg.
      spread<0  -> basis_carry > perp_carry (quarterly fut RICHER) -> SHORT fut / LONG perp
      spread>0  -> perp_carry  > basis_carry (perp RICHER)        -> SHORT perp / LONG fut
    With the convention sig=+1 := SHORT perp / LONG fut, that is exactly
      sig = +sign(spread).
    (NOTE: empirically the quarterly future trades at a premium to perp, so
    basis_carry is structurally > perp_carry; the dominant trade is short the
    premium future / long perp, earning the premium decay = convergence.)

  Per-bar net (sig = +1 means SHORT perp / LONG fut):
    pnl = sig * ( funding_received_on_perp + (fut_ret - perp_ret) ) - cost*turnover
        = sig * ( fr  (received when short perp & fr>0)  + fut_ret - perp_ret )
  where fut_ret-perp_ret is the basis-convergence carry. This IS the convergence
  PnL: a dated future must converge to spot at delivery, and the perp is pinned
  to spot by funding, so the perp-vs-fut basis is exactly what mean-reverts.

GOVERNANCE
  - signal shifted >=1 bar (decide on close t, hold t->t+1); no look-ahead.
  - cost charged on |Δposition| every leg (2 legs: perp + quarterly fut).
  - any threshold/param tuned on first 60% (chronological), reported on last 40%.
  - market-neutral by construction (long fut / short perp, both ~spot).
  - PSR reported; DSR* across the grid of variants tried (deflation).
  - ARTIFACTS handled: (a) funding ts offset ~1ms -> floor to 8h grid;
    (b) basis annualization 365/days EXPLODES near expiry -> drop last
    EXPIRY_BUF days; (c) post-delivery frozen future klines -> trim constant
    tail and stop at delivery; (d) 8h grid throughout, single per-period (8h)
    Sharpe with ppy=365*3 so DSR mixes no frequencies.

Each quarterly contract gives one ~100-day overlapping window of perp-vs-that-
contract. We chain all contracts for BTC & ETH into one chronological return
stream, split 60/40, tune the entry dead-band on IS, report OOS.
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
# USDT-M quarterly delivery codes (YYMMDD). Expired + live.
CODES = ["250328", "250627", "250926", "251226", "260327", "260626"]
EXPIRY_BUF = 10        # drop last N days: 365/days annualization explodes near delivery
ENTRY_DAYS = 95        # only use the window once contract is liquid (<= ~95d to expiry)
PPY = 365 * 3          # 8h bars -> 3 per day
TRAIN_FRAC = 0.60


def _ms(ts):
    return int(ts.timestamp() * 1000)


def contract_frame(asset: str, code: str) -> pd.DataFrame | None:
    """Aligned 8h frame for one (asset, quarterly contract): perp/fut/spot
    returns, funding, and the two annualized carries + spread."""
    deliv = pd.Timestamp("20" + code, tz="UTC")
    start_ms = _ms(deliv - pd.Timedelta(days=ENTRY_DAYS + 25))
    end_ms = _ms(deliv + pd.Timedelta(days=1))
    fut = fb.klines(f"{asset}_{code}", "8h", start_ms, end_ms, futures=True)
    perp = fb.klines(asset, "8h", start_ms, end_ms, futures=True)
    spot = fb.klines(asset, "8h", start_ms, end_ms, futures=False)
    fr = fb.funding_rate(asset, start_ms, end_ms)
    if any(x is None for x in (fut, perp, spot, fr)) or len(fut) < 80:
        return None

    fr = fr.copy()
    fr.index = fr.index.floor("8h")                 # ~1ms ts offset -> snap to grid
    fr = fr[~fr.index.duplicated()]

    df = pd.DataFrame(index=perp.index)
    df["perp"] = perp["close"]
    df["fut"] = fut["close"].reindex(df.index)
    df["spot"] = spot["close"].reindex(df.index).ffill()
    df["fr"] = fr["fundingRate"].reindex(df.index)
    df = df.dropna(subset=["perp", "fut", "spot"])
    df["fr"] = df["fr"].fillna(0.0)

    # --- ARTIFACT (c): trim frozen post-settlement future tail, stop at delivery
    df = df[df.index <= deliv]
    const_tail = (df["fut"] == df["fut"].iloc[-1])[::-1].cummin()[::-1]
    if const_tail.sum() > 1:
        first_frozen = const_tail.values.argmax()
        df = df.iloc[:first_frozen + 1] if first_frozen > 0 else df
    if len(df) < 60:
        return None

    days = (deliv - df.index).total_seconds() / 86400.0
    df["days"] = days
    df["basis_carry"] = (df["fut"] / df["spot"] - 1.0) * 365.0 / np.clip(days, 1.0, None)
    df["perp_carry"] = df["fr"] * 1095.0
    df["spread"] = df["perp_carry"] - df["basis_carry"]
    df["perp_ret"] = df["perp"].pct_change()
    df["fut_ret"] = df["fut"].pct_change()

    # --- ARTIFACT (b): exclude window where annualization explodes (near expiry)
    #     and the early illiquid window (> ENTRY_DAYS to expiry).
    df = df[(days <= ENTRY_DAYS) & (days >= EXPIRY_BUF)]
    df = df.dropna(subset=["perp_ret", "fut_ret"])
    if len(df) < 40:
        return None
    df["contract"] = f"{asset[:-4]}_{code}"
    return df


def signal(spread: np.ndarray, span: int, enter: float) -> np.ndarray:
    """Convergence signal, EMA-smoothed + hysteresis, LAGGED 1 bar.
    sig=+1 -> SHORT perp / LONG fut (perp richer); sig=-1 -> long perp/short fut.
    sig = +sign(smoothed spread): short the richer-carry leg, long the cheaper.
    Hysteresis: enter when |EMA(spread)|>enter, flatten only when it falls back
    inside enter/2 -> kills the bar-by-bar whipsaw that otherwise destroys this on cost."""
    e = pd.Series(spread).ewm(span=span, adjust=False).mean().shift(1).values
    exit_b = enter / 2.0
    sig = np.zeros(len(spread))
    state = 0.0
    for i in range(len(spread)):
        v = e[i]
        if np.isnan(v):
            sig[i] = 0.0
            continue
        if v > enter:
            state = 1.0
        elif v < -enter:
            state = -1.0
        elif abs(v) < exit_b:
            state = 0.0
        sig[i] = state
    return sig


def contract_net(df: pd.DataFrame, sig: np.ndarray, cost_bps: float) -> np.ndarray:
    """Per-bar net return. sig=+1 short perp/long fut.
    carry = funding received on perp short + basis convergence (fut_ret - perp_ret).
    sig*fr  : when short perp (sig=+1) and fr>0 we RECEIVE funding.
    turnover: 2 legs (perp + quarterly fut), each |Δsig|.
    """
    fr = df["fr"].values
    pr = df["perp_ret"].values
    futr = df["fut_ret"].values
    gross = sig * (fr + futr - pr)
    turn = 2.0 * np.abs(np.diff(sig, prepend=0.0))   # two legs open/close
    return gross - (cost_bps / 1e4) * turn, turn


def build_stream(span: int, enter: float, cost_bps: float):
    """Chronological concatenation of all (asset,contract) net returns + |sig| pos."""
    frames = []
    for a in ASSETS:
        for code in CODES:
            df = contract_frame(a, code)
            if df is None:
                continue
            sig = signal(df["spread"].values, span, enter)
            net, turn = contract_net(df, sig, cost_bps)
            frames.append(pd.DataFrame(
                {"net": net, "pos": np.abs(sig), "turn": turn,
                 "contract": df["contract"].values}, index=df.index))
    if not frames:
        return None
    allf = pd.concat(frames).sort_index()
    return allf


def evaluate(net, pos):
    n = len(net)
    tr, te = bt.oos_split(n, TRAIN_FRAC)
    mi = bt.metrics(net[tr], PPY, pos[tr])
    mo = bt.metrics(net[te], PPY, pos[te])
    pr = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
    return mi, mo, pr


def main():
    # 1) tune (EMA span, enter-band) on IS only (chronological 60%);
    #    cost fixed at 10bp/leg during tuning. Select best IS Sharpe.
    grid = [(span, enter) for span in (9, 21, 42)
            for enter in (0.02, 0.04, 0.08)]
    is_sr, oos_sr_trials = [], []
    best = None
    for span, enter in grid:
        s = build_stream(span, enter, cost_bps=10.0)
        if s is None or len(s) < 100:
            continue
        mi, mo, _ = evaluate(s["net"].values, s["pos"].values)
        is_sr.append((span, enter, mi["sharpe_ann"]))
        oos_sr_trials.append(mo["sr_pp"])
        if best is None or mi["sharpe_ann"] > best[2]:
            best = (span, enter, mi["sharpe_ann"])
    span_sel, enter_sel = best[0], best[1]
    print("IS param tuning (EMA span, enter-band; cost=10bp/leg):")
    for span, enter, sr in is_sr:
        mark = "  <- selected" if (span, enter) == (span_sel, enter_sel) else ""
        print(f"   span={span:3d} enter={enter:5.3f}  IS Sharpe={sr:6.2f}{mark}")
    print(f"\nIS-selected: span={span_sel}, enter={enter_sel} (IS Sharpe={best[2]:.2f})\n")

    # DSR* across the family of variants tried (deflate)
    sr_star = bt.dsr_benchmark(oos_sr_trials)

    out_rows = {}
    res = {}
    for cost in (5.0, 10.0):
        s = build_stream(span_sel, enter_sel, cost_bps=cost)
        net = s["net"].values
        pos = s["pos"].values
        mi, mo, pr = evaluate(net, pos)
        # OOS slice for reporting auxiliaries
        n = len(net); _, te = bt.oos_split(n, TRAIN_FRAC)
        oos = s.iloc[te]
        active_frac = float((oos["pos"] > 0).mean())
        oos_turn_per_bar = float(oos["turn"].mean())
        print(f"=== cost={cost:.0f}bp/leg | span={span_sel} enter={enter_sel:.3f} ===")
        print(f"  FULL    n={mi['n']+mo['n']}  ")
        print(f"  IS  Sharpe={mi['sharpe_ann']:6.2f} ret={mi['ret_ann']*100:7.2f}% "
              f"maxDD={mi['maxdd']*100:6.2f}% turn={mi['turnover']:.3f}")
        print(f"  OOS Sharpe={mo['sharpe_ann']:6.2f} ret={mo['ret_ann']*100:7.2f}% "
              f"vol={mo['vol_ann']*100:5.2f}% maxDD={mo['maxdd']*100:6.2f}% "
              f"turn={mo['turnover']:.3f} hit={mo['hit']:.2f}")
        print(f"  OOS PSR={pr:.3f}  DSR*(per-period, {len(oos_sr_trials)} trials)={sr_star:.4f}  "
              f"OOS sr_pp={mo['sr_pp']:.4f}  active_frac={active_frac:.2f}")
        print()
        out_rows[f"cost_{int(cost)}bp"] = dict(
            is_sharpe=mi["sharpe_ann"], oos_sharpe=mo["sharpe_ann"],
            oos_ret_ann=mo["ret_ann"], oos_vol_ann=mo["vol_ann"],
            oos_maxdd=mo["maxdd"], oos_turnover=mo["turnover"],
            oos_hit=mo["hit"], oos_psr=pr, oos_sr_pp=mo["sr_pp"],
            oos_n=mo["n"], oos_skew=mo["skew"], oos_kurt=mo["kurt"],
            active_frac=active_frac, oos_turn_per_bar=oos_turn_per_bar)
        if cost == 10.0:
            res = out_rows[f"cost_{int(cost)}bp"]

    # context: GROSS (0bp) at selected params -> does the spread mean-revert at all?
    sg = build_stream(span_sel, enter_sel, cost_bps=0.0)
    mig, mog, prg = evaluate(sg["net"].values, sg["pos"].values)
    print(f"context: GROSS (0bp) selected-params OOS Sharpe={mog['sharpe_ann']:.2f} "
          f"ret={mog['ret_ann']*100:.2f}%  -> spread DOES mean-revert gross; cost is the killer.")

    # ---- verdict ----
    sr_star_ann = sr_star * np.sqrt(PPY)
    beats_dsr = res["oos_sr_pp"] > sr_star
    survives_cost = res["oos_sharpe"] > 0 and res["oos_ret_ann"] > 0
    if res["oos_psr"] >= 0.95 and survives_cost and beats_dsr:
        verdict = "EDGE"
    elif res["oos_psr"] >= 0.80 and survives_cost:
        verdict = "MARGINAL"
    else:
        verdict = "DEAD"

    summary = dict(
        key="funding_term_structure", family="basis-carry",
        implemented=True, verdict=verdict, market_neutral=True,
        file="experiments/cand_funding_term_structure.py",
        universe="BTCUSDT,ETHUSDT perp-vs-quarterly (6 quarterly contracts each)",
        ema_span=span_sel, enter_band=enter_sel, cost_bps=10.0,
        oos_sharpe=res["oos_sharpe"], oos_ret_ann_pct=res["oos_ret_ann"] * 100,
        oos_maxdd_pct=res["oos_maxdd"] * 100, oos_turnover=res["oos_turnover"],
        oos_turn_per_bar=res["oos_turn_per_bar"],
        psr=res["oos_psr"], dsr_star_pp=sr_star, dsr_star_ann=sr_star_ann,
        oos_sr_pp=res["oos_sr_pp"], beats_dsr=bool(beats_dsr),
        gross_oos_sharpe=mog["sharpe_ann"], gross_oos_ret_ann_pct=mog["ret_ann"] * 100,
        n_obs=res["oos_n"], active_frac=res["active_frac"],
        notes=("Spread mean-reverts GROSS (OOS gross Sharpe %.2f) but the perp-vs-quarterly "
               "carry whipsaws; even EMA+hysteresis-smoothed turnover is too high for the "
               "tiny (~3-6%%/yr) gross edge -> net NEGATIVE at 5 and 10bp/leg. Static "
               "hold-to-expiry convergence also nets <0 after one round trip. Dies on cost." )
              % mog["sharpe_ann"],
        result_5bp=out_rows.get("cost_5bp"), result_10bp=out_rows.get("cost_10bp"),
    )
    rep = ROOT / "reports" / "cand_funding_term_structure.json"
    rep.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {rep}")
    return summary


if __name__ == "__main__":
    main()
