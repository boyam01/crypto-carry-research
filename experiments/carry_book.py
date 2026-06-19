"""TRACK A -- multi-coin / multi-venue CARRY BOOK (deployable).

Combines the THREE low-turnover STRUCTURAL-premia sleeves into one sized,
risk-managed, vol-targeted book on a common DAILY PnL grid:

  1. funding-carry sleeve    : delta-neutral EMA(21)+hysteresis funding harvest
                               across ~10 liquid coins (reuses carry_edge).
                               8h funding/PnL -> aggregated to DAILY.
  2. calendar-basis sleeve   : BTC/ETH quarterly cash-and-carry, hold-to-
                               convergence (reuses basis_carry_spec logic).
                               daily delta-neutral convergence stream.
  3. cross-venue funding     : Binance-vs-Bybit funding-spread, delta-neutral.
                               CRITICAL: align by CLOSE time. Binance funds 8h
                               (close=open+8h), Bybit 4h (close=open+4h). If you
                               naively merge by stamp you fabricate 150-330bp
                               phantom vol. We resample each venue's funding to a
                               common DAILY total-funding (sum over the day) and
                               difference the dailies -> alignment-safe.

DISCIPLINE: signals lagged >=1 bar; explicit cost; chronological OOS (tune IS
60%, report OOS 40%); inverse-vol sleeve weights fit on IS ONLY; vol-target the
COMBINED book to ~10%/yr using IS vol only; PSR reported; close-to-close maxDD
flagged as understating intrabar liquidation on the short legs.

Each sleeve returns a DAILY net-return series (per $1 of that sleeve's gross
notional, delta-neutral). We intersect on the COMMON LIVE WINDOW (all 3 have
data), then build the combined book.
"""
from __future__ import annotations
import sys, time, json, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import fetch_binance as fb
from engine import fetch_funding as ff
from engine import backtest as bt
import experiments.carry_edge as ce
import experiments.basis_carry_spec as bcs

START = "2023-01-01"          # broad fetch window; common-window is intersected later
TRAIN_FRAC = 0.60
PPY = 365                      # daily grid
VOL_TARGET = 0.10             # 10%/yr on the combined book
FUND_COST_BPS = 5.0           # per-leg, per-rebalance (carry sleeves are low-turnover)
XVENUE_COST_BPS = 5.0


def _ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


# --------------------------------------------------------------------------- #
# Sleeve 1: funding carry (delta-neutral, EMA+hysteresis), 8h -> daily
# --------------------------------------------------------------------------- #
def sleeve_funding_carry(start_ms, end_ms, span, thr, cost_bps=FUND_COST_BPS):
    """Reuse carry_edge. Equal-weight across coins. Return DAILY net series."""
    P = ce.panel(ce.UNIVERSE, start_ms, end_ms)
    nets = {}
    for s, df in P.items():
        sig = ce.hysteresis_signal(df["funding"].values, span, thr)
        net8h = ce.carry_net(df, sig, cost_bps)          # per-8h net
        nets[s] = pd.Series(net8h, index=df.index)
    port8h = pd.DataFrame(nets).fillna(0.0).mean(axis=1)   # equal-wt portfolio, per-8h
    # aggregate 8h -> daily: a day's return is the SUM of its three 8h legs
    # (small returns; sum is an accurate daily approximation and keeps additivity)
    daily = port8h.groupby(port8h.index.normalize()).sum()
    daily.index = daily.index.tz_convert("UTC") if daily.index.tz else daily.index
    return daily.rename("funding_carry"), len(P)


# --------------------------------------------------------------------------- #
# Sleeve 2: calendar basis (quarterly cash-and-carry, hold-to-convergence)
# --------------------------------------------------------------------------- #
def sleeve_calendar_basis(start_ms, end_ms, cost_rt=bcs.RT_COST):
    """Reuse basis_carry_spec mechanics. Build a DAILY book = mean across all
    currently-held taken contracts of (spot_ret - fut_ret). Amortize round-trip
    cost over each contract's held days."""
    streams = []
    for coin in bcs.COINS:
        spot = fb.klines(coin, "1d", start_ms, end_ms, futures=False)
        if spot is None:
            continue
        spot = spot["close"]
        for code in bcs.CODES:
            try:
                f = fb.klines(f"{coin}_{code}", "1d", start_ms, end_ms, futures=True)
            except Exception:
                continue
            if f is None or len(f) < 20:
                continue
            f = f["close"]
            exp = bcs.expiry(code)
            dte = np.array([(exp - ix).days for ix in f.index])
            hold = (dte >= bcs.MIN_DTE) & (dte <= bcs.MAX_DTE)
            if hold.sum() < 10:
                continue
            sp = spot.reindex(f.index)
            ann_basis = (f / sp - 1) * 365 / np.maximum(dte, 1)
            entry_i = np.where(hold)[0][0]
            entry_basis = float(ann_basis.iloc[entry_i])
            # entry filter: only take contango with entry basis above cost hurdle
            if entry_basis <= (cost_rt * 365 / 80):
                continue
            sret = np.log(sp / sp.shift(1)); fret = np.log(f / f.shift(1))
            dstream = (sret - fret)[hold].dropna()
            days = len(dstream)
            if days < 5:
                continue
            # amortize round-trip cost evenly over the held days
            dstream = dstream - (cost_rt / days)
            streams.append(dstream)
    if not streams:
        return pd.Series(dtype=float, name="calendar_basis"), 0
    book = pd.concat(streams).groupby(level=0).mean().sort_index()
    book.index = book.index.normalize()
    return book.rename("calendar_basis"), len(streams)


# --------------------------------------------------------------------------- #
# Sleeve 3: cross-venue funding spread (Binance vs Bybit), CLOSE-aligned
# --------------------------------------------------------------------------- #
def sleeve_xvenue_funding(start_ms, end_ms, ema_span, thr, cost_bps=XVENUE_COST_BPS):
    """Long-funding-receiver-leg on the high venue, short on the low venue, both
    delta-neutral-hedged so the COIN exposure cancels and you capture the funding
    SPREAD. Position sign from EMA-smoothed daily funding spread (lagged).

    ALIGNMENT: do NOT merge raw 8h(Binance)/4h(Bybit) stamps -- that fabricates
    phantom vol from the 4h vs 8h offset. Instead aggregate EACH venue's funding
    to a DAILY total (sum of that day's funding payments, each settled at its own
    close), then difference the dailies. Day boundary = UTC date of CLOSE."""
    coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]
    sp_daily = {}
    for c in coins:
        bnf = fb.funding_rate(c, start_ms, end_ms)               # 8h, stamped at settlement(=close)
        byf = ff.bybit_funding(c, start_ms, end_ms)              # 4h or 8h, stamped at close
        if bnf is None or byf is None or len(bnf) < 50 or len(byf) < 50:
            continue
        # Each stamp IS the close (settlement) time. Bin by UTC date and SUM the
        # day's funding -> daily total funding earned holding short-perp on each venue.
        bn_d = bnf["fundingRate"].groupby(bnf.index.normalize()).sum()
        by_d = byf["fundingRate"].groupby(byf.index.normalize()).sum()
        j = pd.concat([bn_d.rename("bn"), by_d.rename("by")], axis=1).dropna()
        if len(j) < 50:
            continue
        sp_daily[c] = (j["bn"] - j["by"])                        # daily funding spread

    if not sp_daily:
        return pd.Series(dtype=float, name="xvenue_funding"), 0

    nets = {}
    for c, spread in sp_daily.items():
        spread = spread.sort_index()
        sig = ce.hysteresis_signal(spread.values, ema_span, thr)  # lagged inside
        # PnL of capturing the spread = sig * spread (the coin legs are hedged on
        # BOTH venues, so price exposure nets; you earn the funding differential).
        turn = 2 * np.abs(np.diff(sig, prepend=0.0))             # 2 venues * 2 legs ~ amortized as 2
        net = sig * spread.values - (cost_bps / 1e4) * turn
        nets[c] = pd.Series(net, index=spread.index)
    book = pd.DataFrame(nets).fillna(0.0).mean(axis=1).sort_index()
    book.index = book.index.normalize()
    return book.rename("xvenue_funding"), len(nets)


# --------------------------------------------------------------------------- #
# IS tuning helpers
# --------------------------------------------------------------------------- #
def _is_metrics(net):
    n = len(net)
    tr, te = bt.oos_split(n, TRAIN_FRAC)
    return bt.metrics(net[tr], PPY), bt.metrics(net[te], PPY)


def tune_funding(start_ms, end_ms):
    """Pick EMA span/threshold for the funding sleeve on IS Sharpe only."""
    best = None
    for span in (9, 21):
        for thr in (0.0, 0.5e-4, 1e-4):
            s, _ = sleeve_funding_carry(start_ms, end_ms, span, thr)
            if len(s) < 50:
                continue
            mi, _ = _is_metrics(s.values)
            if best is None or (np.isfinite(mi["sharpe_ann"]) and mi["sharpe_ann"] > best[2]):
                best = (span, thr, mi["sharpe_ann"])
    return (best[0], best[1]) if best else (21, 1e-4)


def tune_xvenue(start_ms, end_ms):
    best = None
    for span in (5, 13, 21):
        for thr in (0.0, 1e-4, 3e-4):
            s, _ = sleeve_xvenue_funding(start_ms, end_ms, span, thr)
            if len(s) < 50:
                continue
            mi, _ = _is_metrics(s.values)
            if best is None or (np.isfinite(mi["sharpe_ann"]) and mi["sharpe_ann"] > best[2]):
                best = (span, thr, mi["sharpe_ann"])
    return (best[0], best[1]) if best else (13, 1e-4)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    end = int(time.time() * 1000)
    start = _ms(START)

    print("building sleeves ...")
    fspan, fthr = tune_funding(start, end)
    xspan, xthr = tune_xvenue(start, end)
    print(f"  IS-tuned funding sleeve  : ema={fspan} thr={fthr*1e4:.1f}bp")
    print(f"  IS-tuned xvenue  sleeve  : ema={xspan} thr={xthr*1e4:.1f}bp")

    s1, n1 = sleeve_funding_carry(start, end, fspan, fthr)
    s2, n2 = sleeve_calendar_basis(start, end)
    s3, n3 = sleeve_xvenue_funding(start, end, xspan, xthr)
    print(f"  funding-carry : {len(s1):4d} days, {n1} coins, {s1.index.min()} .. {s1.index.max()}")
    print(f"  calendar-basis: {len(s2):4d} days, {n2} contracts, {s2.index.min()} .. {s2.index.max()}")
    print(f"  xvenue-funding: {len(s3):4d} days, {n3} coins, {s3.index.min()} .. {s3.index.max()}")

    # ---- COMMON LIVE WINDOW (apples-to-apples; all sleeves have data) ---- #
    book = pd.concat([s1, s2, s3], axis=1)
    common = book.dropna()
    if len(common) < 60:
        print(f"\nWARNING: common window only {len(common)} days -- intersection too thin.")
    print(f"\nCOMMON LIVE WINDOW: {len(common)} days  "
          f"{common.index.min().date()} .. {common.index.max().date()}")

    cols = ["funding_carry", "calendar_basis", "xvenue_funding"]
    tr, te = bt.oos_split(len(common), TRAIN_FRAC)

    # ---- inverse-vol sleeve weights, fit on IS ONLY ---- #
    is_block = common.iloc[tr]
    inv_vol = 1.0 / is_block[cols].std()
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    w = (inv_vol / inv_vol.sum()).values
    print(f"\ninverse-vol sleeve weights (IS): "
          + ", ".join(f"{c}={wi:.3f}" for c, wi in zip(cols, w)))

    # raw inverse-vol-weighted combined daily return
    combined_raw = (common[cols].values * w).sum(axis=1)

    # ---- vol-target the combined book to 10%/yr using IS vol only ---- #
    # NOTE: sleeve returns are per $1 of DELTA-NEUTRAL GROSS notional. Delta-
    # neutral carry has tiny per-dollar daily vol, so hitting 10%/yr vol on NAV
    # demands large gross/NAV leverage. We compute the *required* scale but CAP
    # it at a deployable leverage limit -- the achievable vol is then reported
    # honestly. Sharpe/PSR are leverage-invariant, so OOS Sharpe is unchanged.
    LEV_CAP = 4.0           # max gross delta-neutral notional per $1 NAV (deployable)
    is_vol_ann = combined_raw[tr].std() * np.sqrt(PPY)
    scale_unconstrained = VOL_TARGET / is_vol_ann if is_vol_ann > 0 else 1.0
    scale = min(scale_unconstrained, LEV_CAP)
    combined = combined_raw * scale
    capped = scale < scale_unconstrained
    print(f"combined IS vol (pre-scale) = {is_vol_ann*100:.3f}%/yr")
    print(f"  scale for 10% vol = {scale_unconstrained:.1f}x  |  LEVERAGE CAP = {LEV_CAP:.0f}x"
          f"  -> applied scale = {scale:.2f}x  {'(CAPPED: vol < target)' if capped else ''}")
    ach_is_vol = combined[tr].std() * np.sqrt(PPY)
    print(f"  achievable IS vol at {scale:.2f}x = {ach_is_vol*100:.2f}%/yr"
          + ("  <-- carry-per-dollar too small to reach 10% within deployable leverage" if capped else ""))

    # ---- per-sleeve OOS metrics (each sleeve's own raw net, OOS slice) ---- #
    print("\n=== PER-SLEEVE OOS (raw, delta-neutral, per $1 sleeve notional) ===")
    print(f"{'sleeve':16} {'IS_Shrp':>8} {'OOS_Shrp':>9} {'OOS_ret%':>9} {'OOS_vol%':>9} {'maxDD%':>8} {'PSR':>6}")
    sleeve_oos = {}
    for c in cols:
        x = common[c].values
        mi = bt.metrics(x[tr], PPY)
        mo = bt.metrics(x[te], PPY)
        psr = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
        sleeve_oos[c] = dict(is_sharpe=mi["sharpe_ann"], **mo, psr=psr)
        print(f"{c:16} {mi['sharpe_ann']:8.2f} {mo['sharpe_ann']:9.2f} "
              f"{mo['ret_ann']*100:9.2f} {mo['vol_ann']*100:9.2f} {mo['maxdd']*100:8.2f} {psr:6.3f}")

    # ---- sleeve correlation (OOS) ---- #
    print("\n=== SLEEVE CORRELATION MATRIX (OOS) ===")
    corr = common[cols].iloc[te].corr()
    print(corr.round(3).to_string())

    # ---- COMBINED OOS ---- #
    mo_c = bt.metrics(combined[te], PPY)
    mi_c = bt.metrics(combined[tr], PPY)
    psr_c = bt.psr(mo_c["sr_pp"], mo_c["n"], mo_c["skew"], mo_c["kurt"])
    print("\n=== COMBINED BOOK (inverse-vol wt, vol-targeted 10%/yr) ===")
    print(f"  IS  Sharpe={mi_c['sharpe_ann']:6.2f}  ret={mi_c['ret_ann']*100:6.2f}%  vol={mi_c['vol_ann']*100:5.2f}%")
    print(f"  OOS Sharpe={mo_c['sharpe_ann']:6.2f}  ret={mo_c['ret_ann']*100:6.2f}%  vol={mo_c['vol_ann']*100:5.2f}%  "
          f"maxDD={mo_c['maxdd']*100:6.2f}%  PSR={psr_c:.3f}  ({mo_c['n']} days)")
    # full-window combined (for context)
    mo_full = bt.metrics(combined, PPY)
    print(f"  FULL-window Sharpe={mo_full['sharpe_ann']:.2f}  ret={mo_full['ret_ann']*100:.2f}%  "
          f"maxDD={mo_full['maxdd']*100:.2f}%")

    # ---- DEPLOYABLE SPEC ---- #
    print("\n" + "=" * 70)
    print("DEPLOYABLE SPEC -- multi-coin / multi-venue carry book")
    print("=" * 70)
    cap = 1_000_000
    ach_vol_oos = mo_c["vol_ann"]
    print(f"  VOL               : target {VOL_TARGET*100:.0f}%/yr, but delta-neutral carry-per-$")
    print(f"                      is small -> CAPPED at {LEV_CAP:.0f}x gross/NAV leverage.")
    print(f"                      Achievable OOS vol = {ach_vol_oos*100:.1f}%/yr (scale {scale:.2f}x).")
    print(f"  SLEEVE WEIGHTS    : (inverse-vol, IS-fit)")
    for c, wi in zip(cols, w):
        print(f"      {c:16}: {wi*100:5.1f}% of gross carry notional")
    print(f"  SIZING (per $1 NAV, after capped vol-target):")
    for c, wi in zip(cols, w):
        print(f"      {c:16}: ~${wi*scale:.2f} delta-neutral gross notional")
    print(f"  REBALANCE         : funding sleeve daily (8h funding), basis weekly +")
    print(f"                      at quarter roll, xvenue daily. Hysteresis dead-band")
    print(f"                      keeps turnover low (flip only on threshold breach).")
    print(f"  MARGIN / LIQ      : short-perp & short-future legs need a liquidation")
    print(f"                      buffer. Keep <=2x leg leverage; hold >=40-50% of leg")
    print(f"                      notional as free margin so a +30-40% spot spike does")
    print(f"                      not liquidate the short before the spot long offsets.")
    print(f"  CAPITAL REQUIRED  : on ${cap:,.0f} NAV, gross delta-neutral notional ~="
          f" ${cap*scale:,.0f}")
    print(f"                      (long-spot capital + posted margin on the short legs).")
    print(f"  HONEST NOTES      : close-to-close maxDD UNDERSTATES intrabar liquidation")
    print(f"                      risk on the short legs. Funding & xvenue carry BLEED")
    print(f"                      in adverse regimes (crowded shorts, funding flips);")
    print(f"                      calendar basis is the most ROBUST (convergence is an")
    print(f"                      exchange settlement rule, regime-independent).")
    print(f"                      Numbers are the RECENT COMMON WINDOW -- period-sensitive.")

    out = pathlib.Path(__file__).resolve().parent.parent / "reports" / "carry_book.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(
        common_days=int(len(common)),
        common_start=str(common.index.min().date()),
        common_end=str(common.index.max().date()),
        weights={c: float(wi) for c, wi in zip(cols, w)},
        vol_target=VOL_TARGET, vol_scale=float(scale),
        sleeve_oos={c: {k: float(v) for k, v in d.items()} for c, d in sleeve_oos.items()},
        corr_oos=corr.round(4).to_dict(),
        combined_oos=dict(sharpe=float(mo_c["sharpe_ann"]), ret=float(mo_c["ret_ann"]),
                          vol=float(mo_c["vol_ann"]), maxdd=float(mo_c["maxdd"]),
                          psr=float(psr_c), n=int(mo_c["n"])),
        combined_is=dict(sharpe=float(mi_c["sharpe_ann"])),
        combined_full=dict(sharpe=float(mo_full["sharpe_ann"]), maxdd=float(mo_full["maxdd"])),
    ), indent=2, default=float))
    print(f"\nwrote {out}")

    return dict(combined_oos=mo_c, psr_c=psr_c, sleeve_oos=sleeve_oos,
                corr=corr, n_common=len(common),
                cstart=common.index.min().date(), cend=common.index.max().date(),
                weights=dict(zip(cols, w)), scale=scale)


if __name__ == "__main__":
    main()
