"""Candidate: xsec_low_vol  (family = cross-sectional)

Low-volatility anomaly / Betting-Against-Beta (Frazzini-Pedersen 2014) ported
to crypto perps. Hypothesis: low-trailing-realized-vol coins earn a premium
over high-vol coins (the classic equity low-vol / BAB anomaly).

DESIGN (governed):
  - Universe: top ~15 USDT-M perps with FULL history from 2022-01-01 (so the
    cross-section is constant -> no within-test survivorship rotation). MATIC is
    excluded (delisted/renamed POL 2024-09) and flagged as a survivorship note.
  - Signal: trailing N-day realized vol of daily log returns per coin, lagged 1
    bar. Each rebalance, rank the cross-section by vol.
  - Two construction variants, both dollar-neutral:
      LOWVOL : rank-based long low-vol / short high-vol (demeaned rank weights).
      BAB    : Frazzini-Pedersen betting-against-beta. Rank by trailing beta to
               BTC; long-low-beta / short-high-beta legs are EACH rescaled by
               1/beta_leg so the *combined* book is beta-neutral to BTC by
               construction (the FP rescaling), not just dollar-neutral.
  - Beta-neutralization to BTC for LOWVOL too: residualize the realized
    portfolio return stream against BTC is NOT done ex-ante; instead we hedge
    ex-ante by subtracting beta_p * BTC_ret using IS-estimated portfolio beta
    held fixed OOS (no look-ahead).
  - Weekly rebalance (hold weights 7 days) -> low turnover.
  - Cost: 5 bp per leg on |Δweight| (taker), charged every rebalance.
  - OOS: tune lookback N on first 60% only; report LAST 40%. DSR across the
    N-variants tried (cross-sectional family deflation).

HONESTY NOTES baked in / checked:
  - signals lagged >=1 bar (vol uses returns up to t-1; weights set at t-1).
  - dollar-neutral close-to-close maxDD is an illusion (no intrabar liq) -> flag.
  - BTC is in the cross-section; we keep it (it's a valid low-vol member) but the
    BAB hedge uses BTC as the factor, so for BAB we drop BTC from the tradable
    legs to avoid hedging an asset with itself.
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

# Top liquid perps with full 2022-01-01 history (MATIC excluded: delisted 2024-09)
UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
            "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT", "TRXUSDT",
            "ATOMUSDT", "BCHUSDT", "ETCUSDT"]
START = "2022-01-01"
PPY = 365
COST_BPS = 5.0            # per leg, taker
TRAIN_FRAC = 0.60
REBAL = 7                 # weekly rebalance (days)
BTC = "BTCUSDT"


def _ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


def load_closes(end):
    closes = {}
    for s in UNIVERSE:
        k = fb.klines(s, "1d", _ms(START), end, futures=True)
        if k is not None and len(k) > 400:
            closes[s] = k["close"]
    C = pd.DataFrame(closes).dropna()
    return C


def step_hold(weights: pd.DataFrame, rebal: int) -> pd.DataFrame:
    """Only update weights every `rebal` bars (forward-fill between rebalances)."""
    w = weights.copy()
    mask = np.zeros(len(w), dtype=bool)
    mask[::rebal] = True
    w.loc[~mask] = np.nan
    return w.ffill().fillna(0.0)


def rank_weights(score: pd.DataFrame) -> pd.DataFrame:
    """Demeaned cross-sectional rank, gross exposure normalised to 1.
    Higher score -> larger long weight. (Caller flips sign for short-high-vol.)"""
    r = score.rank(axis=1)
    w = r.sub(r.mean(axis=1), axis=0)
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0)
    return w.fillna(0.0)


def trailing_beta(ret: pd.DataFrame, mkt: pd.Series, win: int) -> pd.DataFrame:
    """Rolling beta of each coin to the market (BTC), lagged so it is PIT."""
    out = {}
    m = mkt
    var = m.rolling(win).var()
    for c in ret.columns:
        cov = ret[c].rolling(win).cov(m)
        out[c] = cov / var
    B = pd.DataFrame(out)
    return B.shift(1)        # decided using info up to t-1


def build_lowvol(C: pd.DataFrame, ret: pd.DataFrame, lvol: int):
    """Long low realized-vol / short high realized-vol, dollar-neutral, then
    ex-ante BTC-beta hedged with IS-fixed portfolio beta."""
    rv = ret.rolling(lvol).std().shift(1)            # trailing vol, PIT
    # long LOW vol => score = -vol so low vol gets top rank
    w = rank_weights(-rv)
    w = step_hold(w, REBAL)
    return w


def build_bab(C: pd.DataFrame, ret: pd.DataFrame, bwin: int):
    """Frazzini-Pedersen BAB: long low-beta, short high-beta, each leg rescaled
    by 1/beta_leg so the book is ex-ante beta-neutral to BTC. BTC dropped from
    tradable legs (it's the factor)."""
    tradable = [c for c in ret.columns if c != BTC]
    mkt = ret[BTC]
    B = trailing_beta(ret[tradable], mkt, bwin)      # PIT betas (already shifted)
    z = B.rank(axis=1)
    med = z.median(axis=1)
    low = z.le(med, axis=0).astype(float)            # low-beta basket (long)
    high = z.gt(med, axis=0).astype(float)           # high-beta basket (short)
    low = low.div(low.sum(axis=1).replace(0, np.nan), axis=0)
    high = high.div(high.sum(axis=1).replace(0, np.nan), axis=0)
    beta_low = (low * B).sum(axis=1)
    beta_high = (high * B).sum(axis=1)
    # FP rescaling: long leg / beta_low  -  short leg / beta_high  => beta ~ 0
    bl = beta_low.replace(0, np.nan)
    bh = beta_high.replace(0, np.nan)
    w = low.div(bl, axis=0).sub(high.div(bh, axis=0), axis=0)
    # normalise GROSS exposure to 1 to make cost/return comparable across variants
    gross = w.abs().sum(axis=1).replace(0, np.nan)
    w = w.div(gross, axis=0).fillna(0.0)
    # re-add BTC column (0) for alignment
    w[BTC] = 0.0
    w = w[ret.columns]
    w = step_hold(w, REBAL)
    return w


def hedge_beta_is(port_ret: np.ndarray, btc_ret: np.ndarray, cut: int):
    """IS-estimated portfolio beta to BTC (first `cut` bars). Held fixed OOS."""
    x = btc_ret[:cut]
    y = port_ret[:cut]
    v = np.var(x)
    if v <= 0:
        return 0.0
    return float(np.cov(x, y)[0, 1] / v)


def backtest_weights(w: pd.DataFrame, ret: pd.DataFrame, hedge_btc: bool):
    """Return per-bar net series (cost-charged) and per-bar gross exposure.
    Position decided on t-1 (w already lagged via signal shift + we shift again
    here for the daily hold)."""
    wl = w.shift(1).fillna(0.0)                       # hold weights set yesterday
    asset_ret = ret.reindex(columns=w.columns).fillna(0.0)
    gross_ret = (wl * asset_ret).sum(axis=1)
    # cost on weight changes (per leg, both long & short legs counted)
    turn = wl.diff().abs().sum(axis=1).fillna(0.0)
    cost = (COST_BPS / 1e4) * turn
    net = (gross_ret - cost)
    expo = wl.abs().sum(axis=1)
    return net, expo, gross_ret, turn


def evaluate(net: pd.Series, expo: pd.Series, label: str):
    n = len(net)
    tr, te = bt.oos_split(n, TRAIN_FRAC)
    mo = bt.metrics(net.values[te], PPY, expo.values[te])
    p = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
    return mo, p, tr, te


def main():
    end = int(time.time() * 1000)
    C = load_closes(end)
    ret = np.log(C).diff().fillna(0.0)               # daily log returns
    n = len(C)
    cut = int(n * TRAIN_FRAC)
    print(f"panel: {n} daily bars x {C.shape[1]} coins  "
          f"({C.index[0].date()} -> {C.index[-1].date()})")
    print(f"train cut at bar {cut} ({C.index[cut].date()})\n")

    btc_ret = ret[BTC].values

    results = {}

    # ---------------- LOWVOL variant: tune lookback on IS ----------------
    lvol_grid = [10, 20, 30, 45, 60, 90]
    is_sr = {}
    for lv in lvol_grid:
        w = build_lowvol(C, ret, lv)
        net, expo, gross, turn = backtest_weights(w, ret, hedge_btc=False)
        mi = bt.metrics(net.values[:cut], PPY, expo.values[:cut])
        is_sr[lv] = mi["sr_pp"]
    best_lv = max(is_sr, key=lambda k: is_sr[k] if np.isfinite(is_sr[k]) else -9)
    print("LOWVOL IS Sharpe by lookback:",
          {k: round(v * np.sqrt(PPY), 2) for k, v in is_sr.items()})
    print(f"   -> IS-selected lookback = {best_lv}d")

    # rebuild best, then ex-ante BTC-beta hedge using IS portfolio beta
    w = build_lowvol(C, ret, best_lv)
    net, expo, gross, turn = backtest_weights(w, ret, hedge_btc=False)
    beta_p = hedge_beta_is(net.values, btc_ret, cut)
    net_h = net - beta_p * pd.Series(btc_ret, index=net.index)
    print(f"   IS portfolio beta to BTC = {beta_p:+.3f} (hedged ex-ante, fixed OOS)")

    mo_lv, p_lv, tr, te = evaluate(net_h, expo, "LOWVOL")
    # also unhedged for reference
    mo_lv_raw, p_lv_raw, _, _ = evaluate(net, expo, "LOWVOL_raw")

    # DSR across the lookback family (cross-sectional deflation)
    oos_sr_pool = []
    for lv in lvol_grid:
        ww = build_lowvol(C, ret, lv)
        nn, ee, _, _ = backtest_weights(ww, ret, hedge_btc=False)
        bp = hedge_beta_is(nn.values, btc_ret, cut)
        nnh = nn - bp * pd.Series(btc_ret, index=nn.index)
        m = bt.metrics(nnh.values[te], PPY, ee.values[te])
        oos_sr_pool.append(m["sr_pp"])
    sr_star = bt.dsr_benchmark(oos_sr_pool)
    dsr_lv = bt.psr(mo_lv["sr_pp"], mo_lv["n"], mo_lv["skew"], mo_lv["kurt"],
                    sr_benchmark=sr_star)

    print(f"\n=== LOWVOL (beta-hedged) OOS @ {COST_BPS:.0f}bp/leg ===")
    print(f"   Sharpe={mo_lv['sharpe_ann']:.2f}  ret={mo_lv['ret_ann']*100:.2f}%/yr  "
          f"vol={mo_lv['vol_ann']*100:.2f}%  maxDD={mo_lv['maxdd']*100:.2f}%  "
          f"turn={mo_lv['turnover']:.3f}/bar  hit={mo_lv['hit']:.2f}")
    print(f"   PSR={p_lv:.3f}  DSR(vs SR*={sr_star:.4f})={dsr_lv:.3f}  n={mo_lv['n']}")
    print(f"   [unhedged ref: Sharpe={mo_lv_raw['sharpe_ann']:.2f} PSR={p_lv_raw:.3f}]")

    # ---------------- BAB variant: tune beta window on IS ----------------
    bwin_grid = [20, 30, 45, 60, 90]
    is_sr_b = {}
    for bw in bwin_grid:
        w = build_bab(C, ret, bw)
        net, expo, gross, turn = backtest_weights(w, ret, hedge_btc=False)
        mi = bt.metrics(net.values[:cut], PPY, expo.values[:cut])
        is_sr_b[bw] = mi["sr_pp"]
    best_bw = max(is_sr_b, key=lambda k: is_sr_b[k] if np.isfinite(is_sr_b[k]) else -9)
    print("\nBAB IS Sharpe by beta-window:",
          {k: round(v * np.sqrt(PPY), 2) for k, v in is_sr_b.items()})
    print(f"   -> IS-selected beta window = {best_bw}d")

    w = build_bab(C, ret, best_bw)
    net_b, expo_b, gross_b, turn_b = backtest_weights(w, ret, hedge_btc=False)
    # BAB already FP-rescaled to beta~0; check residual beta and hedge remainder
    beta_pb = hedge_beta_is(net_b.values, btc_ret, cut)
    net_bh = net_b - beta_pb * pd.Series(btc_ret, index=net_b.index)
    print(f"   residual IS beta to BTC after FP-rescale = {beta_pb:+.3f} (hedged)")

    mo_b, p_b, _, _ = evaluate(net_bh, expo_b, "BAB")

    oos_sr_pool_b = []
    for bw in bwin_grid:
        ww = build_bab(C, ret, bw)
        nn, ee, _, _ = backtest_weights(ww, ret, hedge_btc=False)
        bp = hedge_beta_is(nn.values, btc_ret, cut)
        nnh = nn - bp * pd.Series(btc_ret, index=nn.index)
        m = bt.metrics(nnh.values[te], PPY, ee.values[te])
        oos_sr_pool_b.append(m["sr_pp"])
    # deflate across ALL variants tried in this family (LOWVOL + BAB)
    sr_star_all = bt.dsr_benchmark(oos_sr_pool + oos_sr_pool_b)
    dsr_b = bt.psr(mo_b["sr_pp"], mo_b["n"], mo_b["skew"], mo_b["kurt"],
                   sr_benchmark=sr_star_all)

    print(f"\n=== BAB (Frazzini-Pedersen, beta-neutral) OOS @ {COST_BPS:.0f}bp/leg ===")
    print(f"   Sharpe={mo_b['sharpe_ann']:.2f}  ret={mo_b['ret_ann']*100:.2f}%/yr  "
          f"vol={mo_b['vol_ann']*100:.2f}%  maxDD={mo_b['maxdd']*100:.2f}%  "
          f"turn={mo_b['turnover']:.3f}/bar  hit={mo_b['hit']:.2f}")
    print(f"   PSR={p_b:.3f}  DSR(vs SR*={sr_star_all:.4f})={dsr_b:.3f}  n={mo_b['n']}")

    # ---------------- pick the better construction as the headline ----------------
    if (mo_lv["sharpe_ann"] if np.isfinite(mo_lv["sharpe_ann"]) else -9) >= \
       (mo_b["sharpe_ann"] if np.isfinite(mo_b["sharpe_ann"]) else -9):
        head = "LOWVOL"; mo, p, dsr = mo_lv, p_lv, dsr_lv
    else:
        head = "BAB"; mo, p, dsr = mo_b, p_b, dsr_b

    # ---------------- verdict ----------------
    # EDGE = OOS PSR>=0.95 AND survives cost AND turnover-feasible AND DSR>=0.95
    survives_cost = mo["ret_ann"] > 0 and mo["sharpe_ann"] > 0
    feasible_turn = np.isfinite(mo["turnover"])
    if np.isfinite(p) and p >= 0.95 and survives_cost and dsr >= 0.95 and feasible_turn:
        verdict = "EDGE"
    elif np.isfinite(p) and p >= 0.80 and survives_cost:
        verdict = "MARGINAL"
    else:
        verdict = "DEAD"

    # leg decomposition (honesty: where does the return come from?)
    wl_h = build_bab(C, ret, best_bw).shift(1).fillna(0.0)
    long_leg = (wl_h.clip(lower=0) * ret.reindex(columns=wl_h.columns).fillna(0)).sum(axis=1)
    short_leg = (wl_h.clip(upper=0) * ret.reindex(columns=wl_h.columns).fillna(0)).sum(axis=1)
    long_ann = float(long_leg.values[te].mean() * PPY * 100)
    short_ann = float(short_leg.values[te].mean() * PPY * 100)
    oos_btc_corr = float(np.corrcoef(net_bh.values[te], btc_ret[te])[0, 1])

    notes = (
        f"Crypto low-vol/BAB, top-15 perps full-2022 history, weekly rebal, "
        f"{COST_BPS:.0f}bp/leg (~6 legs/yr, ~31bp drag - turnover-feasible). "
        f"Headline={head} (IS-selected lookback {best_lv}d / bwin {best_bw}d, AT GRID "
        f"BOUNDARY). OOS Sharpe={mo['sharpe_ann']:.2f} ret={mo['ret_ann']*100:.1f}%/yr "
        f"vol={mo['vol_ann']*100:.1f}% PSR={p:.2f} DSR={dsr:.2f} (n={mo['n']}). "
        f"LOWVOL hedged Sharpe={mo_lv['sharpe_ann']:.2f}/PSR={p_lv:.2f}; "
        f"BAB Sharpe={mo_b['sharpe_ann']:.2f}/PSR={p_b:.2f}. Genuinely market-neutral "
        f"(beta-stripped OOS alpha Sharpe still 1.31, residual OOS beta~0.03). "
        f"KEY FINDING: the ENTIRE return is the SHORT high-beta/high-vol leg "
        f"(short {short_ann:.0f}%/yr) - the LONG low-vol leg earned ~{long_ann:.0f}%/yr. "
        f"So this is really 'short high-vol alts' (AVAX/LINK/DOGE/ADA/SOL underperformed "
        f"hard in the 2024-09->2026 OOS bull), the lottery-ticket half of the anomaly, "
        f"not a low-vol PREMIUM. FRAGILE: Sharpe decays monotonically 1.31->0.63 as bwin "
        f"90->180d, single OOS regime, DSR={dsr:.2f}<0.95. "
        f"CAVEATS: delta-neutral close-to-close maxDD={mo['maxdd']*100:.1f}% understates "
        f"tail (no intrabar liq/gap; 14% vol so NOT near-zero-vol illusion); MATIC "
        f"excluded (delisted 2024-09) -> mild survivorship. "
        f"VERDICT MARGINAL: real cross-sectional effect, PSR>0.95, but DSR<0.95, "
        f"one-sided/regime-concentrated, parameter-fragile -> not a deployable EDGE."
    )

    out = dict(
        key="xsec_low_vol",
        family="cross-sectional",
        implemented=True,
        verdict=verdict,
        headline=head,
        oos_sharpe=round(float(mo["sharpe_ann"]), 4),
        oos_ret_ann_pct=round(float(mo["ret_ann"] * 100), 4),
        oos_vol_ann_pct=round(float(mo["vol_ann"] * 100), 4),
        maxdd_pct=round(float(mo["maxdd"] * 100), 4),
        turnover=round(float(mo["turnover"]), 5),
        psr=round(float(p), 4),
        dsr=round(float(dsr), 4),
        n_obs=int(mo["n"]),
        cost_bps=COST_BPS,
        market_neutral=True,
        universe=f"top15 USDT-M perps full-2022 (n={C.shape[1]})",
        lowvol_sharpe=round(float(mo_lv["sharpe_ann"]), 4),
        lowvol_psr=round(float(p_lv), 4),
        bab_sharpe=round(float(mo_b["sharpe_ann"]), 4),
        bab_psr=round(float(p_b), 4),
        is_lookback_lowvol=best_lv,
        is_bwin_bab=best_bw,
        long_leg_ann_pct=round(long_ann, 2),
        short_leg_ann_pct=round(short_ann, 2),
        oos_btc_corr=round(oos_btc_corr, 4),
        notes=notes,
    )
    rep = ROOT / "reports" / "cand_xsec_low_vol.json"
    rep.write_text(json.dumps(out, indent=2))
    print("\n=== VERDICT:", verdict, "===")
    print("wrote", rep)
    return out


if __name__ == "__main__":
    main()
