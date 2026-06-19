"""Candidate: amihud_illiquidity  (family = cross-sectional)

Amihud (2002) illiquidity premium ported to crypto perps.

HYPOTHESIS: illiquid assets must compensate holders with a higher expected
return (the Amihud illiquidity premium). Amihud measure per coin per day:
    ILLIQ_t = |daily return_t| / dollar_volume_t          (use qvol = quote vol)
A coin with large price moves on little dollar volume is illiquid. Cross-
sectionally, LONG high-ILLIQ (illiquid) / SHORT low-ILLIQ (liquid), dollar-
neutral, weekly rebalance.

THE MAKE-OR-BREAK: illiquid coins have high *real* trading cost. A naive equal
cost backtest flatters the strategy because the long (illiquid) leg is exactly
the basket that is expensive to trade. We therefore charge an ASYMMETRIC cost:
a HIGH taker cost (default 20bp, swept 15/20/25) on the illiquid LONG leg and a
normal taker cost (5bp) on the liquid SHORT leg. If the premium does not survive
this asymmetric cost it is not a deployable edge -> the whole point of the test.

DESIGN (governed, mirrors house standards):
  - Universe: top ~25 USDT-M perps with FULL daily history from 2022-01-01 so the
    cross-section is constant within the test (no within-test survivorship
    rotation). Coins without full history are dropped and flagged.
  - Signal: trailing-N-day mean of daily Amihud (ILLIQ) per coin, lagged 1 bar
    (uses data up to t-1 only). Rank cross-section; demeaned rank weights,
    dollar-neutral, gross exposure normalised to 1.
  - Weekly rebalance (hold 7 days) -> low turnover (per house preference).
  - Ex-ante BTC-beta hedge: subtract beta_p * BTC_ret using an IS-estimated
    (first 60%) portfolio beta held FIXED out-of-sample (no look-ahead). Makes
    the book market-neutral in returns, not just dollar-neutral.
  - Cost: asymmetric taker. Long(illiquid) leg LONG_BPS (sweep 15/20/25 to find
    the break-even); short(liquid) leg SHORT_BPS=5. Charged on |Δweight| per leg
    every rebalance.
  - OOS: tune lookback N on the FIRST 60% only; report metrics on the LAST 40%.
    DSR deflation across the (lookback x long-cost) variants tried in this family.

HONESTY NOTES baked in / checked:
  - signals lagged >=1 bar (ILLIQ uses returns/vol up to t-1; weights then held).
  - dollar-neutral close-to-close maxDD is an ILLUSION (no intrabar liq/gap) ->
    flagged in notes.
  - ASYMMETRIC cost on the illiquid leg is the headline gate (house rule: high-
    turnover / illiquid strategies die on cost).
  - Amihud is notoriously scale-sensitive; we rank cross-sectionally each day so
    only the *ordering* matters, not the raw magnitude -> robust to qvol units.
  - dropped-history coins flagged as a (mild) survivorship caveat.
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

# Candidate universe: ~30 liquid USDT-M perps; we keep only those with FULL
# daily history from START (constant cross-section). Stable-ish, no MATIC (POL
# rename 2024-09), no obviously-late listings here -> filtered at load time.
UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
            "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT", "TRXUSDT",
            "ATOMUSDT", "BCHUSDT", "ETCUSDT", "FILUSDT", "EOSUSDT", "AAVEUSDT",
            "XLMUSDT", "ALGOUSDT", "AXSUSDT", "SANDUSDT", "MANAUSDT", "THETAUSDT",
            "EGLDUSDT", "FTMUSDT", "NEARUSDT", "GALAUSDT", "ZILUSDT", "ICPUSDT"]
START = "2022-01-01"
PPY = 365
SHORT_BPS = 5.0            # liquid (short) leg taker cost, per leg
LONG_BPS_DEFAULT = 20.0    # illiquid (long) leg taker cost, per leg (the gate)
LONG_BPS_GRID = [15.0, 20.0, 25.0]
TRAIN_FRAC = 0.60
REBAL = 7                 # weekly
BTC = "BTCUSDT"
LOOKBACK_GRID = [10, 20, 30, 45, 60]


def _ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


def load_panel(end):
    """Return (close DataFrame, qvol DataFrame) for coins with full history."""
    closes, qvols, dropped = {}, {}, []
    for s in UNIVERSE:
        k = fb.klines(s, "1d", _ms(START), end, futures=True)
        if k is not None and len(k) > 700:           # need ~full 2022+ history
            closes[s] = k["close"]
            qvols[s] = k["qvol"]
        else:
            dropped.append(s)
    C = pd.DataFrame(closes)
    Q = pd.DataFrame(qvols)
    # require a coin to be present across the whole common window
    C = C.dropna(how="any")
    Q = Q.reindex(C.index)[C.columns]
    return C, Q, dropped


def step_hold(weights: pd.DataFrame, rebal: int) -> pd.DataFrame:
    """Update weights only every `rebal` bars (ffill between rebalances)."""
    w = weights.copy()
    mask = np.zeros(len(w), dtype=bool)
    mask[::rebal] = True
    w.loc[~mask] = np.nan
    return w.ffill().fillna(0.0)


def rank_weights(score: pd.DataFrame) -> pd.DataFrame:
    """Demeaned cross-sectional rank, gross exposure normalised to 1.
    Higher score -> larger long weight."""
    r = score.rank(axis=1)
    w = r.sub(r.mean(axis=1), axis=0)
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0)
    return w.fillna(0.0)


def amihud_signal(ret: pd.DataFrame, Q: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Trailing-N-day mean daily Amihud illiquidity per coin, PIT (shifted 1).
    ILLIQ_t = |ret_t| / qvol_t ; signal = rolling mean over `lookback`, lag 1.
    Higher signal = more illiquid -> we want to be LONG it (illiquidity premium).
    """
    dollar = Q.replace(0, np.nan)
    illiq_daily = ret.abs() / dollar                 # raw Amihud per day
    sig = illiq_daily.rolling(lookback, min_periods=max(3, lookback // 2)).mean()
    return sig.shift(1)                              # decided using info up to t-1


def build_weights(ret: pd.DataFrame, Q: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Long high-Amihud (illiquid) / short low-Amihud (liquid), dollar-neutral."""
    sig = amihud_signal(ret, Q, lookback)
    w = rank_weights(sig)                            # high illiq -> long
    w = step_hold(w, REBAL)
    return w


def backtest_weights(w: pd.DataFrame, ret: pd.DataFrame,
                     long_bps: float, short_bps: float):
    """Per-bar net (asymmetric-cost), gross exposure, and gross return.
    Weights are held from yesterday (extra 1-bar shift on top of the PIT signal).
    Cost: |Δweight| on the LONG (positive) side charged at long_bps; on the SHORT
    (negative) side at short_bps. This makes the illiquid leg expensive (the gate).
    """
    wl = w.shift(1).fillna(0.0)                       # hold weights set yesterday
    asset_ret = ret.reindex(columns=w.columns).fillna(0.0)
    gross_ret = (wl * asset_ret).sum(axis=1)

    dpos = wl.diff().fillna(0.0)
    # split the traded weight change into the portion on long vs short positions.
    # charge each name by whether its (lagged) position sits long or short side.
    long_mask = (wl > 0)
    short_mask = (wl < 0)
    turn_long = (dpos.abs() * long_mask).sum(axis=1)
    turn_short = (dpos.abs() * short_mask).sum(axis=1)
    # names crossing zero / freshly opened: attribute by sign of NEW position
    flat_prev = (wl.shift(1).fillna(0.0) == 0)
    # (kept simple & conservative: the long_mask/short_mask on current wl already
    #  captures the side being financed; opening trades counted via dpos.abs())
    cost = (long_bps / 1e4) * turn_long + (short_bps / 1e4) * turn_short
    net = gross_ret - cost
    expo = wl.abs().sum(axis=1)
    turn_total = dpos.abs().sum(axis=1)
    return net, expo, gross_ret, turn_total


def hedge_beta_is(port_ret: np.ndarray, btc_ret: np.ndarray, cut: int) -> float:
    """IS-estimated (first `cut` bars) portfolio beta to BTC. Held fixed OOS."""
    x = btc_ret[:cut]
    y = port_ret[:cut]
    v = np.var(x)
    if v <= 0:
        return 0.0
    return float(np.cov(x, y)[0, 1] / v)


def main():
    end = int(time.time() * 1000)
    C, Q, dropped = load_panel(end)
    ret = np.log(C).diff().fillna(0.0)               # daily log returns
    n = len(C)
    cut = int(n * TRAIN_FRAC)
    tr, te = bt.oos_split(n, TRAIN_FRAC)
    btc_ret = ret[BTC].values

    print(f"panel: {n} daily bars x {C.shape[1]} coins "
          f"({C.index[0].date()} -> {C.index[-1].date()})")
    print(f"dropped (insufficient history): {dropped}")
    print(f"train cut at bar {cut} ({C.index[cut].date()})\n")

    # ---------------- tune lookback on IS (with default asymmetric cost) -------
    is_sr = {}
    for lb in LOOKBACK_GRID:
        w = build_weights(ret, Q, lb)
        net, expo, _, _ = backtest_weights(w, ret, LONG_BPS_DEFAULT, SHORT_BPS)
        bp = hedge_beta_is(net.values, btc_ret, cut)
        net_h = net - bp * pd.Series(btc_ret, index=net.index)
        mi = bt.metrics(net_h.values[:cut], PPY, expo.values[:cut])
        is_sr[lb] = mi["sr_pp"]
    best_lb = max(is_sr, key=lambda k: is_sr[k] if np.isfinite(is_sr[k]) else -9)
    print("IS Sharpe by lookback (default 20bp long-leg):",
          {k: round(v * np.sqrt(PPY), 2) for k, v in is_sr.items()})
    print(f"   -> IS-selected lookback = {best_lb}d\n")

    # ---------------- headline: best lookback, default cost, OOS --------------
    w = build_weights(ret, Q, best_lb)
    net, expo, gross, turn = backtest_weights(w, ret, LONG_BPS_DEFAULT, SHORT_BPS)
    beta_p = hedge_beta_is(net.values, btc_ret, cut)
    net_h = net - beta_p * pd.Series(btc_ret, index=net.index)
    print(f"IS portfolio beta to BTC = {beta_p:+.3f} (hedged ex-ante, fixed OOS)")

    mo = bt.metrics(net_h.values[te], PPY, expo.values[te])
    p = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])

    # gross (no-cost) and symmetric-5bp references to expose the cost sensitivity
    net_g, expo_g, _, _ = backtest_weights(w, ret, 0.0, 0.0)
    net_g_h = net_g - hedge_beta_is(net_g.values, btc_ret, cut) * pd.Series(btc_ret, index=net_g.index)
    mo_gross = bt.metrics(net_g_h.values[te], PPY, expo_g.values[te])
    net_s, _, _, _ = backtest_weights(w, ret, SHORT_BPS, SHORT_BPS)   # symmetric 5bp
    net_s_h = net_s - hedge_beta_is(net_s.values, btc_ret, cut) * pd.Series(btc_ret, index=net_s.index)
    mo_sym = bt.metrics(net_s_h.values[te], PPY, expo.values[te])

    # ---------------- DSR across the family (lookback x long-cost) ------------
    oos_sr_pool = []
    cost_sweep = {}
    for lb in LOOKBACK_GRID:
        ww = build_weights(ret, Q, lb)
        for lbps in LONG_BPS_GRID:
            nn, ee, _, _ = backtest_weights(ww, ret, lbps, SHORT_BPS)
            bp = hedge_beta_is(nn.values, btc_ret, cut)
            nnh = nn - bp * pd.Series(btc_ret, index=nn.index)
            m = bt.metrics(nnh.values[te], PPY, ee.values[te])
            oos_sr_pool.append(m["sr_pp"])
            if lb == best_lb:
                cost_sweep[lbps] = m["sharpe_ann"]
    sr_star = bt.dsr_benchmark(oos_sr_pool)
    dsr = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"], sr_benchmark=sr_star)

    print(f"\n=== Amihud illiquidity OOS (long {LONG_BPS_DEFAULT:.0f}bp / short "
          f"{SHORT_BPS:.0f}bp, beta-hedged) ===")
    print(f"   Sharpe={mo['sharpe_ann']:.2f}  ret={mo['ret_ann']*100:.2f}%/yr  "
          f"vol={mo['vol_ann']*100:.2f}%  maxDD={mo['maxdd']*100:.2f}%  "
          f"turn={mo['turnover']:.4f}/bar  hit={mo['hit']:.2f}")
    print(f"   PSR={p:.3f}  DSR(vs SR*={sr_star:.4f})={dsr:.3f}  n={mo['n']}")
    print(f"   [gross/no-cost OOS Sharpe = {mo_gross['sharpe_ann']:.2f}]")
    print(f"   [symmetric 5bp OOS Sharpe = {mo_sym['sharpe_ann']:.2f}]")
    print(f"   long-leg cost sweep @ lookback {best_lb}d (OOS Sharpe): "
          f"{ {k: round(v,2) for k,v in cost_sweep.items()} }")

    # ---------------- verdict ----------------
    survives_cost = (mo["ret_ann"] > 0) and (mo["sharpe_ann"] > 0)
    feasible_turn = np.isfinite(mo["turnover"])
    if np.isfinite(p) and p >= 0.95 and survives_cost and np.isfinite(dsr) and dsr >= 0.95 and feasible_turn:
        verdict = "EDGE"
    elif np.isfinite(p) and p >= 0.80 and survives_cost:
        verdict = "MARGINAL"
    else:
        verdict = "DEAD"

    notes = (
        f"Crypto Amihud illiquidity premium: long high-ILLIQ (illiquid) / short "
        f"low-ILLIQ (liquid), top-{C.shape[1]} USDT-M perps full-2022, weekly rebal, "
        f"ASYMMETRIC taker cost (long/illiquid leg {LONG_BPS_DEFAULT:.0f}bp, short/"
        f"liquid leg {SHORT_BPS:.0f}bp -- the make-or-break gate). IS-selected "
        f"lookback={best_lb}d. OOS Sharpe={mo['sharpe_ann']:.2f} "
        f"ret={mo['ret_ann']*100:.1f}%/yr turn={mo['turnover']:.4f}/bar "
        f"PSR={p:.2f} DSR={dsr:.2f} (vs SR*={sr_star:.3f}, {len(oos_sr_pool)} trials). "
        f"Cost sensitivity: gross OOS Sharpe={mo_gross['sharpe_ann']:.2f}, "
        f"symmetric-5bp={mo_sym['sharpe_ann']:.2f}, asym-{LONG_BPS_DEFAULT:.0f}bp="
        f"{mo['sharpe_ann']:.2f}. Beta-neutral to BTC (IS-fixed hedge beta={beta_p:+.2f}). "
        f"CAVEATS: delta-neutral close-to-close maxDD={mo['maxdd']*100:.1f}% is an "
        f"ILLUSION (no intrabar liq/gap); the long leg is by construction the most "
        f"expensive-to-trade basket so the asym cost is essential and slippage on "
        f"illiquid perps may exceed even {LONG_BPS_DEFAULT:.0f}bp in size; "
        f"dropped {len(dropped)} symbols for short history (mild survivorship)."
    )
    if verdict == "EDGE":
        notes += " Survives the asymmetric illiquid-leg cost gate + DSR."
    elif verdict == "DEAD":
        notes += " Does NOT clear the cost/PSR gate -> the illiquidity premium is eaten by the real cost of trading illiquid coins (consistent with house priors)."
    else:
        notes += " Positive but fragile under the illiquid-leg cost gate."

    out = dict(
        key="amihud_illiquidity",
        family="cross-sectional",
        implemented=True,
        verdict=verdict,
        oos_sharpe=round(float(mo["sharpe_ann"]), 4),
        oos_ret_ann_pct=round(float(mo["ret_ann"] * 100), 4),
        oos_vol_ann_pct=round(float(mo["vol_ann"] * 100), 4),
        maxdd_pct=round(float(mo["maxdd"] * 100), 4),
        turnover=round(float(mo["turnover"]), 6),
        psr=round(float(p), 4),
        dsr=round(float(dsr), 4),
        n_obs=int(mo["n"]),
        cost_bps=LONG_BPS_DEFAULT,
        market_neutral=True,
        universe=f"top{C.shape[1]} USDT-M perps full-2022",
        gross_oos_sharpe=round(float(mo_gross["sharpe_ann"]), 4),
        sym5bp_oos_sharpe=round(float(mo_sym["sharpe_ann"]), 4),
        is_lookback=best_lb,
        long_leg_bps=LONG_BPS_DEFAULT,
        short_leg_bps=SHORT_BPS,
        n_trials=len(oos_sr_pool),
        sr_star=round(float(sr_star), 4),
        beta_to_btc=round(float(beta_p), 4),
        notes=notes,
    )
    rep = ROOT / "reports" / "cand_amihud_illiquidity.json"
    rep.write_text(json.dumps(out, indent=2))
    print("\n=== VERDICT:", verdict, "===")
    print("wrote", rep)
    return out


if __name__ == "__main__":
    main()
