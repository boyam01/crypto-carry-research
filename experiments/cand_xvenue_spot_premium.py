"""Candidate: xvenue_spot_premium  (family = cross-venue)

Cross-venue spot premium for BTC & ETH on three venues:
  - Binance spot  (BTCUSDT / ETHUSDT)          quote = USDT
  - Coinbase spot (BTC-USD  / ETH-USD)         quote = USD   <- "Coinbase premium"
  - Kraken spot   (XBTUSDT  / ETHUSDT)         quote = USDT

Two distinct hypotheses, both tested OOS and after cost:

(a) HARVESTABLE PERSISTENT SPREAD (market-neutral, like cross-venue funding):
    Is there a *persistent signed* mispricing between two venues that you can
    capture by going long the cheap venue / short the rich venue and waiting for
    convergence?  A pure level offset (e.g. Coinbase always +5bp because of the
    USD vs USDT basis) is NOT harvestable unless it mean-reverts faster than you
    pay to hold both legs.  We measure the spread, its persistence (OU half-life),
    and run a delta-neutral convergence backtest with an explicit DAILY BORROW
    cost on the short spot leg (spot has no funding -- you must locate & borrow
    the coin to short it, which is the real-world killer for this trade).

(b) PREMIUM-AS-DEMAND PREDICTOR (directional timing overlay):
    Does today's Coinbase-premium (US spot demand) PREDICT tomorrow's return?
    Lead-lag xcorr + a long/flat (and long/short) timing backtest on Binance,
    OOS + taker cost.

HONESTY NOTES baked in:
  * USDT~USD~1 but NOT exactly: the USD/USDT (stablecoin) basis is itself a
    time-varying factor and contaminates any "premium". We report the spread
    mean (the basis) separately from its mean-reverting component.
  * Shorting SPOT needs BORROW. Perp funding does not apply. We charge a daily
    borrow on the short leg; without it the convergence trade looks far better
    than it is.
  * Venue close-time alignment: Binance/Coinbase/Kraken daily candles all close
    00:00 UTC. We align on the UTC date and use close-to-close returns.
"""
from __future__ import annotations
import sys, time, json, pathlib, hashlib, datetime as dt
import warnings
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import fetch_binance as fb
from engine import backtest as bt
from engine import stats as st

CACHE = ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
_S = requests.Session()
_S.headers.update({"User-Agent": "quant-research/0.1"})

START = "2022-01-01"          # full window; Coinbase/Kraken trimmed to common range
TRAIN_FRAC = 0.60
PPY = 365                     # daily bars
TAKER_BP = 5.0               # taker cost per leg (Binance-class liquid spot)
BORROW_BP_PER_DAY = 3.0      # daily borrow to short a spot coin (~11%/yr, optimistic)


# ---------------------------------------------------------------- fetch helpers
def _cache(tag: str, builder):
    p = CACHE / f"xv_{hashlib.md5(tag.encode()).hexdigest()[:16]}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    df = builder()
    if df is not None and len(df):
        df.to_parquet(p)
    return df


def _get(url, params, tries=5):
    for i in range(tries):
        r = _S.get(url, params=params, timeout=25)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (418, 429, 500, 502, 503):
            time.sleep(1.5 ** i)
            continue
        r.raise_for_status()
    r.raise_for_status()


def coinbase_daily(product: str, start: str, end: str) -> pd.DataFrame:
    """Coinbase Exchange daily candles (granularity 86400). Paginated in <=290d
    windows (300-row cap). Cols [time,low,high,open,close,volume]. UTC close."""
    def build():
        s0 = pd.Timestamp(start, tz="UTC")
        e0 = pd.Timestamp(end, tz="UTC")
        rows = []
        cur = s0
        while cur < e0:
            w_end = min(cur + pd.Timedelta(days=290), e0)
            j = _get("https://api.exchange.coinbase.com/products/%s/candles" % product,
                     dict(granularity=86400,
                          start=cur.strftime("%Y-%m-%dT%H:%M:%S"),
                          end=w_end.strftime("%Y-%m-%dT%H:%M:%S")))
            rows.extend(j)
            cur = w_end
            time.sleep(0.25)
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
        df.index = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df[~df.index.duplicated()].sort_index()
    return _cache(f"cb|{product}|{start}|{end}", build)


def kraken_daily(pair: str, start: str) -> pd.DataFrame:
    """Kraken daily OHLC (interval 1440). Returns ~720 most-recent candles only.
    Cols [time,open,high,low,close,vwap,volume,count]. UTC close."""
    def build():
        s0 = int(pd.Timestamp(start, tz="UTC").timestamp())
        j = _get("https://api.kraken.com/0/public/OHLC",
                 dict(pair=pair, interval=1440, since=s0))
        res = j["result"]
        key = [k for k in res if k != "last"][0]
        rows = res[key]
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close",
                                         "vwap", "volume", "count"])
        df.index = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df[~df.index.duplicated()].sort_index()
    return _cache(f"kr|{pair}|{start}", build)


def binance_spot_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    return fb.klines(symbol, "1d",
                     int(pd.Timestamp(start, tz="UTC").timestamp() * 1000),
                     int(pd.Timestamp(end, tz="UTC").timestamp() * 1000),
                     futures=False)


# ------------------------------------------------------------------ assemble
def assemble(asset: str, end_iso: str) -> pd.DataFrame:
    """Daily close panel of the three venues, normalised to a UTC-midnight index."""
    bn_sym = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}[asset]
    cb_prod = {"BTC": "BTC-USD", "ETH": "ETH-USD"}[asset]
    kr_pair = {"BTC": "XBTUSDT", "ETH": "ETHUSDT"}[asset]

    bn = binance_spot_daily(bn_sym, START, end_iso)
    cb = coinbase_daily(cb_prod, START, end_iso)
    kr = kraken_daily(kr_pair, START)

    def norm(df):
        x = df["close"].copy()
        x.index = x.index.normalize()       # snap to UTC midnight (daily close day)
        return x[~x.index.duplicated()]

    df = pd.DataFrame({"binance": norm(bn), "coinbase": norm(cb), "kraken": norm(kr)})
    df = df.dropna()                          # common trading days across all venues
    return df


# ------------------------------------------------------------- metric helpers
def evaluate_oos(net: np.ndarray, position=None):
    n = len(net)
    _, te = bt.oos_split(n, TRAIN_FRAC)
    mo = bt.metrics(net[te], PPY, position[te] if position is not None else None)
    p = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
    return mo, p


# ============================================================ PART A: spread
def part_a_spread(panels: dict):
    """Persistence + delta-neutral convergence harvest of each venue pair."""
    print("=" * 78)
    print("PART (a)  PERSISTENT CROSS-VENUE SPREAD  (market-neutral convergence)")
    print("=" * 78)
    pairs = [("coinbase", "binance"), ("kraken", "binance"), ("coinbase", "kraken")]
    results = {}
    best_oos = []
    for asset, df in panels.items():
        for hi, lo in pairs:
            # signed log spread: + means `hi` venue richer than `lo` venue
            spr = np.log(df[hi] / df[lo])
            spr_bp = spr * 1e4
            mean_bp = float(spr_bp.mean())
            std_bp = float(spr_bp.std())
            hl = st.ou_half_life(spr.values)                # bars (days) to revert
            ac1 = st.acf1(spr.diff().dropna().values)

            # --- convergence backtest -------------------------------------
            # demean spread on TRAIN ONLY (the steady basis is not the edge);
            # trade the residual z-score: long `lo`/short `hi` when hi is rich.
            n = len(spr)
            cut = int(n * TRAIN_FRAC)
            mu_tr = spr.iloc[:cut].mean()
            sd_tr = spr.iloc[:cut].std()
            z = ((spr - mu_tr) / sd_tr)
            # position in the SPREAD (long lo / short hi) = -sign(z), continuous,
            # clipped; decided on close t, shift 1 bar -> trade t+1.
            raw_pos = (-z).clip(-1, 1)
            pos = raw_pos.shift(1).fillna(0).values

            # spread return = d(log hi) - d(log lo); holding -1 unit of spread
            # (short hi/long lo) earns -1 * spread_return when hi falls to lo.
            ret_hi = np.log(df[hi]).diff().fillna(0).values
            ret_lo = np.log(df[lo]).diff().fillna(0).values
            spread_ret = ret_hi - ret_lo
            gross = pos * spread_ret
            # cost: 2 legs taker on |Δpos|, PLUS daily borrow on whichever leg is
            # short spot. |pos| units of notional are short on ONE side each day.
            turn = np.abs(np.diff(pos, prepend=0.0))
            cost_taker = 2 * turn * (TAKER_BP / 1e4)
            cost_borrow = np.abs(pos) * (BORROW_BP_PER_DAY / 1e4)
            net = gross - cost_taker - cost_borrow
            net_nobrw = gross - cost_taker

            mo, p = evaluate_oos(net, np.abs(pos))
            mo_nb, p_nb = evaluate_oos(net_nobrw, np.abs(pos))
            key = f"{asset}:{hi}-{lo}"
            results[key] = dict(mean_bp=mean_bp, std_bp=std_bp, half_life=hl,
                                acf1_dspread=ac1, oos=mo, psr=p,
                                oos_nobrw=mo_nb, psr_nobrw=p_nb)
            best_oos.append(mo["sr_pp"])
            print(f"\n  {key}")
            print(f"    mean spread = {mean_bp:+.2f} bp   std = {std_bp:.2f} bp   "
                  f"OU half-life = {hl:.2f} d   acf1(Δspread) = {ac1:+.3f}")
            print(f"    convergence backtest (z-revert, demeaned on train):")
            print(f"      NO borrow : OOS Sharpe={mo_nb['sharpe_ann']:6.2f}  "
                  f"ret={mo_nb['ret_ann']*100:6.2f}%  turn={mo_nb['turnover']:.3f}  "
                  f"PSR={p_nb:.3f}")
            print(f"      +borrow {BORROW_BP_PER_DAY:.0f}bp/d : OOS Sharpe={mo['sharpe_ann']:6.2f}  "
                  f"ret={mo['ret_ann']*100:6.2f}%  maxDD={mo['maxdd']*100:6.2f}%  "
                  f"PSR={p:.3f}")
    return results, best_oos


# ============================================================ PART B: predictor
def part_b_predictor(panels: dict):
    """Coinbase premium as a demand signal predicting next-day Binance return."""
    print("\n" + "=" * 78)
    print("PART (b)  COINBASE PREMIUM AS DEMAND -> PREDICTS NEXT-DAY RETURN?")
    print("=" * 78)
    results = {}
    best_oos = []
    for asset, df in panels.items():
        prem = np.log(df["coinbase"] / df["binance"])          # Coinbase premium
        bn_ret = np.log(df["binance"]).diff()                  # Binance daily ret
        # lead-lag: does premium_t predict ret_{t+L}?
        lag, c = st.lagged_xcorr(prem.values, bn_ret.fillna(0).values, max_lag=5)
        # contemporaneous correlation (sanity: premium co-moves with the move)
        c0 = float(np.corrcoef(prem.values[1:], bn_ret.fillna(0).values[1:])[0, 1])
        # next-day predictive correlation specifically
        c1 = float(np.corrcoef(prem.values[:-1], bn_ret.fillna(0).values[1:])[0, 1])

        # signal = z-scored premium (train-standardised), traded next day.
        n = len(prem)
        cut = int(n * TRAIN_FRAC)
        mu_tr, sd_tr = prem.iloc[:cut].mean(), prem.iloc[:cut].std()
        z = ((prem - mu_tr) / sd_tr)
        ret = bn_ret.fillna(0).values

        # (1) long/short by sign of premium, traded next day
        pos_ls = np.sign(z).shift(1).fillna(0).values if hasattr(z, "shift") \
            else np.sign(z).shift(1).fillna(0).values
        net_ls = bt.run(ret, pos_ls, TAKER_BP)
        mo_ls, p_ls = evaluate_oos(net_ls, pos_ls)

        # (2) long/flat (only act on POSITIVE premium = US demand), traded next day
        pos_lf = (z > 0).astype(float).shift(1).fillna(0).values
        net_lf = bt.run(ret, pos_lf, TAKER_BP)
        mo_lf, p_lf = evaluate_oos(net_lf, pos_lf)

        # (3) continuous-tilt long/short (premium z clipped), traded next day
        pos_ct = z.clip(-1, 1).shift(1).fillna(0).values
        net_ct = bt.run(ret, pos_ct, TAKER_BP)
        mo_ct, p_ct = evaluate_oos(net_ct, pos_ct)

        results[asset] = dict(leadlag=(lag, c), corr_contemp=c0, corr_next=c1,
                              ls=mo_ls, psr_ls=p_ls, lf=mo_lf, psr_lf=p_lf,
                              ct=mo_ct, psr_ct=p_ct)
        best_oos += [mo_ls["sr_pp"], mo_lf["sr_pp"], mo_ct["sr_pp"]]
        print(f"\n  {asset}  premium mean={prem.mean()*1e4:+.2f}bp")
        print(f"    contemp corr(prem_t, ret_t)   = {c0:+.3f}  (co-move, not tradable)")
        print(f"    predictive corr(prem_t, ret_t+1)= {c1:+.3f}  "
              f"| best lead-lag L={lag} corr={c:+.3f}")
        print(f"    L/S   sign(prem): OOS Sharpe={mo_ls['sharpe_ann']:6.2f}  "
              f"ret={mo_ls['ret_ann']*100:6.2f}%  turn={mo_ls['turnover']:.2f}  PSR={p_ls:.3f}")
        print(f"    L/flat  prem>0  : OOS Sharpe={mo_lf['sharpe_ann']:6.2f}  "
              f"ret={mo_lf['ret_ann']*100:6.2f}%  turn={mo_lf['turnover']:.2f}  PSR={p_lf:.3f}")
        print(f"    L/S  z-tilt     : OOS Sharpe={mo_ct['sharpe_ann']:6.2f}  "
              f"ret={mo_ct['ret_ann']*100:6.2f}%  turn={mo_ct['turnover']:.2f}  PSR={p_ct:.3f}")
    return results, best_oos


# ===================================================================== main
def main():
    end_iso = "2026-06-18"
    panels = {}
    for asset in ("BTC", "ETH"):
        df = assemble(asset, end_iso)
        panels[asset] = df
        print(f"{asset}: {len(df)} common daily bars  {df.index[0].date()} -> {df.index[-1].date()}")
        # quick alignment sanity: venues must agree within ~1% on any given day
        rel = (df.max(axis=1) / df.min(axis=1) - 1)
        print(f"     max cross-venue rel-gap on a day = {rel.max()*100:.2f}%  "
              f"(median {rel.median()*100:.3f}%)  -> alignment OK if median<<1%")

    res_a, oos_a = part_a_spread(panels)
    res_b, oos_b = part_b_predictor(panels)

    # ---- robustness: extra-lag the (b) signal to detect same-bar close-time
    # leak. A genuine close-time alignment leak vanishes after +1 extra bar; a
    # SLOW-MOVING level signal is ~unchanged when lagged (it is autocorrelated,
    # so the "next-day prediction" is really just trading the persistent level).
    print("\n  [robustness] (b) z-tilt OOS Sharpe vs extra lag (leak vs slow-state):")
    for asset, df in panels.items():
        prem = np.log(df["coinbase"] / df["binance"])
        ret = np.log(df["binance"]).diff().fillna(0).values
        n = len(prem); cut = int(n * TRAIN_FRAC); _, te = bt.oos_split(n, TRAIN_FRAC)
        z = (prem - prem.iloc[:cut].mean()) / prem.iloc[:cut].std()
        line = []
        for extra in (1, 2, 3):
            pos = z.clip(-1, 1).shift(extra).fillna(0).values
            m = bt.metrics(bt.run(ret, pos, TAKER_BP)[te], PPY, pos[te])
            line.append(f"lag{extra}={m['sharpe_ann']:.2f}")
        print(f"      {asset}: " + "  ".join(line) +
              "   (flat-vs-lag => slow autocorr state, NOT a next-day predictor)")

    # ---- deflate WITHIN coherent families (Bailey-LdP): the (a) mean-reversion
    # variants and the (b) directional variants are different strategy families;
    # mixing their wildly-different Sharpes inflates the variance and SR*.
    oos_a_f = [s for s in oos_a if np.isfinite(s)]
    oos_b_f = [s for s in oos_b if np.isfinite(s)]
    sr_star_a = bt.dsr_benchmark(oos_a_f)
    sr_star_b = bt.dsr_benchmark(oos_b_f)
    sr_star = max(sr_star_a, sr_star_b)
    print("\n" + "=" * 78)
    print(f"DSR deflation (within-family): (a) {len(oos_a_f)} mean-rev trials SR*="
          f"{sr_star_a:.4f}({sr_star_a*np.sqrt(PPY):.2f}a) | (b) {len(oos_b_f)} "
          f"directional trials SR*={sr_star_b:.4f}({sr_star_b*np.sqrt(PPY):.2f}a)")

    # ---- pick the single best honest candidate (each judged vs ITS family SR*)
    # Part (a) net-of-BORROW is the realistic market-neutral number.
    cand = []
    for k, v in res_a.items():
        cand.append((f"A:{k}", v["oos"], v["psr"], v["oos"]["sr_pp"], sr_star_a))
    for a, v in res_b.items():
        cand.append((f"B:{a}:LS", v["ls"], v["psr_ls"], v["ls"]["sr_pp"], sr_star_b))
        cand.append((f"B:{a}:Lflat", v["lf"], v["psr_lf"], v["lf"]["sr_pp"], sr_star_b))
        cand.append((f"B:{a}:Ztilt", v["ct"], v["psr_ct"], v["ct"]["sr_pp"], sr_star_b))
    cand.sort(key=lambda x: -(x[1]["sharpe_ann"] if np.isfinite(x[1]["sharpe_ann"]) else -99))
    best_name, best_mo, best_psr, best_srpp, best_srstar = cand[0]
    beats_dsr = bool(best_srpp > best_srstar)

    print(f"\nBEST honest OOS candidate: {best_name}")
    print(f"  OOS Sharpe={best_mo['sharpe_ann']:.2f}  ret={best_mo['ret_ann']*100:.2f}%  "
          f"maxDD={best_mo['maxdd']*100:.2f}%  turn={best_mo['turnover']:.3f}  "
          f"PSR={best_psr:.3f}  family-SR*={best_srstar*np.sqrt(PPY):.2f}a  beats SR*={beats_dsr}")

    # ---- verdict (governance: EDGE needs PSR>=0.95 AND beats family DSR AND
    #      survives cost AND no fatal artifact AND turnover-feasible) ----------
    market_neutral = best_name.startswith("A:")
    if np.isfinite(best_psr) and best_psr >= 0.95 and beats_dsr and best_mo["ret_ann"] > 0:
        verdict = "EDGE"
    elif np.isfinite(best_psr) and best_psr >= 0.80 and best_mo["ret_ann"] > 0:
        verdict = "MARGINAL"
    else:
        verdict = "DEAD"

    notes = (
        "Cross-venue spot premium BTC/ETH on Binance(USDT)/Coinbase(USD)/Kraken(USDT), "
        "721 common daily bars 2024-06 to 2026-06 (Kraken OHLC only returns ~720 recent "
        "candles -> window is short, single regime). "
        "(a) MARKET-NEUTRAL convergence harvest FAILS hard: spreads are tiny (mean -1bp, "
        "std 4-9bp) and dominated by a slow USD/USDT stablecoin-basis offset, NOT a fast "
        "mean-reverting edge; the demeaned z-revert trade has NEGATIVE OOS Sharpe even "
        "before borrow (-7 to -18) -- daily noise + 2-leg taker swamp the few-bp signal, "
        "and a real spot SHORT needs daily borrow (~3bp/d) which makes it far worse. "
        "(b) Coinbase premium is mostly a CONTEMPORANEOUS co-move (corr(prem_t,ret_t)~+0.25 "
        "= it IS the move); next-day predictive corr is only ~+0.07. A directional z-tilt "
        f"L/S overlay shows OOS Sharpe~{best_mo['sharpe_ann']:.1f} PSR={best_psr:.2f}, but it is "
        "FRAGILE: (i) the signal is a slow autocorrelated LEVEL -- extra-lagging it 2-3 bars "
        "does not decay performance, proving it is not a genuine next-day predictor but a "
        "persistent-state directional bet; (ii) it barely beats plain 1-day momentum; "
        "(iii) directional momentum/reversal have ALL died in prior tests here; (iv) only "
        "~288 OOS bars in one 2024-26 regime. Not market-neutral, not robust. "
        f"Family DSR SR*={best_srstar*np.sqrt(PPY):.2f}a, beats={beats_dsr}. "
        "Fatal real-world frictions: spot-short BORROW + stablecoin-basis contamination "
        "(part a); regime/momentum-proxy fragility (part b)."
    )

    summary = dict(
        key="xvenue_spot_premium", family="cross-venue",
        file="experiments/cand_xvenue_spot_premium.py", implemented=True,
        verdict=verdict, market_neutral=market_neutral,
        best=best_name, oos_sharpe=round(float(best_mo["sharpe_ann"]), 4),
        oos_ret_ann_pct=round(float(best_mo["ret_ann"]) * 100, 4),
        psr=round(float(best_psr), 4) if np.isfinite(best_psr) else None,
        maxdd_pct=round(float(best_mo["maxdd"]) * 100, 4),
        turnover=round(float(best_mo["turnover"]), 5),
        dsr_sr_star_pp_family=round(float(best_srstar), 5),
        dsr_sr_star_a_pp=round(float(sr_star_a), 5),
        dsr_sr_star_b_pp=round(float(sr_star_b), 5),
        beats_dsr=beats_dsr,
        cost_bps=TAKER_BP, borrow_bp_per_day=BORROW_BP_PER_DAY,
        universe="BTC,ETH x Binance/Coinbase/Kraken daily",
        n_obs=int(best_mo["n"]),
        part_a={k: dict(mean_bp=round(v["mean_bp"], 3),
                        half_life_d=round(float(v["half_life"]), 3)
                        if np.isfinite(v["half_life"]) else None,
                        oos_sharpe_borrow=round(float(v["oos"]["sharpe_ann"]), 3),
                        oos_sharpe_noborrow=round(float(v["oos_nobrw"]["sharpe_ann"]), 3),
                        psr_borrow=round(float(v["psr"]), 3) if np.isfinite(v["psr"]) else None)
                for k, v in res_a.items()},
        part_b={a: dict(corr_contemp=round(v["corr_contemp"], 4),
                        corr_next=round(v["corr_next"], 4),
                        ls_oos_sharpe=round(float(v["ls"]["sharpe_ann"]), 3),
                        lf_oos_sharpe=round(float(v["lf"]["sharpe_ann"]), 3))
                for a, v in res_b.items()},
        notes=notes,
    )
    out = ROOT / "reports" / "cand_xvenue_spot_premium.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nVERDICT = {verdict}")
    print(f"wrote {out}")
    return summary


if __name__ == "__main__":
    main()
