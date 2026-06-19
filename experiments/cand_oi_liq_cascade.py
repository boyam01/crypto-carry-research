"""Candidate: oi_liq_cascade  (family = microstructure)

Liquidation-cascade reversal / exhaustion fade.

Thesis: a forced-deleveraging cascade prints as a SHARP open-interest DROP
together with a LARGE 1h price move. After the forced flow clears, price tends
to snap back over the next few hours (exhaustion mean-reversion). We fade the
cascade move for k hours.

Data (ALL Binance public, no key):
  - openInterestHist  period=1h   -> ~21d / 500 bars ONLY  (HARD LIMITATION)
  - 1h futures klines              -> price / returns
  - takerlongshortRatio period=1h -> aggressor side (gating)
prewarmed by experiments/_prewarm_oi-and-taker.py into data/cache/oi_*.parquet,
taker_*.parquet ; 1h klines pulled via engine.fetch_binance (parquet-cached).

Signal at bar t (decided on CLOSE of t, info <= t only):
  doi_t   = pct change of OI over the last hour
  r_t     = 1h price return
  ENTER a fade when  doi_t <= P_x(doi over trailing window)   (OI collapse)
              AND  |r_t| >= Y                                 (large move)
  position over the NEXT k hours = -sign(r_t)  (fade the cascade)
  optional taker gate: only fade DOWN-cascades where sells dominate, and
  UP-cascades where buys dominate (true aggressor exhaustion).

Discipline:
  - signal shifted +1 bar (hold from t+1).  no look-ahead.
  - cost charged on |Δposition| (one perp leg; directional, NOT delta-neutral).
  - chronological OOS: percentile X, threshold Y, horizon k tuned on first 60%
    of the POOLED timeline; metrics reported on last 40% only.
  - PSR on OOS; DSR benchmark across the param grid actually tried.
  - pooled across coins to lift the tiny event count; per-coin reported too.

HARD LIMITATION: ~21d of OI history. This is a PILOT. Even a high number here is
NOT a confirmed edge — far too few independent cascade events. Reported as such.
"""
from __future__ import annotations
import sys, time, json, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import fetch_binance as fb
from engine import backtest as bt

CACHE = ROOT / "data" / "cache"
PPY = 365 * 24                       # hourly bars per year
TRAIN_FRAC = 0.60
COST_BPS = 5.0                       # taker, one leg (directional)
# coins that have prewarmed OI/taker AND are real liquid majors (skip the
# odd tokenised-equity / metal perps that showed up in the top-vol snapshot)
COINS = ["BTC", "ETH", "SOL", "HYPE", "ZEC"]


def load_coin(c: str):
    """Build a 1h aligned frame: ret, doi (1h OI pct change), taker ratio."""
    oi_p = CACHE / f"oi_{c}.parquet"
    tk_p = CACHE / f"taker_{c}.parquet"
    if not oi_p.exists():
        return None
    oi = pd.read_parquet(oi_p).set_index("ts").sort_index()
    sym = f"{c}USDT"
    start = int(oi.index[0].timestamp() * 1000) - 3600_000
    end = int(oi.index[-1].timestamp() * 1000) + 3600_000
    kl = fb.klines(sym, "1h", start, end, futures=True)
    if kl is None or len(kl) < 100:
        return None
    df = pd.DataFrame(index=kl.index)
    df["ret"] = kl["close"].pct_change()
    # OI is timestamped at the bar; align to kline open_time index by reindex+ffill
    oi_s = oi["sumOpenInterest"].reindex(df.index).ffill()
    df["doi"] = oi_s.pct_change()
    if tk_p.exists():
        tk = pd.read_parquet(tk_p).set_index("ts").sort_index()
        df["taker"] = tk["buySellRatio"].reindex(df.index).ffill()
    else:
        df["taker"] = np.nan
    return df.dropna(subset=["ret", "doi"])


def build_position(df: pd.DataFrame, x_pct: float, y_thr: float, k: int,
                   taker_gate: bool) -> np.ndarray:
    """Target exposure series. Entry uses only info up to bar t; held k bars.
    Returned position[t] is the exposure DECIDED at t (caller shifts +1)."""
    ret = df["ret"].values
    doi = df["doi"].values
    taker = df["taker"].values
    n = len(df)
    # trailing percentile threshold for OI drop, expanding (no future info).
    # use a rolling window of trailing doi; percentile x_pct (low tail = big drop)
    win = 72                                   # 3 days trailing
    thr = np.full(n, np.nan)
    s = pd.Series(doi)
    thr = s.rolling(win, min_periods=24).quantile(x_pct).values
    pos = np.zeros(n)
    hold = 0
    cur = 0.0
    for t in range(n):
        if hold > 0:
            pos[t] = cur
            hold -= 1
            continue
        if np.isfinite(thr[t]) and doi[t] <= thr[t] and abs(ret[t]) >= y_thr:
            side = -np.sign(ret[t])            # fade the move
            ok = True
            if taker_gate and np.isfinite(taker[t]):
                # down-cascade (ret<0) should have sells dominating (taker<1);
                # up-cascade should have buys dominating (taker>1)
                if ret[t] < 0 and taker[t] >= 1.0:
                    ok = False
                if ret[t] > 0 and taker[t] <= 1.0:
                    ok = False
            if ok and side != 0:
                cur = side
                pos[t] = cur
                hold = k - 1
    return pos


def pooled_net(frames: dict, x_pct, y_thr, k, taker_gate, cost):
    """Equal-weight pooled net return series on a common hourly grid."""
    nets = {}
    poss = {}
    for c, df in frames.items():
        pos = build_position(df, x_pct, y_thr, k, taker_gate)
        # shift +1 bar: decide on close t, hold over t+1
        pos_held = np.concatenate([[0.0], pos[:-1]])
        net = bt.run(df["ret"].values, pos_held, cost)
        nets[c] = pd.Series(net, index=df.index)
        poss[c] = pd.Series(pos_held, index=df.index)
    netdf = pd.DataFrame(nets)
    posdf = pd.DataFrame(poss)
    # equal-weight across coins that are ACTIVE (nonzero position) that bar;
    # if none active, return is 0. Average only over coins present.
    port_net = netdf.mean(axis=1)
    port_pos = posdf.abs().mean(axis=1)
    return port_net, port_pos, netdf, posdf


def evaluate(net: pd.Series, pos: pd.Series | None):
    nv = net.values
    n = len(nv)
    tr, te = bt.oos_split(n, TRAIN_FRAC)
    pv = pos.values if pos is not None else None
    mi = bt.metrics(nv[tr], PPY, pv[tr] if pv is not None else None)
    mo = bt.metrics(nv[te], PPY, pv[te] if pv is not None else None)
    p = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
    return mi, mo, p


def count_events(frames, x_pct, y_thr, k, taker_gate):
    tot = 0
    for c, df in frames.items():
        pos = build_position(df, x_pct, y_thr, k, taker_gate)
        # count entries = transitions from 0 to nonzero
        prev = np.concatenate([[0.0], pos[:-1]])
        tot += int(np.sum((pos != 0) & (prev == 0)))
    return tot


def main():
    frames = {}
    for c in COINS:
        f = load_coin(c)
        if f is not None and len(f) > 100:
            frames[c] = f
    if not frames:
        print("NO DATA")
        _write(dict(verdict="ERROR", note="no OI/price frames built"))
        return
    spans = {c: (str(f.index[0]), str(f.index[-1]), len(f)) for c, f in frames.items()}
    print("loaded coins:", list(frames.keys()))
    for c, (a, b, n) in spans.items():
        print(f"  {c:6} {a} -> {b}  ({n} 1h bars)")
    n_bars = len(next(iter(frames.values())))
    print(f"\nHARD LIMITATION: ~{n_bars} hourly bars (~{n_bars/24:.0f}d). PILOT ONLY.\n")

    # ---- IS grid search (tune on first 60% of POOLED timeline only) ----
    x_grid = [0.02, 0.05, 0.10]          # OI-drop percentile (low tail)
    y_grid = [0.004, 0.007, 0.012]       # |1h ret| threshold
    k_grid = [2, 4, 6]                   # fade horizon (hours)
    gate_grid = [False, True]

    trials = []
    best = None
    for x_pct in x_grid:
        for y_thr in y_grid:
            for k in k_grid:
                for gate in gate_grid:
                    pn, pp, _, _ = pooled_net(frames, x_pct, y_thr, k, gate, COST_BPS)
                    mi, mo, _ = evaluate(pn, pp)
                    ev = count_events(frames, x_pct, y_thr, k, gate)
                    trials.append(mi["sr_pp"])
                    # selection metric = IS Sharpe, require >= some events to be real
                    if np.isfinite(mi["sharpe_ann"]) and ev >= 8:
                        if best is None or mi["sharpe_ann"] > best["is_shrp"]:
                            best = dict(x=x_pct, y=y_thr, k=k, gate=gate,
                                        is_shrp=mi["sharpe_ann"], n_events=ev)
    if best is None:
        print("NO param combo produced >=8 IS cascade events -> NO SIGNAL. KILL.")
        _write(dict(verdict="DEAD", note="no param combo yields >=8 in-sample cascade events; "
                    "~21d OI history too thin", n_obs=n_bars))
        return

    print(f"IS-selected: x_pct={best['x']} y_thr={best['y']*1e4:.0f}bp k={best['k']}h "
          f"gate={best['gate']} | IS Sharpe={best['is_shrp']:.2f} | IS events={best['n_events']}\n")

    # ---- OOS evaluation at selected params ----
    pn, pp, netdf, posdf = pooled_net(frames, best["x"], best["y"], best["k"],
                                      best["gate"], COST_BPS)
    mi, mo, p = evaluate(pn, pp)
    # OOS event count
    te_slice = bt.oos_split(len(pn), TRAIN_FRAC)[1]
    oos_events = 0
    for c, df in frames.items():
        pos = build_position(df, best["x"], best["y"], best["k"], best["gate"])
        prev = np.concatenate([[0.0], pos[:-1]])
        ent = (pos != 0) & (prev == 0)
        oos_events += int(ent[te_slice].sum())

    # DSR benchmark across the grid actually tried (deflate)
    sr_star = bt.dsr_benchmark(trials)
    dsr = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"], sr_benchmark=sr_star)

    print("=== POOLED OOS (last 40% of timeline) ===")
    print(f"  n_bars={mo['n']}  OOS cascade entries={oos_events}")
    print(f"  Sharpe_ann={mo['sharpe_ann']:.2f}  ret_ann={mo['ret_ann']*100:.2f}%  "
          f"vol_ann={mo['vol_ann']*100:.2f}%")
    print(f"  maxDD={mo['maxdd']*100:.2f}%  hit={mo['hit']:.3f}  turnover={mo['turnover']:.4f}")
    print(f"  PSR={p:.3f}  DSR_SR*(pp)={sr_star:.4f}  DSR-PSR={dsr:.3f}")

    # ---- per-coin OOS (directional, cost 5bp) ----
    print("\n--- per-coin OOS, selected params ---")
    print(f"{'coin':6} {'OOS_Shrp':>9} {'OOS_ret%':>9} {'PSR':>6} {'entries':>7}")
    per_coin = {}
    for c, df in frames.items():
        pos = build_position(df, best["x"], best["y"], best["k"], best["gate"])
        pos_held = np.concatenate([[0.0], pos[:-1]])
        net = bt.run(df["ret"].values, pos_held, COST_BPS)
        s = pd.Series(net, index=df.index)
        mi_c, mo_c, p_c = evaluate(s, pd.Series(pos_held, index=df.index))
        prev = np.concatenate([[0.0], pos[:-1]])
        ent = (pos != 0) & (prev == 0)
        ent_oos = int(ent[bt.oos_split(len(pos), TRAIN_FRAC)[1]].sum())
        per_coin[c] = dict(oos_sharpe=mo_c["sharpe_ann"], oos_ret=mo_c["ret_ann"],
                           psr=p_c, oos_entries=ent_oos)
        print(f"{c:6} {mo_c['sharpe_ann']:9.2f} {mo_c['ret_ann']*100:9.2f} "
              f"{p_c:6.3f} {ent_oos:7d}")

    # ---- verdict ----
    # Honest gate: too few OOS events => cannot be EDGE no matter the Sharpe.
    if oos_events < 5:
        verdict = "DEAD"
        note = (f"PILOT only (~{n_bars/24:.0f}d OI). Only {oos_events} OOS cascade "
                f"entries -> statistically empty; OOS Sharpe={mo['sharpe_ann']:.2f} "
                f"is noise. KILL per data-thinness rule.")
    elif p >= 0.95 and dsr >= 0.95 and mo["sharpe_ann"] > 0:
        verdict = "MARGINAL"   # NEVER EDGE on ~21d: hard cap at MARGINAL
        note = (f"PILOT (~{n_bars/24:.0f}d OI history): OOS Sharpe={mo['sharpe_ann']:.2f} "
                f"PSR={p:.2f} DSR={dsr:.2f} over only {oos_events} OOS events. "
                f"Suggestive fade but NOT a confirmed edge - far too few independent "
                f"cascades. Capped at MARGINAL by hard ~21d-window limitation.")
    elif mo["sharpe_ann"] > 0 and p >= 0.80:
        verdict = "MARGINAL"
        note = (f"PILOT (~{n_bars/24:.0f}d): OOS Sharpe={mo['sharpe_ann']:.2f} PSR={p:.2f} "
                f"DSR={dsr:.2f}, {oos_events} OOS events. Weak/fragile, not deflation-proof.")
    else:
        verdict = "DEAD"
        note = (f"PILOT (~{n_bars/24:.0f}d): OOS Sharpe={mo['sharpe_ann']:.2f} PSR={p:.2f} "
                f"DSR={dsr:.2f}. Fade does not survive cost/deflation. KILL.")

    print(f"\nVERDICT: {verdict}\n{note}")

    out = dict(
        key="oi_liq_cascade", family="microstructure", file="experiments/cand_oi_liq_cascade.py",
        implemented=True, verdict=verdict,
        universe="+".join(frames.keys()), n_obs=int(mo["n"]),
        oos_sharpe=float(mo["sharpe_ann"]), oos_ret_ann_pct=float(mo["ret_ann"] * 100),
        psr=float(p), dsr=float(dsr), sr_star_pp=float(sr_star),
        oos_events=int(oos_events), is_params=dict(x=best["x"], y=best["y"],
            k=best["k"], gate=best["gate"]), is_events=int(best["n_events"]),
        cost_bps=COST_BPS, turnover=float(mo["turnover"]), maxdd_pct=float(mo["maxdd"] * 100),
        market_neutral=False, n_trials=len(trials),
        hist_days=round(n_bars / 24, 1), per_coin=per_coin, note=note,
        mechanism="OI-collapse + large 1h move => fade the cascade for k hours "
                  "(forced-deleveraging exhaustion mean-reversion)",
        data_caveats="ONLY ~21d Binance openInterestHist history -> PILOT; "
                     "OI aligned to kline open_time by reindex+ffill; directional single-leg.",
    )
    _write(out)


def _write(d):
    p = ROOT / "reports" / "cand_oi_liq_cascade.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2, default=float))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
