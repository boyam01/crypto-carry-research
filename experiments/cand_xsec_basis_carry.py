"""Cross-sectional quarterly futures-basis carry.

Question tested:
    Does weekly cross-sectional selection on annualized futures basis add value
    versus naive equal-weight all-contango carry?

Universe:
    - USDT-M dated quarterlies for BTC/ETH via engine.fetch_binance.klines:
      BTCUSDT_<YYMMDD>, ETHUSDT_<YYMMDD>
    - COIN-M inverse quarterlies for BTC/ETH/XRP/BNB/SOL via Binance dapi:
      <COIN>USD_<YYMMDD>

Signal:
    annualized_basis = (future_close / spot_close - 1) * 365 / days_to_expiry

Trading convention:
    spread_ret = spot_ret - future_ret
      +weight means long-carry:  long spot / short future
      -weight means short-carry: short spot / long future

Governance:
    - weekly rebalance only
    - all desired weights are shifted by 1 daily bar before trading
    - cost is 30bp round-trip for the two-leg spread package
      (=15bp per spread turnover, 7.5bp/leg one-way, >= 5bp/leg)
    - chronological OOS = last 40%
    - report PSR and selection-minus-baseline PSR
    - daily klines around delivery can freeze; post-settlement bars are removed

Run:
    cd D:/量化交易CLAUDE
    python experiments/cross_sectional_basis_carry.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import backtest as bt
from engine import fetch_binance as fb


DAPI = "https://dapi.binance.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "quant-research/0.1"})

CODES = ["250328", "250627", "250926", "251226", "260327", "260626", "260925"]
EXPIRED_CODES = ["250328", "250627", "250926", "251226", "260327"]
COINM_COINS = ["BTC", "ETH", "XRP", "BNB", "SOL"]
USDTM_COINS = ["BTC", "ETH"]

# Primary research universe: COIN-M breadth, one venue per coin to avoid
# over-weighting BTC/ETH by also including their USDT-M twins.
PRIMARY_SOURCES = ("coinm",)

TRAIN_FRAC = 0.60
PPY = 365
TOP_N = 2
BOTTOM_N = 2
MIN_DTE_SIGNAL_DAYS = 7
MIN_CONTRACT_ROWS = 30
FETCH_LOOKBACK_DAYS = 240
FETCH_POST_EXPIRY_DAYS = 14

COST_ROUND_TRIP_BPS = 30.0
COST_PER_SPREAD_TURN_BPS = COST_ROUND_TRIP_BPS / 2.0

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(exist_ok=True)
CACHE = ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ContractMeta:
    key: str
    venue: str
    coin: str
    symbol: str
    code: str
    expiry: str
    rows: int
    first_date: str
    last_date: str
    frozen_rows_dropped: int
    post_expiry_rows_seen: int
    first_basis: float
    median_ann_basis: float
    pct_contango: float
    terminal_basis: float
    is_expired: bool


def code_to_expiry(code: str) -> pd.Timestamp:
    return pd.Timestamp(f"20{code[:2]}-{code[2:4]}-{code[4:6]}", tz="UTC")


def ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def cache_path(tag: str) -> pathlib.Path:
    digest = hashlib.md5(tag.encode()).hexdigest()[:16]
    return CACHE / f"xsec_basis_{digest}.parquet"


def read_cached(tag: str, builder):
    path = cache_path(tag)
    if path.exists():
        return pd.read_parquet(path)
    df = builder()
    if df is not None and len(df):
        df.to_parquet(path)
    return df


def get_json(url: str, params: dict, tries: int = 5):
    last = None
    for i in range(tries):
        r = SESSION.get(url, params=params, timeout=20)
        last = r
        if r.status_code == 200:
            return r.json()
        # Unknown symbol / not-listed contract: treat as no data.
        if r.status_code == 400:
            try:
                msg = r.json()
            except Exception:
                msg = {}
            if msg.get("code") in {-1121, -1122, -1100, -1128}:
                return []
        if r.status_code in (418, 429, 500, 502, 503, 504):
            time.sleep(2**i)
            continue
        r.raise_for_status()
    last.raise_for_status()


def parse_klines(rows: list) -> pd.DataFrame | None:
    if not rows:
        return None
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "qvol",
        "trades",
        "tbbav",
        "tbqav",
        "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    for col in ["open", "high", "low", "close", "volume", "qvol", "tbbav", "tbqav"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce")
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["open", "high", "low", "close", "volume", "qvol", "trades", "tbbav", "tbqav"]]
    return df[~df.index.duplicated()].sort_index()


def dapi_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame | None:
    """COIN-M klines with local parquet cache.

    Binance COIN-M kline requests have a finite time-window limit, so this
    paginates in <=180d chunks even for daily bars.
    """
    tag = f"dapi_kl|{symbol}|{interval}|{start_ms}|{end_ms}"

    def build():
        rows: list = []
        cur = start_ms
        max_span = 180 * 24 * 3600 * 1000
        while cur < end_ms:
            chunk_end = min(end_ms, cur + max_span)
            data = get_json(
                DAPI + "/dapi/v1/klines",
                dict(symbol=symbol, interval=interval, startTime=cur, endTime=chunk_end, limit=1500),
            )
            if data:
                rows.extend(data)
                last_open = int(data[-1][0])
                nxt = last_open + 1
                cur = nxt if nxt > cur else chunk_end + 1
                if len(data) < 1500 and cur < chunk_end:
                    cur = chunk_end + 1
            else:
                cur = chunk_end + 1
            time.sleep(0.08)
        return parse_klines(rows)

    return read_cached(tag, build)


def usdtm_klines(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame | None:
    return fb.klines(symbol, "1d", start_ms, end_ms, futures=True)


def spot_klines(coin: str, start_ms: int, end_ms: int) -> pd.DataFrame | None:
    return fb.klines(f"{coin}USDT", "1d", start_ms, end_ms, futures=False)


def trailing_constant_rows(s: pd.Series) -> int:
    if len(s) < 3:
        return 0
    eq_last = s.eq(s.iloc[-1]).iloc[::-1].cummin().iloc[::-1]
    return int(eq_last.sum())


def load_contract(venue: str, coin: str, code: str, now: pd.Timestamp):
    expiry = code_to_expiry(code)
    start = expiry - pd.Timedelta(days=FETCH_LOOKBACK_DAYS)
    end = min(now + pd.Timedelta(days=1), expiry + pd.Timedelta(days=FETCH_POST_EXPIRY_DAYS))
    start_ms, end_ms = ms(start), ms(end)

    if venue == "coinm":
        symbol = f"{coin}USD_{code}"
        fut = dapi_klines(symbol, "1d", start_ms, end_ms)
    elif venue == "usdtm":
        symbol = f"{coin}USDT_{code}"
        fut = usdtm_klines(symbol, start_ms, end_ms)
    else:
        raise ValueError(f"unknown venue: {venue}")

    spot = spot_klines(coin, start_ms, end_ms)
    if fut is None or spot is None or len(fut) < MIN_CONTRACT_ROWS:
        return None

    all_df = pd.DataFrame(index=fut.index)
    all_df["fut"] = fut["close"]
    all_df["fut_volume"] = fut["volume"]
    all_df["spot"] = spot["close"].reindex(all_df.index).ffill()
    all_df = all_df.dropna()
    if len(all_df) < MIN_CONTRACT_ROWS:
        return None

    post_expiry_rows = int((all_df.index > expiry).sum())
    df = all_df.loc[all_df.index <= expiry].copy()

    # Secondary guard: some delivered contracts can keep returning a frozen
    # final price near/after settlement. Keep one final row, drop extra repeats.
    frozen_dropped = post_expiry_rows
    tail_n = trailing_constant_rows(df["fut"])
    if tail_n > 2:
        frozen_dropped += tail_n - 1
        keep_until = len(df) - tail_n + 1
        df = df.iloc[:keep_until].copy()

    if len(df) < MIN_CONTRACT_ROWS:
        return None

    dte = (expiry.normalize() - df.index.normalize()).days.astype(float)
    df["days_to_expiry"] = dte
    df["basis"] = df["fut"] / df["spot"] - 1.0
    df["ann_basis"] = df["basis"] * 365.0 / df["days_to_expiry"].replace(0, np.nan)
    df.loc[df["days_to_expiry"] < MIN_DTE_SIGNAL_DAYS, "ann_basis"] = np.nan
    df["spread_ret"] = df["spot"].pct_change().fillna(0.0) - df["fut"].pct_change().fillna(0.0)

    key = f"{venue}:{coin}_{code}"
    valid_basis = df["ann_basis"].replace([np.inf, -np.inf], np.nan).dropna()
    meta = ContractMeta(
        key=key,
        venue=venue,
        coin=coin,
        symbol=symbol,
        code=code,
        expiry=str(expiry.date()),
        rows=int(len(df)),
        first_date=str(df.index[0].date()),
        last_date=str(df.index[-1].date()),
        frozen_rows_dropped=int(frozen_dropped),
        post_expiry_rows_seen=int(post_expiry_rows),
        first_basis=float(df["basis"].iloc[0]),
        median_ann_basis=float(valid_basis.median()) if len(valid_basis) else float("nan"),
        pct_contango=float((valid_basis > 0).mean()) if len(valid_basis) else float("nan"),
        terminal_basis=float(df["basis"].iloc[-1]),
        is_expired=bool(expiry < now),
    )
    return key, df[["spread_ret", "ann_basis", "basis", "days_to_expiry"]], meta


def load_universe(now: pd.Timestamp):
    contracts: dict[str, pd.DataFrame] = {}
    metas: list[ContractMeta] = []

    jobs = []
    for coin in COINM_COINS:
        for code in CODES:
            jobs.append(("coinm", coin, code))
    for coin in USDTM_COINS:
        for code in CODES:
            jobs.append(("usdtm", coin, code))

    for venue, coin, code in jobs:
        try:
            loaded = load_contract(venue, coin, code, now)
        except Exception as exc:
            print(f"[WARN] load failed {venue} {coin} {code}: {exc}")
            continue
        if loaded is None:
            print(f"[MISS] {venue:5} {coin:4} {code}")
            continue
        key, df, meta = loaded
        contracts[key] = df
        metas.append(meta)
        print(
            f"[OK] {key:18} rows={meta.rows:3d} {meta.first_date}->{meta.last_date} "
            f"annBasisMed={meta.median_ann_basis * 100:7.2f}% "
            f"contango={meta.pct_contango:4.0%} frozenDrop={meta.frozen_rows_dropped}"
        )

    return contracts, metas


def make_panels(
    contracts: dict[str, pd.DataFrame],
    metas: Iterable[ContractMeta],
    sources: tuple[str, ...],
    include_live: bool,
):
    allowed = {
        m.key
        for m in metas
        if m.venue in sources and (include_live or m.code in EXPIRED_CODES)
    }
    if not allowed:
        raise RuntimeError(f"no contracts after source/live filter: sources={sources} include_live={include_live}")

    returns = pd.DataFrame({k: contracts[k]["spread_ret"] for k in sorted(allowed)}).sort_index()
    ann_basis = pd.DataFrame({k: contracts[k]["ann_basis"] for k in sorted(allowed)}).sort_index()
    basis = pd.DataFrame({k: contracts[k]["basis"] for k in sorted(allowed)}).sort_index()

    # Keep dates where either signal or return exists; missing returns are zero
    # only after multiplying by zero weights.
    idx = returns.index.union(ann_basis.index).sort_values()
    returns = returns.reindex(idx)
    ann_basis = ann_basis.reindex(idx)
    basis = basis.reindex(idx)
    return returns, ann_basis, basis


def weekly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if len(index) == 0:
        return index
    local_naive = index.tz_convert(None) if index.tz is not None else index
    weeks = pd.Series(local_naive, index=index).dt.to_period("W-FRI")
    positions = pd.Series(np.arange(len(index)), index=index)
    last_positions = positions.groupby(weeks).tail(1).to_numpy()
    return index[last_positions]


def shifted_weekly_weights(raw: pd.DataFrame, ann_basis: pd.DataFrame) -> pd.DataFrame:
    rebals = weekly_rebalance_dates(ann_basis.index)
    sparse = pd.DataFrame(np.nan, index=ann_basis.index, columns=ann_basis.columns)
    sparse.loc[rebals] = raw.loc[rebals]
    desired = sparse.ffill().fillna(0.0)
    # Mandatory governance: trade one bar after the signal is known.
    return desired.shift(1).fillna(0.0)


def xsec_rank_neutral_weights(ann_basis: pd.DataFrame, top_n: int, bottom_n: int) -> pd.DataFrame:
    """Long highest basis, short lowest basis, equal gross on both sides.

    This is the strict cross-sectional factor version. If all contracts are in
    contango, the bottom side is still short the relatively weakest carry.
    """
    raw = pd.DataFrame(0.0, index=ann_basis.index, columns=ann_basis.columns)
    for dt in weekly_rebalance_dates(ann_basis.index):
        scores = ann_basis.loc[dt].replace([np.inf, -np.inf], np.nan).dropna()
        if len(scores) < 2:
            continue
        n_side = min(top_n, bottom_n, len(scores) // 2)
        if n_side < 1:
            continue
        ranked = scores.sort_values(ascending=False)
        longs = list(ranked.head(n_side).index)
        shorts = list(ranked.tail(n_side).index)
        raw.loc[dt, longs] = 0.5 / len(longs)
        raw.loc[dt, shorts] = -0.5 / len(shorts)
    return shifted_weekly_weights(raw, ann_basis)


def signed_extreme_weights(ann_basis: pd.DataFrame, top_n: int, bottom_n: int) -> pd.DataFrame:
    """Long strongest contango and short actual backwardation only.

    This answers the literal "most-backwardated" version. It can be one-sided
    when no backwardated contracts exist, so it is less strictly market-neutral
    in carry-factor space than xsec_rank_neutral_weights.
    """
    raw = pd.DataFrame(0.0, index=ann_basis.index, columns=ann_basis.columns)
    for dt in weekly_rebalance_dates(ann_basis.index):
        scores = ann_basis.loc[dt].replace([np.inf, -np.inf], np.nan).dropna()
        longs = list(scores[scores > 0].nlargest(top_n).index)
        shorts = list(scores[scores < 0].nsmallest(bottom_n).index)
        if longs:
            raw.loc[dt, longs] = 0.5 / len(longs) if shorts else 1.0 / len(longs)
        if shorts:
            raw.loc[dt, shorts] = -0.5 / len(shorts) if longs else -1.0 / len(shorts)
    return shifted_weekly_weights(raw, ann_basis)


def all_contango_weights(ann_basis: pd.DataFrame) -> pd.DataFrame:
    """Naive baseline: equal-weight every available contango contract."""
    raw = pd.DataFrame(0.0, index=ann_basis.index, columns=ann_basis.columns)
    for dt in weekly_rebalance_dates(ann_basis.index):
        scores = ann_basis.loc[dt].replace([np.inf, -np.inf], np.nan).dropna()
        eligible = list(scores[scores > 0].index)
        if eligible:
            raw.loc[dt, eligible] = 1.0 / len(eligible)
    return shifted_weekly_weights(raw, ann_basis)


def all_signed_weights(ann_basis: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight all positive basis long-carry and all negative basis short-carry."""
    raw = pd.DataFrame(0.0, index=ann_basis.index, columns=ann_basis.columns)
    for dt in weekly_rebalance_dates(ann_basis.index):
        scores = ann_basis.loc[dt].replace([np.inf, -np.inf], np.nan).dropna()
        longs = list(scores[scores > 0].index)
        shorts = list(scores[scores < 0].index)
        if longs:
            raw.loc[dt, longs] = 0.5 / len(longs) if shorts else 1.0 / len(longs)
        if shorts:
            raw.loc[dt, shorts] = -0.5 / len(shorts) if longs else -1.0 / len(shorts)
    return shifted_weekly_weights(raw, ann_basis)


def portfolio_backtest(returns: pd.DataFrame, weights: pd.DataFrame):
    r = returns.reindex_like(weights).fillna(0.0)
    w = weights.fillna(0.0)
    gross = (w * r).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = w.iloc[0].abs().sum()
    cost = turnover * COST_PER_SPREAD_TURN_BPS / 1e4
    net = gross - cost
    exposure = w.abs().sum(axis=1)
    return pd.DataFrame(
        {
            "gross": gross,
            "cost": cost,
            "net": net,
            "turnover": turnover,
            "gross_exposure": exposure,
        }
    )


def evaluate_series(net: pd.Series, position: pd.Series | None = None):
    n = len(net)
    tr, te = bt.oos_split(n, TRAIN_FRAC)
    pos_values = position.values if position is not None else None
    is_m = bt.metrics(net.values[tr], PPY, pos_values[tr] if pos_values is not None else None)
    oos_m = bt.metrics(net.values[te], PPY, pos_values[te] if pos_values is not None else None)
    psr = bt.psr(oos_m["sr_pp"], oos_m["n"], oos_m["skew"], oos_m["kurt"])
    return is_m, oos_m, psr, tr, te


def summarize_strategy(name: str, result: pd.DataFrame):
    is_m, oos_m, psr, tr, te = evaluate_series(result["net"], result["gross_exposure"])
    return {
        "name": name,
        "IS": is_m,
        "OOS": oos_m,
        "OOS_psr": psr,
        "avg_daily_turnover_IS": float(result["turnover"].iloc[tr].mean()),
        "avg_daily_turnover_OOS": float(result["turnover"].iloc[te].mean()),
        "avg_cost_bps_per_day_OOS": float(result["cost"].iloc[te].mean() * 1e4),
        "gross_exposure_OOS": float(result["gross_exposure"].iloc[te].mean()),
    }


def print_summary(rows: list[dict]):
    print("\n=== Strategy comparison (chronological OOS = last 40%) ===")
    print(
        f"{'strategy':28} {'IS_sh':>7} {'OOS_sh':>8} {'OOS_ret%':>9} "
        f"{'maxDD%':>8} {'turn':>8} {'costbp/d':>9} {'PSR':>7}"
    )
    for row in rows:
        is_m, oos_m = row["IS"], row["OOS"]
        print(
            f"{row['name'][:28]:28} {is_m['sharpe_ann']:7.2f} {oos_m['sharpe_ann']:8.2f} "
            f"{oos_m['ret_ann'] * 100:9.2f} {oos_m['maxdd'] * 100:8.2f} "
            f"{row['avg_daily_turnover_OOS']:8.4f} {row['avg_cost_bps_per_day_OOS']:9.4f} "
            f"{row['OOS_psr']:7.3f}"
        )


def evaluate_suite(
    label: str,
    contracts: dict[str, pd.DataFrame],
    metas: list[ContractMeta],
    sources: tuple[str, ...],
    include_live: bool,
):
    returns, ann_basis, _ = make_panels(contracts, metas, sources, include_live)
    # Drop leading dates before any signal exists.
    active = ann_basis.notna().any(axis=1) | returns.notna().any(axis=1)
    returns = returns.loc[active]
    ann_basis = ann_basis.loc[active]

    weights = {
        "xsec_rank_neutral": xsec_rank_neutral_weights(ann_basis, TOP_N, BOTTOM_N),
        "signed_extremes": signed_extreme_weights(ann_basis, TOP_N, BOTTOM_N),
        "naive_all_contango": all_contango_weights(ann_basis),
        "naive_all_signed": all_signed_weights(ann_basis),
    }
    results = {name: portfolio_backtest(returns, w) for name, w in weights.items()}
    summaries = [summarize_strategy(name, res) for name, res in results.items()]

    # Direct value-add test: same dates, selection net minus naive baseline net.
    diff = results["xsec_rank_neutral"]["net"] - results["naive_all_contango"]["net"]
    diff_df = pd.DataFrame(
        {
            "net": diff,
            "gross_exposure": pd.Series(1.0, index=diff.index),
            "turnover": pd.Series(0.0, index=diff.index),
            "cost": pd.Series(0.0, index=diff.index),
        }
    )
    summaries.append(summarize_strategy("xsec_minus_all_contango", diff_df))

    print(f"\n\n##### SUITE: {label} | sources={sources} include_live={include_live} #####")
    print(f"panel: {len(returns):,} daily rows, {returns.shape[1]} instruments")
    print_summary(summaries)

    oos = {r["name"]: r["OOS"] for r in summaries}
    psr = {r["name"]: r["OOS_psr"] for r in summaries}
    value_add = oos["xsec_minus_all_contango"]["ret_ann"]
    robust = (
        value_add > 0
        and psr["xsec_minus_all_contango"] >= 0.95
        and oos["xsec_rank_neutral"]["sharpe_ann"] > oos["naive_all_contango"]["sharpe_ann"]
    )
    verdict = (
        "PASS: selection adds robust OOS value over naive all-contango"
        if robust
        else "FAIL/WEAK: selection does not robustly beat naive all-contango"
    )
    print(f"verdict: {verdict}")
    print(
        "risk note: close-to-close maxDD is a delta-neutral marking illusion; "
        "it excludes intraday basis spikes, margin/liquidation risk, and settlement execution."
    )

    return {
        "label": label,
        "sources": list(sources),
        "include_live": include_live,
        "n_days": int(len(returns)),
        "n_instruments": int(returns.shape[1]),
        "summaries": summaries,
        "verdict": verdict,
    }


def main():
    now = pd.Timestamp.utcnow()
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    print("=== Cross-sectional quarterly futures-basis carry ===")
    print(f"now_utc={now.isoformat()}")
    print(
        f"cost model: {COST_ROUND_TRIP_BPS:.1f}bp RT per two-leg spread package "
        f"({COST_PER_SPREAD_TURN_BPS / 2:.2f}bp/leg one-way)"
    )

    contracts, metas = load_universe(now)
    if not contracts:
        raise RuntimeError("No contracts loaded; check network/API availability.")

    meta_rows = [asdict(m) for m in metas]
    meta_path = OUT_DIR / "cross_sectional_basis_contracts.csv"
    pd.DataFrame(meta_rows).sort_values(["venue", "coin", "code"]).to_csv(meta_path, index=False)

    suites = []
    suites.append(
        evaluate_suite(
            label="primary_coinm_expired_only",
            contracts=contracts,
            metas=metas,
            sources=PRIMARY_SOURCES,
            include_live=False,
        )
    )
    suites.append(
        evaluate_suite(
            label="primary_coinm_with_live",
            contracts=contracts,
            metas=metas,
            sources=PRIMARY_SOURCES,
            include_live=True,
        )
    )
    suites.append(
        evaluate_suite(
            label="usdtm_btc_eth_expired_sanity",
            contracts=contracts,
            metas=metas,
            sources=("usdtm",),
            include_live=False,
        )
    )

    report = {
        "generated_at_utc": now.isoformat(),
        "config": {
            "codes": CODES,
            "expired_codes": EXPIRED_CODES,
            "coinm_coins": COINM_COINS,
            "usdtm_coins": USDTM_COINS,
            "primary_sources": PRIMARY_SOURCES,
            "train_frac": TRAIN_FRAC,
            "ppy": PPY,
            "top_n": TOP_N,
            "bottom_n": BOTTOM_N,
            "min_dte_signal_days": MIN_DTE_SIGNAL_DAYS,
            "cost_round_trip_bps": COST_ROUND_TRIP_BPS,
            "cost_per_spread_turn_bps": COST_PER_SPREAD_TURN_BPS,
        },
        "contracts": meta_rows,
        "suites": suites,
        "interpretation_rules": [
            "Primary answer should come from primary_coinm_expired_only; live contracts are diagnostic only.",
            "xsec_minus_all_contango OOS return and PSR are the direct selection value-add test.",
            "Frozen post-settlement rows are dropped before signal/return construction.",
            "COIN-M inverse PnL is approximated with USD price spread returns; exact production PnL needs contract multiplier/collateral accounting.",
            "Close-to-close maxDD is not liquidation-aware and should be treated as optimistic.",
        ],
    }
    out = OUT_DIR / "cross_sectional_basis_carry.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {meta_path}")
    print(f"wrote {out}")

    # ---- canonical one-line candidate summary (governance verdict) ----
    # Primary judgement = COIN-M expired-only suite. Headline strategy = the
    # cross-sectional rank-neutral selection portfolio; also report whether
    # selection beats the naive all-contango baseline (xsec_minus_all_contango).
    primary = suites[0]
    by_name = {s["name"]: s for s in primary["summaries"]}
    head = by_name["xsec_rank_neutral"]
    base = by_name["naive_all_contango"]
    add = by_name["xsec_minus_all_contango"]
    oos, ism = head["OOS"], head["IS"]

    # all per-strategy OOS per-period Sharpes in this family -> deflated benchmark
    fam_sr = [s["OOS"]["sr_pp"] for s in primary["summaries"]
              if np.isfinite(s["OOS"].get("sr_pp", np.nan))]
    sr_star = bt.dsr_benchmark(fam_sr) if len(fam_sr) >= 2 else 0.0
    dsr = (bt.psr(oos["sr_pp"], oos["n"], oos["skew"], oos["kurt"], sr_benchmark=sr_star)
           if np.isfinite(oos.get("sr_pp", np.nan)) else float("nan"))

    summary = {
        "key": "xsec_basis_carry",
        "family": "basis-carry",
        "file": "experiments/cand_xsec_basis_carry.py",
        "universe": f"COIN-M quarterlies {COINM_COINS} expired {EXPIRED_CODES} (primary); USDT-M BTC/ETH sanity",
        "n_obs": int(oos["n"]),
        "market_neutral": True,
        "turnover": float(head["avg_daily_turnover_OOS"]),
        "cost_bps": COST_ROUND_TRIP_BPS,
        "headline_strategy": "xsec_rank_neutral",
        "oos_sharpe": float(oos["sharpe_ann"]),
        "oos_ret_ann_pct": float(oos["ret_ann"] * 100),
        "maxdd_pct": float(oos["maxdd"] * 100),
        "is_sharpe": float(ism["sharpe_ann"]),
        "psr": float(head["OOS_psr"]),
        "dsr": float(dsr),
        "sr_star_family": float(sr_star),
        "baseline_all_contango_oos_sharpe": float(base["OOS"]["sharpe_ann"]),
        "selection_value_add_oos_ret_pct": float(add["OOS"]["ret_ann"] * 100),
        "selection_value_add_psr": float(add["OOS_psr"]),
    }
    cand = OUT_DIR / "cand_xsec_basis_carry.json"
    cand.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"wrote {cand}")
    print("\nCANDIDATE SUMMARY:\n" + json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
