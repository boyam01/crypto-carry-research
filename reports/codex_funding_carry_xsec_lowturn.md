以下是 `experiments/cand_funding_carry_xsec_lowturn.py` 的完整內容。我沒有執行任何命令或回測。

```python
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from engine import backtest as bt
from engine import fetch_binance as fb
from engine import fetch_funding as ff
from engine import stats as st  # noqa: F401


DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LTCUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "TRXUSDT",
    "DOTUSDT",
]

PPY_DAILY = 365.0


@dataclass(frozen=True)
class VariantConfig:
    lookback_weeks: int
    basket_n: int
    hysteresis_ranks: int
    min_abs_weekly_funding: float
    require_score_sign: bool = True

    @property
    def key(self) -> str:
        sign = "sign" if self.require_score_sign else "rel"
        return (
            f"lb{self.lookback_weeks}_b{self.basket_n}"
            f"_h{self.hysteresis_ranks}_min{self.min_abs_weekly_funding:g}_{sign}"
        )


def _as_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _to_ms(value: str | pd.Timestamp) -> int:
    return int(_as_utc_timestamp(value).timestamp() * 1000)


def _normalize_symbol(symbol: str) -> str:
    s = symbol.strip().upper().replace("/", "").replace("-", "").replace("_", "")
    if s.endswith("USDT"):
        return s
    return f"{s}USDT"


def _parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = raw.replace(",", " ").split()
    return [_normalize_symbol(x) for x in parts if x.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _standardize_time_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "open_time" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
        out = out.set_index("open_time")

    idx = out.index
    if np.issubdtype(idx.dtype, np.number):
        out.index = pd.to_datetime(idx, unit="ms", utc=True)
    else:
        out.index = pd.to_datetime(idx, utc=True)

    out = out[~out.index.duplicated(keep="last")].sort_index()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _fetch_funding_rate(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    getter = getattr(fb, "funding_rate", None)
    if getter is None:
        getter = getattr(ff, "funding_rate")
    fund = getter(symbol, start_ms, end_ms)
    fund = _standardize_time_index(fund)

    if "fundingRate" not in fund.columns:
        if len(fund.columns) == 0:
            raise ValueError(f"{symbol}: funding dataframe has no columns")
        fund["fundingRate"] = fund.iloc[:, 0]

    fund["fundingRate"] = pd.to_numeric(fund["fundingRate"], errors="coerce")
    if fund["fundingRate"].dropna().empty:
        raise ValueError(f"{symbol}: fundingRate is empty")
    return fund


def _frozen_bar_count(df: pd.DataFrame) -> int:
    if "close" not in df.columns:
        return 0
    vol_col = "qvol" if "qvol" in df.columns else "volume"
    close = pd.to_numeric(df["close"], errors="coerce")
    vol = pd.to_numeric(df.get(vol_col, pd.Series(index=df.index, data=np.nan)), errors="coerce")
    frozen = close.diff().abs().eq(0) & vol.fillna(0).eq(0)
    return int(frozen.sum())


def _fetch_one_symbol(symbol: str, start_ms: int, end_ms: int) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    fut = _standardize_time_index(fb.klines(symbol, "1d", start_ms, end_ms, futures=True))
    spot = _standardize_time_index(fb.klines(symbol, "1d", start_ms, end_ms, futures=False))
    fund = _fetch_funding_rate(symbol, start_ms, end_ms)

    if "close" not in fut.columns or "close" not in spot.columns:
        raise ValueError(f"{symbol}: missing close column")

    idx = fut.index.intersection(spot.index).sort_values()
    if len(idx) < 120:
        raise ValueError(f"{symbol}: too few aligned daily bars: {len(idx)}")

    spot_close = spot.loc[idx, "close"]
    fut_close = fut.loc[idx, "close"]

    spot_ret = spot_close.pct_change()
    fut_ret = fut_close.pct_change()

    fund_s = fund["fundingRate"].dropna().sort_index()
    fund_daily = fund_s.resample("1D").sum(min_count=1)
    daily_key = idx.normalize()
    fund_aligned = fund_daily.reindex(daily_key)
    fund_aligned = pd.Series(fund_aligned.to_numpy(), index=idx, name=symbol)

    # +1 carry unit means long spot / short perp.
    # Positive funding is received by short perp holders.
    carry_ret = spot_ret - fut_ret + fund_aligned
    carry_ret.name = symbol

    fund_counts = fund_s.resample("1D").count().reindex(daily_key).fillna(0)
    quality = {
        "n_daily_aligned": int(len(idx)),
        "first_bar": idx.min().isoformat(),
        "last_bar": idx.max().isoformat(),
        "spot_frozen_zero_close_zero_qvol": _frozen_bar_count(spot.loc[idx]),
        "futures_frozen_zero_close_zero_qvol": _frozen_bar_count(fut.loc[idx]),
        "funding_days_lt_2_events": int((fund_counts < 2).sum()),
        "funding_days_eq_3_events": int((fund_counts == 3).sum()),
    }

    return carry_ret, fund_aligned, quality


def _load_panel(
    symbols: list[str],
    start_ms: int,
    end_ms: int,
    min_symbol_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    carry_map: dict[str, pd.Series] = {}
    funding_map: dict[str, pd.Series] = {}
    quality: dict[str, Any] = {}

    for symbol in symbols:
        try:
            carry, funding, q = _fetch_one_symbol(symbol, start_ms, end_ms)
            carry_map[symbol] = carry
            funding_map[symbol] = funding
            quality[symbol] = q
        except Exception as exc:
            quality[symbol] = {"error": repr(exc)}

    if not carry_map:
        raise RuntimeError("No usable symbols loaded")

    carry_panel = pd.DataFrame(carry_map).sort_index()
    funding_panel = pd.DataFrame(funding_map).reindex(carry_panel.index)

    coverage = carry_panel.notna().mean()
    keep = coverage[coverage >= min_symbol_coverage].index.tolist()
    dropped = sorted(set(carry_panel.columns) - set(keep))

    for symbol in dropped:
        quality.setdefault(symbol, {})
        quality[symbol]["dropped_for_coverage"] = float(coverage.loc[symbol])

    carry_panel = carry_panel[keep]
    funding_panel = funding_panel[keep]

    if len(keep) < 4:
        raise RuntimeError(f"Too few symbols after coverage filter: {keep}")

    return carry_panel, funding_panel, quality


def _weekly_scores(funding_daily: pd.DataFrame, lookback_weeks: int) -> pd.DataFrame:
    weekly_funding = funding_daily.resample("W-FRI").sum(min_count=5)
    return weekly_funding.rolling(lookback_weeks, min_periods=lookback_weeks).mean()


def _build_weekly_targets(scores: pd.DataFrame, cfg: VariantConfig) -> pd.DataFrame:
    symbols = list(scores.columns)
    current_long: set[str] = set()
    current_short: set[str] = set()
    rows: list[pd.Series] = []

    def long_allowed(value: float) -> bool:
        if not cfg.require_score_sign:
            return True
        return value >= cfg.min_abs_weekly_funding

    def short_allowed(value: float) -> bool:
        if not cfg.require_score_sign:
            return True
        return value <= -cfg.min_abs_weekly_funding

    for _, row in scores.iterrows():
        weights = pd.Series(0.0, index=symbols)
        valid = row.dropna()

        if valid.empty:
            rows.append(weights)
            continue

        desc = valid.sort_values(ascending=False)
        asc = valid.sort_values(ascending=True)
        desc_rank = pd.Series(np.arange(1, len(desc) + 1), index=desc.index)
        asc_rank = pd.Series(np.arange(1, len(asc) + 1), index=asc.index)

        exit_rank = cfg.basket_n + cfg.hysteresis_ranks

        current_long = {
            s
            for s in current_long
            if s in desc_rank.index and desc_rank.loc[s] <= exit_rank and long_allowed(valid.loc[s])
        }
        current_short = {
            s
            for s in current_short
            if s in asc_rank.index and asc_rank.loc[s] <= exit_rank and short_allowed(valid.loc[s])
        }
        current_long -= current_short

        for symbol in desc.index:
            if len(current_long) >= cfg.basket_n:
                break
            if symbol in current_short:
                continue
            if long_allowed(valid.loc[symbol]):
                current_long.add(symbol)

        for symbol in asc.index:
            if len(current_short) >= cfg.basket_n:
                break
            if symbol in current_long:
                continue
            if short_allowed(valid.loc[symbol]):
                current_short.add(symbol)

        long_list = [s for s in desc.index if s in current_long]
        short_list = [s for s in asc.index if s in current_short]
        selected = long_list + short_list

        if selected:
            unit = 1.0 / len(selected)
            weights.loc[long_list] = unit
            weights.loc[short_list] = -unit

        rows.append(weights)

    return pd.DataFrame(rows, index=scores.index, columns=symbols)


def _as_series(value: Any, index: pd.Index, name: str) -> pd.Series:
    if isinstance(value, pd.Series):
        out = value.copy()
    elif isinstance(value, pd.DataFrame):
        out = value.iloc[:, 0].copy() if value.shape[1] == 1 else value.sum(axis=1)
    else:
        out = pd.Series(value, index=index)

    out = out.reindex(index)
    out.name = name
    return pd.to_numeric(out, errors="coerce").fillna(0.0)


def _run_engine(carry_ret: pd.DataFrame, position: pd.DataFrame, cost_bps: float) -> pd.Series:
    aligned_pos = position.fillna(0.0)
    aligned_ret = carry_ret.reindex(index=aligned_pos.index, columns=aligned_pos.columns).fillna(0.0)
    net = bt.run(aligned_ret, aligned_pos, cost_bps)
    return _as_series(net, aligned_pos.index, "net")


def _empty_metrics() -> dict[str, float]:
    return {
        "n": 0.0,
        "sharpe_ann": np.nan,
        "sr_pp": np.nan,
        "ret_ann": np.nan,
        "vol_ann": np.nan,
        "maxdd": np.nan,
        "hit": np.nan,
        "turnover": np.nan,
        "skew": np.nan,
        "kurt": np.nan,
    }


def _engine_metrics(net: pd.Series, position: pd.DataFrame, ppy: float) -> dict[str, float]:
    if len(net) < 3:
        return _empty_metrics()
    try:
        raw = bt.metrics(net, ppy, position.reindex(net.index).fillna(0.0))
        return {k: float(v) if v is not None else np.nan for k, v in raw.items()}
    except Exception as exc:
        out = _empty_metrics()
        out["metrics_error"] = repr(exc)
        return out


def _metric_block(
    net: pd.Series,
    position: pd.DataFrame,
    split: tuple[slice, slice],
    ppy: float,
) -> dict[str, dict[str, float]]:
    train_slice, oos_slice = split
    return {
        "full": _engine_metrics(net, position, ppy),
        "train": _engine_metrics(net.iloc[train_slice], position.iloc[train_slice], ppy),
        "oos": _engine_metrics(net.iloc[oos_slice], position.iloc[oos_slice], ppy),
    }


def _turnover_stats(
    position: pd.DataFrame,
    period_slice: slice,
    cost_bps_per_leg: float,
    ppy: float,
) -> dict[str, float]:
    pair_turnover = position.diff().abs().sum(axis=1).fillna(0.0)
    period_pair = pair_turnover.iloc[period_slice]
    period_pos = position.iloc[period_slice]

    pair_per_bar = float(period_pair.mean()) if len(period_pair) else np.nan
    leg_per_bar = 2.0 * pair_per_bar if np.isfinite(pair_per_bar) else np.nan
    implied_cost_ann = leg_per_bar * cost_bps_per_leg / 1e4 * ppy if np.isfinite(leg_per_bar) else np.nan

    return {
        "pair_turnover_per_bar": pair_per_bar,
        "leg_turnover_per_bar": leg_per_bar,
        "implied_cost_ann_from_leg_turnover": implied_cost_ann,
        "rebalance_days_with_trade": int((period_pair > 1e-12).sum()),
        "avg_gross_pair_exposure": float(period_pos.abs().sum(axis=1).mean()),
        "avg_active_names": float(period_pos.abs().gt(0).sum(axis=1).mean()),
    }


def _prefix(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}{k}": v for k, v in metrics.items()}


def _evaluate_variant(
    carry_ret: pd.DataFrame,
    funding_daily: pd.DataFrame,
    cfg: VariantConfig,
    split: tuple[slice, slice],
    cost_bps_per_leg: float,
    signal_lag_bars: int,
    keep_detail: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if signal_lag_bars < 1:
        raise ValueError("signal_lag_bars must be >= 1 to avoid look-ahead")

    scores = _weekly_scores(funding_daily, cfg.lookback_weeks)
    weekly_target = _build_weekly_targets(scores, cfg)

    daily_target = weekly_target.reindex(carry_ret.index).ffill().fillna(0.0)
    position = daily_target.shift(signal_lag_bars).fillna(0.0)

    active_missing = int(((position.abs() > 0) & carry_ret.isna()).sum().sum())

    # One synthetic carry unit contains two traded legs:
    #   +w means +w spot and -w perp.
    #   -w means -w spot and +w perp.
    # Engine cost is charged on abs(position change), so 10bp/leg becomes 20bp
    # per synthetic carry-unit turnover.
    engine_pair_cost_bps = 2.0 * cost_bps_per_leg

    gross = _run_engine(carry_ret, position, cost_bps=0.0)
    net = _run_engine(carry_ret, position, cost_bps=engine_pair_cost_bps)

    net_metrics = _metric_block(net, position, split, PPY_DAILY)
    gross_metrics = _metric_block(gross, position, split, PPY_DAILY)

    train_slice, oos_slice = split
    train_turnover = _turnover_stats(position, train_slice, cost_bps_per_leg, PPY_DAILY)
    oos_turnover = _turnover_stats(position, oos_slice, cost_bps_per_leg, PPY_DAILY)

    summary: dict[str, Any] = {
        "key": cfg.key,
        **asdict(cfg),
        "active_missing_return_cells": active_missing,
        **_prefix("train_net_", net_metrics["train"]),
        **_prefix("oos_net_", net_metrics["oos"]),
        **_prefix("oos_gross_", gross_metrics["oos"]),
        **_prefix("train_turnover_", train_turnover),
        **_prefix("oos_turnover_", oos_turnover),
    }

    detail = None
    if keep_detail:
        detail = {
            "scores": scores,
            "weekly_target": weekly_target,
            "daily_target_unshifted": daily_target,
            "position": position,
            "gross": gross,
            "net": net,
            "net_metrics": net_metrics,
            "gross_metrics": gross_metrics,
            "train_turnover": train_turnover,
            "oos_turnover": oos_turnover,
            "active_missing_return_cells": active_missing,
        }

    return summary, detail


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _safe_psr_zero(metrics: dict[str, float]) -> float:
    sr_pp = metrics.get("sr_pp", np.nan)
    n = metrics.get("n", np.nan)
    skew = metrics.get("skew", np.nan)
    kurt = metrics.get("kurt", np.nan)

    if not all(np.isfinite([sr_pp, n, skew, kurt])) or n <= 2:
        return np.nan

    try:
        return float(bt.psr(sr_pp, n, skew, kurt))
    except Exception:
        return _psr_against_benchmark(sr_pp, n, skew, kurt, 0.0)


def _psr_against_benchmark(
    sr_pp: float,
    n: float,
    skew: float,
    kurt: float,
    benchmark_sr_pp: float,
) -> float:
    if not all(np.isfinite([sr_pp, n, skew, kurt, benchmark_sr_pp])) or n <= 2:
        return np.nan

    denom_sq = 1.0 - skew * sr_pp + ((kurt - 1.0) / 4.0) * sr_pp * sr_pp
    if denom_sq <= 0:
        return np.nan

    z = (sr_pp - benchmark_sr_pp) * math.sqrt(n - 1.0) / math.sqrt(denom_sq)
    return float(_normal_cdf(z))


def _extract_dsr_benchmark(raw: Any) -> float:
    if isinstance(raw, dict):
        for key in ("sr_benchmark", "benchmark", "dsr_benchmark", "sr0"):
            if key in raw:
                return float(raw[key])
    if isinstance(raw, (tuple, list, np.ndarray)) and len(raw) > 0:
        return float(raw[0])
    return float(raw)


def _safe_dsr_benchmark(sr_pp_list: list[float]) -> tuple[float, Any]:
    clean = [float(x) for x in sr_pp_list if np.isfinite(x)]
    if len(clean) < 2:
        return np.nan, None

    try:
        raw = bt.dsr_benchmark(clean)
        return _extract_dsr_benchmark(raw), raw
    except Exception as exc:
        return np.nan, {"error": repr(exc)}


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if not np.isfinite(v) else v
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _fmt_float(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{100.0 * value:.2f}%"


def _survival_verdict(oos_net: dict[str, float], psr: float, dsr_prob: float) -> str:
    fails = []
    if not np.isfinite(oos_net.get("ret_ann", np.nan)) or oos_net["ret_ann"] <= 0:
        fails.append("OOS net return <= 0")
    if not np.isfinite(oos_net.get("sharpe_ann", np.nan)) or oos_net["sharpe_ann"] <= 1.0:
        fails.append("OOS net Sharpe <= 1")
    if np.isfinite(psr) and psr < 0.95:
        fails.append("PSR < 95%")
    if np.isfinite(dsr_prob) and dsr_prob < 0.95:
        fails.append("DSR probability < 95%")

    if fails:
        return "DOES_NOT_SURVIVE_FIRST_PASS: " + "; ".join(fails)
    return "SURVIVES_FIRST_PASS_CANDIDATE: positive OOS net, Sharpe > 1, PSR/DSR gates pass"


def _select_symbols(args: argparse.Namespace) -> list[str]:
    explicit = _parse_symbols(args.symbols)
    if explicit:
        return explicit[: args.top]

    if args.use_exchange_universe:
        try:
            universe = fb.exchange_universe(args.min_quote_vol, args.top)
            symbols = [_normalize_symbol(x) for x in universe]
            if symbols:
                return symbols[: args.top]
        except Exception as exc:
            print(f"[WARN] exchange_universe failed, falling back to static cached list: {exc!r}")

    return DEFAULT_SYMBOLS[: args.top]


def _make_grid(args: argparse.Namespace, n_symbols: int) -> list[VariantConfig]:
    lookbacks = _parse_int_list(args.lookback_weeks)
    baskets = _parse_int_list(args.basket_n)
    hysteresis = _parse_int_list(args.hysteresis_ranks)
    min_abs = _parse_float_list(args.min_abs_weekly_funding)

    configs = [
        VariantConfig(
            lookback_weeks=lb,
            basket_n=b,
            hysteresis_ranks=h,
            min_abs_weekly_funding=m,
            require_score_sign=not args.allow_same_sign_baskets,
        )
        for lb in lookbacks
        for b in baskets
        for h in hysteresis
        for m in min_abs
        if 2 * b <= n_symbols
    ]

    if not configs:
        raise RuntimeError(f"No valid configs for n_symbols={n_symbols}")
    return configs


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Weekly low-turnover cross-sectional funding carry backtest."
    )
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--symbols", default=None, help="Comma/space separated symbols or bases.")
    p.add_argument("--use-exchange-universe", action="store_true")
    p.add_argument("--min-quote-vol", type=float, default=50_000_000)
    p.add_argument("--train-frac", type=float, default=0.60)
    p.add_argument("--cost-bps-per-leg", type=float, default=10.0)
    p.add_argument("--signal-lag-bars", type=int, default=1)
    p.add_argument("--min-symbol-coverage", type=float, default=0.90)
    p.add_argument("--lookback-weeks", default="4,8,12")
    p.add_argument("--basket-n", default="2,3,4")
    p.add_argument("--hysteresis-ranks", default="1,2,3")
    p.add_argument("--min-abs-weekly-funding", default="0,0.0005,0.001")
    p.add_argument("--allow-same-sign-baskets", action="store_true")
    p.add_argument(
        "--outdir",
        default="artifacts/cand_funding_carry_xsec_lowturn",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    start_ts = _as_utc_timestamp(args.start)
    end_ts = _as_utc_timestamp(args.end) if args.end else pd.Timestamp.utcnow().tz_convert("UTC").normalize()

    symbols = _select_symbols(args)
    start_ms = _to_ms(start_ts)
    end_ms = _to_ms(end_ts)

    print(f"[INFO] symbols={symbols}")
    print(f"[INFO] window={start_ts.isoformat()} to {end_ts.isoformat()}")
    print("[INFO] loading daily spot/futures and 8h funding")

    carry_ret, funding_daily, quality = _load_panel(
        symbols=symbols,
        start_ms=start_ms,
        end_ms=end_ms,
        min_symbol_coverage=args.min_symbol_coverage,
    )

    carry_ret = carry_ret.sort_index()
    funding_daily = funding_daily.reindex(carry_ret.index).sort_index()

    split = bt.oos_split(len(carry_ret), train_frac=args.train_frac)
    train_slice, oos_slice = split

    configs = _make_grid(args, n_symbols=len(carry_ret.columns))
    print(f"[INFO] usable_symbols={list(carry_ret.columns)}")
    print(f"[INFO] variants={len(configs)}")
    print(f"[INFO] train={carry_ret.index[train_slice][0]} to {carry_ret.index[train_slice][-1]}")
    print(f"[INFO] oos={carry_ret.index[oos_slice][0]} to {carry_ret.index[oos_slice][-1]}")

    summaries: list[dict[str, Any]] = []
    for cfg in configs:
        summary, _ = _evaluate_variant(
            carry_ret=carry_ret,
            funding_daily=funding_daily,
            cfg=cfg,
            split=split,
            cost_bps_per_leg=args.cost_bps_per_leg,
            signal_lag_bars=args.signal_lag_bars,
            keep_detail=False,
        )
        summaries.append(summary)

    grid = pd.DataFrame(summaries)
    grid = grid.sort_values(
        ["train_net_sharpe_ann", "train_net_ret_ann"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    if grid.empty:
        raise RuntimeError("No variant results")

    selected_key = str(grid.loc[0, "key"])
    selected_cfg = next(cfg for cfg in configs if cfg.key == selected_key)

    selected_summary, detail = _evaluate_variant(
        carry_ret=carry_ret,
        funding_daily=funding_daily,
        cfg=selected_cfg,
        split=split,
        cost_bps_per_leg=args.cost_bps_per_leg,
        signal_lag_bars=args.signal_lag_bars,
        keep_detail=True,
    )
    assert detail is not None

    train_trial_sr = grid["train_net_sr_pp"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().tolist()
    dsr_benchmark_sr_pp, dsr_raw = _safe_dsr_benchmark(train_trial_sr)

    oos_net_metrics = detail["net_metrics"]["oos"]
    psr_oos_net = _safe_psr_zero(oos_net_metrics)
    dsr_prob_oos = _psr_against_benchmark(
        oos_net_metrics.get("sr_pp", np.nan),
        oos_net_metrics.get("n", np.nan),
        oos_net_metrics.get("skew", np.nan),
        oos_net_metrics.get("kurt", np.nan),
        dsr_benchmark_sr_pp,
    )

    verdict = _survival_verdict(oos_net_metrics, psr_oos_net, dsr_prob_oos)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    grid.to_csv(outdir / "grid_results.csv", index=False)
    detail["position"].to_csv(outdir / "selected_positions_executable_shifted.csv")
    detail["daily_target_unshifted"].to_csv(outdir / "selected_daily_target_unshifted.csv")
    detail["weekly_target"].to_csv(outdir / "selected_weekly_target.csv")
    detail["scores"].to_csv(outdir / "selected_weekly_scores.csv")
    pd.DataFrame({"gross": detail["gross"], "net": detail["net"]}).to_csv(
        outdir / "selected_daily_pnl.csv"
    )
    carry_ret.to_csv(outdir / "carry_unit_returns.csv")
    funding_daily.to_csv(outdir / "funding_daily.csv")

    payload = {
        "experiment": "cand_funding_carry_xsec_lowturn",
        "design": {
            "carry_unit": "+1 = long spot / short perp, -1 = short spot / long perp",
            "signal": "weekly sum funding, rolling weekly mean score",
            "rebalance": "weekly target, hysteresis rank bands, daily ffill",
            "execution_lag_bars": args.signal_lag_bars,
            "cost": "10bp per leg, implemented as 20bp on synthetic carry-unit turnover",
            "oos_rule": f"engine bt.oos_split train_frac={args.train_frac}; select by train net Sharpe only",
            "alignment": "daily kline index is open_time; returns are close-to-close; signal shifted before bt.run",
            "maxdd_caveat": "delta-neutral close-to-close maxDD excludes intraday marks, margin, borrow, and execution gaps",
        },
        "window": {
            "start": start_ts.isoformat(),
            "end": end_ts.isoformat(),
            "train_start": carry_ret.index[train_slice][0].isoformat(),
            "train_end": carry_ret.index[train_slice][-1].isoformat(),
            "oos_start": carry_ret.index[oos_slice][0].isoformat(),
            "oos_end": carry_ret.index[oos_slice][-1].isoformat(),
        },
        "symbols_requested": symbols,
        "symbols_used": list(carry_ret.columns),
        "selected_config": asdict(selected_cfg),
        "selected_key": selected_key,
        "selected_summary": selected_summary,
        "selected_metrics": {
            "net": detail["net_metrics"],
            "gross": detail["gross_metrics"],
            "train_turnover": detail["train_turnover"],
            "oos_turnover": detail["oos_turnover"],
            "psr_oos_net": psr_oos_net,
            "dsr_benchmark_sr_pp_from_train_trials": dsr_benchmark_sr_pp,
            "dsr_probability_oos_net": dsr_prob_oos,
            "dsr_raw": dsr_raw,
            "verdict": verdict,
        },
        "data_quality": quality,
        "artifacts": {
            "grid_results": str(outdir / "grid_results.csv"),
            "positions": str(outdir / "selected_positions_executable_shifted.csv"),
            "daily_pnl": str(outdir / "selected_daily_pnl.csv"),
            "weekly_scores": str(outdir / "selected_weekly_scores.csv"),
            "summary": str(outdir / "summary.json"),
        },
    }

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, indent=2, sort_keys=True)

    print("\nSELECTED CONFIG")
    print(json.dumps(_jsonify(asdict(selected_cfg)), indent=2, sort_keys=True))

    print("\nOOS GROSS")
    oos_gross = detail["gross_metrics"]["oos"]
    print(
        "n={n:.0f} sharpe={sharpe} ret_ann={ret} vol_ann={vol} maxdd={dd}".format(
            n=oos_gross.get("n", np.nan),
            sharpe=_fmt_float(oos_gross.get("sharpe_ann", np.nan)),
            ret=_fmt_pct(oos_gross.get("ret_ann", np.nan)),
            vol=_fmt_pct(oos_gross.get("vol_ann", np.nan)),
            dd=_fmt_pct(oos_gross.get("maxdd", np.nan)),
        )
    )

    print("\nOOS NET, 10BP/LEG")
    print(
        "n={n:.0f} sharpe={sharpe} sr_pp={sr_pp} ret_ann={ret} vol_ann={vol} "
        "maxdd={dd} hit={hit} psr={psr} dsr_prob={dsr}".format(
            n=oos_net_metrics.get("n", np.nan),
            sharpe=_fmt_float(oos_net_metrics.get("sharpe_ann", np.nan)),
            sr_pp=_fmt_float(oos_net_metrics.get("sr_pp", np.nan), 6),
            ret=_fmt_pct(oos_net_metrics.get("ret_ann", np.nan)),
            vol=_fmt_pct(oos_net_metrics.get("vol_ann", np.nan)),
            dd=_fmt_pct(oos_net_metrics.get("maxdd", np.nan)),
            hit=_fmt_pct(oos_net_metrics.get("hit", np.nan)),
            psr=_fmt_float(psr_oos_net, 4),
            dsr=_fmt_float(dsr_prob_oos, 4),
        )
    )

    print("\nREALIZED TURNOVER")
    oos_turnover = detail["oos_turnover"]
    print(
        "pair_turnover/bar={pair} leg_turnover/bar={leg} implied_cost_ann={cost} "
        "trade_days={days} avg_active_names={names}".format(
            pair=_fmt_float(oos_turnover["pair_turnover_per_bar"], 6),
            leg=_fmt_float(oos_turnover["leg_turnover_per_bar"], 6),
            cost=_fmt_pct(oos_turnover["implied_cost_ann_from_leg_turnover"]),
            days=int(oos_turnover["rebalance_days_with_trade"]),
            names=_fmt_float(oos_turnover["avg_active_names"], 2),
        )
    )

    print("\nMULTIPLE TESTING")
    print(
        f"K={len(configs)} dsr_benchmark_sr_pp={_fmt_float(dsr_benchmark_sr_pp, 6)} "
        f"selection_rule=train_net_sharpe"
    )

    print("\nVERDICT")
    print(verdict)

    print("\nCAVEAT")
    print(
        "MaxDD is close-to-close on a synthetic delta-neutral pair. Treat it as optimistic: "
        "it excludes intraday basis moves, margin/liquidation path, borrow/short constraints, "
        "real spot/perp execution, and funding timestamp microstructure."
    )

    print(f"\n[INFO] artifacts written to {outdir}")


if __name__ == "__main__":
    main()
```

設計重點：

- `+1` 定義為 `long spot / short perp`，所以正 funding 會加到 carry return；`-1` 則是反向，用來吃負 funding。
- 訊號只用 weekly funding sum，再用 rolling weekly mean 做 persistent score，並且每日執行前 `shift(1)`，避免 look-ahead。
- 成本用 `10bp/leg`，但 position 是一個合成 carry unit，所以傳給 engine 的成本是 `20bp`。
- 參數只用前 60% train 選，排序依據是 train net Sharpe，最後 40% OOS 才報 gross/net、PSR、DSR、realized turnover。
- hysteresis 是 rank exit band：已持有標的只要還在 `basket_n + hysteresis_ranks` 內就不換，降低 churn。
- artifacts 會輸出 grid、selected positions、weekly scores、daily PnL、summary JSON，方便你或 Claude 查 turnover 和資料品質。