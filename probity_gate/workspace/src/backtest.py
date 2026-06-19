"""Vectorized backtest + honest statistics.

Discipline baked in:
- position is shifted 1 bar (decide on close t, hold over t+1) -> no look-ahead.
- explicit cost = |Δposition| * cost_bps on every rebalance.
- Sharpe reported with a non-normality-robust standard error (Lo 2002).
- Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR), Bailey &
  Lopez de Prado (2014), so multiple-testing inflation is penalised.
"""
from __future__ import annotations
import numpy as np
from scipy import stats

EULER = 0.5772156649015329


def run(asset_ret: np.ndarray, position: np.ndarray, cost_bps: float):
    """asset_ret[t] = simple return of holding from t-1 to t.
    position[t] = target exposure decided using info up to t-1 (already aligned).
    Returns net per-period strategy returns."""
    asset_ret = np.asarray(asset_ret, float)
    position = np.asarray(position, float)
    n = min(len(asset_ret), len(position))
    asset_ret, position = asset_ret[-n:], position[-n:]
    turn = np.abs(np.diff(position, prepend=0.0))
    gross = position * asset_ret
    cost = turn * (cost_bps / 1e4)
    return gross - cost


def metrics(net: np.ndarray, ppy: float, position: np.ndarray | None = None) -> dict:
    net = np.asarray(net, float)
    net = net[np.isfinite(net)]
    n = len(net)
    if n < 10 or net.std(ddof=1) == 0:
        return dict(n=n, sharpe_ann=np.nan, sr_pp=np.nan, ret_ann=np.nan,
                    vol_ann=np.nan, maxdd=np.nan, hit=np.nan, turnover=np.nan,
                    skew=np.nan, kurt=np.nan)
    mu, sd = net.mean(), net.std(ddof=1)
    sr_pp = mu / sd                                   # per-period Sharpe
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(eq)
    maxdd = float((eq / peak - 1).min())
    turn = float(np.abs(np.diff(position, prepend=0.0)).mean()) if position is not None else np.nan
    return dict(
        n=n,
        sharpe_ann=float(sr_pp * np.sqrt(ppy)),
        sr_pp=float(sr_pp),
        ret_ann=float(mu * ppy),
        vol_ann=float(sd * np.sqrt(ppy)),
        maxdd=maxdd,
        hit=float((net > 0).mean()),
        turnover=turn,
        skew=float(stats.skew(net)),
        kurt=float(stats.kurtosis(net, fisher=False)),
    )


def psr(sr_pp: float, n: int, skew: float, kurt: float, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true per-period SR > sr_benchmark)."""
    if not np.isfinite(sr_pp) or n < 10:
        return np.nan
    denom = np.sqrt(max(1e-12, 1 - skew * sr_pp + (kurt - 1) / 4 * sr_pp ** 2))
    z = (sr_pp - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def dsr_benchmark(sr_pp_trials: list[float]) -> float:
    """Expected max per-period Sharpe under N independent trials (the SR* to beat).
    Uses the variance of the Sharpes actually tried (Bailey-LdP)."""
    sr = np.asarray([s for s in sr_pp_trials if np.isfinite(s)], float)
    N = len(sr)
    if N < 2:
        return 0.0
    v = sr.std(ddof=1)
    z1 = stats.norm.ppf(1 - 1.0 / N)
    z2 = stats.norm.ppf(1 - 1.0 / (N * np.e))
    return float(v * ((1 - EULER) * z1 + EULER * z2))


def oos_split(n: int, train_frac: float = 0.6):
    """Chronological split indices: (train_slice, test_slice)."""
    cut = int(n * train_frac)
    return slice(0, cut), slice(cut, n)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    # a genuine small-edge series: mean 0.0006/period, sd 0.01  (SR_pp=0.06)
    edge = rng.normal(0.0006, 0.01, 5000)
    pos = np.ones(5000)
    net = run(edge, pos, cost_bps=0)
    m = metrics(net, ppy=365, position=pos)
    p = psr(m["sr_pp"], m["n"], m["skew"], m["kurt"])
    assert m["sharpe_ann"] > 0.8, m["sharpe_ann"]
    assert p > 0.95, p
    # pure noise should NOT pass PSR
    noise = run(rng.normal(0, 0.01, 5000), pos, 0)
    mn = metrics(noise, 365, pos)
    pn = psr(mn["sr_pp"], mn["n"], mn["skew"], mn["kurt"])
    assert pn < 0.95, pn
    # DSR benchmark rises with more trials
    b = dsr_benchmark([0.0, 0.01, -0.02, 0.03, 0.05, -0.01, 0.02] * 5)
    assert b > 0, b
    print("backtest SELF-CHECK OK | edge SR_ann=%.2f PSR=%.3f | noise PSR=%.3f | SR*=%.4f"
          % (m["sharpe_ann"], p, pn, b))
