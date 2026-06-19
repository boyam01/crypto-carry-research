"""Math toolkit: turn classic stochastic-process / time-series formulas into
edge detectors that consume price/return series.

Each function is pure (numpy in, scalar/tuple out) so it can be unit-checked.
References are named so a method maps back to a paper.
"""
from __future__ import annotations
import numpy as np


def _clean(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def variance_ratio(returns, q: int) -> tuple[float, float]:
    """Lo-MacKinlay (1988) variance ratio test on log returns.

    VR(q) = Var(q-period return)/(q*Var(1-period)). VR>1 => trending /
    positive autocorr; VR<1 => mean-reverting. Returns (VR, het-robust z).
    """
    r = _clean(returns)
    n = len(r)
    if n < 2 * q + 2:
        return np.nan, np.nan
    mu = r.mean()
    var1 = np.sum((r - mu) ** 2) / (n - 1)
    if var1 == 0:
        return np.nan, np.nan
    # overlapping q-period sums
    rq = np.convolve(r, np.ones(q), "valid")          # length n-q+1
    m = q * (n - q + 1) * (1 - q / n)
    varq = np.sum((rq - q * mu) ** 2) / m
    vr = varq / var1
    # heteroskedasticity-robust standard error (Lo-MacKinlay eq. 18)
    theta = 0.0
    e2 = (r - mu) ** 2
    for j in range(1, q):
        w = 2 * (q - j) / q
        cov = np.sum(e2[j:] * e2[:-j]) / (np.sum(e2) ** 2 / n)
        theta += (w ** 2) * cov
    z = (vr - 1) / np.sqrt(theta / n) if theta > 0 else np.nan
    return float(vr), float(z)


def hurst_dfa(x, scales=None) -> float:
    """Hurst exponent via Detrended Fluctuation Analysis (Peng 1994).

    H~0.5 random walk; H>0.5 persistent/trending; H<0.5 anti-persistent.
    Input is a price/level series (integrated). Returns H.
    """
    x = _clean(x)
    n = len(x)
    if n < 100:
        return np.nan
    y = np.cumsum(x - x.mean())
    if scales is None:
        scales = np.unique(np.floor(np.logspace(np.log10(8),
                            np.log10(n // 4), 20)).astype(int))
    F = []
    for s in scales:
        if s < 4:
            continue
        nseg = n // s
        rms = []
        for i in range(nseg):
            seg = y[i * s:(i + 1) * s]
            t = np.arange(s)
            c = np.polyfit(t, seg, 1)
            fit = np.polyval(c, t)
            rms.append(np.sqrt(np.mean((seg - fit) ** 2)))
        if rms:
            F.append((s, np.mean(rms)))
    if len(F) < 4:
        return np.nan
    s_arr = np.log([f[0] for f in F])
    f_arr = np.log([f[1] for f in F])
    return float(np.polyfit(s_arr, f_arr, 1)[0])


def ou_half_life(level) -> float:
    """Mean-reversion half-life from AR(1) / Ornstein-Uhlenbeck fit.

    Regress dX_t on X_{t-1}: dX = a + b*X_{t-1}. lambda=-b, half-life=ln2/lambda.
    Returns bars to revert half-way; inf if not mean-reverting (b>=0).
    """
    x = _clean(level)
    if len(x) < 30:
        return np.inf
    xlag = x[:-1]
    dx = np.diff(x)
    b = np.polyfit(xlag, dx, 1)[0]
    if b >= 0:
        return np.inf
    return float(np.log(2) / -b)


def acf1(returns) -> float:
    """First-lag autocorrelation of returns."""
    r = _clean(returns)
    if len(r) < 10:
        return np.nan
    return float(np.corrcoef(r[:-1], r[1:])[0, 1])


def ljung_box(returns, lags: int = 10) -> tuple[float, float]:
    """Ljung-Box Q statistic for autocorrelation up to `lags`. Returns (Q, p)."""
    from scipy import stats
    r = _clean(returns)
    n = len(r)
    if n < lags + 5:
        return np.nan, np.nan
    r = r - r.mean()
    c0 = np.sum(r ** 2)
    Q = 0.0
    for k in range(1, lags + 1):
        ck = np.sum(r[k:] * r[:-k])
        rho = ck / c0
        Q += rho ** 2 / (n - k)
    Q *= n * (n + 2)
    p = 1 - stats.chi2.cdf(Q, lags)
    return float(Q), float(p)


def hawkes_branching(counts) -> float:
    """Crude self-excitation (branching ratio) proxy for an event-count series.

    Branching ratio n in (0,1): fraction of events that are 'children'.
    Estimated as lag-1 autocorr of counts clipped to [0,1). High => clustered
    (self-exciting) order flow (Hawkes, Bacry-Mastromatteo-Muzy 2015).
    """
    a = acf1(counts)
    if not np.isfinite(a):
        return np.nan
    return float(min(max(a, 0.0), 0.999))


def lagged_xcorr(lead, follow, max_lag: int = 6) -> tuple[int, float]:
    """Best positive lag where `lead` predicts `follow` (lead-lag detection).

    Returns (lag*, corr) with lag*>0 meaning lead leads follow by lag* bars.
    """
    a, b = _clean(lead), _clean(follow)
    n = min(len(a), len(b))
    a, b = a[-n:], b[-n:]
    best = (0, 0.0)
    for L in range(1, max_lag + 1):
        if n - L < 20:
            break
        c = np.corrcoef(a[:-L], b[L:])[0, 1]
        if np.isfinite(c) and abs(c) > abs(best[1]):
            best = (L, float(c))
    return best


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # white noise: VR~1, H~0.5, no LB autocorr
    wn = rng.standard_normal(4000)
    vr, z = variance_ratio(wn, 8)
    assert abs(vr - 1) < 0.15, vr
    assert 0.4 < hurst_dfa(wn) < 0.6
    # trending (random walk drift in returns via AR(1) +): H should rise
    ar = np.zeros(4000)
    for i in range(1, 4000):
        ar[i] = 0.4 * ar[i - 1] + rng.standard_normal()
    assert acf1(ar) > 0.2
    vr2, _ = variance_ratio(ar, 4)
    assert vr2 > 1.1, vr2
    # mean-reverting OU: finite half-life
    ou = np.zeros(2000)
    for i in range(1, 2000):
        ou[i] = ou[i - 1] - 0.1 * ou[i - 1] + rng.standard_normal()
    hl = ou_half_life(ou)
    assert 3 < hl < 12, hl
    q, p = ljung_box(ar, 10)
    assert p < 0.01, (q, p)
    print("stats SELF-CHECK OK  | VR(wn)=%.3f H(wn)=%.3f acf1(ar)=%.3f OU-HL=%.2f"
          % (vr, hurst_dfa(wn), acf1(ar), hl))
