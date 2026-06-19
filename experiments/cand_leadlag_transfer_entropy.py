"""cand_leadlag_transfer_entropy  (family = lead-lag)

HYPOTHESIS: BTC is the market beta leader. If BTC's return at bar t-L carries
predictive information about an alt's return at bar t (BTC LEADS the alt by L
bars), we can trade the alt on BTC's *lagged, already-observed* return. We
detect the lead with two independent measures, BOTH computed in-sample only:

  (1) Binned TRANSFER ENTROPY  TE(BTC -> alt) at lag L, on 3-bin discretized
      returns (sign-ish terciles). TE_{X->Y} = sum p(y_t, y_{t-1}, x_{t-L})
      * log[ p(y_t | y_{t-1}, x_{t-L}) / p(y_t | y_{t-1}) ]  (Schreiber 2000).
      Significance via a circular-shift / shuffle null (permute BTC) -> z-score.
  (2) st.lagged_xcorr(BTC_ret, alt_ret) -> best lag* and signed corr.

  A coin "qualifies" iff TE is significant (z >= TE_Z_GATE) AND xcorr agrees:
  same lag bucket and |corr| above XC_GATE. The TRADE on the qualifying lag L:
      position_alt[t] = sign(corr) * sign( BTC_ret[t-L] )     (decided pre-bar)
  i.e. follow BTC's lagged move. Optionally hedge out BTC beta to be market-
  neutral (subtract beta*BTC_ret from the alt leg). All gating params are fixed
  on the first 60% (IS); the LAST 40% is the untouched OOS.

HONEST PRIORS (governance rule 7): raw directional momentum/reversal already
DIED here, and a BTC-lead signal is a *cross-asset momentum* bet, so the strong
prior is DEAD-on-cost. We test it properly anyway. Turnover is reported and is
the main killer: a sign(BTC_ret[t-L]) signal flips almost every bar.

Outputs reports/cand_leadlag_transfer_entropy.json with the OOS verdict.
"""
from __future__ import annotations
import sys, json, time, warnings, pathlib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import fetch_binance as fb
from engine import backtest as bt
from engine import stats as st

# ---- universe / config -------------------------------------------------------
ALTS = ["ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
        "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT"]
LEADER = "BTCUSDT"
START = "2022-01-01"
TRAIN_FRAC = 0.60
MAX_LAG = 3            # bars (4h: up to 12h lead; 1d: up to 3d lead)
N_BINS = 3            # tercile discretization for TE
N_SHUFFLE = 200       # circular-shift null for TE significance
TE_Z_GATE = 2.0      # TE z-score vs shuffled null
XC_GATE = 0.05      # |lagged xcorr| floor on IS
COST_BPS = 5.0        # taker per leg per |Δposition|

# grids we sweep on IS only (counts toward DSR family size)
CONFIGS = [("4h", "4h", 2190.0), ("1d", "1d", 365.0)]   # (label, interval, ppy)


def _ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


# ---- transfer entropy --------------------------------------------------------
def _discretize(r: np.ndarray, n_bins: int) -> np.ndarray:
    """Map returns to {0..n_bins-1} by IS-style quantile bins (computed on r)."""
    r = np.asarray(r, float)
    qs = np.quantile(r, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(r, qs)


def transfer_entropy(x_lead: np.ndarray, y_follow: np.ndarray, lag: int,
                     n_bins: int = N_BINS) -> float:
    """TE(x -> y) at `lag`: extra info x_{t-lag} gives about y_t beyond y_{t-1}.

    History order 1 on the target (y_{t-1}); single source lag x_{t-lag}.
    Plug-in estimator on joint counts; returns nats (>=0 up to estimator noise).
    """
    x = _discretize(x_lead, n_bins)
    y = _discretize(y_follow, n_bins)
    n = len(y)
    if n <= lag + 2:
        return 0.0
    yt = y[lag:]          # y_t
    yp = y[lag - 1:-1]    # y_{t-1}
    xs = x[:n - lag]      # x_{t-lag}
    m = len(yt)
    # joint counts p(yt, yp, xs)
    joint = np.zeros((n_bins, n_bins, n_bins))
    for a, b, c in zip(yt, yp, xs):
        joint[a, b, c] += 1.0
    joint /= m
    te = 0.0
    p_yp = joint.sum(axis=(0, 2))                     # p(y_{t-1})
    p_yp_xs = joint.sum(axis=0)                       # p(y_{t-1}, x_{t-lag})
    p_yt_yp = joint.sum(axis=2)                       # p(y_t, y_{t-1})
    for a in range(n_bins):
        for b in range(n_bins):
            for c in range(n_bins):
                pj = joint[a, b, c]
                if pj <= 0:
                    continue
                # p(y_t | y_{t-1}, x_{t-lag}) / p(y_t | y_{t-1})
                num = pj / p_yp_xs[b, c] if p_yp_xs[b, c] > 0 else 0.0
                den = p_yt_yp[a, b] / p_yp[b] if p_yp[b] > 0 else 0.0
                if num > 0 and den > 0:
                    te += pj * np.log(num / den)
    return float(max(te, 0.0))


def te_significance(x_lead, y_follow, lag, n_shuffle=N_SHUFFLE, seed=0):
    """z-score of observed TE vs a circular-shift null on the source series."""
    obs = transfer_entropy(x_lead, y_follow, lag)
    rng = np.random.default_rng(seed)
    x = np.asarray(x_lead, float)
    n = len(x)
    null = np.empty(n_shuffle)
    for i in range(n_shuffle):
        shift = int(rng.integers(lag + 2, n - lag - 2))
        xs = np.roll(x, shift)          # break temporal coupling, keep marginal
        null[i] = transfer_entropy(xs, y_follow, lag)
    mu, sd = null.mean(), null.std()
    z = (obs - mu) / sd if sd > 0 else 0.0
    return float(obs), float(z)


# ---- data --------------------------------------------------------------------
def load_panel(interval, end):
    close = {}
    led = fb.klines(LEADER, interval, _ms(START), end, futures=True)
    if led is None:
        return None, None
    close[LEADER] = led["close"]
    for s in ALTS:
        k = fb.klines(s, interval, _ms(START), end, futures=True)
        if k is not None and len(k) > 300:
            close[s] = k["close"]
    C = pd.DataFrame(close).dropna()
    R = C.pct_change().dropna()
    return C, R


# ---- one coin ----------------------------------------------------------------
def evaluate_coin(btc_ret, alt_ret, ppy, hedge):
    """Returns dict with IS gate diagnostics and OOS metrics (None if no trade)."""
    n = len(alt_ret)
    tr, te_slice = bt.oos_split(n, TRAIN_FRAC)
    b_is, a_is = btc_ret[tr], alt_ret[tr]

    # (2) xcorr on IS
    xlag, xcorr = st.lagged_xcorr(b_is, a_is, MAX_LAG)

    # (1) TE at the xcorr-preferred lag (and scan all lags for the best TE z)
    best = (0, 0.0, 0.0)   # (lag, te, z)
    for L in range(1, MAX_LAG + 1):
        te_val, te_z = te_significance(b_is, a_is, L, seed=L)
        if te_z > best[2]:
            best = (L, te_val, te_z)
    te_lag, te_val, te_z = best

    # gate: TE significant AND xcorr agrees in lag and is strong enough
    qualifies = (te_z >= TE_Z_GATE and xlag == te_lag and abs(xcorr) >= XC_GATE)
    L = te_lag
    direction = np.sign(xcorr) if xcorr != 0 else 1.0

    # build signal on FULL series, evaluate OOS only
    pos = direction * np.sign(np.concatenate([np.zeros(L), btc_ret[:-L]]))
    pos = np.nan_to_num(pos)

    if hedge:
        # market-neutral: trade (alt - beta*BTC). beta from IS.
        beta = np.polyfit(b_is, a_is, 1)[0]
        traded_ret = alt_ret - beta * btc_ret
        # two legs -> double the cost on |Δpos| (alt leg + btc hedge leg)
        net = bt.run(traded_ret, pos, COST_BPS * 2)
    else:
        net = bt.run(alt_ret, pos, COST_BPS)

    mo = bt.metrics(net[te_slice], ppy, pos[te_slice])
    mi = bt.metrics(bt.run(alt_ret if not hedge else
                           alt_ret - np.polyfit(b_is, a_is, 1)[0] * btc_ret,
                           pos, COST_BPS * (2 if hedge else 1))[tr], ppy, pos[tr])
    p = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
    return dict(lag=L, te=te_val, te_z=te_z, xlag=xlag, xcorr=xcorr,
                qualifies=bool(qualifies), IS_sharpe=mi["sharpe_ann"],
                OOS=mo, OOS_psr=p, net_oos=net[te_slice])


def main():
    end = int(time.time() * 1000)
    summary = dict(generated=pd.Timestamp.utcnow().isoformat(), start=START,
                   cost_bps=COST_BPS, train_frac=TRAIN_FRAC, max_lag=MAX_LAG,
                   te_z_gate=TE_Z_GATE, xc_gate=XC_GATE, runs=[])

    all_sr_pp = []          # for DSR family deflation (every coin-config-hedge trial)
    best_overall = None

    for label, interval, ppy in CONFIGS:
        C, R = load_panel(interval, end)
        if R is None or LEADER not in R.columns:
            print(f"[{label}] no data"); continue
        print(f"\n=== {label} panel: {R.shape[0]} bars x {R.shape[1]} coins "
              f"({R.index[0].date()} -> {R.index[-1].date()}) ===")
        btc = R[LEADER].values

        for hedge in (False, True):
            tag = "BTC-hedged(MN)" if hedge else "directional"
            print(f"\n  -- {label} / {tag} --")
            qual_nets, qual_names, rows = [], [], []
            for s in [c for c in R.columns if c != LEADER]:
                res = evaluate_coin(btc, R[s].values, ppy, hedge)
                all_sr_pp.append(res["OOS"]["sr_pp"])
                rows.append((s, res))
                if res["qualifies"]:
                    qual_nets.append(pd.Series(res["net_oos"],
                                     index=R.index[-len(res["net_oos"]):]))
                    qual_names.append(s)
            # print per-coin
            print(f"     {'coin':9} {'lag':>3} {'TE':>7} {'TEz':>6} {'xL':>3} "
                  f"{'xcorr':>7} {'qual':>4} {'OOSshrp':>8} {'OOSret%':>8} "
                  f"{'turn':>6} {'PSR':>5}")
            for s, res in sorted(rows, key=lambda x: -x[1]["OOS"]["sharpe_ann"]):
                o = res["OOS"]
                print(f"     {s:9} {res['lag']:3d} {res['te']:7.4f} "
                      f"{res['te_z']:6.2f} {res['xlag']:3d} {res['xcorr']:7.3f} "
                      f"{('Y' if res['qualifies'] else '.'):>4} "
                      f"{o['sharpe_ann']:8.2f} {o['ret_ann']*100:8.1f} "
                      f"{o['turnover']:6.2f} {res['OOS_psr']:5.2f}")

            # qualifying-coin equal-weight OOS portfolio
            if qual_nets:
                port = pd.concat(qual_nets, axis=1).fillna(0).mean(axis=1).values
                pm = bt.metrics(port, ppy)
                pp = bt.psr(pm["sr_pp"], pm["n"], pm["skew"], pm["kurt"])
                print(f"     >> QUALIFYING PORTFOLIO ({len(qual_names)} coins: "
                      f"{','.join(c[:-4] for c in qual_names)})")
                print(f"        OOS Sharpe={pm['sharpe_ann']:.2f} "
                      f"ret={pm['ret_ann']*100:.1f}% maxDD={pm['maxdd']*100:.1f}% "
                      f"turn={pm['turnover']:.2f} PSR={pp:.3f}")
                run = dict(config=label, mode=tag, ppy=ppy,
                           n_qual=len(qual_names), qual=qual_names,
                           OOS=pm, OOS_psr=pp)
                summary["runs"].append(run)
                cand = (pm["sharpe_ann"], pm["sr_pp"], pp, pm, label, tag,
                        len(qual_names), ppy)
                if best_overall is None or cand[0] > best_overall[0]:
                    best_overall = cand
            else:
                print("     >> no coin passed the IS TE+xcorr gate")
                summary["runs"].append(dict(config=label, mode=tag, ppy=ppy,
                                            n_qual=0, qual=[], OOS=None,
                                            OOS_psr=None))

    # ---- DSR deflation across the whole family of trials -----------------
    sr_star = bt.dsr_benchmark(all_sr_pp)
    summary["n_trials"] = len(all_sr_pp)
    summary["sr_star_pp"] = sr_star

    if best_overall is not None:
        shrp, sr_pp, pp, pm, label, tag, nq, ppy = best_overall
        dsr = bt.psr(pm["sr_pp"], pm["n"], pm["skew"], pm["kurt"],
                     sr_benchmark=sr_star)
        # verdict per VERDICT RULE: EDGE iff OOS PSR>=0.95 & survives cost &
        # no fatal artifact & turnover feasible. Cost already charged; turnover
        # is the flagged feasibility risk for a per-bar sign flip.
        turn = pm["turnover"]
        if pp >= 0.95 and dsr >= 0.95 and shrp > 0:
            verdict = "EDGE"
        elif pp >= 0.80 and shrp > 0:
            verdict = "MARGINAL"
        else:
            verdict = "DEAD"
        summary["best"] = dict(config=label, mode=tag, n_qual=nq,
                               OOS_sharpe=shrp, OOS_ret_ann=pm["ret_ann"],
                               OOS_maxdd=pm["maxdd"], turnover=turn,
                               OOS_psr=pp, DSR_psr=dsr, sr_star_pp=sr_star)
        summary["verdict"] = verdict
        print(f"\n=== BEST QUALIFYING: {label}/{tag} ({nq} coins) ===")
        print(f"    OOS Sharpe={shrp:.2f} ret={pm['ret_ann']*100:.1f}% "
              f"maxDD={pm['maxdd']*100:.1f}% turn={turn:.2f}")
        print(f"    PSR={pp:.3f}  DSR(vs SR*={sr_star:.4f})={dsr:.3f}  "
              f"[{len(all_sr_pp)} trials]")
        print(f"    VERDICT = {verdict}")
    else:
        summary["best"] = None
        summary["verdict"] = "DEAD"
        print("\n=== NO COIN QUALIFIED on the IS TE+xcorr gate -> DEAD ===")

    out = ROOT / "reports" / "cand_leadlag_transfer_entropy.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {out}")
    return summary


if __name__ == "__main__":
    main()
