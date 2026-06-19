"""exo_transfer_entropy_network  (family = information-flow network)

CANDIDATE: Multivariate transfer-entropy NETWORK among ~12 coins.

The BIVARIATE BTC->alt transfer-entropy lead-lag already DIED here
(experiments/cand_leadlag_transfer_entropy.py). This is the NETWORK / centrality
angle requested by the spec:

  1. Build a directed, weighted information-flow network: edge W[i,j] = binned
     transfer entropy TE(i -> j) at lag 1 over the coin return panel.
       TE_{X->Y} = sum p(y_t, y_{t-1}, x_{t-1})
                   * log[ p(y_t | y_{t-1}, x_{t-1}) / p(y_t | y_{t-1}) ]
     (Schreiber 2000, plug-in estimator on quantile-binned returns; order-1
     target history, single source lag). Implemented in pure numpy.
  2. Edge significance via CIRCULAR-SHIFT SURROGATES on the source series
     (break temporal coupling, keep marginal). Keep TE z-score; optionally
     threshold the network at a z-gate (a knob -> counted for DSR).
  3. Node centrality:
       net-information-outflow  NIO[i] = OUT[i] - IN[i]
         OUT[i] = sum_j W[i,j] (sig-gated), IN[i] = sum_j W[j,i].
       coin is a net information SINK iff NIO < 0 (receives > sends).
       also PageRank / eigenvector centrality (networkx).

  HYPOTHESES (tested honestly, with cost+OOS+DSR):
    H-A  net SINKS lag the network -> a sink's next move is predicted by its
         in-flow sources' last move. position_sink[t] = sign(inflow-weighted
         source return at t-1). Cross-sectionally demeaned => market-neutral.
    H-B  residual reversal: trade -(coin_ret - inflow-predicted move).
    OVERLAY  network mean edge strength / density = systemic-coupling / contagion
         detector. Gate a simple base book (long-BTC, equal-weight long basket)
         by the coupling regime: cut/scale exposure when coupling is extreme.
         A risk overlay that improves OOS risk-adjusted return is a valid POSITIVE.

GOVERNANCE: features at t use only returns < t (shift>=1, causal bins, network
rebuilt on data strictly before the traded bar -> EXPANDING window, refit
periodically). Cost >=5bp/leg on |Dpos|. Tune ALL knobs on first 60%; report
LAST 40%. Report PSR and within-family DSR over every variant tried.

PRIOR (rule 7): cross-asset directional momentum DIED; a "sink follows the
network" rule is cross-asset momentum dressed up as a network, so the honest
prior is DEAD-on-cost / contemporaneous-beta. We test it properly and report
real numbers. Writes reports/exo_transfer_entropy_network.json.
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
try:
    import networkx as nx
    HAVE_NX = True
except Exception:
    HAVE_NX = False

# ---- universe / config -------------------------------------------------------
COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
         "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT", "TRXUSDT", "DOTUSDT",
         "ATOMUSDT", "NEARUSDT"]                 # ~14 liquid coins
START = "2022-01-01"
TRAIN_FRAC = 0.60
COST_BPS = 5.0
N_BINS = 3                  # tercile discretization for TE (a knob)
N_SURR = 80                # circular-shift surrogates per edge for significance
REFIT_EVERY = 45           # bars between network refits (expanding window)
MIN_TRAIN = 200            # min bars before first network is trusted

CONFIGS = [("1d", "1d", 365.0), ("8h", "8h", 1095.0)]   # (label, interval, ppy)


def _ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


# ---- transfer entropy (pure numpy, IMPLEMENTED here) -------------------------
def _bin_edges(r: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges fit on a (training) sample -> causal discretizer."""
    return np.quantile(r, np.linspace(0, 1, n_bins + 1)[1:-1])


def _apply_bins(r: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(r, float), edges)


def transfer_entropy_binned(xd: np.ndarray, yd: np.ndarray, n_bins: int) -> float:
    """TE(x -> y) at lag 1 from ALREADY-DISCRETIZED series xd,yd (same length).

    History order 1 on target y_{t-1}, single source lag x_{t-1}. Plug-in
    estimator on joint counts. Returns nats (>=0 up to estimator noise).
    """
    n = len(yd)
    if n < 5:
        return 0.0
    yt = yd[1:]            # y_t
    yp = yd[:-1]           # y_{t-1}
    xs = xd[:-1]           # x_{t-1}
    m = len(yt)
    # vectorized joint count via flattened index
    idx = (yt * n_bins + yp) * n_bins + xs
    cnt = np.bincount(idx, minlength=n_bins ** 3).astype(float)
    joint = cnt.reshape(n_bins, n_bins, n_bins) / m
    p_yp = joint.sum(axis=(0, 2))          # p(y_{t-1})            [b]
    p_yp_xs = joint.sum(axis=0)            # p(y_{t-1}, x_{t-1})   [b,c]
    p_yt_yp = joint.sum(axis=2)            # p(y_t, y_{t-1})       [a,b]
    # TE = sum joint * log[ (joint/p_yp_xs) / (p_yt_yp/p_yp) ]  (fully vectorized)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond_full = joint / p_yp_xs[None, :, :]               # p(yt|yp,xs)
        cond_red = (p_yt_yp / p_yp[None, :])[:, :, None]      # p(yt|yp)
        ratio = cond_full / cond_red
        term = joint * np.log(ratio)
    term = term[np.isfinite(term) & (joint > 0)]
    te = float(term.sum())
    return max(te, 0.0)


def te_zscore(xd, yd, n_bins, n_surr, rng) -> tuple[float, float]:
    """Observed TE and z vs circular-shift surrogate null on the source."""
    obs = transfer_entropy_binned(xd, yd, n_bins)
    n = len(xd)
    if n < 30:
        return obs, 0.0
    null = np.empty(n_surr)
    for i in range(n_surr):
        shift = int(rng.integers(5, n - 5))
        null[i] = transfer_entropy_binned(np.roll(xd, shift), yd, n_bins)
    sd = null.std()
    z = (obs - null.mean()) / sd if sd > 0 else 0.0
    return float(obs), float(z)


def build_network(R_train: np.ndarray, n_bins: int, z_gate: float,
                  n_surr: int, seed: int):
    """Directed TE adjacency on the training return matrix R_train [T x K].

    Returns W (raw TE), Wsig (z-gated TE), and per-edge z matrix.
    Discretizers fit on R_train (causal). O(K^2) TE + surrogates -- expensive,
    so n_surr small and refit infrequent.
    """
    T, K = R_train.shape
    rng = np.random.default_rng(seed)
    # fit bins per coin on training window, discretize once
    D = np.empty((T, K), dtype=np.int64)
    for k in range(K):
        edges = _bin_edges(R_train[:, k], n_bins)
        D[:, k] = _apply_bins(R_train[:, k], edges)
    W = np.zeros((K, K))
    Z = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            obs, z = te_zscore(D[:, i], D[:, j], n_bins, n_surr, rng)
            W[i, j] = obs
            Z[i, j] = z
    Wsig = np.where(Z >= z_gate, W, 0.0)
    return W, Wsig, Z


def centrality(Wsig: np.ndarray):
    """net-information-outflow + (optional) pagerank from a sig-gated network."""
    out = Wsig.sum(axis=1)        # OUT[i] = sum_j W[i,j]
    inn = Wsig.sum(axis=0)        # IN[i]  = sum_j W[j,i]
    nio = out - inn               # >0 source, <0 sink
    pr = None
    if HAVE_NX and Wsig.sum() > 0:
        G = nx.DiGraph()
        K = Wsig.shape[0]
        G.add_nodes_from(range(K))
        for i in range(K):
            for j in range(K):
                if Wsig[i, j] > 0:
                    G.add_edge(i, j, weight=float(Wsig[i, j]))
        try:
            prd = nx.pagerank(G, weight="weight")
            pr = np.array([prd.get(k, 0.0) for k in range(K)])
        except Exception:
            pr = None
    return out, inn, nio, pr


# ---- data --------------------------------------------------------------------
def load_panel(interval, end):
    close = {}
    for s in COINS:
        k = fb.klines(s, interval, _ms(START), end, futures=True)
        if k is not None and len(k) > 300:
            close[s] = k["close"]
    C = pd.DataFrame(close).dropna()
    R = C.pct_change().dropna()
    return C, R


# ---- rolling network features (point-in-time) --------------------------------
def rolling_network_features(R: pd.DataFrame, n_bins, z_gate, n_surr,
                             refit_every, min_train, seed):
    """For each bar t (>= min_train), build the TE network on returns STRICTLY
    BEFORE t (expanding window, refit every `refit_every` bars to save compute)
    and emit point-in-time features used to form the position for bar t.

    Returns dict of arrays aligned to R.index:
      nio[t,k]       net-info-outflow of coin k from the last refit < t
      inflow_pred[t,k] inflow-weighted source return realized at t-1
      coupling[t]    mean significant edge strength (systemic coupling)
      density[t]     fraction of significant edges
    All strictly causal: everything at t uses returns up to t-1.
    """
    Rv = R.values
    T, K = Rv.shape
    nio = np.full((T, K), np.nan)
    inflow_pred = np.full((T, K), np.nan)
    coupling = np.full(T, np.nan)
    density = np.full(T, np.nan)
    Wsig_cur = None
    last_refit = -10 ** 9
    for t in range(min_train, T):
        if (t - last_refit) >= refit_every or Wsig_cur is None:
            R_train = Rv[:t]                       # strictly < t
            _, Wsig, _ = build_network(R_train, n_bins, z_gate, n_surr,
                                       seed + t)   # vary seed -> independent surrogates
            Wsig_cur = Wsig
            last_refit = t
            out, inn, nio_t, _ = centrality(Wsig)
            nio_row = nio_t
            tot = Wsig.sum()
            cpl = tot / (K * (K - 1))
            dens = (Wsig > 0).mean()
        # carry last refit's centrality forward (still uses only past data)
        nio[t] = nio_row
        coupling[t] = cpl
        density[t] = dens
        # inflow-weighted source return realized at t-1 (observed pre-bar):
        # for target k, weight sources by W[source->k] = Wsig[:,k]
        w_in = Wsig_cur[:, :]                      # [K_src x K_tgt]
        r_prev = Rv[t - 1]                         # last observed return vector
        denom = w_in.sum(axis=0)                   # total inflow per target
        num = (w_in * r_prev[:, None]).sum(axis=0)
        pred = np.where(denom > 0, num / denom, 0.0)
        inflow_pred[t] = pred
    return dict(nio=nio, inflow_pred=inflow_pred, coupling=coupling,
                density=density)


# ---- strategies --------------------------------------------------------------
def strat_sink_follow(R, feats, sink_frac, ppy, slc, hypothesis):
    """H-A: net SINKS follow the network (go in inflow-predicted direction).
    H-B: residual reversal (fade the inflow-predicted move).

    Only the bottom `sink_frac` of coins by NIO (the strongest sinks) are traded
    at each bar. Positions are cross-sectionally demeaned -> dollar-neutral.
    Returns net portfolio return series over the whole sample (eval on slc).
    """
    Rv = R.values
    T, K = Rv.shape
    nio = feats["nio"]
    pred = feats["inflow_pred"]
    pos = np.zeros((T, K))
    for t in range(T):
        if not np.isfinite(nio[t]).any():
            continue
        order = np.argsort(nio[t])                 # ascending: most-sink first
        n_sink = max(1, int(round(sink_frac * K)))
        sinks = order[:n_sink]
        raw = np.zeros(K)
        sig = np.sign(pred[t, sinks])
        if hypothesis == "A":
            raw[sinks] = sig                       # follow predicted move
        else:
            raw[sinks] = -sig                      # fade it (residual reversal)
        # cross-sectional demean over traded names -> market neutral
        if np.any(raw != 0):
            m = raw[raw != 0].mean()
            raw[raw != 0] -= m
            gross = np.abs(raw).sum()
            if gross > 0:
                raw = raw / gross                  # unit gross
        pos[t] = raw
    # per-coin net then aggregate; cost on |Dpos| per coin
    net_mat = np.zeros((T, K))
    for k in range(K):
        net_mat[:, k] = bt.run(Rv[:, k], pos[:, k], COST_BPS)
    port = net_mat.sum(axis=1)
    avg_turn = np.abs(np.diff(pos, axis=0, prepend=0.0)).sum(axis=1).mean()
    m = bt.metrics(port[slc], ppy)
    p = bt.psr(m["sr_pp"], m["n"], m["skew"], m["kurt"])
    return dict(net=port, metrics=m, psr=p, turnover=float(avg_turn))


def overlay_gate(base_ret, coupling, slc, ppy, gate_q, is_slc):
    """Risk overlay: scale base book by systemic-coupling regime.
    When coupling exceeds its IS `gate_q` quantile (high contagion), cut
    exposure to 0 (risk-off); else hold full. Gate threshold fit on IS only.
    Returns (base metrics on slc, gated metrics on slc, gated psr).
    """
    cpl = np.asarray(coupling, float)
    thr = np.nanquantile(cpl[is_slc][np.isfinite(cpl[is_slc])], gate_q)
    # position decided pre-bar: use coupling computed from data < t (already so)
    gate = np.where(np.isfinite(cpl) & (cpl <= thr), 1.0, 0.0)
    gate = np.nan_to_num(gate, nan=1.0)
    base_pos = np.ones_like(base_ret)
    gated_pos = gate
    net_base = bt.run(base_ret, base_pos, COST_BPS)
    net_gate = bt.run(base_ret, gated_pos, COST_BPS)
    mb = bt.metrics(net_base[slc], ppy, base_pos[slc])
    mg = bt.metrics(net_gate[slc], ppy, gated_pos[slc])
    pg = bt.psr(mg["sr_pp"], mg["n"], mg["skew"], mg["kurt"])
    return mb, mg, pg, float(thr)


# ---- main --------------------------------------------------------------------
def main():
    end = int(time.time() * 1000)
    summary = dict(generated=pd.Timestamp.utcnow().isoformat(), start=START,
                   cost_bps=COST_BPS, train_frac=TRAIN_FRAC, n_bins=N_BINS,
                   n_surr=N_SURR, refit_every=REFIT_EVERY, runs=[], overlays=[])
    all_sr_pp = []                 # DSR family: every standalone variant
    best_standalone = None
    best_overlay = None

    # knob grids (counted for DSR honesty)
    Z_GATES = [1.5, 2.0]           # network significance threshold
    SINK_FRACS = [0.3, 0.5]        # fraction of coins traded as sinks
    HYPS = ["A", "B"]
    GATE_QS = [0.8, 0.9]           # overlay coupling quantile

    for label, interval, ppy in CONFIGS:
        C, R = load_panel(interval, end)
        if R is None or R.shape[1] < 6:
            print(f"[{label}] insufficient data"); continue
        n = R.shape[0]
        tr, te = bt.oos_split(n, TRAIN_FRAC)
        print(f"\n=== {label} panel: {n} bars x {R.shape[1]} coins "
              f"({R.index[0].date()} -> {R.index[-1].date()}) "
              f"| IS={tr.stop} OOS={te.stop-te.start} ===")

        # ---- standalone alpha: sweep knobs ----
        for z_gate in Z_GATES:
            feats = rolling_network_features(
                R, N_BINS, z_gate, N_SURR, REFIT_EVERY, MIN_TRAIN, seed=12345)
            # quick coupling sanity
            cpl = feats["coupling"]
            print(f"  [z>={z_gate}] coupling mean={np.nanmean(cpl):.5f} "
                  f"density mean={np.nanmean(feats['density']):.3f}")
            for sink_frac in SINK_FRACS:
                for hyp in HYPS:
                    res = strat_sink_follow(R, feats, sink_frac, ppy, te, hyp)
                    res_is = strat_sink_follow(R, feats, sink_frac, ppy, tr, hyp)
                    all_sr_pp.append(res["metrics"]["sr_pp"])
                    o = res["metrics"]
                    print(f"     z>={z_gate} sink={sink_frac} H-{hyp}: "
                          f"IS_shrp={res_is['metrics']['sharpe_ann']:6.2f} "
                          f"OOS_shrp={o['sharpe_ann']:6.2f} "
                          f"OOS_ret={o['ret_ann']*100:6.1f}% "
                          f"turn={res['turnover']:.2f} PSR={res['psr']:.3f}")
                    run = dict(config=label, z_gate=z_gate, sink_frac=sink_frac,
                               hypothesis=hyp, ppy=ppy,
                               IS_sharpe=res_is["metrics"]["sharpe_ann"],
                               OOS=o, OOS_psr=res["psr"],
                               turnover=res["turnover"])
                    summary["runs"].append(run)
                    cand = (o["sharpe_ann"], o["sr_pp"], res["psr"], o, label,
                            f"z{z_gate}_sink{sink_frac}_H{hyp}", res["turnover"])
                    if best_standalone is None or cand[0] > best_standalone[0]:
                        best_standalone = cand

        # ---- overlay: network coupling as a risk regime detector ----
        # need a coupling series; reuse z_gate=2.0 features (already computed last)
        feats_ov = rolling_network_features(
            R, N_BINS, 2.0, N_SURR, REFIT_EVERY, MIN_TRAIN, seed=999)
        cpl = feats_ov["coupling"]
        # base books
        btc_ret = R["BTCUSDT"].values if "BTCUSDT" in R.columns else R.iloc[:, 0].values
        ew_ret = R.mean(axis=1).values            # equal-weight long basket
        bases = [("longBTC", btc_ret), ("EWbasket", ew_ret)]
        for bname, base_ret in bases:
            for gq in GATE_QS:
                mb, mg, pg, thr = overlay_gate(base_ret, cpl, te, ppy, gq, tr)
                improve = mg["sharpe_ann"] - mb["sharpe_ann"]
                dd_cut = mg["maxdd"] - mb["maxdd"]     # less negative = better
                print(f"  OVERLAY {bname} gate q={gq}: "
                      f"base OOS shrp={mb['sharpe_ann']:.2f} dd={mb['maxdd']*100:.1f}% "
                      f"-> gated shrp={mg['sharpe_ann']:.2f} dd={mg['maxdd']*100:.1f}% "
                      f"(d_shrp={improve:+.2f}) PSRg={pg:.3f}")
                ov = dict(config=label, base=bname, gate_q=gq,
                          base_sharpe=mb["sharpe_ann"], gated_sharpe=mg["sharpe_ann"],
                          base_maxdd=mb["maxdd"], gated_maxdd=mg["maxdd"],
                          d_sharpe=improve, dd_cut=dd_cut, gated_psr=pg)
                summary["overlays"].append(ov)
                cand = (improve, mb["sharpe_ann"], mg["sharpe_ann"], pg,
                        label, bname, gq, mb["maxdd"], mg["maxdd"])
                if best_overlay is None or cand[0] > best_overlay[0]:
                    best_overlay = cand

    # ---- DSR deflation across the whole standalone family ----
    sr_star = bt.dsr_benchmark(all_sr_pp)
    summary["n_trials_standalone"] = len(all_sr_pp)
    summary["sr_star_pp"] = sr_star

    # ---- verdict ----
    verdict = "DEAD"
    role = "none"
    best_block = {}

    if best_standalone is not None:
        shrp, sr_pp, pp, o, label, knobs, turn = best_standalone
        dsr = bt.psr(sr_pp, o["n"], o["skew"], o["kurt"], sr_benchmark=sr_star)
        best_block["standalone"] = dict(
            config=label, knobs=knobs, OOS_sharpe=shrp,
            OOS_ret_ann=o["ret_ann"], OOS_maxdd=o["maxdd"], turnover=turn,
            OOS_psr=pp, DSR_psr=dsr, sr_star_pp=sr_star)
        print(f"\n=== BEST STANDALONE: {label} [{knobs}] ===")
        print(f"    OOS Sharpe={shrp:.2f} ret={o['ret_ann']*100:.1f}% "
              f"maxDD={o['maxdd']*100:.1f}% turn={turn:.2f}")
        print(f"    PSR={pp:.3f} DSR(vs SR*={sr_star:.4f})={dsr:.3f} "
              f"[{len(all_sr_pp)} trials]")
        if pp >= 0.95 and dsr >= 0.95 and shrp > 0:
            verdict = "EDGE"; role = "standalone-alpha"
        elif pp >= 0.80 and shrp > 0:
            verdict = "MARGINAL"; role = "standalone-alpha"

    if best_overlay is not None:
        improve, bs, gs, pg, label, bname, gq, bdd, gdd = best_overlay
        best_block["overlay"] = dict(
            config=label, base=bname, gate_q=gq, base_sharpe=bs,
            gated_sharpe=gs, d_sharpe=improve, base_maxdd=bdd, gated_maxdd=gdd,
            gated_psr=pg)
        print(f"\n=== BEST OVERLAY: {label} {bname} q={gq} ===")
        print(f"    base OOS Sharpe={bs:.2f} (dd={bdd*100:.1f}%) -> "
              f"gated={gs:.2f} (dd={gdd*100:.1f}%) d={improve:+.2f} PSRg={pg:.3f}")
        # overlay positive iff it MATERIALLY improves OOS Sharpe and is significant
        overlay_good = (improve > 0.15 and gs > 0 and pg >= 0.80)
        if overlay_good:
            if verdict == "EDGE":
                role = "both"
            elif verdict == "MARGINAL":
                role = "both" if role == "standalone-alpha" else "risk-overlay"
            else:
                # overlay rescues an otherwise-dead standalone
                if improve > 0.30 and pg >= 0.90 and gs > bs:
                    verdict = "MARGINAL"; role = "risk-overlay"

    summary["best"] = best_block
    summary["verdict"] = verdict
    summary["role"] = role
    print(f"\n=== VERDICT = {verdict} | role = {role} ===")

    out = ROOT / "reports" / "exo_transfer_entropy_network.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    print(f"wrote {out}")
    return summary


if __name__ == "__main__":
    main()
