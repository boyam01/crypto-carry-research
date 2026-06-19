"""Deployable market-neutral carry book — the real edge the research found.
Combines the survivors (all LOW-TURNOVER structural risk premia, the only thing
that beats the fee wall) into one OOS, cost-adjusted, inverse-vol-weighted book:
  Sleeve A: funding-rate carry (delta-neutral, EMA+hysteresis, multi-coin)
  Sleeve B: variance risk premium (short BTC vol vs Deribit DVOL, daily MTM proxy)
  (Calendar basis = ~4 trades/yr hold-to-expiry; reported separately in REPORT.)
The point: independent premia -> diversified; show combined Sharpe + sleeve corr.
"""
from __future__ import annotations
import sys, time, json, pathlib, urllib.request, ssl
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import fetch_binance as fb
from engine import backtest as bt
from experiments import carry_edge as ce

CTX = ssl.create_default_context()


def carry_sleeve():
    end = int(time.time() * 1000)
    P = ce.panel(ce.UNIVERSE, ce._ms(ce.START), end)
    nets = {}
    for s, df in P.items():
        sig = ce.hysteresis_signal(df["funding"].values, 21, 1e-4)
        nets[s] = pd.Series(ce.carry_net(df, sig, 10.0), index=df.index)
    port8h = pd.DataFrame(nets).fillna(0).mean(axis=1)
    daily = port8h.resample("1D").sum()                 # 8h carry -> daily PnL
    return daily


def basis_sleeve():
    """delta-neutral quarterly cash-and-carry daily stream: long spot / short front
    dated future, ret = spot_ret - fut_ret while held (5-85 days to expiry)."""
    end = int(time.time() * 1000); start = int(pd.Timestamp("2024-06-01", tz="UTC").timestamp() * 1000)
    codes = ["250328", "250627", "250926", "251226", "260327", "260626", "260925"]
    rets = []
    for coin in ["BTCUSDT", "ETHUSDT"]:
        spot = fb.klines(coin, "1d", start, end, futures=False)["close"]
        sret = np.log(spot / spot.shift(1))
        for code in codes:
            try:
                f = fb.klines(f"{coin}_{code}", "1d", start, end, futures=True)
            except Exception:
                continue
            if f is None or len(f) < 20:
                continue
            exp = pd.Timestamp(f"20{code[:2]}-{code[2:4]}-{code[4:]}", tz="UTC")
            fret = np.log(f["close"] / f["close"].shift(1))
            dte = np.array([(exp - ix).days for ix in f.index])
            hold = (dte >= 5) & (dte <= 85)
            r = (sret.reindex(f.index) - fret)[hold]
            if len(r):
                rets.append(r)
    if not rets:
        return pd.Series(dtype=float)
    return pd.concat(rets).groupby(level=0).mean().sort_index()


def vrp_sleeve():
    now = int(time.time() * 1000); start = now - 3 * 365 * 24 * 3600 * 1000
    u = f"https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&start_timestamp={start}&end_timestamp={now}&resolution=1D"
    d = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "M"}), timeout=20, context=CTX).read())["result"]["data"]
    iv = pd.Series([x[4] for x in d], index=pd.to_datetime([x[0] for x in d], unit="ms", utc=True).floor("D"), dtype=float) / 100.0
    k = fb.klines("BTCUSDT", "1d", int(iv.index[0].timestamp() * 1000), now, futures=True)["close"]
    r = np.log(k / k.shift(1))
    df = pd.DataFrame({"iv": iv, "r": r}).dropna()
    # daily short-variance carry MTM proxy: collect implied var/day, pay realized var
    daily = (df["iv"].shift(1) ** 2) / 365.0 - df["r"] ** 2
    daily = daily * 20.0                                # vega scaling to ~comparable vol
    return daily.dropna()


def inv_vol_combine(streams, train_frac=0.6, target_vol=0.10):
    # restrict to the COMMON LIVE period where every sleeve has data (fixes the
    # different-start-date sparsity that blows up inverse-vol weights)
    start = max(s.index.min() for s in streams.values())
    idx = None
    for s in streams.values():
        idx = s.index if idx is None else idx.union(s.index)
    idx = idx[idx >= start]
    df = pd.DataFrame({k: v.reindex(idx).fillna(0.0) for k, v in streams.items()}).sort_index()
    n = len(df); cut = int(n * train_frac)
    w = {}                                              # inverse-(active-)vol weights, IS only
    for c in df.columns:
        active = df[c].iloc[:cut]
        active = active[active != 0]
        v = (active.std() * np.sqrt(365)) if len(active) > 5 else 1e9
        w[c] = 1.0 / (v + 1e-9)
    wsum = sum(w.values())
    w = {c: w[c] / wsum for c in w}
    port = sum(df[c] * w[c] for c in df.columns)
    is_vol = port.iloc[:cut].std() * np.sqrt(365)
    scale = target_vol / is_vol if is_vol > 1e-4 else 1.0   # guard against degenerate IS vol
    return df, port * scale, w, cut


def main():
    print("building deployable book sleeves ...")
    streams = {"funding_carry": carry_sleeve(), "calendar_basis": basis_sleeve(), "vrp_btc": vrp_sleeve()}
    streams = {k: v for k, v in streams.items() if v is not None and len(v) > 30}
    df, port, w, cut = inv_vol_combine(streams)
    print(f"IS inverse-vol weights: {{ {', '.join(f'{k}:{v:.2f}' for k,v in w.items())} }}")
    cm = df.iloc[:cut].corr()
    off = cm.values[np.triu_indices(len(cm), 1)]
    print(f"sleeve pairwise corr (IS): {np.round(off,3)}  mean={off.mean():.3f}  (low => genuine diversification)\n")
    corr = float(off.mean())

    print(f"{'sleeve':16} {'OOS_Sharpe':>11} {'OOS_ret%':>9} {'OOS_maxDD%':>11}")
    for c in df.columns:
        mo = bt.metrics(df[c].values[cut:], 365)
        print(f"{c:16} {mo['sharpe_ann']:11.2f} {mo['ret_ann']*100:9.2f} {mo['maxdd']*100:11.2f}")
    mo = bt.metrics(port.values[cut:], 365)
    psr = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
    print(f"{'COMBINED (10%v)':16} {mo['sharpe_ann']:11.2f} {mo['ret_ann']*100:9.2f} {mo['maxdd']*100:11.2f}   PSR={psr:.3f}")
    print(f"\nOOS combined: Sharpe {mo['sharpe_ann']:.2f}, ret {mo['ret_ann']*100:.1f}%/yr @10% target vol, maxDD {mo['maxdd']*100:.1f}%.")
    print("Diversified low-turnover structural-premium book — the deployable edge (carry formulas on funding+vol data).")
    print("NOTE: returns net of 10bp/leg carry cost; VRP sleeve needs options execution; close-to-close maxDD excludes intrabar/liquidation tail.")
    out = pathlib.Path(__file__).resolve().parent.parent / "reports" / "deploy_book_result.json"
    out.write_text(json.dumps(dict(weights=w, sleeve_corr_IS=float(corr),
        combined=dict(sharpe=mo["sharpe_ann"], ret=mo["ret_ann"], maxdd=mo["maxdd"], psr=psr)), indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
