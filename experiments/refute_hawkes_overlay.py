"""Adversarial refutation of hawkes_intensity OVERLAY (the only positive claim).
Lens = OVERFITTING & ROBUSTNESS. Standalone is conceded DEAD.

Claimed overlay: EW long basket, gauge=branching-ratio n, event mode=activity,
kappa=3.0, fw=120, pct=0.80, risk_off_w=0.0.
Claimed OOS: base Sharpe 0.359 -> gated 0.666 (delta +0.31), maxDD -61.9% -> -54.9%,
gated PSR 0.81, gated DSR 0.59.

Probes:
 (1) reproduce selected overlay OOS numbers.
 (2) regime/time concentration: where does the OOS delta-PnL come from? Top-day,
     and the gate's PnL contribution split (does de-risking help only via a few bars?).
 (3) parameter fragility: perturb pct in {0.70..0.95}, roff in {0,0.25,0.5},
     gauge in {n,ratio}, fw in {120,240}, kappa in {2,3}. Distribution of OOS delta.
 (4) shifted train/test cut (0.55,0.65) -> does the SELECTED param still beat base OOS?
 (5) is the gate just a realized-vol / beta proxy? Compare to a trivial trailing-vol
     gate (de-risk when 30-bar realized vol is in top 20% IS) -- if a dumb vol gate
     does as well or better, Hawkes adds nothing.
 (6) honest base: is gated>base OOS robust, or did IS-selection cherry-pick?
"""
from __future__ import annotations
import sys, pathlib
import numpy as np, pandas as pd
from datetime import datetime, timezone
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import fetch_binance as fb
from engine import backtest as bt
import exo_hawkes_intensity as H  # reuse exact event/feature/overlay code

START = int(datetime(2022,1,1,tzinfo=timezone.utc).timestamp()*1000)
END = int(datetime.now(timezone.utc).timestamp()*1000)
COINS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT"]
COST=5.0; PPY=365*3; EVWIN=90; STRIDE=3

def load():
    frames={}
    for s in COINS:
        f=H.load_coin(s)
        if f is not None: frames[s]=f
    return frames

def ann_sharpe(net):
    net=np.asarray(net,float); net=net[np.isfinite(net)]
    if len(net)<10 or net.std(ddof=1)==0: return np.nan
    return net.mean()/net.std(ddof=1)*np.sqrt(PPY)

def maxdd(net):
    net=np.asarray(net,float); eq=np.cumprod(1+net); pk=np.maximum.accumulate(eq)
    return float((eq/pk-1).min())

def build_gauge(frames, mode, kappa, fw, gauge):
    gmat=[]
    for s,df in frames.items():
        flag=H.build_events(df,mode,kappa,EVWIN)
        ratio,nbr=H.rolling_hawkes_feature(flag,fw,STRIDE)
        gv = nbr if gauge=="n" else ratio
        gmat.append(pd.Series(gv,index=df.index))
    return pd.concat(gmat,axis=1).mean(axis=1)

def overlay_net(base_ret, gser, thr, roff):
    g=np.asarray(gser,float); on=(g>=thr)&np.isfinite(g)
    w=np.where(on,roff,1.0); wh=H.shift_hold(w)
    net=bt.run(base_ret.values, wh, COST)
    return net, wh

def main():
    frames=load()
    ew=pd.concat([frames[s]["ret"] for s in frames],axis=1).mean(axis=1)
    n=len(ew); tr,te=bt.oos_split(n,0.6)
    print(f"n={n}  OOS span {ew.index[te.start]} -> {ew.index[-1]}")

    base_net=bt.run(ew.values, np.ones(n), COST)
    base_oos_sr=ann_sharpe(base_net[te]); base_oos_dd=maxdd(base_net[te])
    print(f"\n[BASE EW] OOS Sharpe={base_oos_sr:.3f} maxDD={base_oos_dd*100:.1f}%")

    # ---- (1) reproduce selected overlay ----
    sel=dict(mode="activity",kappa=3.0,fw=120,gauge="n",pct=0.80,roff=0.0)
    g_sel=build_gauge(frames, sel["mode"], sel["kappa"], sel["fw"], sel["gauge"])
    g_sel=g_sel.reindex(ew.index)
    gtr=g_sel.values[tr]; gtr=gtr[np.isfinite(gtr)]
    thr_sel=np.quantile(gtr, sel["pct"])
    net_sel,wh_sel=overlay_net(ew, g_sel.values, thr_sel, sel["roff"])
    sel_oos_sr=ann_sharpe(net_sel[te]); sel_oos_dd=maxdd(net_sel[te])
    print(f"\n[SELECTED OVERLAY] thr={thr_sel:.4f}")
    print(f"  OOS gated Sharpe={sel_oos_sr:.3f} (base {base_oos_sr:.3f}, delta {sel_oos_sr-base_oos_sr:+.3f})")
    print(f"  OOS gated maxDD={sel_oos_dd*100:.1f}% (base {base_oos_dd*100:.1f}%)")
    roff_frac=np.mean(np.abs(wh_sel[te])<1.0)
    print(f"  OOS risk-off fraction={roff_frac:.3f}")

    # ---- (2) regime / time concentration of the DELTA pnl ----
    # delta pnl = gated_net - base_net (per OOS bar). Where does the gain come from?
    dpnl = (net_sel - base_net)[te]
    didx = ew.index[te]
    order=np.argsort(dpnl)  # most-negative first
    cum=dpnl.sum()
    # the gate only ACTS when risk-off; gain = avoided base return on those bars minus cost
    print(f"\n[DELTA-PNL CONCENTRATION] total OOS delta-pnl (sum)={cum:.4f}")
    top5_gain=np.sort(dpnl)[-5:].sum()
    top5_loss=np.sort(dpnl)[:5].sum()
    print(f"  sum of top-5 positive delta bars={top5_gain:.4f}  ({100*top5_gain/abs(cum) if cum!=0 else 0:.0f}% of net delta)")
    print(f"  sum of top-5 negative delta bars={top5_loss:.4f}")
    # how many distinct risk-off episodes in OOS?
    on=(np.abs(wh_sel[te])<1.0).astype(int)
    episodes=int(np.sum(np.diff(np.concatenate([[0],on]))==1))
    print(f"  distinct OOS risk-off episodes={episodes}, total risk-off bars={on.sum()}")
    # biggest single avoided drawdown day
    worst_base_days = np.argsort(base_net[te])[:10]
    avoided = np.sum((on[worst_base_days]==1))
    print(f"  of the 10 worst BASE OOS bars, gate was risk-off on {avoided}/10")

    # ---- (3) parameter fragility ----
    print("\n[PARAMETER FRAGILITY] OOS delta-Sharpe across the knob neighborhood:")
    deltas=[]
    rows=[]
    for mode in ["activity","move"]:
        for kappa in [2.0,3.0]:
            for fw in [120,240]:
                for gauge in ["n","ratio"]:
                    g=build_gauge(frames,mode,kappa,fw,gauge).reindex(ew.index)
                    gt=g.values[tr]; gt=gt[np.isfinite(gt)]
                    if len(gt)<50: continue
                    for pct in [0.70,0.80,0.90,0.95]:
                        thr=np.quantile(gt,pct)
                        for roff in [0.0,0.25,0.5]:
                            net,_=overlay_net(ew,g.values,thr,roff)
                            d=ann_sharpe(net[te])-base_oos_sr
                            deltas.append(d)
                            rows.append((mode,kappa,fw,gauge,pct,roff,d))
    deltas=np.array(deltas)
    print(f"  variants probed={len(deltas)}")
    print(f"  OOS delta-Sharpe: mean={np.nanmean(deltas):+.3f} median={np.nanmedian(deltas):+.3f} "
          f"std={np.nanstd(deltas):.3f}")
    print(f"  frac delta>0 = {np.mean(deltas>0):.2%}   frac delta>+0.10 = {np.mean(deltas>0.10):.2%}")
    print(f"  best={np.nanmax(deltas):+.3f}  worst={np.nanmin(deltas):+.3f}")
    # neighbors of the selected param (mode=activity,kappa=3,fw=120,gauge=n)
    neigh=[r for r in rows if r[0]=="activity" and r[1]==3.0 and r[2]==120 and r[3]=="n"]
    print("  immediate neighbors (activity,k3,fw120,n) varying pct,roff:")
    for r in neigh:
        print(f"     pct={r[4]} roff={r[5]}  deltaSR={r[6]:+.3f}")

    # ---- (4) shifted train/test cut, SELECTED param re-evaluated ----
    print("\n[SHIFTED CUT] selected param, OOS delta-Sharpe under different splits:")
    for frac in [0.55,0.60,0.65]:
        tr2,te2=bt.oos_split(n,frac)
        gt=g_sel.values[tr2]; gt=gt[np.isfinite(gt)]
        thr2=np.quantile(gt,sel["pct"])
        net2,_=overlay_net(ew,g_sel.values,thr2,sel["roff"])
        b2=ann_sharpe(base_net[te2]); s2=ann_sharpe(net2[te2])
        print(f"  train_frac={frac}: base OOS SR={b2:.3f} gated={s2:.3f} delta={s2-b2:+.3f}")

    # ---- (5) dumb realized-vol gate benchmark ----
    print("\n[DUMB VOL-GATE BENCHMARK] de-risk when trailing realized vol high (no Hawkes):")
    rv=ew.rolling(30,min_periods=15).std().shift(1)  # causal trailing 30-bar vol
    rvtr=rv.values[tr]; rvtr=rvtr[np.isfinite(rvtr)]
    for pct in [0.80,0.90]:
        thr=np.quantile(rvtr,pct)
        netv,_=overlay_net(ew, rv.values, thr, 0.0)
        print(f"  volgate pct={pct}: OOS Sharpe={ann_sharpe(netv[te]):.3f} "
              f"delta={ann_sharpe(netv[te])-base_oos_sr:+.3f} maxDD={maxdd(netv[te])*100:.1f}%")

    # ---- (6) gauge correlation with realized vol (is n just a vol proxy?) ----
    gv=g_sel.reindex(ew.index).values
    rvv=ew.rolling(30,min_periods=15).std().shift(1).values
    m=np.isfinite(gv)&np.isfinite(rvv)
    c=np.corrcoef(gv[m],rvv[m])[0,1]
    print(f"\n[GAUGE vs VOL] corr(branching-ratio n, trailing 30-bar realized vol)={c:+.3f}")

if __name__=="__main__":
    main()
