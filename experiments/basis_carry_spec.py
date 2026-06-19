"""Full deployable spec for the most robust edge found: quarterly-futures
cash-and-carry (basis convergence). Per-contract analysis + a basis-threshold
entry filter + honest capital-efficiency / risk metrics.

Mechanism: long spot / short dated quarterly future, held while 5-85 days to
expiry. The future MUST converge to spot at delivery (exchange settlement rule),
so the entry annualized basis is structurally earned regardless of price path.
This is regime-INDEPENDENT (unlike funding carry / VRP) — its only real risks are
the short-future margin/liquidation path and spot-leg capital.
"""
from __future__ import annotations
import sys, time, json, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import fetch_binance as fb
from engine import backtest as bt

CODES = ["250328", "250627", "250926", "251226", "260327", "260626", "260925"]
COINS = ["BTCUSDT", "ETHUSDT"]
RT_COST = 0.0030          # 30bp round-trip (enter+exit, both legs)
MIN_DTE, MAX_DTE = 5, 85


def expiry(code):
    return pd.Timestamp(f"20{code[:2]}-{code[2:4]}-{code[4:]}", tz="UTC")


def main():
    end = int(time.time() * 1000); start = int(pd.Timestamp("2024-06-01", tz="UTC").timestamp() * 1000)
    rows, daily_all = [], []
    for coin in COINS:
        spot = fb.klines(coin, "1d", start, end, futures=False)["close"]
        for code in CODES:
            try:
                f = fb.klines(f"{coin}_{code}", "1d", start, end, futures=True)["close"]
            except Exception:
                continue
            if f is None or len(f) < 20:
                continue
            exp = expiry(code)
            dte = np.array([(exp - ix).days for ix in f.index])
            hold = (dte >= MIN_DTE) & (dte <= MAX_DTE)
            if hold.sum() < 10:
                continue
            sp = spot.reindex(f.index)
            ann_basis = (f / sp - 1) * 365 / np.maximum(dte, 1)        # annualized basis path
            entry_i = np.where(hold)[0][0]
            entry_basis = float(ann_basis.iloc[entry_i])               # basis at entry (known)
            # delta-neutral daily carry while held = spot_ret - fut_ret
            sret = np.log(sp / sp.shift(1)); fret = np.log(f / f.shift(1))
            dstream = (sret - fret)[hold].dropna()
            realized = float(dstream.sum())                            # total convergence captured
            days = len(dstream)
            ann_realized = realized * 365 / max(days, 1)
            rows.append(dict(coin=coin[:-4], code=code, entry_basis_ann=entry_basis,
                             realized_ann=ann_realized, days=days,
                             net_ann=ann_realized - RT_COST * 365 / max(days, 1)))
            daily_all.append(dstream)
    R = pd.DataFrame(rows)
    # entry-filter rule: only take contango with entry annualized basis > round-trip cost hurdle
    R["take"] = R["entry_basis_ann"] > (RT_COST * 365 / 80)            # hurdle ~ cost over a ~80d hold
    print("Per-contract cash-and-carry (entry basis -> realized convergence, annualized):")
    print(R.round(4).to_string(index=False))
    taken = R[R["take"]]
    print(f"\ntaken (entry basis > hurdle): {len(taken)}/{len(R)} contracts; "
          f"win rate {100*(taken['realized_ann']>0).mean():.0f}%")
    print(f"mean entry basis {taken['entry_basis_ann'].mean()*100:.2f}%/yr -> mean realized "
          f"{taken['realized_ann'].mean()*100:.2f}%/yr (net of 30bp RT: {taken['net_ann'].mean()*100:.2f}%/yr)")

    # pooled daily book (all taken contracts) for Sharpe/maxDD
    book = pd.concat(daily_all).groupby(level=0).mean().sort_index()
    mo = bt.metrics(book.values, 365)
    psr = bt.psr(mo["sr_pp"], mo["n"], mo["skew"], mo["kurt"])
    print(f"\npooled daily book: Sharpe {mo['sharpe_ann']:.2f}  ret {mo['ret_ann']*100:.2f}%/yr  "
          f"maxDD {mo['maxdd']*100:.2f}%  PSR {psr:.3f}  ({len(book)} days)")

    print("\n=== DEPLOYABLE SPEC ===")
    print(f"  universe: BTC/ETH USDT-M quarterly futures (extendable to COIN-M for XRP/BNB/SOL)")
    print(f"  entry:    when front quarter has ~{MAX_DTE}d to expiry AND annualized basis > hurdle (contango)")
    print(f"  position: long spot / short dated future, delta-neutral, equal notional")
    print(f"  exit:     close ~{MIN_DTE}d before delivery (avoid settlement-print noise), roll to next quarter")
    print(f"  expected: ~{taken['entry_basis_ann'].mean()*100:.1f}%/yr gross carry, lock-in by convergence (regime-independent)")
    print(f"  risks:    short-future margin/liquidation on sharp rallies (keep low leverage), spot-leg capital,")
    print(f"            settlement execution; basis can compress if entered late. NOT free — capital-intensive carry.")
    out = pathlib.Path(__file__).resolve().parent.parent / "reports" / "basis_carry_spec.json"
    out.write_text(json.dumps(dict(contracts=rows, pooled=dict(sharpe=mo["sharpe_ann"],
        ret=mo["ret_ann"], maxdd=mo["maxdd"], psr=psr),
        mean_entry_basis=float(taken["entry_basis_ann"].mean()),
        mean_net=float(taken["net_ann"].mean())), indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
