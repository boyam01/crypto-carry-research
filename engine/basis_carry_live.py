"""Executable spec for the calendar-basis cash-and-carry edge.

READ-ONLY decision engine: scans live quarterly contracts, applies the validated
entry filter, and emits TARGET positions + margin/capital plan. It NEVER places
orders (place_order is hard-gated per the Probity governance invariant). The user
executes manually unless scope is explicitly upgraded.

Validated basis (basis_carry_spec.py): 12/12 BTC/ETH quarterly contracts positive,
entry annualized basis ~= realized (convergence locked by settlement rule),
pooled Sharpe 4.16, net ~4.25%/yr unleveraged. This module turns that into rules.
"""
from __future__ import annotations
import json, ssl, urllib.request
import pandas as pd

CTX = ssl.create_default_context()

# ---- validated/deployable parameters ----
PARAMS = dict(
    entry_dte_max=100,       # enter when far quarter has <= this many days (it lists ~97 DTE)
    entry_dte_min=20,        # ...and >= this (avoid the noisy last weeks)
    exit_dte=5,              # close & roll this many days before delivery
    min_ann_basis=0.025,     # 2.5%/yr hurdle: comfortably above ~1.4%/yr round-trip cost
    backwardation_stop=-0.01,# exit if annualized basis falls below -1% (carry inverted)
    max_gross_leverage=3.0,  # gross notional / NAV cap (short-leg liquidation safety)
    per_coin_max=0.40,       # max NAV fraction per coin
    min_free_margin=0.50,    # keep >=50% of futures-margin account free (rally buffer)
    fee_rt_bp=30,            # assumed round-trip cost (both legs, enter+exit)
)


def _get(u):
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "M"}),
                                              timeout=20, context=CTX).read())


def scan_opportunities():
    """Return DataFrame of live quarterly contracts with annualized basis (USDT-M + COIN-M)."""
    now = pd.Timestamp.now(tz="UTC")
    spot = {d["symbol"]: float(d["price"]) for d in _get("https://api.binance.com/api/v3/ticker/price")}
    rows = []

    def harvest(info_url, px_url, venue, inverse):
        info = _get(info_url)
        px = {d["symbol"]: float(d["price"]) for d in _get(px_url)}
        for s in info["symbols"]:
            if s.get("contractType") not in ("CURRENT_QUARTER", "NEXT_QUARTER") or "_" not in s["symbol"]:
                continue
            sym = s["symbol"]
            base = (s.get("baseAsset", "") + "USDT") if inverse else sym.split("_")[0]
            fut, sp, dl = px.get(sym), spot.get(base), s.get("deliveryDate")
            if not (fut and sp and dl):
                continue
            dte = (pd.Timestamp(dl, unit="ms", tz="UTC") - now).days
            if dte <= 1:
                continue
            rows.append(dict(venue=venue, contract=sym, coin=base[:-4], inverse=inverse,
                             spot=sp, fut=fut, dte=dte,
                             ann_basis=(fut / sp - 1) * 365 / dte))
    harvest("https://fapi.binance.com/fapi/v1/exchangeInfo", "https://fapi.binance.com/fapi/v1/ticker/price", "USDT-M", False)
    try:
        harvest("https://dapi.binance.com/dapi/v1/exchangeInfo", "https://dapi.binance.com/dapi/v1/ticker/price", "COIN-M", True)
    except Exception:
        pass
    return pd.DataFrame(rows).sort_values("ann_basis", ascending=False)


def select_and_size(df, nav_usd, p=PARAMS):
    """Apply entry filter + risk-limited sizing. Returns target positions."""
    elig = df[(df.dte <= p["entry_dte_max"]) & (df.dte >= p["entry_dte_min"])
              & (df.ann_basis >= p["min_ann_basis"])].copy()
    # one (front) contract per coin: highest basis among eligible
    elig = elig.sort_values("ann_basis", ascending=False).drop_duplicates("coin")
    if elig.empty:
        return elig
    # weight by basis (richer carry -> more), cap per coin, cap gross leverage
    elig["raw_w"] = elig["ann_basis"].clip(lower=0)
    elig["w"] = (elig["raw_w"] / elig["raw_w"].sum()).clip(upper=p["per_coin_max"])
    elig["w"] = elig["w"] / elig["w"].sum()
    gross = nav_usd * p["max_gross_leverage"]
    elig["leg_notional_usd"] = (elig["w"] * gross).round(0)
    elig["spot_qty"] = (elig["leg_notional_usd"] / elig["spot"]).round(4)
    elig["fut_short_notional_usd"] = elig["leg_notional_usd"]
    elig["expected_carry_usd_yr"] = (elig["leg_notional_usd"] * (elig["ann_basis"] - p["fee_rt_bp"] / 1e4 * 365 / 60)).round(0)
    return elig[["venue", "contract", "coin", "dte", "ann_basis", "w",
                 "leg_notional_usd", "spot_qty", "fut_short_notional_usd", "expected_carry_usd_yr"]]


def main(nav_usd=1_000_000):
    p = PARAMS
    df = scan_opportunities()
    print(f"=== LIVE quarterly-basis opportunity scan ({pd.Timestamp.now(tz='UTC').date()}) ===")
    print(df.assign(ann_basis_pct=(df.ann_basis * 100).round(2)).drop(columns="ann_basis")
          [["venue", "contract", "coin", "dte", "spot", "fut", "ann_basis_pct"]].to_string(index=False))

    tgt = select_and_size(df, nav_usd, p)
    print(f"\n=== TARGET BOOK (NAV ${nav_usd:,.0f}, max {p['max_gross_leverage']}x gross, "
          f"entry {p['entry_dte_min']}-{p['entry_dte_max']} DTE, hurdle {p['min_ann_basis']*100:.1f}%/yr) ===")
    cost_hurdle = p["fee_rt_bp"] / 1e4 * 365 / 60      # ~breakeven annualized basis
    watch = df[(df.dte <= p["entry_dte_max"]) & (df.dte >= p["entry_dte_min"])
               & (df.ann_basis > cost_hurdle) & (df.ann_basis < p["min_ann_basis"])]
    if tgt.empty:
        print("No contracts clear the preferred hurdle right now.")
        print("-> Hold cash / keep funding-carry satellite running; re-scan daily.")
    else:
        view = tgt.copy(); view["ann_basis_pct"] = (view["ann_basis"] * 100).round(2)
        print(view.drop(columns="ann_basis")[["venue", "contract", "coin", "dte", "ann_basis_pct",
              "w", "leg_notional_usd", "spot_qty", "fut_short_notional_usd", "expected_carry_usd_yr"]].to_string(index=False))
        gross = tgt["leg_notional_usd"].sum()
        carry = tgt["expected_carry_usd_yr"].sum()
        fut_margin = gross / 1.0  # short legs at 1x isolated would need full; with cross use leverage
        print(f"\nCapital plan: gross spot+fut notional 2x${gross:,.0f}=${2*gross:,.0f}; gross/NAV={gross/nav_usd:.2f}x")
        print(f"  spot leg: ${gross:,.0f} (long, full capital). futures leg: ${gross:,.0f} short.")
        print(f"  futures margin @ keep {p['min_free_margin']*100:.0f}% free: budget >= ${gross*(1+p['min_free_margin']):,.0f} margin headroom")
        print(f"  expected net carry: ${carry:,.0f}/yr  = {carry/nav_usd*100:.2f}%/yr on NAV (after ~{p['fee_rt_bp']}bp RT)")
    if not watch.empty:
        print(f"\nWATCHLIST (net-positive above ~{cost_hurdle*100:.1f}%/yr cost but below {p['min_ann_basis']*100:.1f}%/yr preferred hurdle):")
        print(watch.assign(ann_basis_pct=(watch.ann_basis*100).round(2))
              [["venue","contract","coin","dte","ann_basis_pct"]].to_string(index=False))
    return tgt


if __name__ == "__main__":
    main()
