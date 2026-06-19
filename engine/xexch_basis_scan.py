"""Cross-exchange basis-carry scanner: Binance + OKX + Bitget dated futures.
Builds a coin x venue x expiry grid of annualized basis so you can (a) put the
carry on the RICHEST venue/expiry per coin, and (b) spot cross-venue basis spreads.
Read-only. Spot reference = Binance spot (common index for fair comparison).
"""
from __future__ import annotations
import json, ssl, urllib.request
import pandas as pd
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine.basis_carry_live import scan_opportunities

CTX = ssl.create_default_context()
NOW = pd.Timestamp.now(tz="UTC")
def _get(u):
    try:
        return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "M"}), timeout=15, context=CTX).read())
    except Exception as e:
        return {"_err": str(e)[:120]}

def binance_spot():
    return {d["symbol"]: float(d["price"]) for d in _get("https://api.binance.com/api/v3/ticker/price")}

SPOT = binance_spot()
def spot_of(coin):
    return SPOT.get(coin + "USDT")


def binance_rows():
    b = scan_opportunities()
    out = []
    for _, r in b.iterrows():
        out.append(dict(venue="Binance", coin=r["coin"], expiry=r["contract"].split("_")[-1],
                        dte=int(r["dte"]), ann_basis_pct=round(r["ann_basis"] * 100, 2)))
    return out


def okx_rows():
    inst = {x["instId"]: x for x in _get("https://www.okx.com/api/v5/public/instruments?instType=FUTURES").get("data", [])}
    ftk = {x["instId"]: x for x in _get("https://www.okx.com/api/v5/market/tickers?instType=FUTURES").get("data", [])}
    out = []
    for iid, meta in inst.items():
        if "XPERP" in iid:                      # expiry-perps, not real dated futures
            continue
        et = meta.get("expTime")
        if not et:
            continue
        dte = (pd.Timestamp(int(et), unit="ms", tz="UTC") - NOW).days
        if not (1 < dte < 300):
            continue
        base = iid.split("-")[0]
        f = ftk.get(iid, {}).get("last"); s = spot_of(base)
        if f and s:
            out.append(dict(venue="OKX", coin=base, expiry=iid.split("-")[-1],
                            dte=dte, ann_basis_pct=round((float(f) / s - 1) * 365 / dte * 100, 2)))
    return out


def bitget_rows():
    out = []
    cons = _get("https://api.bitget.com/api/v2/mix/market/contracts?productType=coin-futures").get("data", [])
    tks = {x.get("symbol"): x for x in _get("https://api.bitget.com/api/v2/mix/market/tickers?productType=coin-futures").get("data", [])}
    code_mon = {"H": 3, "M": 6, "U": 9, "Z": 12}
    for c in cons:
        sym = c.get("symbol", "")
        # delivery date: prefer explicit field, else parse CME-style code e.g. BTCUSDU26
        dt = c.get("deliveryTime") or c.get("deliveryDate")
        exp = None
        if dt and str(dt).isdigit():
            exp = pd.Timestamp(int(dt), unit="ms", tz="UTC")
        elif len(sym) >= 3 and sym[-3] in code_mon and sym[-2:].isdigit():
            mon = code_mon[sym[-3]]; yr = 2000 + int(sym[-2:])
            exp = pd.Timestamp(year=yr, month=mon, day=26, tz="UTC")
        if exp is None:
            continue
        dte = (exp - NOW).days
        if not (1 < dte < 300):
            continue
        base = sym.replace("USD", "").rstrip("HMUZ0123456789") or sym[:3]
        base = sym[:3] if sym.startswith(("BTC", "ETH", "SOL", "XRP", "BNB")) else base
        f = tks.get(sym, {}).get("lastPr"); s = spot_of(base)
        if f and s:
            out.append(dict(venue="Bitget", coin=base, expiry=sym[-3:],
                            dte=dte, ann_basis_pct=round((float(f) / s - 1) * 365 / dte * 100, 2)))
    return out


def main():
    rows = binance_rows()
    for fn in (okx_rows, bitget_rows):
        try:
            rows += fn()
        except Exception as e:
            print(f"  {fn.__name__} error: {str(e)[:100]}")
    df = pd.DataFrame(rows)
    if df.empty:
        print("no dated futures found"); return
    print(f"=== CROSS-EXCHANGE basis grid ({NOW.date()}), {len(df)} contracts ===")
    print(df.sort_values(["coin", "dte"]).to_string(index=False))

    print("\n=== BEST harvestable carry per coin (DTE 20-200, contango, pick richest venue/expiry) ===")
    elig = df[(df.dte >= 20) & (df.dte <= 200) & (df.ann_basis_pct > 1.5)]
    if elig.empty:
        print("  none in window above 1.5%/yr right now")
    else:
        best = elig.sort_values("ann_basis_pct", ascending=False).drop_duplicates("coin")
        print(best.to_string(index=False))
        print(f"\n  -> {len(best)} coins harvestable now; richest: {best.iloc[0]['coin']} "
              f"{best.iloc[0]['ann_basis_pct']}%/yr on {best.iloc[0]['venue']} ({best.iloc[0]['expiry']}, {best.iloc[0]['dte']}d)")

    print("\n=== cross-venue basis SPREAD (same coin, similar DTE) — relative-value ===")
    for coin in df.coin.unique():
        sub = df[(df.coin == coin) & (df.dte >= 60) & (df.dte <= 130)]
        if len(sub) >= 2:
            hi, lo = sub.loc[sub.ann_basis_pct.idxmax()], sub.loc[sub.ann_basis_pct.idxmin()]
            sprd = hi.ann_basis_pct - lo.ann_basis_pct
            if sprd > 0.5:
                print(f"  {coin}: long {lo.venue}({lo.expiry}) {lo.ann_basis_pct}% / short {hi.venue}({hi.expiry}) {hi.ann_basis_pct}% -> spread {sprd:.2f}%/yr")
    print("\nNOTE: OKX gives a term structure (multiple expiries) -> pick the richest annualized;")
    print("      cross-venue spreads need positions on BOTH venues (capital + transfer); thin but uncorrelated.")


if __name__ == "__main__":
    main()
