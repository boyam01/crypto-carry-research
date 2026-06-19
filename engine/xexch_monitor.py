"""Cross-exchange carry MONITOR (Qwen roadmap step 1): for the 5 core coins,
consolidate across Binance/OKX/Bitget -> best dated basis + perp funding +
liquidity. The 'scanner instrument' that turns manual watching into a dashboard.
Read-only.
"""
from __future__ import annotations
import json, ssl, urllib.request
import pandas as pd
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import engine.xexch_basis_scan as xb

CTX = ssl.create_default_context()
CORE = ["BTC", "ETH", "BNB", "SOL", "XRP"]
def _get(u):
    try:
        return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "M"}), timeout=15, context=CTX).read())
    except Exception as e:
        return {"_err": str(e)[:90]}


def funding_all():
    out = {c: {} for c in CORE}
    # Binance USDT perp
    bp = {d["symbol"]: float(d["lastFundingRate"]) for d in _get("https://fapi.binance.com/fapi/v1/premiumIndex") if "lastFundingRate" in d}
    bv = {d["symbol"]: float(d["quoteVolume"]) for d in _get("https://fapi.binance.com/fapi/v1/ticker/24hr")}
    for c in CORE:
        f = bp.get(c + "USDT")
        if f is not None:
            out[c]["Binance"] = f * 3 * 365 * 100
        out[c]["vol_musd"] = bv.get(c + "USDT", 0) / 1e6
    # OKX SWAP
    okf = {d["instId"]: d for d in _get("https://www.okx.com/api/v5/public/funding-rate?instType=SWAP").get("data", [])} if False else {}
    for c in CORE:
        d = _get(f"https://www.okx.com/api/v5/public/funding-rate?instId={c}-USDT-SWAP").get("data", [])
        if d and d[0].get("fundingRate"):
            out[c]["OKX"] = float(d[0]["fundingRate"]) * 3 * 365 * 100
    # Bitget USDT perp
    for c in CORE:
        d = _get(f"https://api.bitget.com/api/v2/mix/market/current-fund-rate?symbol={c}USDT&productType=usdt-futures").get("data", [])
        if d and d[0].get("fundingRate"):
            iv = float(d[0].get("fundingRateInterval", 8) or 8)
            out[c]["Bitget"] = float(d[0]["fundingRate"]) * (24 / iv) * 365 * 100
    return out


def main():
    # basis grid (reuse cross-exchange scanner)
    rows = xb.binance_rows()
    for fn in (xb.okx_rows, xb.bitget_rows):
        try:
            rows += fn()
        except Exception:
            pass
    bdf = pd.DataFrame(rows)
    fund = funding_all()

    print(f"=== CROSS-EXCHANGE CARRY MONITOR ({xb.NOW.date()}) — 5 core coins ===\n")
    print(f"{'coin':5} {'best_basis%/yr (venue/exp/dte)':34} {'funding %/yr  B / O / Bg':28} {'Binance_vol_$M':>13}")
    for c in CORE:
        sub = bdf[(bdf.coin == c) & (bdf.dte >= 20) & (bdf.dte <= 200)] if not bdf.empty else pd.DataFrame()
        if len(sub):
            b = sub.loc[sub.ann_basis_pct.idxmax()]
            basis_s = f"{b.ann_basis_pct:+6.2f}  {b.venue}/{b.expiry}/{int(b.dte)}d"
        else:
            basis_s = "  (none in 20-200d window)"
        fd = fund[c]
        f_s = f"{fd.get('Binance',float('nan')):+6.1f}/{fd.get('OKX',float('nan')):+6.1f}/{fd.get('Bitget',float('nan')):+6.1f}"
        print(f"{c:5} {basis_s:34} {f_s:28} {fd.get('vol_musd',0):13,.0f}")

    print("\nHOW TO READ:")
    print("- best_basis>hurdle(~2.5%) & contango -> put on cash-and-carry on that venue/expiry (settlement-locked, robust).")
    print("- funding %/yr: if persistently HIGH positive on a venue -> short-perp/long-spot funding carry there (regime-dependent).")
    print("- pick the RICHEST venue per coin; liquidity (vol) tells you executable size.")
    print("- basis carry = robust core; funding carry = opportunistic satellite (bleeds when funding compresses/flips).")


if __name__ == "__main__":
    main()
