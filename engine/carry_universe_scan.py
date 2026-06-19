"""Full Binance carry opportunity scan across ALL instruments:
  1. BASIS carry  -> every dated quarterly (USDT-M + COIN-M), annualized basis
  2. FUNDING carry -> every liquid perp, current annualized funding rate
Shows where the carry edge is harvestable beyond BTC/ETH, incl. tokenized
stock/commodity perps. Read-only.
"""
from __future__ import annotations
import json, ssl, urllib.request
import pandas as pd
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine.basis_carry_live import scan_opportunities

CTX = ssl.create_default_context()
def _get(u):
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "M"}), timeout=20, context=CTX).read())

TOKENIZED = ("XAU", "XAG", "SOXL", "SPCX", "ESPORTS", "SOX", "NVDA", "TSLA", "MSTR", "COIN", "HOOD")


def funding_scan(min_qvol=3e7):
    prem = {d["symbol"]: float(d["lastFundingRate"]) for d in _get("https://fapi.binance.com/fapi/v1/premiumIndex")
            if "lastFundingRate" in d}
    vol = {d["symbol"]: float(d["quoteVolume"]) for d in _get("https://fapi.binance.com/fapi/v1/ticker/24hr")}
    rows = []
    for s, f in prem.items():
        if not s.endswith("USDT") or vol.get(s, 0) < min_qvol:
            continue
        rows.append(dict(perp=s, coin=s[:-4], funding_8h_bp=f * 1e4, ann_funding_pct=f * 3 * 365 * 100,
                         qvol_musd=vol[s] / 1e6, tokenized=any(t in s for t in TOKENIZED)))
    return pd.DataFrame(rows).sort_values("ann_funding_pct", ascending=False)


def main():
    print("=== 1) BASIS CARRY universe (all Binance dated quarterlies) ===")
    b = scan_opportunities()
    print(b.assign(ann_basis_pct=(b.ann_basis * 100).round(2))[["venue", "contract", "coin", "dte", "ann_basis_pct"]].to_string(index=False))
    coins_with_q = sorted(b.coin.unique())
    print(f"-> only {len(coins_with_q)} coins have dated quarterlies on Binance: {coins_with_q}")
    print("   (basis carry universe is structurally SMALL — only the largest coins get quarterly listings)")

    print("\n=== 2) FUNDING CARRY universe (liquid perps, current annualized funding) ===")
    fd = funding_scan()
    print(f"   {len(fd)} perps with >$30M 24h volume. Funding carry: short perp/long spot collects +funding.")
    print("\n  TOP positive funding (richest long-spot/short-perp carry right now):")
    print(fd.head(12)[["perp", "ann_funding_pct", "funding_8h_bp", "qvol_musd", "tokenized"]].round(2).to_string(index=False))
    print("\n  MOST NEGATIVE funding (long-perp/short-spot carry, or shorts pay):")
    print(fd.tail(6)[["perp", "ann_funding_pct", "funding_8h_bp", "qvol_musd", "tokenized"]].round(2).to_string(index=False))
    tok = fd[fd.tokenized]
    if len(tok):
        print("\n  TOKENIZED-ASSET perps (gold/silver/equities/ETFs on Binance) — funding reflects synthetic-exposure cost:")
        print(tok[["perp", "ann_funding_pct", "qvol_musd"]].round(2).to_string(index=False))
    n_rich = (fd.ann_funding_pct.abs() > 10).sum()
    print(f"\n-> {n_rich}/{len(fd)} perps have |annualized funding| > 10%/yr — a much WIDER carry universe than basis,")
    print("   but funding carry is REGIME-DEPENDENT (compresses/flips), unlike settlement-locked basis.")


if __name__ == "__main__":
    main()
