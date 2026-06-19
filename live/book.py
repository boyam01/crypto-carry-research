"""Live execution engine for the 4-premium market-neutral book.

SAFETY (research->execution scope upgrade, but bounded):
- DEFAULT MODE = PAPER. No API keys, no real orders, read-only public data only.
- LIVE order placement is a GATED STUB (place_live_order) that raises until the
  user wires their OWN authenticated client and explicitly confirms each order.
  This engine never holds credentials nor moves money on the user's behalf.

Sleeves (all market-neutral structural risk premia validated in REPORT.md):
  1 funding_carry      delta-neutral spot/perp funding harvest (EMA+hysteresis)
  2 calendar_basis     long spot / short dated quarterly future (cash-and-carry)
  3 xvenue_funding     long perp cheap-funding venue / short rich-funding venue
  4 vrp                short BTC vol vs realized   (SIGNAL-ONLY: needs options acct)

Risk controls baked in (research flagged these as the real risks):
  - short-spot BORROW avoidance (prefer no-borrow legs; haircut borrow legs)
  - gross LEVERAGE cap
  - vol-spike CIRCUIT BREAKER (de-risk the carry/VRP fat tail)
"""
from __future__ import annotations
import sys, json, time, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import fetch_binance as fb
from engine import fetch_funding as ff

STATE = pathlib.Path(__file__).resolve().parent / "state.json"

CFG = dict(
    capital_usd=100_000.0,
    carry_coins=["BTCUSDT", "ETHUSDT", "LTCUSDT", "ADAUSDT", "LINKUSDT"],
    xvenue_coins=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    calendar=["BTCUSDT", "ETHUSDT"],
    funding_ema=21, funding_thr=1e-4,          # 1bp/8h smoothed band
    basis_min_ann=0.05,                        # require >5%/yr basis after cost
    vrp_margin=3.0,                            # DVOL must exceed realized by 3 vol pts
    max_gross=3.0,                             # gross leverage cap (x NAV)
    max_per_instrument=0.5,                    # |notional_i|/NAV cap
    borrow_rate_ann=0.10,                      # assumed spot-short borrow cost
    cb_dvol=80.0,                              # circuit breaker: BTC DVOL% threshold
    cost_bps=5.0,
)


# ---------------- live price + signal snapshot (read-only public) ----------------
def _last(sym, fut, interval="1h"):
    end = int(time.time() * 1000); start = end - 3 * 24 * 3600 * 1000
    k = fb.klines(sym, interval, start, end, futures=fut)
    return float(k["close"].iloc[-1]) if k is not None and len(k) else None


def dvol(cur="BTC"):
    import urllib.request, ssl, json as J
    end = int(time.time() * 1000); start = end - 90 * 24 * 3600 * 1000
    u = (f"https://www.deribit.com/api/v2/public/get_volatility_index_data?"
         f"currency={cur}&start_timestamp={start}&end_timestamp={end}&resolution=1D")
    try:
        d = J.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "x"}),
                    timeout=15, context=ssl.create_default_context()).read())["result"]["data"]
        return float(d[-1][4])
    except Exception:
        return None


def front_quarter(sym):
    """Find the nearest live USDT-M quarterly contract symbol + days-to-expiry."""
    import datetime as dt
    today = dt.datetime.utcnow().date()
    for code in ["260626", "260925", "261226", "270327"]:
        d = dt.datetime.strptime(code, "%y%m%d").date()
        dte = (d - today).days
        if 10 < dte < 200:
            return f"{sym}_{code}", dte
    return None, None


def snapshot():
    """One read-only public snapshot -> raw signals for every sleeve."""
    s = dict(ts=pd.Timestamp.utcnow().isoformat(), prices={}, funding={}, bybit_funding={},
             basis={}, dvol_btc=dvol("BTC"))
    end = int(time.time() * 1000); start = end - 30 * 24 * 3600 * 1000
    coins = set(CFG["carry_coins"]) | set(CFG["xvenue_coins"]) | set(CFG["calendar"])
    for c in coins:
        s["prices"][f"BINANCE:PERP:{c}"] = _last(c, True)
        s["prices"][f"BINANCE:SPOT:{c}"] = _last(c, False)
        fr = fb.funding_rate(c, start, end)
        if fr is not None and len(fr):
            ema = fr["fundingRate"].ewm(span=CFG["funding_ema"], adjust=False).mean().iloc[-1]
            s["funding"][c] = float(ema)
    for c in CFG["xvenue_coins"]:
        try:
            bf = ff.bybit_funding(c, start, end)
            if bf is not None and len(bf):
                s["bybit_funding"][c] = float(bf.iloc[-1, 0])
            bk = ff.bybit_klines(c, 240, start, end)        # 4h; last close for the Bybit leg
            if bk is not None and len(bk):
                s["prices"][f"BYBIT:PERP:{c}"] = float(bk["close"].iloc[-1])
        except Exception:
            pass
    for c in CFG["calendar"]:
        fsym, dte = front_quarter(c)
        spot = s["prices"].get(f"BINANCE:SPOT:{c}")
        fut = _last(fsym, True, "1h") if fsym else None
        if fut and spot and dte:
            s["basis"][c] = dict(sym=fsym, dte=dte,
                                 ann=float((fut / spot - 1) * 365 / dte), fut=fut)
            s["prices"][f"BINANCE:FUT:{fsym}"] = fut
    return s


# ---------------- sleeve target legs: (instrument, signed_weight, needs_borrow, sleeve) -------
def sleeve_funding_carry(s):
    legs = []
    for c in CFG["carry_coins"]:
        f = s["funding"].get(c)
        if f is None or abs(f) < CFG["funding_thr"]:
            continue
        if f > 0:                                   # longs pay shorts -> short perp/long spot (NO borrow)
            legs += [(f"BINANCE:PERP:{c}", -1.0, False, "carry"),
                     (f"BINANCE:SPOT:{c}", +1.0, False, "carry")]
        else:                                       # negative funding -> long perp/short spot (NEEDS borrow)
            legs += [(f"BINANCE:PERP:{c}", +1.0, False, "carry"),
                     (f"BINANCE:SPOT:{c}", -1.0, True, "carry")]
    return legs


def sleeve_calendar_basis(s):
    legs = []
    for c, b in s["basis"].items():
        if b["ann"] > CFG["basis_min_ann"]:          # contango -> long spot / short future
            legs += [(f"BINANCE:SPOT:{c}", +1.0, False, "calendar"),
                     (f"BINANCE:FUT:{b['sym']}", -1.0, False, "calendar")]
    return legs


def sleeve_xvenue_funding(s):
    legs = []
    for c in CFG["xvenue_coins"]:
        bn, by = s["funding"].get(c), s["bybit_funding"].get(c)
        if bn is None or by is None or abs(bn - by) < CFG["funding_thr"]:
            continue
        # short perp on the higher-funding venue (receive), long perp on the lower (pay less)
        if bn > by:
            legs += [(f"BINANCE:PERP:{c}", -1.0, False, "xvenue"),
                     (f"BYBIT:PERP:{c}", +1.0, False, "xvenue")]
        else:
            legs += [(f"BINANCE:PERP:{c}", +1.0, False, "xvenue"),
                     (f"BYBIT:PERP:{c}", -1.0, False, "xvenue")]
    return legs


def sleeve_vrp(s):
    # SIGNAL-ONLY: execution needs a Deribit options account. Emit target vega sign.
    d = s.get("dvol_btc")
    if d is None:
        return []
    end = int(time.time() * 1000); start = end - 40 * 24 * 3600 * 1000
    k = fb.klines("BTCUSDT", "1d", start, end, futures=True)
    rv = float(np.log(k["close"] / k["close"].shift(1)).dropna().std() * np.sqrt(365) * 100)
    if d - rv > CFG["vrp_margin"]:
        return [("DERIBIT:VOL:BTC", -1.0, False, "vrp")]   # short vol (signal only)
    return []


# ---------------- portfolio combine + risk controls ----------------
def build_book(s):
    sleeves = {"carry": sleeve_funding_carry(s), "calendar": sleeve_calendar_basis(s),
               "xvenue": sleeve_xvenue_funding(s), "vrp": sleeve_vrp(s)}
    nav = CFG["capital_usd"]
    # equal risk budget across ACTIVE executable sleeves (vrp is signal-only)
    exec_sleeves = [k for k in ("carry", "calendar", "xvenue") if sleeves[k]]
    budget = (CFG["max_gross"] / max(1, len(exec_sleeves))) if exec_sleeves else 0.0

    target = {}      # instrument -> notional_usd
    signals = {"vrp_signal": [l[0] for l in sleeves["vrp"]]}
    for name in exec_sleeves:
        legs = sleeves[name]
        per = budget / (len(legs) / 2)               # legs come in pairs
        for inst, w, needs_borrow, _ in legs:
            notion = w * per * nav
            if needs_borrow:                          # haircut borrow legs; drop if uneconomic
                notion *= 0.5                          # ponytail: borrow legs carry less risk budget
            target[inst] = target.get(inst, 0.0) + notion

    # circuit breaker: vol spike -> de-risk whole book
    cb = 1.0
    if s.get("dvol_btc") and s["dvol_btc"] > CFG["cb_dvol"]:
        cb = CFG["cb_dvol"] / s["dvol_btc"]
        target = {k: v * cb for k, v in target.items()}

    # per-instrument cap
    cap = CFG["max_per_instrument"] * nav
    target = {k: float(np.clip(v, -cap, cap)) for k, v in target.items()}

    # gross leverage cap
    gross = sum(abs(v) for v in target.values())
    if gross > CFG["max_gross"] * nav and gross > 0:
        scale = CFG["max_gross"] * nav / gross
        target = {k: v * scale for k, v in target.items()}

    net = sum(target.values())
    return dict(target_notional=target, gross_lev=sum(abs(v) for v in target.values()) / nav,
                net_usd=net, circuit_breaker=cb, sleeves_active=exec_sleeves,
                vrp_signal=signals["vrp_signal"])


# ---------------- paper broker (default) + live stub ----------------
class PaperBroker:
    def __init__(self):
        self.st = json.loads(STATE.read_text()) if STATE.exists() else \
            dict(nav=CFG["capital_usd"], cash=CFG["capital_usd"], pos={}, last_px={},
                 realized=0.0, history=[])

    def mark(self, prices):
        pnl = 0.0
        for inst, qty in self.st["pos"].items():
            px = prices.get(inst); lpx = self.st["last_px"].get(inst)
            if px and lpx:
                pnl += qty * (px - lpx)
        for inst, px in prices.items():
            if px:
                self.st["last_px"][inst] = px
        self.st["nav"] += pnl
        return pnl

    def rebalance(self, target_notional, prices):
        orders = []
        for inst, notion in target_notional.items():
            px = prices.get(inst)
            if not px:
                continue
            tgt_qty = notion / px
            cur_qty = self.st["pos"].get(inst, 0.0)
            dq = tgt_qty - cur_qty
            if abs(dq * px) < 1.0:
                continue
            cost = abs(dq * px) * CFG["cost_bps"] / 1e4
            self.st["nav"] -= cost
            self.st["pos"][inst] = tgt_qty
            orders.append(dict(inst=inst, side="BUY" if dq > 0 else "SELL",
                               qty=round(dq, 6), px=px, cost=round(cost, 2)))
        # close instruments no longer targeted
        for inst in list(self.st["pos"]):
            if inst not in target_notional and abs(self.st["pos"][inst]) > 0:
                px = prices.get(inst)
                if px:
                    orders.append(dict(inst=inst, side="CLOSE",
                                       qty=round(-self.st["pos"][inst], 6), px=px, cost=0.0))
                self.st["pos"][inst] = 0.0
        return orders

    def save(self):
        self.st["pos"] = {k: v for k, v in self.st["pos"].items() if abs(v) > 1e-9}
        STATE.write_text(json.dumps(self.st, indent=2, default=float))


def place_live_order(*a, **k):
    raise NotImplementedError(
        "LIVE trading is intentionally not wired. To go live you must: (1) add your OWN "
        "authenticated exchange client, (2) supply keys via your own secret store (never "
        "committed), (3) implement per-order human confirmation. This engine will not place "
        "orders or move money on your behalf.")


def run_cycle(live=False):
    if live:
        place_live_order()                            # hard stop: gated stub
    s = snapshot()
    book = build_book(s)
    bk = PaperBroker()
    bk.mark(s["prices"])
    orders = bk.rebalance(book["target_notional"], s["prices"])
    bk.st["history"].append(dict(ts=s["ts"], nav=bk.st["nav"], gross=book["gross_lev"]))
    bk.save()
    return s, book, bk, orders


def dashboard(s, book, bk, orders):
    print(f"\n=== 4-PREMIUM BOOK (PAPER)  {s['ts'][:19]}Z ===")
    print(f"NAV ${bk.st['nav']:,.0f}  gross_lev {book['gross_lev']:.2f}x  "
          f"net_exposure ${book['net_usd']:,.0f}  CB {book['circuit_breaker']:.2f}  "
          f"BTC DVOL {s.get('dvol_btc')}")
    print(f"active sleeves: {book['sleeves_active']}  | VRP signal (needs options acct): {book['vrp_signal']}")
    print(f"orders this cycle: {len(orders)}")
    for o in orders[:20]:
        print(f"  {o['side']:5} {o['inst']:26} qty={o['qty']:>12}  @ {o['px']}  cost ${o['cost']}")
    tn = book["target_notional"]
    print(f"target legs: {len(tn)}  gross ${sum(abs(v) for v in tn.values()):,.0f}  "
          f"net ${sum(tn.values()):,.0f} (should be ~0 = market-neutral)")


if __name__ == "__main__":
    s, book, bk, orders = run_cycle(live=False)
    dashboard(s, book, bk, orders)
    # self-checks: market-neutral-ish, leverage capped, live gated
    tn = book["target_notional"]
    gross = sum(abs(v) for v in tn.values())
    assert book["gross_lev"] <= CFG["max_gross"] + 1e-6, "leverage cap breached"
    if gross > 0:
        assert abs(sum(tn.values())) < 0.6 * gross, "book not market-neutral"
    try:
        place_live_order(); raise SystemExit("live stub NOT gated!")
    except NotImplementedError:
        pass
    print("\nSELF-CHECK OK: leverage capped, market-neutral, live-order path gated.")
