# Crypto Carry Research — an honest hunt for a real edge

> I ran **~40 methods across 3 waves** of crypto alpha research — from WorldQuant‑style
> formula mining to RMT, path signatures, TDA, Hawkes processes and order‑flow
> imbalance — under **deflated‑Sharpe statistics + cross‑engine adversarial
> verification**, proved that the cross‑sectional formula alphas are **non‑stationary
> and untradeable** (in‑sample‑vs‑out‑of‑sample quintile‑shape correlation **−0.507**,
> negative **gross of cost**), and isolated the **one structurally robust edge**:
> settlement‑locked **quarterly‑futures cash‑and‑carry basis** (12/12 contracts
> positive, ~4%/yr unlevered) — packaged as a read‑only, order‑gated live engine.

**This project's value is not a magic edge — it's the machinery to tell a *real* edge
from a *fitted* one, used to kill my own hype.** Most quant portfolios show one
cherry‑picked backtest. This one shows the kills.

![Research funnel + deployable edge](reports/showcase.png)

*Left: ~40 methods → almost all killed (formula alphas gross‑of‑cost = 0; signal
inverts out‑of‑sample) → 1 robust survivor. Right: the surviving basis‑carry book,
OOS region shaded, OOS Sharpe ≈ 4.*

---

## What survived, and what died (honestly)

| Family | Methods tried | Verdict | Why |
|---|---|---|---|
| **Quarterly basis cash‑and‑carry** | calendar/term‑structure | ✅ **ROBUST** | settlement rule *guarantees* convergence → entry basis earned regardless of price path; 12/12 BTC/ETH contracts positive |
| Funding‑rate carry | static / hysteresis / x‑sectional | ⚠️ marginal | real but **regime‑dependent** (bled −1.2 Sharpe in the recent OOS window); short‑spot borrow erodes it |
| Cross‑sectional formula alphas | **4,120 formulas**, 11 angles | ❌ **DEAD** | **non‑stationary**: IS‑vs‑OOS quintile‑shape corr **−0.507**, gross‑of‑cost ≈ 0; both textbook cures (stability‑selection, walk‑forward) also ≈ 0 |
| Exotic math | RMT+OU, path signatures, TDA, Hawkes, fracdiff, wavelet, entropy, optimal transport, MFDFA, EVT | ❌ DEAD | correct implementations (signature ≡ Chen to 1e‑9; MFDFA validated on cascades) — but no extractable predictive content |
| Momentum / reversal / cointegration pairs | TS/XS momentum, OU pairs, Kalman | ❌ DEAD | fragile / huge drawdowns / spreads break OOS |
| Order‑flow imbalance (OFI) | taker + maker market‑making | ❌ DEAD (retail) | fee‑walled as taker; OFI adds **zero** value to market‑making; viability is pure maker‑rebate (infrastructure, not signal) |

**The through‑line:** in an efficient market, retail‑accessible edge is **not** predictive
alpha (dies to non‑stationarity) or execution alpha (dies to fees/infrastructure) — it's
**harvesting a structural risk premium with a settlement‑guaranteed convergence**.

---

## The credibility backbone (look here first)

1. **Deflation is real, not decoration.** [`engine/backtest.py`](engine/backtest.py)
   implements the Probabilistic Sharpe Ratio and a Bailey–López‑de‑Prado **Deflated
   Sharpe** benchmark, with a self‑test that **PASSES a genuine small edge and REJECTS
   pure noise**. Positions are shifted 1 bar (no look‑ahead); turnover cost is charged
   on every rebalance.
2. **The non‑stationarity kill is the intellectual centerpiece.**
   [`experiments/alpha_stable.py`](experiments/alpha_stable.py) shows the formula signal
   is *real in‑sample* (IS‑A vs IS‑B cross‑formula IC corr **0.901** — not noise) yet the
   tradable quintile shape **regime‑flips out‑of‑sample** (corr **−0.507**) and loses
   gross of cost. *Separating "there is a statistical signal" from "it is tradable" is
   the whole game.* Both correct cures were applied **and shown to fail**: stability‑
   selection (gross −0.33) and monthly walk‑forward re‑fit (gross ≈ +0.08).
3. **Cross‑engine adversarial verification.** Every positive candidate was attacked by an
   independent **Codex** refuter (`reports/codex_refute_*.md`). It killed results I then
   conceded — e.g. short‑spot borrow (~3.5%/yr) wiping a funding‑carry "edge"; a COIN‑M
   settlement‑print bar being 54% of a candidate's PnL. *Inviting a second engine to
   refute your best results, and conceding when it wins, is the point.*
4. **Deterministic engineering gate (Probity).** [`probity_gate/`](probity_gate/) is a
   **zero‑LLM, mutation‑tested** ruler that gates 6 governance invariants — cost‑on‑
   turnover, live‑order hard‑gating, chronological OOS, deflation primitives, keyless
   public data, non‑normality‑robust statistics. The mutation sieve **kills all 6
   mutants** (proving the checks have teeth); Probity verdict = **PASS**.

---

## The deployable edge: quarterly cash‑and‑carry basis

- **Mechanism:** long spot / short dated quarterly future, held to convergence. The
  exchange **settlement rule** forces the future to the spot index at delivery → the
  entry annualized basis is earned **structurally, regardless of price path**. This is a
  *risk premium*, **not** arbitrage and **not** a prediction.
- **Evidence:** 12/12 BTC/ETH quarterly contracts positive, realized basis ≈ entry basis;
  end‑to‑end backtest (basis core + funding satellite, costs in, chronological 60/40 OOS):
  **OOS Sharpe ≈ 4, +4.1%/yr unlevered, +12%/yr at 3×**, over 2024‑06…2026‑06.
- **Live:** [`engine/basis_carry_live.py`](engine/basis_carry_live.py) emits a target book
  (read‑only; order placement is hard‑gated). Cross‑exchange scanner
  [`engine/xexch_monitor.py`](engine/xexch_monitor.py) covers Binance + OKX + Bitget.
  Playbook: [`reports/BASIS_CARRY_PLAYBOOK.md`](reports/BASIS_CARRY_PLAYBOOK.md).

---

## Honest limitations (printing these *is* the portfolio value)

- **No live fills.** Read‑only by design (`place_order` raises `NotImplementedError`).
  Slippage, partial fills, borrow availability, and funding‑timestamp execution are
  *modeled*, never observed. Nothing here has traded.
- **The < 1% max drawdown is cosmetic.** It is close‑to‑close and delta‑neutral; it does
  **not** capture the intra‑bar short‑futures‑leg liquidation path on a sharp rally at
  leverage. The real risk is margin management, not the headline number. This is a
  thin (~4%/yr unlevered), capital‑intensive carry — "picking up pennies" is a fair
  framing of the unlevered version.
- **One OOS regime (~2 years).** The basis edge has not seen a violent backwardation /
  deleveraging cascade out‑of‑sample. Robustness rests on the *settlement mechanism*,
  not on regime diversity.
- **Program‑level deflation is not airtight.** Each script deflates within its local
  variant grid; across 40+ methods and 4,120+ formulas the family‑wide multiple‑testing
  penalty is larger than any single PSR = 1.0 suggests. (The surviving edge is defended
  *mechanistically*, which is why it stands anyway.)
- **Cross‑exchange / COIN‑M legs are scanned live but not backtested** (inverse‑contract
  sizing, cross‑venue transfer/basis risk unvalidated).

---

## Quickstart

```bash
pip install -r requirements.txt          # numpy, pandas, scipy, scikit-learn, pyarrow, requests, matplotlib

# engine self-checks (offline, deterministic)
python engine/stats.py
python engine/backtest.py                # PSR/Deflated-Sharpe: PASSES a real edge, REJECTS noise

# the surviving edge
python experiments/basis_carry_spec.py   # 12/12 contracts positive, Sharpe ~4.2 (uses cached data)
python experiments/basis_carry_backtest.py   # end-to-end OOS book
python engine/basis_carry_live.py        # live target book (read-only; needs internet)
python engine/xexch_monitor.py           # Binance+OKX+Bitget carry dashboard

# the kills (reproduce the honesty)
python experiments/alpha_stable.py       # the -0.507 non-stationarity proof
python experiments/maker_ofi.py          # OFI adds no value to market-making

# regenerate the showcase figure
python experiments/make_showcase.py      # -> reports/showcase.png
```

All market data is fetched from **public, keyless, read‑only** exchange endpoints and
cached locally (`data/cache/`, git‑ignored). No API keys, ever.

---

## Repository map

```
engine/        data fetchers (Binance/Bybit/OKX public), math toolkit, backtester
               (PSR/Deflated-Sharpe), live basis-carry engine, cross-exchange scanners
experiments/   the 3 waves of candidates (battery, exotic math, 4120-formula mining,
               OFI) + the surviving carry books + the showcase generator
probity_gate/  zero-LLM, mutation-tested governance gate (6 invariants, verdict PASS)
reports/       REPORT.md (full 23-section write-up, 中文), BASIS_CARRY_PLAYBOOK.md,
               showcase.png, and per-candidate result JSONs
```

- **Deep dive:** [`reports/REPORT.md`](reports/REPORT.md) — the full 23‑section research log (Chinese).
- **License:** MIT ([`LICENSE`](LICENSE)). **Disclaimer:** [`DISCLAIMER.md`](DISCLAIMER.md) — research only, not financial advice, no live trading.
