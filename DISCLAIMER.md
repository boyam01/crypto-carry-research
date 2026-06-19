# Disclaimer

This repository is a **research project**, published for educational and
portfolio purposes. It is **not financial advice**, not an investment
recommendation, and not an offer to trade.

- **No live trading.** The code is read-only research and a *decision engine*
  that emits target positions for manual review. The live-order entrypoint is
  intentionally hard-gated (`raise NotImplementedError`) and is verified as such
  by the deterministic Probity gate. Wiring real order placement, and any
  consequences thereof, are entirely the user's responsibility.
- **Backtests are not future returns.** All results are historical, out-of-sample
  where stated, and net of modeled costs — but real execution involves slippage,
  margin/liquidation risk on short legs, exchange/counterparty risk, borrow costs,
  and regime changes that backtests understate. Close-to-close drawdowns shown
  here do **not** capture intra-bar liquidation.
- **The edge is thin and risk-compensated.** The surviving strategy (delta-neutral
  basis cash-and-carry) is a structural *risk premium* (~4–8%/yr unleveraged),
  not arbitrage and not "free money." Most candidate strategies tested here were
  honestly **rejected**; see the report.
- **Trade at your own risk.** Cryptocurrency and derivatives trading can result in
  the total loss of capital. Do your own research. Comply with the laws and
  regulations of your jurisdiction. The authors accept no liability (see LICENSE).
