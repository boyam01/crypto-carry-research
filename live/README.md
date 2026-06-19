# Live Engine — 4-Premium Market-Neutral Book (PAPER)

Deploys the 4 validated structural risk premia from [../reports/REPORT.md](../reports/REPORT.md)
as a single market-neutral book. **Paper/shadow by default — no keys, no real orders.**

## Run
```bash
python live/book.py          # one cycle: fetch public data -> targets -> paper-fill -> dashboard
```
State persists in `live/state.json` (NAV, positions, history). Delete it to reset.

## Schedule
Run every **8h aligned to funding settlement** (00:00 / 08:00 / 16:00 UTC) so funding-based
sleeves rebalance with the funding clock. Use Windows Task Scheduler / cron to call the run.
Calendar-basis and VRP only need daily refresh.

## Sleeves & gates (only trades premia ABOVE cost — regime-dependent)
| Sleeve | Trade | Gate | Execution |
|---|---|---|---|
| funding_carry | short perp / long spot (per coin) | smoothed funding > 1bp/8h | spot+perp ✅ |
| calendar_basis | long spot / short dated quarterly | annualized basis > 5%/yr | spot+future ✅ |
| xvenue_funding | short rich-funding venue / long cheap | venue funding diff > 1bp | Binance+Bybit perp ✅ |
| vrp | short BTC vol | DVOL > realized + 3 vol pts | **SIGNAL-ONLY (needs Deribit options acct)** |

If a sleeve is flat, the premium is currently below its cost gate — that is correct, not a bug.

## Risk controls (do not remove)
- **short-spot borrow avoidance** — no-borrow legs preferred; borrow legs get half risk-budget.
- **gross leverage cap** `max_gross=3.0x`, per-instrument cap `0.5x NAV`.
- **vol-spike circuit breaker** — scales the whole book down when BTC DVOL > 80%.
All tunables in `CFG` at the top of `book.py`.

## Going live (YOUR action — the engine will not do this for you)
`place_live_order()` is a hard **NotImplementedError** stub. To trade real money you must:
1. add your own authenticated exchange client (Binance/Bybit) using keys from your own secret
   store — never commit keys;
2. implement per-order human confirmation;
3. flip your own live flag.
This engine never holds credentials nor places orders on your behalf.

## Known simplifications (`ponytail:`)
- Paper PnL is currently **price mark-to-market only**; funding income/cost accrual across
  runs is the next addition for faithful carry PnL. Add when running it on a real 8h schedule.
- maxDD/risk are close-to-close — they do not model intra-bar liquidation; size leverage with
  that in mind (the research showed the 0.3% backtest maxDD is an idealization).
