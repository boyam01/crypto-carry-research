**Verdict**

No. The **cross-sectional basis-carry selection edge does not survive scrutiny** as a deployable/statistically robust edge.

What survives is weaker: the broad **basis/carry mechanism may be real**, but this specific `xsec_rank_neutral` selection backtest is only **marginal** and fails its own robustness gate.

**Why**

- No obvious same-bar look-ahead: weights are explicitly shifted one daily bar after signal construction in [cand_xsec_basis_carry.py](D:/量化交易CLAUDE/experiments/cand_xsec_basis_carry.py:366).
- Open-time vs close-time is handled acceptably for daily Binance bars because close-derived signals are shifted before returns are earned. Still, the code labels close data by `open_time`, so settlement-day precision is not production-grade.
- Frozen post-settlement rows are actively removed: rows after expiry are dropped, and repeated frozen tails are guarded in [cand_xsec_basis_carry.py](D:/量化交易CLAUDE/experiments/cand_xsec_basis_carry.py:251). The report confirms USDT-M frozen rows were dropped and primary COIN-M showed `0` frozen drops.
- Costs are included, but likely optimistic. Current OOS turnover is `0.1073/day`; at the modeled 15bp per spread turn, annual cost is already about `5.88%`. The reported OOS net return is only `4.08%`, so raising execution cost from `30bp` round-trip package to about `50.8bp` would zero it out. That is plausible for COIN-M alt quarterlies.
- Survivorship/universe bias is not solved. The universe is hardcoded to BTC/ETH/XRP/BNB/SOL and available quarterly contracts; there is no point-in-time contract/liquidity universe or delisting/unavailable-contract audit.
- The statistical gate fails: [cand_xsec_basis_carry.json](D:/量化交易CLAUDE/reports/cand_xsec_basis_carry.json:15) reports `PSR=0.870`, `DSR=0.727`, and selection value-add PSR only `0.703`. None clears a 0.95 robustness bar.
- DSR is understated because it deflates only over the 5 local variants in this suite, while the repo has 12 `cand_*.json` candidate reports. It also uses 219 daily OOS observations as if effective sample size were close to daily iid, which is optimistic for weekly-held overlapping futures spreads.
- The report itself says the primary suite verdict is `FAIL/WEAK: selection does not robustly beat naive all-contango`.

**Bottom line**

I would mark this as **FAIL / do not deploy as an edge**. The backtest is not killed by a blatant look-ahead bug, but it is killed by **weak deflated significance, fragile cost assumptions, hardcoded survivor-style universe, short sample, and non-robust value-add over naive carry**.

**Verified**

Read and cross-checked `experiments/cand_xsec_basis_carry.py`, `reports/cand_xsec_basis_carry.json`, `reports/cross_sectional_basis_carry.json`, `reports/cross_sectional_basis_contracts.csv`, and `engine/backtest.py`. I did not rerun the script because the audit target was the existing generated report.