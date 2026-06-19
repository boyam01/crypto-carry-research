Done: 新增完整腳本 [cross_sectional_basis_carry.py](D:/量化交易CLAUDE/experiments/cross_sectional_basis_carry.py:1)。

Verified:
- `python -m py_compile .\experiments\cross_sectional_basis_carry.py`
- mock panel smoke test：週頻 ranking、1-bar shift、gross exposure、成本、OOS/PSR 路徑都跑通。

Run command for Claude:
```powershell
cd "D:\量化交易CLAUDE"
python experiments/cross_sectional_basis_carry.py
```

腳本會輸出：
- `reports/cross_sectional_basis_contracts.csv`
- `reports/cross_sectional_basis_carry.json`

Assumptions used:
- 主結論以 `COIN-M expired-only` 為準，live contracts 只做診斷。
- BTC/ETH 的 USDT-M 另做 sanity suite，不混進 primary，避免 BTC/ETH 因雙 venue 被 overweight。
- COIN-M inverse PnL 先用 USD price spread return 近似；production 需補 contract multiplier/collateral 精算。
- `xsec_minus_all_contango` 的 OOS return + PSR 是 selection 是否加值的直接判準。

Post-flight Fix Pack:
1. Must fix before real deployment: COIN-M inverse 真實保證金/合約乘數 PnL、交割 settlement price、intraday liquidation-aware maxDD。
2. Should fix soon: 加入 liquidity/OI filter，避免冷門 quarterlies 的標記價或成交斷層污染 ranking。
3. Nice to have: 把 `TOP_N/BOTTOM_N/MIN_DTE` 做成 CLI args，方便 Claude 做固定規格 sweep。