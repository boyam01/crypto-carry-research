不，這個 edge **沒有 SURVIVE scrutiny 作為可部署 EDGE**。最合理結論是報告自己的 `MARGINAL`，甚至偏向「不要部署，只保留為候選現象」。

**主要問題**
1. **DSR 不過關**：報告是 PSR `0.9652` 過，但 DSR `0.9432` 低於程式自己的 EDGE 門檻 `0.95`。程式明確要求 `PSR>=0.95` 且 `DSR>=0.95` 才是 EDGE，否則只是 MARGINAL：[cand_xsec_low_vol.py](D:/量化交易CLAUDE/experiments/cand_xsec_low_vol.py:281)、[cand_xsec_low_vol.json](D:/量化交易CLAUDE/reports/cand_xsec_low_vol.json:12)。

2. **多重測試仍偏樂觀**：DSR 只 deflate 這個 candidate 的 LOWVOL/BAB lookback 小網格：[cand_xsec_low_vol.py](D:/量化交易CLAUDE/experiments/cand_xsec_low_vol.py:212)、[cand_xsec_low_vol.py](D:/量化交易CLAUDE/experiments/cand_xsec_low_vol.py:262)。但實際研究 repo 已經跑過多個候選與變體，若把整個 research battery 算進 selection，DSR 只會更差。

3. **不是低波溢酬，主要是 short 高 beta/high vol alts**：JSON 顯示 long leg 年化 `-0.21%`，short leg `+19.16%`，收益幾乎全來自空高波/高 beta alt，而不是「long low-vol premium」：[cand_xsec_low_vol.json](D:/量化交易CLAUDE/reports/cand_xsec_low_vol.json:24)。這使結果高度 regime-dependent。

4. **Survivorship / universe look-ahead 存在**：宇宙是手工固定的「目前看來從 2022 有完整歷史的 top 15」，且 MATIC/POL 被排除：[cand_xsec_low_vol.py](D:/量化交易CLAUDE/experiments/cand_xsec_low_vol.py:8)、[cand_xsec_low_vol.py](D:/量化交易CLAUDE/experiments/cand_xsec_low_vol.py:47)。這避免了測試內輪動，但不是 point-in-time universe。

5. **成本/換手呈現不乾淨**：PnL 內部有用 `wl.diff().abs().sum()` 收 5bp 成本：[cand_xsec_low_vol.py](D:/量化交易CLAUDE/experiments/cand_xsec_low_vol.py:159)。但 JSON 的 `turnover` 來自 gross exposure 變化，不是真實權重 turnover，因 `evaluate()` 傳的是 `expo` 而不是 `turn`：[cand_xsec_low_vol.py](D:/量化交易CLAUDE/experiments/cand_xsec_low_vol.py:208)。另 BTC hedge 是直接扣 `beta * BTC_ret`，沒有 hedge 成本/額外 exposure：[cand_xsec_low_vol.py](D:/量化交易CLAUDE/experiments/cand_xsec_low_vol.py:205)。

**通過的部分**
- 沒看到直接同列 look-ahead：vol/beta 都有 `shift(1)`，持倉又再 `shift(1)`：[cand_xsec_low_vol.py](D:/量化交易CLAUDE/experiments/cand_xsec_low_vol.py:99)、[cand_xsec_low_vol.py](D:/量化交易CLAUDE/experiments/cand_xsec_low_vol.py:155)。
- open-time vs close-time：資料以 `open_time` 當 index 但使用 close 價；因為有額外 shift，較像保守延遲，不是明顯前視：[fetch_binance.py](D:/量化交易CLAUDE/engine/fetch_binance.py:48)。
- frozen post-settlement futures：這支用的是 USDT-M perpetual symbols，不是 dated quarterly futures，所以交割後凍結 K 線問題不適用。

**Verdict**
`xsec_low_vol` **不應標成 surviving EDGE**。它有一個可能真實但脆弱的橫截面 short-high-vol 現象；在 survivorship、完整 multiple-testing、perp funding/hedge/friction、以及更長 OOS regime 之前，不能部署。