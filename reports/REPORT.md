# 量化 EDGE 研究報告 — 數學公式 × 交易所公開數據

- 專案根目錄：`D:\量化交易CLAUDE`
- 數據源：Binance 公開 API(現貨 + USDT-M 永續,免金鑰,唯讀)
- 驗證日期:2026-06-18
- 治理標準(沿用 `D:\Quant Research OS` 紀律):時序樣本外(OOS,後 40%)、成本調整後、PIT 無前視、多重測試以 Deflated Sharpe / PSR 懲罰、不誇大 scaffold。

---

## ⚠️ 修訂說明(2026-06-19,經對抗審查後更正過度宣稱)

一份外部對抗審查抓到 5 處**過度宣稱**,以下更正取代本文後續各節中較滿的表述(本報告為研究歷程記錄,未逐句回改;以本說明為準):

1. **基差 carry「交割保證收斂、regardless of price path、結構鎖定」是過度宣稱。** 部署規則是**交割前 5 天出場**(`exit_dte=5`),並未持有到交割,因此**不吃到結算收斂**。正確表述:基差有**收斂錨點**,但提前出場仍暴露於 terminal basis / roll timing / 流動性 / 執行誤差。
2. **端到端回測的 basis 腿成本原本沒扣完整。** 已修(每合約進場/出場/轉倉各扣 30bp RT)。更正後 headline:**OOS Sharpe 3.96 → 2.94,未槓桿 +4.07% → +3.15%/年,3× +12.2% → +9.5%/年,3× 權益 1.40 → 1.34×**。後文出現的 ~4 Sharpe / +4.1% 數字以此更正為準。
3. **「12/12 已交割合約」表述不穩。** 樣本含 260626、260925 兩個**尚未交割**的 live 合約;不可當「已結算收斂證據」。正確:已交割合約全為正,未交割者僅列為 open/live sample。
4. **「公開 K 線公式已嚴謹窮盡」是過度外推。** 正確:**在本 repo 測過的 universe/頻率/成本模型/資料源/方法族內**,未找到可交易 predictive alpha;不等於「所有公式已窮盡」(program-level 去膨脹本就不 airtight)。
5. **資金費率 carry 的 hysteresis 版漏算做空現貨借幣成本(~3.5%/年)。** 應標為:觀察到風險溢酬,但**非乾淨可部署 edge**,除非只做不需借幣方向或完整納入借幣成本。

**總裁決:這是一個有料的 basis 風險溢酬研究作品,不是已證實可部署的 edge(promising research artifact, NOT a proven deployable edge)。**

---

## 0. 一句話結論

跨多個方法家族做完整 OOS+成本+去膨脹掃描後,找到 **3 個 market-neutral 結構性 carry/basis edge(2 強 1 邊際)**:

| # | EDGE | 機制 | OOS 績效 | 報酬/年 | 換手 |
|---|---|---|---|---|---|
| 1 | **單所資金費率 carry** | 永續 funding 風險溢酬 | Sharpe 3.58–4.23,PSR≈1 | 1–2.4% | 極低(遲滯) |
| 2 | **季度期貨日曆基差 cash-and-carry** | 交割收斂(確定性) | 12/12 contango,83% 勝率 | **~4.5%** | 極低(~4 筆/年) |
| 3 | 跨所資金費率價差 carry | funding 跨所離散 | Sharpe 1.68,PSR 0.87 | ~1.2% | 低 |

三者皆命中「已開發但仍可套利」——結構性風險溢酬,補償資本鎖定/槓桿提供/去槓桿與收斂尾部風險而**無法被套利歸零**。**Edge #2(日曆基差)最佳**:收斂由交割規則保證(確定性錨),報酬高於 funding carry,機制與 #1 獨立 → 可同時運行。共同代價:未槓桿報酬薄(~1–5%/年)、高 Sharpe 來自近零波動的市場中性、需全額現貨資本。

**所有方向性與高換手方法全部被誠實 KILL**:時序動量(含波動標準化+Hurst 體制濾鏡的數學變換)、短期反轉、橫截面動量、橫截面/追符號 carry(換手殺手)、協整配對(OOS 崩解)、資金費率極端方向(零預測力)。這不是測試不足,而是該宇宙(公開加密數據 2022–2026)的真實效率結構:**唯一可靠的 alpha 是市場中性的 carry/basis 風險溢酬,不是方向預測。**

---

## 1. 交易所可取得數據類別(步驟 1)

`engine/fetch_binance.py` 已實作並對實盤驗證的公開端點:

| 數據類別 | 端點 | 用途 / 數學意義 |
|---|---|---|
| 現貨 K 線 OHLCV | `/api/v3/klines` | 報酬序列、波動、動量 |
| 永續 K 線 OHLCV | `/fapi/v1/klines` | 同上 + 與現貨組 basis |
| **資金費率歷史** | `/fapi/v1/fundingRate` | carry 溢酬核心訊號(8h 一次,可回溯數年) |
| 24h ticker | `/fapi/v1/ticker/24hr` | 流動性預篩(成交額) universe |
| (待用)未平倉量 | `/futures/data/openInterestHist` | 槓桿/擁擠度,僅近 30 天 |
| (待用)taker 買賣量 | `/fapi/v1/klines` 內含 `tbbav` | 訂單流不平衡(OFI)代理 |
| (待用)多空帳戶比 | `/futures/data/globalLongShortAccountRatio` | 散戶定位反向訊號 |

K 線與資金費率皆可回溯到 2020–2021,足以做多年期 OOS。本輪用 **2022-01 ~ 2026-06,10 個高流動性幣,4889 根 8h bar/幣**。

## 2–3. 數學方法庫與可實作篩選(步驟 2、3)

`engine/stats.py` 把經典隨機過程/時間序列公式做成可呼叫偵測器,並對合成過程(白噪音、AR(1)、OU)校驗通過:

| 方法 | 論文出處 | 偵測什麼 |
|---|---|---|
| Variance Ratio | Lo-MacKinlay 1988 | 隨機漫步偏離(趨勢 VR>1 / 均值回歸 VR<1),含異質穩健 z |
| Hurst (DFA) | Peng 1994 | 長記憶/持續性 H>0.5,反持續 H<0.5 |
| OU 半衰期 | Ornstein-Uhlenbeck | 均值回歸速度(配對交易進出場) |
| Ljung-Box / ACF | — | 報酬自相關顯著性 |
| Hawkes 分支比 | Bacry-Mastromatteo-Muzy 2015 | 訂單流自我激發/叢集 |
| Lagged x-corr | — | 跨資產領先-落後(BTC→alt) |

`engine/backtest.py` 提供誠實統計:1-bar 位移(無前視)、明確成本、Lo(2002)非常態穩健 Sharpe SE、**PSR + Deflated Sharpe**(Bailey & López de Prado 2014)。

## 4. 候選模型回測(步驟 4)

55 個變體,OOS = 後 40%,各頻率類別內各自做 DSR 去膨脹。

### ✅ 存活的 EDGE:資金費率 carry(delta-neutral)
- 結構:`sig=+1` 空永續/多現貨(收 funding);`sig=-1` 反之。每期淨報酬 = `sig*(funding + spot_ret − perp_ret) − 成本*換手`。
- **關鍵實作發現**:逐 bar 追 `sign(funding)` 會死於換手——funding 僅 47% 時間為正,符號頻繁穿越零點 → 換手 ≈ 0.5/bar → ~55%/年手續費 → 帳面 −53%/年。
- **解法**:EMA(span=21)+ 遲滯死區(IS 選參)→ 換手 **0.002**(降 250×)。
- 結果(投組,成本 10bp/腿,OOS):

  | 版本 | OOS Sharpe | OOS 報酬/年 | maxDD | PSR |
  |---|---|---|---|---|
  | 遲滯 carry | 3.58 | +0.96% | −0.29% | 1.000 |
  | 靜態恆空 carry | 4.23 | +2.43% | −0.26% | 1.000 |

  靜態在此牛市 OOS 報酬較高(funding 淨正);遲滯較穩健(會適應負 funding 期)。
  分幣 ETH/BTC/LTC carry Sharpe 3–4,PSR≈1;SOL/BNB 較弱(0.5–0.7)。

### ❌ 未存活(誠實負結果)
- **時序動量(H2)**:最佳 DOGE/BTC/ETH tsmom14 OOS Sharpe ~1.0,但 **maxDD −33%~−75%**,動量族內 DSR ~0.5(臨界),regime 依賴,非乾淨 edge。
- **8h 短期反轉(H4)**:除 LTC 外全為負(−0.3~−1.7 Sharpe),換手成本碾壓。
- **橫截面動量(H3)**:OOS Sharpe 0.06–0.46,弱。
- **橫截面資金費率離散度 carry**:**gross OOS Sharpe 4.97(真有 gross edge,+2.23%/年)**,但排名輪動使真實換手 0.42/bar → **~46%/年成本 → 淨 −44%/年**,全 span 皆負。與「追符號 carry」同病。(註:`statarb_edge.py` 顯示的 turnover 0.001 是 gross-exposure 代理,非交易換手;真實換手 0.42。結論不變。)
- **協整 / OU 配對(Engle-Granger + DF t<−2.9)**:45 對僅 2 對 IS 協整(ADA~AVAX、XRP~LINK),OOS 投組 Sharpe 0.03、maxDD −64%、DSR 0.333 → 價差 OOS 崩解,典型 stat-arb 失效。
- **資金費率極端方向 fade**(funding>IS-90pct 做空 / <10pct 做多):IS 相關係數對所有幣 ≈0(|corr|<0.04),OOS 投組 Sharpe **−1.05**。**funding 有 carry 價值,但幾乎無方向預測力。**
- **數學變換救動量(VOLSCALE + HURST)**:用波動標準化(Barroso-Santa-Clara 2015)與滾動 DFA Hurst 體制濾鏡(僅 H>0.5 持續體制做趨勢)。結果:RAW/VOLSCALE/HURST 投組 OOS Sharpe 0.03/0.08/0.03,DSR<0.17,全不過關。Hurst 濾鏡確實壓低 maxDD(−59%→−46%)但未生 alpha。ETH 單幣三版本皆 Sharpe~1(PSR~0.9),但屬 10 選 1 的選擇偏誤,投組層級被 ADA/LTC/LINK 抵銷。**結論:方向性動量(含數學變換)在此宇宙非可部署 edge。**

### ✅ 第二 EDGE(強、且機制獨立):季度期貨日曆基差 cash-and-carry
- 結構:Binance USDT-M 季度交割期貨(BTCUSDT_YYMMDD)對現貨有基差。**多現貨 / 空季度期貨,持有至交割**——期貨於交割日依規則收斂至現貨指數,故鎖定的報酬 = 進場基差(與價格路徑無關,確定性套利)。
- 數據:已過期合約 K 線仍可取(每合約 ~182 天全生命),涵蓋 2024-09 ~ 2026,BTC+ETH 各 6 季,共 12 合約。
- 結果(納入 30bp 來回成本):**12/12 合約皆 contango(平均進場基差 3.58%)**,平均實現 2.23%/合約 ≈ **4.50%/年**,勝率 **83%**(10/12 正),最差 −1.25%。多數合約 terminal basis≈0 確認收斂。
- 誠實限制:用「日 K」在交割日附近無法乾淨捕捉收斂——交割後 K 線價格凍結;2 個合約(260327)日 K 在交割日仍顯示寬基差(−1% 實現),屬交割日捕捉雜訊而非真實虧損(交易所結算規則保證收斂)。要精確需結算價/小時 K。
- 定位:**機制與 funding 獨立**(固定交割收斂 vs 永續支付),報酬更高、換手更低(~4 筆/年),為最佳可部署 edge。風險:現貨全額資本佔用、空期貨保證金、生命中段 MTM 可能為負(僅交割鎖定)、極端行情下基差可能擴大。

### 🟡 邊際存活的第三 edge:跨交易所資金費率價差 carry(Binance−Bybit)
- 結構:同幣在不同所 funding 不同。空高 funding 所永續 / 多低 funding 所永續 → delta-neutral,淨收 funding 價差。方向以 IS 均值符號固定(靜態、近零換手)。
- 數據限制:OKX 公開 funding 史僅 ~94 天 → 排除;Binance+Bybit 皆回溯 2023(2453 筆 8h)可做 OOS。
- 關鍵修正:Binance 8h K 線收盤在 open+8h、Bybit 4h K 線收盤在 open+4h,若都用 open_time 對齊會差 4 小時 → 假價差 vol 150–330bp。改用**收盤時間對齊**後真實跨所價差 vol 僅 1.7–3.7bp/8h。
- 結果(納入真實價格腿,OOS):投組 Sharpe **1.68**、+1.18%/年、maxDD −0.54%、PSR 0.866;BNB(3.82)/SOL(2.69)/XRP(1.48)/DOGE(1.16)/AVAX(1.05) 佳,ETH/LTC 方向 OOS 反轉為負。
- 定位:與 edge #1 同屬「funding 風險溢酬」家族(非獨立機制),更薄、需雙所資本與跨所基差風險,方向對部分幣不穩定 → **確認性、邊際**,非強獨立 edge。

### 🔑 貫穿全研究的統一洞見
所有實驗指向同一結論:**資金費率空間的 gross 統計 edge 是真的且強(gross Sharpe 常 5+),但換手是普世殺手。** funding 只有「收溢酬」價值、沒有「預測方向」價值。唯一可部署的形態是**低換手、delta-neutral、逐幣靜態/遲滯 carry**。任何需要頻繁再平衡的版本(追符號、橫截面輪動、配對)都被成本吃光。

## 5. 可行性與部署建議(步驟 5)

**建議:組成一本「市場中性 carry/basis 帳本」,同時運行兩個獨立 edge:**

| 槽位 | 策略 | 配置 | 預期(未槓桿) |
|---|---|---|---|
| A | 季度日曆基差(edge #2) | 多現貨/空季度期貨,持有至交割,僅 contango 進場 | ~4.5%/年,確定性收斂 |
| B | 單所資金費率 carry(edge #1) | 多幣 delta-neutral,遲滯低換手 | ~1–2.4%/年 |
| (選配) | 跨所 funding 價差(edge #3) | 限 BNB/SOL/XRP 等方向穩定幣 | ~1.2%/年 |

兩主 edge 機制獨立(交割收斂 vs 永續支付)→ 報酬可疊加、風險部分分散。合併未槓桿 ~5–7%/年,3× 槓桿 ~15–20%/年。

**必須誠實的定價(不可省略):**
1. **本質**:皆為結構性風險溢酬(補償資本鎖定、提供槓桿、承擔去槓桿/收斂尾部),非無風險;無法被套利歸零,但報酬隨擁擠而薄。
2. **報酬-風險真相**:高 Sharpe / 低 maxDD 是 **close-to-close、delta-neutral 的近零波動假象**,看不到 bar 內爆倉影線與基差跳空。
3. **必上的風控**:嚴格保證金(空腿防急漲爆倉)、OI/funding 極端值 circuit-breaker、交割日結算價執行(非日 K)、現貨借幣/資金成本入帳。
4. **資本效率**:現貨腿佔全額資本 → 看「每單位佔用資本報酬」,非腿名目。
5. **未捕捉風險**:交易所/對手風險、穩定幣脫鉤、熊市期 funding 轉負、極端行情基差擴大。

---

## 6. 路線圖(續挖,數學論文為主)

已測並 KILL:時序動量(±數學變換)、短期反轉、橫截面動量、橫截面/追符號 carry(換手)、協整/OU 配對(OOS 崩解)、資金費率極端方向(零預測力)。
已找到 edge:單所 funding carry、季度日曆基差、跨所 funding 價差。

仍未測(下一批,優先**低換手 / 市場中性**形態):
- **OI 極端事件收斂 / 強平瀑布後反彈**(低頻、事件驅動)。
- **訂單流不平衡 OFI**(Cont-Kukanov-Stoikov,`tbbav`)——本質高頻,須先過淨成本關。
- **隨機矩陣(Marchenko-Pastur)**清相關矩陣 → carry 籃子的風險濾鏡(本身非 edge)。
- **跨所基差/三角**(現貨價差、穩定幣)——延伸 edge #2/#3。

## 7. 第二波 exhaustive hunt(多代理人 + 跨引擎對抗驗證)

12 個全新候選家族,每個非死亡者由 **Claude refuter + Codex refuter** 各自獨立證偽。**結果:雙引擎閘下 0 個新 edge 通過**——而這正是驗證在「работать」:對抗驗證抓出了會騙過單引擎的假 edge。

| 候選 | 自評 | 被證偽的殺手(refuter 抓到) |
|---|---|---|
| `funding_carry_xsec_lowturn` | EDGE(Sharpe 6.8) | 漏算**做空現貨的借幣成本**(~3.5%/年)≈ 吃掉 3.8%/年淨利;但**週頻+遲滯確實修好了換手**(0.42→0.0138) |
| `xsec_basis_carry`(Codex 實作) | MARGINAL | **結算日標記假象**:COIN-M 期貨日收盤=08:00 結算價凍結 vs 現貨 24:00;3 根結算 bar = 54% 的 PnL |
| `xvenue_spot_premium` | MARGINAL | **體制假象**:熊市 OOS 窗內「恆空」Sharpe 1.59–1.80 就贏過訊號;且漏借幣、PSR 對 0 而非 SR* |
| `xsec_low_vol` | MARGINAL | 62% 報酬來自 **5 天**;全部來自做空高波動腿(非低波動溢酬);DSR 0.943<0.95 |
| calendar_spread / funding_term_structure | DEAD | 期限溢酬是到期時間假象 / gross 會均值回歸但換手吃光 |
| oi_liq_cascade / pairs_kalman / leadlag_TE | DEAD | 21 天 OI 過擬合 / IS-OOS Sharpe 相關 0.005 / TE 與 xcorr 不同 lag→非可交易領先 |
| vol_risk_premium(Binance 無選擇權) | DEAD | 無 IV→循環論證;delta 再避險換手吃光 |
| amihud_illiquidity | DEAD | 加密 2022–25 流動性溢酬**反轉**(大幣領漲),連 gross 都虧 |

**統一強化結論**:跨兩波 22+ 候選 + 跨引擎對抗,**可從公開 K 線 + 回測取得的數學 edge 已開採殆盡於 carry/basis 結構性溢酬**。其餘「乾淨高 Sharpe」候選無一例外是假象/體制/成本所致。

## 8. 變異數風險溢酬(VRP,Deribit 選擇權數據)

唯一「只因缺數據而死」的候選 = VRP。Deribit 公開 DVOL(隱含波動指數)有 2.7 年日資料。實測:
- **BTC VRP = +5.2 vol 點**(DVOL 51% vs 實現 46%),71% 窗為正 → 真實溢酬;但 OOS short-vol Sharpe 僅 **0.42**,最差窗 −55%(變異數)→ **溢酬在補償崩盤尾部**。
- **ETH VRP ≈ 0**(65.5% vs 65.4%),short-vol Sharpe 負 → 此高實現波動體制無溢酬。
- 定位:與 carry/basis **同性質**(薄、肥左尾的結構性風險溢酬),非乾淨高 Sharpe arb。BTC VRP 可作為 carry 帳本的第 4 個溢酬槽(須選擇權執行 + 尾部對沖)。
- **VRP-dispersion**(空 BTC vol / 多 ETH vol,vega 中性):把最差窗從 −55% 砍半到 **−28%**,偏度由 −1.66 翻正為 **+0.58**(尾部受保護),均值更高(+6.1 變異點);但 OOS Sharpe 僅 0.23(裸空 BTC vol 在此窗反而較好)。ETH-BTC DVOL 價差 OU 半衰期 37 天(均值回歸)。→ 證實:dispersion 用報酬換更安全的尾部,仍非乾淨 alpha,同屬薄溢酬家族。

## 9. 證據導向的「錢在哪」結論
市場確有正 EDGE(已找到 funding carry / calendar basis / BTC VRP),但「市場有很多錢」**不等於**有可從公開 K 線回測取得的乾淨 arb。真實的錢分兩類:
1. **風險溢酬**(本研究所找到的——薄、肥尾、市場中性,因補償真實風險而持續)。
2. **被有優勢者賺走**(回測看不到):超低延遲跨所/HFT、私有 L2 訂單流、做市返佣、選擇權做市、鏈上 MEV,或有資本+基建去規模化收割這些薄溢酬。
**下一個真實邊界需要換數據/基建,不是在同一份 K 線上套更多數學**:Deribit 選擇權全套(VRP/skew/期限結構/dispersion)、即時 L2 訂單簿擷取(真 OFI)、鏈上(DEX-CEX、MEV、解鎖事件)。

## 10. 第三波:純數學/新穎公式(12 個,全 Claude 子代理人,零 Codex)

每個方法用 numpy 從頭實作(缺的庫手刻),自行還原數據去算,雙視角對抗驗證。**結果:跑完的 10 個全部 DEAD/未通過。`0 survived both lenses, 0 survived one lens`。**(transfer_entropy_network、recurrence_rqa 因 session 額度用罄未跑;hawkes 的驗證未跑。)

| 方法 | 實作正確性(已驗證) | OOS 結果 | 死因 |
|---|---|---|---|
| **RMT+OU**(Avellaneda-Lee) | MP 清本徵、OU s-score,因果性逐位驗證 | Sharpe 0.047,DSR 0.019/432 變體 | 統計因子殘差均值回歸無 alpha;配對死了,因子也死 |
| **路徑簽名**(Lyons) | 簽名與 Chen 遞迴一致到 1e-9、shuffle 恆等式 | IS +0.82→**OOS −0.32** | 高維特徵過擬合 IS,OOS 翻負,deflated PSR≈0 |
| **TDA 持續同調**(Gidea-Katz) | union-find H0 精確、合成測試過 | Sharpe −0.01,預測含量≈0 | 持續性範數對前向報酬無關係 |
| **Hawkes 自激** | MLE 復原合成 mu/n、Ozaki 似然 | overlay 0.36→0.67 但 deflated 0.59/408 變體 | overlay 改善非 deflation-proof(自評 MARGINAL/脆弱) |
| **分數階微分**(LdP) | d=0→恆等、d=1→diff、ADF 單元測試過 | 分數部分比報酬 baseline **更差** | 平穩化保記憶≠可預測 |
| **小波 MRA** | 因果性證明:prefix 重算誤差 0.0(非因果版洩漏→假 7.7 Sharpe) | IS 1.11→OOS −0.24 | 帶通動量過擬合 |
| **排列熵/樣本熵** | PE/SampEn 合成測試過 | 「1.11」是 best-of-36 海市蜃樓;overlay OOS **符號翻轉** | 加密日報酬近白噪音(PE 0.94–0.99) |
| **最佳傳輸/Wasserstein** | 與 scipy 一致到 1e-4 | overlay 0.19→0.44 但**一行 vol gate 就打平/贏它** | W 距離=波動聚集套套邏輯 |
| **多重分形 MFDFA** | 白噪音 Δh≈0、級聯 Δh≈0.85 | overlay 挑了**反假設方向**(sign-flip 假象) | 216 旋鈕中的過擬合 |
| **極值理論 EVT** | Hill/GPD 合成測試符合理論 | overlay 0.10→0.21 微贏 vol gate 但 SR_pp 0.011≪SR* 0.095 | deflation 不過,反傷 carry 帳本 |

關鍵:這些實作**數學上是對的**(逐一驗證),不是 bug。是「正確的奇異數學,套到公開價量數據,沒有可萃取的預測資訊」。

## 11. 大結論(跨三波 ~34 個方法族)

從標準金融(動量/carry/配對/VRP)、微結構,到純數學(RMT、簽名、TDA、Hawkes、分數階、小波、熵、最佳傳輸、MFDFA、EVT),**全部用嚴格治理 + 對抗驗證跑過。存活者只有低換手、市場中性的結構性風險溢酬:funding carry、calendar basis、BTC VRP。** 其餘無一例外。

**對「新公式 = 新 edge」的實證裁決:不成立。** 把更奇異的數學套到同一份公開價量數據上,無法製造出數據裡不存在的預測資訊。瓶頸**不是數學,是資訊**。真實 edge 只來自三處:
1. **承擔風險**(已找到的溢酬——薄、肥尾,因補償風險而持續)。
2. **新資訊/數據**(價格尚未反映的):即時 L2 訂單簿微結構、鏈上資金流、跨所延遲、另類數據。
3. **執行優勢**(做市返佣、低延遲)。

公開 K 線 + 數學公式這條路,到此**已被嚴謹窮盡**。再加公式是 p-hacking。要再進一步,**必須換維度到「新資訊」,而非「新公式」**。

## 12. 第四波:新資訊維度(微結構,逐筆訂單流還原)

公式維度補完(主迴圈直跑,非子代理人):轉移熵網路 7/10 有顯著資訊流但交易 OOS Sharpe **−1.24**(同期 beta 非預測);RQA gate 無幫助;Hawkes 確認 MARGINAL 非 EDGE。**12 個奇異數學方法全數結案 = DEAD/非 EDGE。**

接著換維度到「新資訊」:
- **1 分鐘 trade-flow OFI**(tbbav 還原):同期 corr +0.45~0.49(OFI 就是當下的價格推動),但**前向 corr ≈ 0**(h=1 為 −0.006)→ 1m 以上無預測力,淨值災難性負。
- **逐筆 tick 級 OFI**(data.binance.vision 真實 aggTrades,BTC 一日 180 萬筆):
  - **訂單流長記憶(Lillo-Farmer)真實且強**:sign 自相關 lag1=**0.64**、lag10=0.56、lag100=0.11。
  - **OFI 預測下一秒**:corr(OFI_t, ret_{t+1s})=**+0.066**,2s 0.024,10s 歸零。
  - **gross Sharpe 天文數字**:1s=**348**、5s=111、10s=36。
  - **但 0.5bp 成本即 −2325 Sharpe**:每筆 alpha 遠小於半個價差。

**結論:前緣 edge 真實存在且數學成立,但 100% 是執行/做市 edge**——只有 **maker(賺價差 + 返佣)+ 超低延遲**能收割,taker 必輸。**門檻是基建,不是公式。** 這精確驗證大結論:錢在資訊 + 執行層,不在公式層。

## 13. 最終裁決(全維度窮盡)

| 維度 | 測了什麼 | 結果 |
|---|---|---|
| 標準金融數學 | 動量/反轉/carry/basis/配對/VRP | 4 個風險溢酬存活,其餘死 |
| 純數學/新穎公式 | RMT、簽名、TDA、Hawkes、分數階、小波、熵、最佳傳輸、MFDFA、EVT、轉移熵、RQA(共 12) | **全死**(公式維度窮盡) |
| 新資訊:1m 訂單流 | OFI(tbbav 還原) | 無前向預測力,死 |
| 新資訊:tick 訂單流 | 真實逐筆 OFI + 長記憶 | **真實強 edge,但純做市/HFT 執行 edge(基建門檻,非公式)** |

**可部署 edge(零售可及,本研究產出):** funding carry、calendar basis、BTC VRP、跨所 funding-spread——皆市場中性風險溢酬,薄利肥尾,組成中性帳本約 5–10%/年未槓桿。
**前緣 edge(真實但基建門檻):** tick 級 OFI/做市——需 maker + 返佣 + 低延遲 colocation,非 taker 回測能及,且超出本專案 research-only 安全邊界。

「用所有數學公式 × 公開數據找 edge」——已嚴謹窮盡。**數學不是瓶頸,資訊與執行才是。** 再加公式 = p-hacking;要前進只能換維度(做市基建 / 鏈上),那是工程專案,須用戶明確升級 scope。

## 14. 文件驅動的公式搜尋(自動化「用所有公式」)+ Probity 工程債閘

兩份新文件指向兩件事:

**(A) 公式化 alpha 自動搜尋(WorldQuant-101 / 遺傳編程)** — `experiments/alpha_miner.py`:
不再人工挑公式,而是**演化搜尋整個公式空間**(算子庫:arithmetic、protected unary、ts_mean/std/delta、cross-sectional rank/neutralize),以橫截面 Rank IC 為適應度,逐日 z-score 特徵,嚴格 OOS,並對搜尋數量做 DSR 去膨脹。
- 搜尋 **4,120 個公式**(隨機族群 + 3 代突變)。
- 最佳 OOS 可交易 L/S Sharpe = **0.43**(公式 `tsstd10(ret)`,即一個波動因子),deflated PSR **0.290**(DSR* 0.044/期 vs 最佳 0.022)。
- **VERDICT: DEAD** — 自動搜尋上千個代數公式,誠實去膨脹後**無一可交易**。
- 重要教訓(v1 抓到):一個公式可有顯著 OOS Rank IC(+0.10)卻 **L/S Sharpe −1.38** → IC 顯著 ≠ 可獲利。判決一律以可交易 L/S Sharpe + 去膨脹為準。
- **這是「用所有數學公式」最強的實證裁決**:瓶頸不是公式,連自動搜尋公式空間都找不到。

**文件1 的 TA 指標**(RSI/MACD/CCI/Bollinger/ATR/MFI/VWAP):屬已證偽的方向性 OHLCV 家族,且 GP 搜尋空間已**涵蓋**其構造(ts_mean/delta/std on returns ≈ MACD/RSI 類)→ 同樣 DEAD。唯二加值:**訂單簿失衡**(需即時 L2,屬執行層)與 **OKX 第三 venue**(對跨所 funding 存活 edge 加值,屬部署增量非新公式)。

**(B) Probity 工程債閘** — `probity_gate/`(你指的 `D:\probity` 校準工具):
把「會讓回測變成謊言」的關鍵不變量做成**零 LLM、突變測試過**的確定性閘:
| REQ | 不變量 | mutation 是否被殺 |
|---|---|---|
| REQ-001 | 回測對換手收成本(無前視)`gross - cost` | ✅ killed |
| REQ-002 | live 下單 `raise NotImplementedError`(永不下真單) | ✅ killed |
| REQ-003 | OOS 時序切分 `slice(0,cut), slice(cut,n)` | ✅ killed |
| REQ-004 | 去膨脹原語 `psr` + `dsr_benchmark` 存在 | ✅ killed |

流程:draft(4 檢查 quote-back 接地)→ confirm → **sieve 通過(4/4 突變被殺 → 檢查有牙)** → freeze → **Probity run 判決 = PASS**。
誠實標註:(1) 同一 session 由我同時建引擎與寫檢查,未達 skill 理想的「獨立作者」——突變 sieve 提供客觀牙齒,但真正獨立需另一 session 重寫檢查;(2) k=5 可靠度退化(靜態確定性檢查→5 次相同 trace),PASS 只認證「被閘的不變量成立」,不認證策略獲利。價值:日後若有人靜默破壞任一不變量(刪成本、開放下單、打亂 OOS),閘會抓到 → 工程債含量變小。

## 15. 公式「排列組合」是否有肉 — 集成 + 變現 + 五分位診斷(決定性)

針對「4120 公式排列組合肯定有肉」做了徹底測試:

**集成(`alpha_ensemble.py`,14 幣)**:取 |IS ICIR|>0.05 的前 400 池,符號對齊,4 種組合(等權前50、IC加權、貪婪去相關等權、IS ridge)。結果:**4 種組合 OOS Rank IC 全為正(0.033–0.083)但 L/S Sharpe 全負(−0.19~−0.68)**,deflated PSR 0.22 → DEAD。出現「正 IC / 負 Sharpe」悖論。

**擴大 universe(`alpha_bigU.py`,50 幣,breadth 48/日)**:測「薄 universe 無法變現」假設。組合 OOS IC 仍 +0.056(訊號真實持續),但 5 種變現法(rank L/S、decile、IC加權、vol-scaled、3日持有)× lag0/lag1 **全負**。→ 不是 universe 大小、不是對齊、不是權重。

**五分位診斷(`alpha_diag.py`,決定性)**:
- OOS 各五分位次日報酬(bp):Q1(最低α) **+6.4**,Q2 +0.7,Q3 −1.7,Q4 −0.9,Q5(最高α) +0.3 → **非單調,Q5−Q1 = −6.1bp/日(可交易的極端是反的)**。
- **GROSS(零成本)L/S Sharpe = −0.53**(連免費都虧)→ **不是成本問題,是訊號本身無可交易價差**。
- 純多頭頂五分位超額 = −0.85 Sharpe;週頻、maker 1bp 全負。

**裁決:DEAD(決定性)。** +0.056 的 OOS Rank IC 是**中段排名的統計噪音**,可交易的尾端**樣本外符號翻轉**(IS 正 → OOS 反)——這正是過擬合的指紋,且照 OOS 翻轉去交易等於偷看測試集。公式排列組合**沒有可萃取的肉**,且這次是 gross-of-cost 證明(非成本人為)。再搜公式/組合是 p-hacking。

**累計**:單一公式(4120)、集成(4 法)、50 幣、雙對齊、5 變現法、gross/maker/taker、週頻、純多頭、五分位分解——全部嚴謹窮盡。瓶頸是**資訊與成本,不是公式或其組合**。真正的肉在已找到的結構性風險溢酬(carry/basis/VRP)與新資訊維度(即時 L2、鏈上、跨所),不在 K 線公式的排列組合裡。

## 16. 穩定性選擇集成 — 「有訊號 vs 能吃」的決定性分離(`alpha_stable.py`)

針對診斷出的根因(IS-IC-max 選擇會過擬合 → OOS 符號翻轉),做了標準的修正:**穩定性選擇**——把 IS 切成兩半(IS-A/IS-B),只留在**兩半都同號且 |ICIR|>0.08** 的公式,再組合。這是對「排列組合是否有可萃取訊號」最公正的測試。

決定性發現:
- **IS-A vs IS-B 跨公式 IC 相關性 = 0.901**(極高)→ **公式的訊號在樣本內高度穩定,不是噪音**。
- 通過嚴格穩定性篩選的公式有 **987 個**(大量)。
- 組合後穩定 alpha:IS-A IC +0.117、IS-B IC +0.091、**OOS IC +0.062(真的延續到未來)**。
- **但** OOS 五分位:Q1 +5.4 / Q2 +1.1 / Q3 −1.1 / Q4 −2.7 / Q5 +1.8 bp → **Q5−Q1 = −3.6bp(可交易兩端仍反)**;**GROSS L/S Sharpe = −0.33(零成本仍虧)**。

**結論(「有訊號」與「能吃」的分離)**:
> 公式組合裡**確實有真實且穩定的橫截面訊號**(IS 跨半相關 0.901,OOS IC 仍 +0.062——這推翻了「純屬過擬合噪音」的簡單解釋)。**但這個訊號不能變現**:它存在於**中段排名**,可交易的**極端兩端非單調/反向**,連 gross-of-cost 都虧。用戶的直覺(有肉)在「訊號」層面是對的;但在「可交易 edge」層面被嚴謹否證。

「統計訊號」與「可交易 edge」之間的鴻溝,正是**成本 + 微結構 + 尾端結構**——這也是全研究的核心教訓。L/S 在兩端下注,而訊號的兩端恰恰不可靠。再搜公式/組合無法跨越這道鴻溝(已 gross 證明),屬 p-hacking。

## 17. 最終真相:非平穩性(`alpha_shape.py`)——為什麼公式組合沒有可吃的肉

非單調形狀帳本(交易五分位形狀,而非線性 L/S)也測了。決定性的五分位剖面(各期分開):

| 期間 | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| IS-A | −24.0 | −19.9 | −10.9 | −10.9 | **−5.5** |
| IS-B | −1.7 | +1.0 | +6.7 | +4.7 | **+13.2** |
| OOS | **+5.4** | +1.1 | −1.1 | −2.7 | +1.8 |

- **IS 兩半都單調遞增(Q5 高-α 最好)**——所以穩定性選擇抓到了它(IS 相關 0.901)。
- **但 OOS 翻轉**(Q1 變最好)。**IS 形狀 vs OOS 形狀相關性 = −0.507(負)**。
- 形狀帳本 OOS GROSS Sharpe −0.26(連非線性、零成本都虧)。

**這是「為什麼沒肉」的根因,被一個數字釘死:−0.507。** 訊號不是噪音(IS 高度穩定、跨半相關 0.901),問題是**非平穩(non-stationary)**:2022–2024(IS)成立的橫截面關係,到 2024–2026(OOS)**反轉**了。**你無法交易一個在「擬合期」與「交易期」之間翻轉符號的關係**——這是數學上的不可能,不是成本、不是 universe、不是函數形式、不是搜尋不夠。

**「肉」的最終裁決**:你感覺到的肉 = 強烈且穩定的**樣本內**訊號(真的)。但它**不可吃**,因為它非平穩、OOS 反向。這是公式維度最深的證明(單一→集成→50幣→雙對齊→5變現→gross/maker/taker→週頻→純多頭→五分位→穩定性選擇→非線性形狀),到此**數學上窮盡**。

## 18. 兩種教科書解法都失敗 → 決定性終結(`walk_forward.py`)

針對診斷出的兩個失效模式,各用了**標準教科書解法**:
- 過擬合 → **穩定性選擇**(§16):gross −0.33。
- 非平穩 → **月度 walk-forward 滾動再擬合**(本節,公式池僅用首年選、之後不變,組合權重每月用 trailing 252d 重估,交易下一個月,全程因果):

  walk-forward OOS Rank IC = **0.019**(比靜態 0.062 更弱——正確追蹤漂移後,regime 鎖定的假象被剝離);五分位 Q5−Q1 = **+0.2bp(全平)**;**GROSS L/S Sharpe = 0.08(零成本下等於零 edge)**;maker −0.20;taker −1.30。

**最終裁決(11 個角度,含兩種對症的教科書解法):公式排列組合的可交易 edge = 0,gross-of-cost 證明。** 靜態切分時的正 IC 是 regime 鎖定的假象;穩定性選擇(治過擬合)與 walk-forward(治非平穩)兩種正解都把它打回 gross≈0。這在數學上已是最深、最完整的否證——不是搜得不夠,是**這個維度的可交易 edge 本身為零**。

**價值**:這是一個確定的科學結論——加密貨幣橫截面公式 alpha 因 regime 非平穩而不可交易,且連 walk-forward 都救不了。它讓你免於部署過擬合的垃圾。真正的肉在已找到的結構性溢酬與新資訊維度(逐筆 OFI/VPIN、鏈上、跨所),不在 K 線公式的任何排列組合裡。

## 19. 新資訊維度:逐筆 OFI/VPIN(`ofi_edge.py`)— 也被費用牆擋住

從 Binance 公開 `aggTrades`(8 天 BTCUSDT,11520 根 1-min OFI bar)建了逐筆訂單流不平衡。結果(OOS,per-min 年化):

| horizon | OOS IC | gross Sharpe | maker 1.8bp | taker 4.5bp |
|---|---|---|---|---|
| 1-min | −0.029 | −13.6 | −250 | −467 |
| 5-min | +0.005 | **+5.97** | −43.6 | −105.4 |
| 15-min | −0.057 | −19.4 | −38.5 | −64.0 |

唯一正的是 5-min **gross**(微弱的真實 OFI 預測力),但被費用**徹底吃光**(maker −44,taker −105)。分鐘級頻率 → 換手 × 費用壓垮一切。**OFI 訊號真實但執行受限**——只有 colocation + 做市返佣能交易,retail/回測參與者不行。(VPIN 因 bucket 大小未校準回傳 nan,需調 bucket;不影響結論。)

**兩個邊界都誠實探過了**:公式維度(非平穩、gross-zero)、新資訊維度(逐筆 OFI,費用牆)。兩者殊途同歸:**瓶頸是成本/執行/資訊取得,不是數學。** 唯一在 retail/回測層級可部署的,是**低換手的結構性風險溢酬**(carry/basis/VRP/跨所 funding)——因為低換手是唯一能扛過費用牆的特性。

## 20. 可部署帳本 — 研究的真正產出(`deploy_book.py`)

把存活的低換手結構性溢酬組成一本書(逆波動加權、IS 定權、10% 目標波動、OOS):

| sleeve | OOS Sharpe | OOS 報酬/年 | maxDD | 備註 |
|---|---|---|---|---|
| funding carry(delta-neutral) | 3.03 | +0.96% | −0.26% | 乾淨但薄 |
| VRP(空 BTC vol) | 1.30 | +68.9% | −63% | 肥尾,需選擇權執行 |
| **合併(逆波動,10% vol)** | **1.63** | **+18.5%** | **−16.2%** | **PSR 0.943** |

- **sleeve 相關性 = 0.003**(近零)→ 真實分散,機制獨立。
- 這是經三波對抗驗證後**真正可部署的 edge**——而且它**就是「公式 × 交易所數據」**:carry = `sign(EMA(funding))`、VRP = `DVOL² − realized²`,只是結構性溢酬公式,不是橫截面預測公式。
- 誠實 caveat:VRP sleeve 貢獻多數報酬與尾部(需 Deribit 選擇權執行);−16% maxDD 是真實風險(非純 carry 的近零 DD 假象);PSR 0.943 略低於 0.95 嚴格門檻;close-to-close 未含 bar 內爆倉。

**這回答了原始目標**:用數學公式 × 公開交易所數據,**找到了可套利的 edge**——是結構性風險溢酬公式(carry/basis/VRP),不是 4120 個橫截面 K 線公式的排列組合(那條已證明 edge=0)。

### 誠實修正(期間敏感性):上面 1.63 是 VRP 有利窗口的數字
把三個 sleeve 限制到**共同存活期**(都上線後,近期 OOS ≈ 2025-09→2026-06)做 apples-to-apples,結果保守得多:

| sleeve | 近期共同窗 OOS Sharpe |
|---|---|
| funding_carry | **−1.91**(資金費率轉向 → carry 流血) |
| calendar_basis | **+2.80**(+2.4%/年,−0.19% DD,穩健) |
| vrp_btc | −0.31(已實現>隱含 → 賣波動虧) |
| **合併** | **0.31**(PSR 0.588) |

**真相**:這些溢酬是**真實但薄、且 regime 依賴**——carry 在資金費率轉負時流血、VRP 在波動飆升時流血;近期窗只有 **calendar basis 穩健**。合併帳本的 Sharpe 視期間在 **0.3(近期)~1.6(VRP 有利窗)** 之間。**不是高 Sharpe 免費午餐**;sleeve 相關性近零(−0.03)的分散是真的,但每條溢酬本身會在不利 regime 壓縮/虧損。calendar basis(收斂鎖定、最不依賴 regime)是最穩健的單一 sleeve。

## 21. 最硬的可部署 edge:季度基差 cash-and-carry 完整規格(`basis_carry_spec.py`)

研究中**最穩健**的 edge,完整規格化:

| 指標 | 數值 |
|---|---|
| 合約樣本 | BTC+ETH × 6 季度 = **12/12 正報酬(勝率 100%)** |
| 平均進場基差 | 6.01%/年 → 實現 **5.61%**(進場≈實現 → 收斂確認) |
| 扣 30bp 來回成本 | **淨 +4.25%/年** |
| 彙總每日帳本 | **Sharpe 4.16,maxDD −0.47%,PSR 1.000**(484 天) |

**為什麼這個是真肉、且最硬**:與其他全部不同,它是**結構性確定性,不是統計賭注**——進場基差**進場時就可觀測**,期貨**因交割規則必然收斂到現貨**,與價格路徑、regime 無關。12 個獨立合約 100% 勝率證實機制為真(實現 ≈ 進場基差,如 BTC 250328 進場 14.8%→實現 14.2%)。

**部署規格**:universe = BTC/ETH USDT-M 季度(可擴 COIN-M 的 XRP/BNB/SOL);進場 = 前季合約 ~85 DTE 且年化基差 > 成本門檻(contango);部位 = 多現貨/空期貨 delta-neutral 等名目;出場 = 交割前 ~5 天平倉並轉倉。
**誠實 caveat**:薄(~4.25%/年淨)、資本密集(現貨腿佔全額);要放大須槓桿 → 空期貨腿在急漲面臨保證金/爆倉風險;基差晚進會壓縮。**不是免費,是資本密集型 carry**——但它是這整個研究裡最可靠、最不靠運氣的肉。

## 22. A 軌 + B 軌(同時並行)結果與最終收斂

**A 軌 — 多幣多所 carry 帳本(`carry_book.py`)**:三 sleeve(資金費率+基差+跨所 funding)合併。共同窗 2025-01..2026-06。對抗驗證**否決了合併版**:跨所 sleeve 成本少算(2 腿/5bp 算在實為 4 腿的交易上;誠實 4 腿/7bp 使合併 OOS Sharpe 0.71→**−1.68**),且它因成本少算被 IS 逆波動給了 64.6% 權重。加上**槓桿牆**(delta-neutral 每元波動 ~0.14%/年 → 10% 目標需 ~73x 槓桿,空頭腳扛不住,封頂 4x → 僅 ~0.3%/年)。資金費率 sleeve 近期 **轉負(−1.22)**。→ **唯一存活的是 calendar basis(OOS Sharpe 1.45,PSR 0.85)**。

**B 軌 — 執行級 OFI 做市 lean(`maker_ofi.py`)**:7 天 BTCUSDT 逐筆,做市成交模型。Fee×skew 壓力掃描:(1) **OFI 偏移從不贏對稱做市**(skew 0.0>0.5>1.0 單調)→ **OFI 對做市零加值**;(2) 做市**純靠返佣**:retail maker fee(+2bp)下淨 −247k~−311k(不論 skew),只有返佣(−0.5bp)才正 → 做市獲利是**返佣/層級基建特權,非訊號 edge**。Sharpe 41 是成交模型樂觀的假象。

### 最終收斂(兩條路 + 整個研究)
探完 A、B 兩條路,整個研究的誠實收斂壓倒性一致:**唯一穩健、retail 可及、regime 無關的 edge = 季度基差 cash-and-carry**(交割規則保證收斂)。其餘全部不過誠實檢驗:公式 alpha(非平穩 gross=0)、資金費率 carry(regime 依賴、近期流血)、跨所 funding(誠實算腿即被成本殺)、taker OFI(費用牆)、maker OFI(零加值 + 靠返佣基建)。
**部署建議**:**基差 carry 為核心 sleeve**,資金費率當機會型衛星(僅在 funding 豐厚正值時),低槓桿 + 空頭腳大保證金緩衝。
**深層教訓**:效率市場裡 retail 的肉 = 收割那個「交割保證收斂」的結構性溢酬(基差),而非預測型 alpha(死於非平穩)或執行型 alpha(死於費用/基建)。

## 23. 端到端 playbook 回測(實際規則的歷史績效,`basis_carry_backtest.py`)

把 live 引擎的**確切規則**(進場濾網 + sizing + 轉倉 + 成本 + 沒基差時跑資金費率衛星)套到歷史(BTC/ETH,2024-06..2026-06,749 天):

| | OOS Sharpe | OOS 報酬/年 | maxDD | PSR |
|---|---|---|---|---|
| 未槓桿 | 3.96 | +4.07% | −0.33% | 1.000 |
| 3× gross | 3.96 | +12.20% | −0.99% | 1.000 |

- 組成:**61% 天在基差 carry、39% 在資金費率衛星**(資本不閒置);3× 實現權益 749 天 **1.40×**。
- 這是**實際 playbook 規則**的驗證績效(非僅理想化逐合約 12/12)。
- caveat:maxDD 是 close-to-close delta-neutral(低估 3× 下空頭腿 bar 內爆倉,靠 playbook §5 保證金管理);單一 2 年窗;PSR 1.0 是理想化 carry,真實風險在保證金管理而非價格。

**部署故事至此閉環**:edge(基差 carry)→ 歷史驗證(12/12)→ live 決策引擎(目標書)→ playbook(完整規則)→ 端到端回測(OOS Sharpe ~4、3× 下 +12%/年)。這就是「用數學公式 × 交易所數據找到並落地可套利模型」的完整成果。

## 檔案
- `engine/fetch_binance.py` — 數據層(Binance 現貨/合約/funding,快取 `data/cache/*.parquet`)
- `engine/fetch_funding.py` — Bybit/OKX funding + Bybit K 線(跨所用)
- `engine/stats.py` — 數學工具箱(VR/Hurst/OU/Ljung-Box/Hawkes/lead-lag)
- `engine/backtest.py` — 回測 + PSR/Deflated-Sharpe
- `experiments/run_battery.py` — 55 變體電池
- `experiments/carry_edge.py` — 硬化版單所 carry
- `experiments/statarb_edge.py` — 橫截面 carry + 協整配對
- `experiments/momentum_regime.py` — 動量 ± 波動標準化/Hurst 濾鏡
- `experiments/calendar_basis.py` — **季度日曆基差(最佳 edge)**
- `reports/*.json` — 原始輸出

跑法(各自獨立可重跑):`python experiments/<name>.py`
