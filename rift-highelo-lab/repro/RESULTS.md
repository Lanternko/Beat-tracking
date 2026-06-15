# Reproduction: ProjektZero + LoLDraftAI (2026-06-15)

複現兩個對標 repo，量測校準/CI，驗證本專案核心發現（draft⊥勝率、execution/player≫draft）。
背景見 auto-memory `sr-prior-art-survey` 與 `../README.md` 的「先行研究與定位」。

**一句話**：player/team 強度（ProjektZero，職業賽）→ **64%**；draft（高端 soloQ）→ **52%，幾乎是擲硬幣**。
這兩個數字就是整個專案的論點——matchmaking 把賽前訊號壓平，能預測的只剩「玩家強度」與「賽中 execution」，不在 draft。

---

## 1. ProjektZero（player/team 強度軸，職業賽）

- 資料：Oracle's Elixir 2023–2025（Google Drive，見下方 file IDs），清理後 **31,283 場**。
- 方法：Team-Elo + Player-Elo + TrueSkill + TrueSkill-normalized EGPM + Side-EMA，accuracy-weighted ensemble（原 repo 邏輯，未改演算法）。
- Elo/TrueSkill/EMA 用 pre-game `_before` 值 = 天然 walk-forward；EGPM 是 logistic regression、ensemble 權重來自準確率。

### 忠實複現（全資料，原 repo 口徑）
| model | acc | logloss | brier | ProjektZero 2021 |
|---|---|---|---|---|
| TeamElo | 62.68% | .6431 | .2261 | 61.79 / .6498 / .2290 |
| PlayerElo | 63.68% | .6361 | .2229 | 62.60 / .6423 / .2257 |
| TrueSkill | 63.50% | .6338 | .2220 | 62.48 / .6404 / .2250 |
| EGPM | 59.28% | .6639 | .2359 | 59.53 / .6637 / .2357 |
| SideEMA | 52.30% | 1.5354 | .2668 | 51.93 / 1.662 / .2700 |
| **Ensemble** | **64.33%** | **.6378** | **.2234** | **63.66 / .6425 / .2255** |

→ 在新資料上落在原數字 ±1–2pp，模型排序一致。**忠實複現成功。**

### 誠實版（time-split + bootstrap CI）
原 repo 在全資料上 validate（EGPM-LR 與 ensemble 權重 in-sample）。改成：ratings walk-forward、
EGPM-LR 與 ensemble 權重只用 train、評估在最後 20%（6,257 場 held-out）。

- Ensemble held-out: **63.81%** / logloss .6395 / brier .2243 / **ECE .0428**
  （≈ 全資料 64.33%，in-sample optimism 僅 ~0.5pp，方法穩健）
- **Bootstrap 90% CI（2000 resamples, 6257 場）**：
  - accuracy [62.9%, 64.7%]，**±0.9pp**
  - logloss [.635, .644]；brier [.222, .226]
  - 符合理論 0.82/√6257 ≈ ±1.0pp
- 校準：ECE .043——accuracy-weighted ensemble **resolution 好但校準普通**（平均機率會被拉平）；
  要 deploy 應加 Platt/isotonic（對標 LoLDraftAI ECE .009）。

---

## 2. LoLDraftAI-style draft→win（draft 軸，高端 soloQ）

- 無法原尺度複現（它需數百萬場）。改用「縮小重建」：champion-identity logistic regression
  （blue +1 / red −1，= DraftGap/ARAM 編碼），高端小資料下最不過擬合的誠實 baseline。
- 資料：① 本專案 KR Challenger+GM `data/lol.db`（**9,648 場**，patch 16.x）；
  ② 公開 HF `gptilt/lol-basic-matches-challenger-10k`（patch 15.x，asia + global 對照）。

| 資料 | 場數 | base rate | champ-LR draft→win | 90% CI | logloss vs base | coinflip |
|---|---|---|---|---|---|---|
| **KR Chall+GM (16.x)** | 9,648 | 49.5% | **52.5%** / ECE .040 | [50.8, 54.4] ±1.8pp | .6930 vs .6933 | 外（勉強） |
| HF asia (15.x) | 3,334 | 51.1% | 51.4% / ECE .067 | [48.1, 54.7] ±3.3pp | .707 vs .694（過擬合） | **內** |
| HF global (15.x) | 10,002 | 50.8% | 51.9% / ECE .043 | [50.1, 53.7] ±1.8pp | .696 vs .694 | 外（勉強） |

**結論**：三個獨立資料集、兩個 patch 世代、四個 region → **draft→win 在 Challenger 穩定 ~51–53%，幾乎擲硬幣**，
且 logloss 幾乎贏不過 matchmaking-flattened 的 base rate。比 LoLDraftAI 的 55.88%（Emerald+）、
DraftRec 55.35%（top 0.1% 混 elo）還低——完全符合 Q3：**elo 越高，draft 越不可預測**。

### 每英雄勝率 CI（Q4.1 實證）
KR 9648 場，median **419 場/英雄** → 每英雄 90% CI ≈ ±4pp（width 8pp）。
範例：LeeSin 53.7%±1.5、Ezreal 48.0%±1.6、Varus 47.8%±2.0。
→ tier cutoff 若每 2pp 一階，**相鄰 tier 的 CI 直接重疊**；Rift tier list 必須畫成區間帶，別假裝嚴格排名。

---

## 2b. 前向預測：泛化測試（predict unseen tournaments）

只用「事件開始前」的比賽訓練 ratings，預測沒看過的賽事（`repro/projektzero/enrich_all.py` 含 2026 資料到 6/14、`forward_test.py`）。

| 賽事 | 類型 | 場數 | ensemble acc | 90% CI | 贏擲硬幣? |
|---|---|---|---|---|---|
| **LCK 2026 Rounds 1-2（你的「MSI之路」，6/14 剛打完）** | 區域 | 224 | **69.4%** | [64.5, 74.3] | 是 |
| LCK 2026 Cup | 區域 | 125 | 62.8% | [55.6, 70.0] | 是 |
| First Stand 2026 | 國際 | 45 | 70.0% | [58.9, 81.1] | 是（n 小 CI 寬）|
| EWC 2026 | 混 | 181 | 64.9% | [59.1, 70.4] | 是 |
| MSI 2025 | 國際 | 80 | 56.2% | [47.5, 65.0] | **否（含 50%）** |
| Worlds 2025 | 國際 | 96 | 55.7% | [47.9, 63.5] | **否（含 50%）** |

- **跨年份泛化 OK**：2026 區域賽（LCK road-to-MSI）**69.4%**（logloss .581 ≫ 擲硬幣 .69），比歷史混合還好——模型只用 4/1 前的資料就準確預測未見的 LCK 賽季。
- **失效在「賽事類型」不是年份**：國際賽（MSI/Worlds，只剩各區最強隊）壓縮到 ~55%、CI 含擲硬幣——和高端 soloQ 同一個壓縮現象。**全資料 64% 是被區域常規賽的「強隊輾壓」灌出來的。**
- PlayerElo 單模型在國際賽 ≈ 或勝過 ensemble（其他模型在跨區硬仗加噪音）。
- **MSI 2026 尚未舉辦**（資料無）。當前戰力榜前段：G2、T1、Gen.G、HLE、BLG（cross-region Elo 會高估弱區霸主，謹慎看）。要預測具體對戰把 bracket 給我即可。

## 2c. 那 69% 是強隊輾壓灌的嗎？是（但模型很誠實）

`repro/projektzero/strong_vs_strong.py`，PlayerElo（roster 強度、零英雄資訊）依信心分箱（LCK 2023–26，1873 場）：

| 模型說 | n | favorite 實際勝率 |
|---|---|---|
| 50–55%（勢均力敵）| 287 | **53.3%** |
| 55–60% | 262 | 60.7% |
| 60–65% | 267 | 64.4% |
| 65–70% | 262 | 69.8% |
| **70%+（輾壓）** | **795** | **80.0%** |

- **近乎完美校準**：說 53% 就真贏 53%、說 80% 就真贏 80%。所以勢均力敵的比賽（強隊互打就是這種）模型 ~53%、勉強贏擲硬幣，**而且它誠實地說自己不確定**。69% 是被 795 場輾壓局拉高的。
- 強隊互打（top-6 互相）65.1%（信心 66.7%）vs 其餘 72.4%（信心 70%）。
- 最頂級 T1 vs Gen.G（2026-06-14 Bo5）：每張圖 model P(T1)=42/38/44/49/44%（≈擲硬幣），實際 3–2 T1。叫不出來，因為沒人叫得出來。
- **roster-strength-only 就是這個**（ProjektZero 本來就零英雄）。vs 賭盤：favorites 上相當；賭盤的優勢來自模型看不到的消息（換人/狀態/patch 準備），需要 closing odds 才能精確比。
- **Bo5 是逐圖預測**（每張圖一 row，prob 隨 side + walk-forward 微調），但**無賽中/系列內適應**（counter-pick、momentum）= menu D 缺口；系列勝率要把逐圖 prob 摺積成 P(≥3 勝)。

## 2d. draft 偏離也 ⊥ 勝負（三度確認）

承 §2/§2c：除了 draft **內容**，連「draft **偏離** meta」也不預測勝負。
- [`projektzero/draft_deviation_probe.py`](projektzero/draft_deviation_probe.py)：以 ProjektZero 強度為 baseline，加「5 隻英雄平均稀有度」偏離分數 → held-out log-loss 幾乎不動（.6247→.6243）；off-meta 對熱門/黑馬**都**微負（−2.6/−3.5pp），無「強隊偏離=創新」交互。
- [`last_pick_probe.py`](projektzero/last_pick_probe.py)：連最後一手（紅方 pick5＝免費 counter / 藏招槽）的驚奇選擇，對強隊也是 −1.3pp（非正）。
→ **draft⊥勝負第三次坐實**（soloQ draft、pro draft 內容、pro draft 偏離全平）。能撬動勝負的不在 draft。

## 3. 對專案的意義

- 你的「純 draft AUC≈0.50」不是 bug，是 matchmaking 天花板的必然，且**比文獻/業界數字更極端**（你在最高 elo）。
- 賽中資料（gd@10）當**訓練 target / 特徵來源**才對（你的洞見）；當推論特徵預測勝負是「偷看答案」。
- 產品方向：draft→win 是紅海且天花板低；真正有 resolution 的在 **player 身份（menu B，對標 ProjektZero）**
  與 **賽中動態（menu D）**。tier list/comp（menu E）要附 CI。

---

## 4. 怎麼重跑

### draft→win（本專案資料，秒級）
```bash
python3 repro/draft_winrate.py        # KR Chall+GM (data/lol.db)
python3 repro/draft_winrate_hf.py     # 公開 HF Challenger 對照（需 huggingface_hub, pyarrow）
```

### ProjektZero（外部 repo，見 repro/projektzero/README.md）
```bash
# 1. clone + 套 2 個 pandas-2.x patch（datetime64[ns]、value_counts Series）
# 2. gdown 下載 Oracle's Elixir 2023/2024/2025/2026（file IDs 見該 README）
# 3. 依序：repro_driver.py(忠實複現) → ci_eval.py(time-split+CI)
#         → enrich_all.py(含2026) → forward_test.py(前向預測) → strong_vs_strong.py(校準)
```

環境：python3 + numpy/pandas/scikit-learn/scipy + trueskill + seaborn + huggingface_hub + pyarrow（gdown 抓資料）。

**交接**：見 [`HANDOFF-2026-06-15.md`](HANDOFF-2026-06-15.md)（檔案地圖、重跑步驟、資料地雷、下一步）。

---

## 5. 職業 draft 選角預測 + meta 理解（詳見 [`DRAFT_META.md`](DRAFT_META.md)）

既然 draft⊥勝負三度坐實，轉向**預測選角本身**＋理解 meta（這條有真訊號）。
- **選角預測**（逐手序列、walk-forward）：top-1 **15→28.5%**、recall@5 **55→74%**；lever＝移除已ban/已選(+7.3)≫fearless 2025+(+4.9)≫選手(+1.0)≫隊伍/counter/region(≈0)。**可用性約束主宰、身份訊號皆弱**。vs soloQ `predict_picks.py`（15%/41%）→ pro ≈**2× 可預測**（差在 ban/fearless/順序）。
- **meta**：寬池子（top-5 蓋 55%）、半衰期 **8 patch**；位移非均勻（TVD 0.10–0.45，**pre-Worlds 最小、季前/季中最大**）；偵測＝模型 recall 掉幅（與 TVD 相關 **+0.49**）。
- 完整數字、lever 表、scripts：[`DRAFT_META.md`](DRAFT_META.md)。
