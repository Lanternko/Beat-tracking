# 職業賽選角預測 + meta 理解（2026-06-15）

目標：徹底了解 pro champion-select meta、把「下一手」預測命中率做上去、搞清楚 meta 多久變/變多大/怎麼偵測。
資料：Oracle's Elixir 2023–2026（player+team rows，`champion`/`ban1-5`/`pick1-5`(draft 順序)/`firstPick`/`game`(系列場次)/`playerid`/`teamid`）。全部 **walk-forward**（只用過去資訊）。

> 緣起：先試「draft 偏離 → 勝負」（[draft_deviation_probe.py](projektzero/draft_deviation_probe.py)、[last_pick_probe.py](projektzero/last_pick_probe.py)），**三度確認 draft⊥勝負**（連最後一手 counter 槽、強隊互打都平/微負，corr≈0）。於是轉向：預測**選角本身**＋理解 meta，這條有真訊號且資料已就緒。

---

## 1. 預測「下一手」選角：top-1 15→28.5%、recall@5 55→74%

逐手、照真實 draft 順序（B1,R1,R2,B2,B3,R3,R4,B4,B5,R5）預測；移除已 ban/已選（＋fearless 系列已用），blend 角色 meta 人氣與選手英雄池。消融（[draft_predict_v2.py](projektzero/draft_predict_v2.py)）：

| 模型 | top-1 | recall@5 | 增量 |
|---|---|---|---|
| M0 角色 meta 人氣（無 draft-state） | 15.2% | 55.1% | 地板 |
| M1 +移除已 ban/已選 | 22.5% | 66.7% | **+7.3 / +11.6** |
| M2 +選手選角癖（α=0.5 blend） | 23.5% | 68.8% | +1.0 / +2.1 |
| M3 +fearless（Bo5 序列，2025+） | **28.4%** | **73.6%** | **+4.9 / +4.8** |
| v3 +region meta（+counter β0.2）| **28.6%** | **73.9%** | +0.2（counter +0.1）|

- **分路**（M3）：bot 29.3/76.5、sup 29.3/74.8、jng 29.1/74.9、top 28.1/72.1、**mid 26.0/69.7（最難＝flex 位）**。
- **Bo5 場次**：g1 27.5% → **g5 31.2%**（fearless 掏空池子，後段候選變窄反而更好猜）。

## 2. lever 圖譜（top-1 貢獻，全部實測）

| lever | top-1 增量 | 性質 | 證據 |
|---|---|---|---|
| 移除已 ban/已選 | **+7.3pp** | 大——可用性 | M0→M1 |
| fearless（Bo5 序列，2025+）| **+4.9pp** | 大——可用性 | M2→M3 |
| 選手身份（α=0.5）| +1.0~1.9pp | 弱 | [draft_player_diag.py](projektzero/draft_player_diag.py) |
| 隊伍身份 | +0.1pp | 無（被選手吸收）| [team_diag.py](projektzero/team_diag.py) |
| counter-pick（β=0.2）| +0.1pp | 可忽略 | [counter_sweep.py](projektzero/counter_sweep.py) |
| per-region meta | +0.2pp | 可忽略 | [draft_predict_v3.py](projektzero/draft_predict_v3.py) |

**核心洞見：pro 選角預測由「可用性約束」主宰（ban＋fearless＋meta 人氣），所有「身份/關係」訊號（選手、隊伍、counter、地區）一律弱。** 這跟專案主軸同構——能預測的是**粗約束**不是**細巧思**（一如 draft⊥勝率、player 強度≫draft）。

**選手是弱訊號（反直覺，實證）**：α 掃描 0→1，純選手池（α=1）20.6% **比純 meta（22.5%）還差**，最佳只是淡 blend（α=0.5）；**老手（≥20 場）與新手命中率幾乎一樣（23.5% vs 23.8%）**。→ pro 選手 meta-flexible，不是一招鮮。

**vs soloQ 對比**：soloQ 版（[predict_picks.py](../predict_picks.py)，KR Chall+GM）top-1 **15%** / top-5 41%；pro 版 top-1 **28.5%** / recall@5 74%——**pro ≈ 2× 可預測**，差距正來自 soloQ 沒有的可用性約束（無 ban、無 fearless、無 draft 順序）。

## 3. 理解 meta（goal #1）

- **meta 是寬池子不是窄尖**：unconditioned 每位置 top-5 只蓋 ~55%；「大家都挑最強」在單英雄層級不成立。
- **塑形選角的是「可用性」**：ban（每場 10 個專打高優先）＋ fearless（系列內用過就禁）＝ §1/§2 最大的兩個槓桿。
- **換手速度**：meta 半衰期 **中位數 8 個 patch**（~3–4 個月）才換掉 50% 選用質量（[meta_shift.py](projektzero/meta_shift.py)）。
- 地板版＋走味量化見 [draft_predictability.py](projektzero/draft_predictability.py)：1-月舊窗 −7.6pp recall@5。

## 4. meta 位移：多大、多久、非均勻（goal #3）

以 patch 為單位、相鄰版本英雄 pick-share 的 **Total Variation Distance（TVD∈[0,1]）** 量「實際位移」（major leagues，[meta_shift.py](projektzero/meta_shift.py)）。

- **單 patch 位移 TVD 0.095 ~ 0.451**（~4.7 倍差距）→ **版本大小確實不固定**。
- **非均勻、且符合直覺**（各時期平均 TVD，低→高）：

  | 時期 | 平均 TVD |
  |---|---|
  | **pre-Worlds (8-9月)** | **0.170（最小之一）** |
  | spring/MSI | 0.171 |
  | spring | 0.208 |
  | summer（季中） | 0.228 |
  | MSI | 0.253 |
  | Worlds 期 | 0.353 |
  | **preseason（季前換年）** | **0.373（最大）** |

  → 「世界賽前變動小、季中/季前大」成立。
- **最大單次位移**：14.08→14.11（季中,0.45）、14.16→14.18（Worlds 期,0.44）、13.19→14.01（季前換年,0.37）、14.18→15.01（0.36）。

## 5. 偵測 meta 變動（goal #3）

兩個訊號（[meta_detect.py](projektzero/meta_detect.py)）：
1. **TVD（落後型 ground truth）**：相鄰 patch 分布距離，threshold（>0.3＝大改）標記。需該 patch 累積足夠場次。
2. **模型自身 recall@5 掉幅（即時型，已實證）**：每 patch「早期 15% 場次」recall@5 vs「安定後」的**掉幅，與該 patch 的 TVD 相關 +0.49（n=39）**。例：14.18（TVD 0.44）early **55.3%→settled 67.0%，掉 11.7pp**；小 patch 只掉 1–4pp。
   → **不用算 TVD：盯部署模型的命中率，一掉就是 meta 變了，掉多少 ≈ 變多大。** 模型約 2–4 週重新學會新 meta。

## 6. Scripts（`repro/projektzero/`）

| 檔 | 做什麼 |
|---|---|
| [draft_predict_v2.py](projektzero/draft_predict_v2.py) | **主力**：逐手序列預測器 + 消融 M0–M3、分路、Bo5 場次 |
| [draft_predict_v3.py](projektzero/draft_predict_v3.py) | +region meta +counter-pick 消融 |
| [counter_sweep.py](projektzero/counter_sweep.py) | counter-pick β 掃描（證明弱）|
| [draft_player_diag.py](projektzero/draft_player_diag.py) | 選手癖 α 掃描 + 老手/新手 |
| [team_diag.py](projektzero/team_diag.py) | 隊伍身份增量（≈0）|
| [draft_predictability.py](projektzero/draft_predictability.py) | 地板 + 1-月走味 + patch-week 衝擊 |
| [meta_shift.py](projektzero/meta_shift.py) | 每 patch TVD、半衰期、非均勻、最大位移 |
| [meta_detect.py](projektzero/meta_detect.py) | 偵測：recall 掉幅 vs TVD（+0.49）|
| [draft_deviation_probe.py](projektzero/draft_deviation_probe.py) · [last_pick_probe.py](projektzero/last_pick_probe.py) | （前置）draft 偏離→勝負，三度確認平 |

重跑：資料在 `/tmp/oe_data/{2023,2024,2025,2026}_oe.csv`（gdown 重抓，見 [HANDOFF](HANDOFF-2026-06-15.md)）；各 script `cd /tmp/oe_data` 後 `python3 <path>`，無需 enriched CSV（直接讀原始 OE）。

## 7. 雷與下一步

- **唯一還沒探的前沿（v4，payoff 不確定）**：pairwise counter 既然弱，**高階「五人 comp 整體協同」**（transformer over draft tokens / DraftRec 全序列）是唯一可能再抬 recall@5 的方向，但 pairwise 已弱→上限可能有限。
- **ban-as-signal-for-own-pick**（你 ban 方向的 richer 版）：「為了選 Y 而 ban 掉 Y 的剋星」＝用己方 ban 預測己方 pick，未建模（counter 弱→預期也弱）。
- **雷**：少數 patch 的 date 中位數異常（部分場次 patch 標錯）；TVD 只用 major leagues；fearless 用 `year≥2025` 規則（已驗證 2023-24 系列重複率 8-37%、2025-26≈0）；counter/region 在重權重下會傷，須輕權重。
