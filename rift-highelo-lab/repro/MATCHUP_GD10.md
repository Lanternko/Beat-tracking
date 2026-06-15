# §6 draft → gd@10：選角塑形對線、但塑形的那塊換不到勝場（2026-06-15）

Kojek 原創 task #8（menu A 強化版）。腳本 [`../matchup_gd10.py`](../matchup_gd10.py)，確定性 SEED=42、秒級、不碰 API。
資料：`data/lol.db` KR Challenger+GM soloQ，queue 420，**9648 場 / 96480 對線列 / 172 英雄**，patch 16.5–16.12（16.9–16.12 為主）。全 **walk-forward**（時序最後 30% 當 test，train 67536 / test 28944）。

> 緣起：全專案三度確認 `draft⊥勝率`、`對線⊥勝率`（matchmaking 天花板，見 [`RESULTS.md`](RESULTS.md) §2/§2d、[`DRAFT_META.md`](DRAFT_META.md)）。缺的拼圖：**draft 到底決不決定「對線結果(gd@10)」本身？** gd@10 是賽中數字、沒被 matchmaking 壓平→留得住真訊號；又賽前(看 draft)就能算→pre-game 可行動。

---

## 一句話

**選角確實決定 10 分鐘的 CS/金幣領先（賽前就能預測），但這份由選角決定的領先對勝負毫無作用（AUC 0.495≈擲硬幣）。對線→勝的訊號全在「臨場」，不在選角。** 這就是 `對線⊥勝率`＋`draft⊥勝率` 缺的那塊機制。

因果鏈：

```
draft ──→ gd@10 ──→ win
  │  AUC 0.62   AUC 0.64(actual)
  │  R²  0.06
  └─────────────→ win   AUC 0.495   ← 純 draft 經對線通道 ≈ 0
                  （= 對線→勝那 0.64 裡，可被 draft 預測的部分 ≈ 0）
```

## 1. draft → gd@10：有真訊號，但量級小（時序 out-of-sample）

模型（每路一組、反對稱加性 ridge λ=25）：`gd10 = θ[role][myChamp] − θ[role][oppChamp] + ε`。預測值 = 我方英雄對線評分 − 對方英雄對線評分；資料含雙向 → 自然 sum-to-zero。

| 路 | R²_oos | AUC(誰領先) | MAE(金) | R²_in |
|---|---|---|---|---|
| UTILITY | **0.107** | **0.639** | 431 | 0.114 |
| JUNGLE | 0.068 | 0.625 | 678 | 0.063 |
| MIDDLE | 0.059 | 0.620 | 625 | 0.084 |
| TOP | 0.047 | 0.605 | 700 | 0.109 |
| BOTTOM | 0.043 | 0.591 | 790 | 0.062 |
| **OVERALL** | **0.058** `[.050,.066]` | **0.616** `[.606,.624]` | — | — |

（CI = cluster(match) bootstrap 1000 次。）純對位解釋 ~6% 的 gd@10 變異、AUC 0.62 賽前猜中誰對線領先——**CI 遠離 0.5/0，訊號真實但小**。gd@10 母體 sd 867 金。

- **逐路 AUC**：UTILITY > JUNGLE > MIDDLE > TOP > BOTTOM。
- **補位居首、下路墊底**：補位對位最決定論（engage vs enchanter）、金幣低變異 → 選角解釋占比最大；下路 gd 受擊殺/塔皮/被 gank 汙染最多 → 選角解釋占比最小。

## 2. 換 target：CS 差最吃選角

同模型換 outcome（時序 oos）：

| target | R²_oos | AUC | 解讀 |
|---|---|---|---|
| **cd10（CS 差）** | **0.152** | **0.674** | 最「純對線機制」（清線/射程），選角最決定 |
| gd10（金幣差） | 0.058 | 0.616 | 金幣含擊殺/塔皮/gank 雜訊 |
| gd14 | 0.049 | 0.610 | ≈gd10 → **領先到 14 分大致守得住** |
| xd10（經驗差） | 0.030 | 0.583 | 共享經驗、最不由單英雄決定 |

→ task 指名的 csd@10 是**選角訊號最強的對線 target**（R² 是 gd10 的 ~2.6 倍）。

## 3. counter（專剋）幾乎是雜訊

加性模型抓「英雄各自單體對線強度」。counter＝「A 專剋 B」＝對位殘差。測殘差 split-half 信度（test 對折，每對 ≥8×2 場的平均殘差兩半相關）：

- **split-half r = +0.072**（n_pair = 1144）。加性模型已吃掉 8% 變異，殘差 sd 832 vs gd10 sd 867。
- → **「專剋」不可複現＝雜訊**。對線 ≈ 各英雄單體對線強度相加，specific counter 幾無結構。**呼應專案「counter 對 win 弱」**（[`DRAFT_META.md`](DRAFT_META.md) §2、[counter_sweep.py](projektzero/counter_sweep.py)）——counter 對 gd@10 也弱。

## 4. 閉合迴路：對線優勢 → 勝負（punchline）

per-lane → 該玩家勝負，時序 test：

| 訊號 | AUC | 含義 |
|---|---|---|
| **actual** gd@10 → win | **0.639** | 真實對線結果（含臨場）能預測贏這局 |
| **predicted** gd@10 → win | **0.495** | 純 draft 對位（賽前可算）→ **≈擲硬幣** |

逐路 predicted→win：TOP .49 / JUN .51 / MID .50 / BOT .49 / UTI .49（全 ≈0.50）。

→ 對線→勝確有訊號（0.639），但**其中「可被 draft 預測的部分」對勝負 AUC 0.495 ≈ 0**。差距（0.639 − 0.495）＝對線→勝的訊號**全在臨場 execution、非選角**。與團隊層級 `純 draft → win ≈ 0.50`（[draft_vs_execution.py](../draft_vs_execution.py)、[`RESULTS.md`](RESULTS.md) §2）一致，這裡給出**機制版**：draft 的確進到對線（gd@10），只是在對線→勝這一步斷掉。

## 5. 誠實註記

舊跨對話記憶記過「gd@10 預測 AUC 0.78、打野>中>下>上>輔」——**與本條對不上**。本條是**純 draft、walk-forward、out-of-sample** 的乾淨隔離 → AUC **0.62、UTILITY 居首 BOTTOM 墊底**。那 0.78 應是不同設定（含更多特徵／in-sample／或 win-target），非同一問題。本條才是「**純選角**對對線結果的預測力」的誠實天花板。

## 6. 方法/重跑

```bash
cd ~/side_projects/rift-highelo-lab
python3 matchup_gd10.py          # 秒級、確定性、讀 data/lol.db
```

- 反對稱加性 ridge（純 numpy normal equations，不 materialize X）；time-split walk-forward；rank-based AUC；cluster(match) bootstrap CI。
- 母體：queue 420、champion/opp_champion/gd10 非空。

## 7. 準確率天花板：玩家實力 / 隊伍 context 都加不動（§7 [`../gd10_ceiling.py`](../gd10_ceiling.py) + §8 [`../gd10_teamctx.py`](../gd10_teamctx.py)）

「提升準確率」：從 §1 純對位（R²=0.058）往上，把所有 pre-game 槓桿逐一加進去測。

**§7 準確率階梯（時序 oos，雙因子加性 backfit `gd10 = θ[champ] + φ[player]`，θ/φ 皆 train-only）**：

| 模型 | R²_oos | AUC | MAE |
|---|---|---|---|
| M0 純對位（§1） | 0.059 | 0.616 | 645 |
| M1 +玩家對線實力 | **0.062** | **0.622** | 643 |
| 只有玩家 | 0.005 | 0.552 | 663 |

→ **最大候選槓桿（玩家實力）幾乎沒用：只 +0.3% R² / +0.6pp AUC**。test 玩家僅 51% 有 ≥10 場 train 歷史（冷啟動）；且 Chall+GM 單一窄帶、技術變異被壓縮。

**變異分解（oos，可預測天花板）**：選誰(champ) **5.9%** ／ 誰在玩(+player) **+0.3%** ／ **純不可測(臨場/variance) 93.8%**。**高端 10 分鐘對線結果 ≈94% 不可由任何賽前資訊預測。**

**§8 隊伍/跨路 context（最後一個槓桿）**：`gd10 ~ own + 打野θ差 + 其餘四路θ差`——打野係數 **+0.00**、其餘四路 **+0.03**、R² 全停在 0.056。非打野列的打野壓力外溢 Δ R²=+0.0001。**隊伍 draft 對個別路 gd@10 零外溢，對線是孤島。**（own 係數 +1.10 確認 θ 尺度對。）

**win-flow 精修（接 §4，CI 排除 .50）**：

| 對線預測力的成分 | → win AUC |
|---|---|
| champ-component（純選角） | 0.498（死路）|
| **player-component（玩家對線實力）** | **0.516** `[.506,.525]` |
| full predicted | 0.507 |
| actual gd@10（含臨場） | 0.639 |

→ **player 對線實力是唯一微弱流向勝負的對線成分**（soloQ matchmaking 把它壓到 .516、殘留一絲：smurf/排位膨脹）；champ 那塊完全死路。精修 §4：不是全部對線預測力都⊥勝負，玩家那一丁點會穿過去。

**§9 GBM 封棺**（[`../gd10_gbm.py`](../gd10_gbm.py)）：給 non-linear 最強配置——native-categorical 雙英雄交互（能 split on myChamp×oppChamp＝pairwise）+ 玩家實力 + role + patch，掃 4 組正則化（靈活→極保守）。**oos R² 全負（−.027 ~ −.042），in-sample 全 0.30+**＝GBM 把 champ-pair/玩家細節記成噪音、完全不泛化（呼應 §3 counter 殘差 r=.072≈雜訊）。**non-linear 連加性的 6% 都打不過**；加性的「反對稱 + 強 shrinkage」才是對的歸納偏置，那之外無真訊號值得更高彈性。

**結論**：**高端 gd@10 預測天花板 ≈6% R² / 0.62 AUC，純對位已抵達**；玩家實力、打野壓力、隊伍 draft、non-linear 交互 **全加不動**。能預測的 6% 幾乎全是英雄對位，其餘 94% 是賽前碰不到的臨場。這從對線角度再證 matchmaking 天花板：頂端連「對線結果本身」都主要由不可測執行決定。

## 8. 雷與下一步

- **雷**：時間跨度窄（單賽季 patch 16.5–16.12）、單 elo 帶（Chall+GM，技術變異被壓縮→player 槓桿天生小）；ridge λ=25 未調（對 R² 量級不敏感）；player φ 用 2-pass backfit shrunk-mean（近似 joint ridge）、冷啟動 φ=0；per-lane→win 是「你這路優勢 vs 你隊伍贏」。
- **已收束（四槓桿 + 封棺）**：champ ＝ 唯一非零可預測來源；player / jungle / team-context / non-linear 皆 ≈0 或負。§9 GBM 已封棺（4 組正則化 oos R² 全負）——線性非線性、所有 pre-game 特徵都到頂。
- **可延伸（payoff 不確定）**：①把 §1 的 θ 當「對線強度 tier list」輸出（menu E，幾乎免費）；②跨 elo 帶比較（低分段技術變異大→player 槓桿可能才顯著，需更多資料）。
