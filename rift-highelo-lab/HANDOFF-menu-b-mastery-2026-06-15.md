# 交接：menu B step 1 — 熟練紅利＝mastery 還是 survivorship（玩家×英雄身份 +11pp）（2026-06-15）

新對話接這份就能續 menu B。配 auto-memory `sr_player_identity`（本軸主檔）、`sr_pick_prediction`（§4/§5）、`sr_highelo_project`、`sr_prior_art_survey`。
腳本 = [`player_identity.py`](player_identity.py)（根目錄，96480 picks `data/lol.db`，**0.5s 可重跑、確定性 SEED=42**）。
這是 **menu B（玩家身份軸，全專案第三軸）首個實質結果**，續 §5「familiarity ≫ meta」。
> 編號註：§6–§8 已被另一條線（matchup_gd10 / `repro/MATCHUP_GD10.md`）佔用，本軸不搶 §number，以「menu B」標示；README findings list 接 #6。

---

## 這次做了什麼（一句話）

把 §5 的「familiarity 紅利」做因果拆解：用巢狀 fixed-effect 控掉玩家強度與英雄強度後，
**同一個玩家打自己的招牌 vs 偶爾玩的，勝率差 +11pp、對線差 +207 金** → 高端最大的賽前槓桿是
**玩家×英雄的匹配（這個人有沒有 main 這隻）**，draft/meta（~0–2pp）望塵莫及。

## 關鍵數字（at a glance）

**familiarity → 結果，逐步加控制（每 +0.1 familiarity；限 ≥10 場玩家、2648 人）：**

| 控制 | Δ勝率 | Δgd10 |
|---|---|---|
| RAW (pooled) | +0.66pp | +1 金 |
| **+ within-player ★（控玩家）** | **+2.04pp** [1.76,2.32] | **+48 金** [42,53] |
| + within-champ（控英雄） | +0.70pp | +4 金 |
| + two-way（玩家+英雄都控） | **+2.19pp** | +37 金 |

**headline（同一玩家：招牌 famil≥0.3 減 偶爾玩 ≤0.1，n=996）：Δ勝率 +11.1pp [9.0,13.1] · Δgd10 +207 金 [170,245]**

**學習曲線（同(玩家,英雄) ≥4 場、窗內後半減前半，n_pair=4221）：Δ勝率 −3.6pp [−4.8,−2.5]**（見限制②）

**四句結論**：
① **玩家×英雄身份是 ±11pp 槓桿，壓過 §3–§5 一切**：同人打招牌 vs 偶爾玩差 11pp 勝率 / 207 金對線（控玩家後）。對照 純 draft ~+2pp、off-meta ~0、對線⊥勝率。**高端殘存賽前訊號就活在「這人有沒有 main 這隻」**。
② **非 survivorship 假象**：within-player 斜率**不縮反漲**（RAW +0.66 → within-player +2.04，suppression：高 familiarity 玩家平均略弱、pooled 蓋掉效果）；**控英雄身份（two-way +2.19）仍在** → 不是「招牌剛好強」。
③ **但拆不出 treatment vs selection**：+11pp 無法分「練出來的」vs「你只 main 你天生擅長的」。最純的學習曲線測試被 **regression-to-mean 汙染**（「打 ≥4 場某英雄」這篩選選到早期手感好、之後回歸均值，−3.6pp；也與 matchmaking 把贏的爬回 50% 一致），**不能當乾淨因果證據**，parked。
④ **收束全專案機制故事**：draft⊥勝率（選哪隻不決定勝負）**但** player×champ-familiarity → +11pp（這人有沒有 main 它，決定很大）。**不是英雄的事，是玩家與英雄匹配的事。**

## 任務定義 / 口徑（為什麼這樣做）

- **familiarity** = 該英雄佔該玩家（crawl 窗）場次比；限 ≥10 場玩家（讓 familiarity 有意義）。
- **巢狀 fixed-effect**：RAW(pooled OLS) → within-player（減玩家均值 + cluster(player)-bootstrap CI）→ within-champ（減英雄均值）→ two-way（玩家+英雄交替投影 demean）。**讀法＝看斜率縮不縮**：selection 主導 → 控 player 後塌；mastery 真 → within-player 仍 > 0。
- **headline 離散對照**：同一玩家「招牌(famil≥0.3) 減 偶爾玩(≤0.1)」的 Δ，cluster(player)-bootstrap。
- **學習曲線**：同 (player,champ) ≥4 場、依 `game_creation` 排序、後半減前半（同人同角、連英雄都固定 → 最純的 mastery 測試，但見限制②）。
- 兩 outcome：對線 `gd@10`、`win`。

## 資料限制（誠實，接手必讀）

- **treatment vs selection 拆不開**：cross-section FE 證明「招牌身份」值 +11pp，但無法分「練出來(treatment)」與「選你擅長的(selection)」。要拆得靠**窗外玩家歷史 / champion-mastery 累積**（見下一步④）。
- **學習曲線是 regression-to-mean confounded**：「窗內 ≥4 場某英雄」這個篩選本身選到早期手感好者 → 後半必回歸。所以 −3.6pp **不代表「越打越爛」**，是 selection artifact（與 matchmaking 動態 clawback 一致但不能證明）。**別把 §3 當正向 mastery 證據**。
- **familiarity 只在 crawl 窗（median 7 場）算** → 低估真實累積熟練；≥10 場門檻只是緩解。
- **單一 elo 帶**（Challenger+GM）：所有效應都在「同強度池內」量到，跨 elo 是否更大未知。
- within-champ 斜率（+0.70）≈ RAW（+0.66）＝ familiarity 效果**不是英雄強度的偽裝**（控英雄幾乎不改斜率）；真正壓低 pooled 斜率的是 player 維度（suppression）。

## 怎麼重跑

```bash
cd ~/side_projects/rift-highelo-lab
python3 player_identity.py     # 0.5s，不碰 API，確定性（SEED=42）
```

環境：python3 + numpy（`~/venvs/dac` 已裝；`~/venvs/dac/bin/python3 player_identity.py`）。

## 下一步（推薦序）

1. **player latent skill（menu B 主線下一步）**：對 PUUID 跑 TrueSkill/Elo（對標 ProjektZero player-Elo），取「控玩家強度後的殘差訊號」——smurf 偵測（新帳號高勝殘差）、per-player stability/variance、招牌池廣度 vs 勝率。⚠️ `sr_prior_art_survey` 的雷：玩家歷史特徵 AUC 0.97 是 leakage 教訓，務必 time-split。
2. **產品洞見**：每路「最被低估的 pocket 招牌」＝high familiarity × low pick-rate × ≥53% 勝率的 (玩家,英雄)；以及一句話「高端＝熟練度 ≫ 選角，練好你的招牌池勝過追 meta」。
3. **champ×player 廣度 vs 深度**：one-trick（窄深）vs 多面手（廣淺）在高端誰勝率高、誰 variance 低（接 §5 的「off-meta one-trick 免疫勝率稅」）。
4. **拆 treatment vs selection**：抓 champion-mastery API 或玩家賽季歷史，把 familiarity 升級成「真實累積場次」→ 真學習曲線（窗外、非 regression-to-mean）。

## 與既有發現的接點

- 收束 **draft⊥勝率 / 對線⊥勝率（§3–§6 + survey 的 matchmaking 天花板）**：賽前訊號不在「選哪隻」，在「這人 main 不 main 這隻」(player×champ identity)。
- 續 **§5**：§5 發現「熟練 off-meta one-trick = 熟練 on-meta」；本檔量出那個「熟練」值多少（+11pp）、並證明它在 player+champ 雙控後仍在。
- 學習曲線 §3 的負值 ＝ **matchmaking 天花板**可能的動態版（贏→升 MMR→更難→回歸），但被 selection 汙染，留作假說。
