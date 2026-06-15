# rift-highelo-lab

KR 高端 soloQ（Challenger + GrandMaster）對線分析。ARAM-Mayhem-Database 的姊妹作，改用 Riot 官方 API。
三軸：**分路 / 對線 / 玩家身份(ID)**。

> 這份 README 是交接入口。新對話要接手：讀這裡 + auto-memory `sr-highelo-project`、`sr-repro-results`，挑下面一個「研究方向」即可。
>
> **2026-06-15 複現/前向預測成果**：交接見 [`repro/HANDOFF-2026-06-15.md`](repro/HANDOFF-2026-06-15.md)，完整結果見 [`repro/RESULTS.md`](repro/RESULTS.md)。下一步推薦 **draft→gd@10**（menu A 強化版）。
>
> **2026-06-15 §4 選角預測 / meta 收斂成果**：交接見 [`HANDOFF-pick-prediction-2026-06-15.md`](HANDOFF-pick-prediction-2026-06-15.md)（腳本 `predict_picks.py`）。選角 top-1 15% / top-5 41%、有效英雄池 BOT25<…<TOP51、off-meta 無勝率稅。**§5 已完成**（見下）。
>
> **2026-06-15 §5 off-meta 因果 / 玩家身份首動**：交接見 [`HANDOFF-offmeta-causal-2026-06-15.md`](HANDOFF-offmeta-causal-2026-06-15.md)（腳本 `offmeta_causal.py`）。**真軸是 familiarity 不是 meta**：熟練 off-meta one-trick 勝率 53.1% ＝ 熟練 on-meta；off-meta 只傷對線（英雄屬性）、控玩家後不傷勝率。＝ finding #1 玩家層級版、menu B 首次動工。
>
> **2026-06-15 職業 draft 選角預測 + meta 理解**：詳見 [`repro/DRAFT_META.md`](repro/DRAFT_META.md)（交接併入 [`repro/HANDOFF-2026-06-15.md`](repro/HANDOFF-2026-06-15.md)）。職業逐手序列預測 top-1 **28.5%** / recall@5 **74%**（pro ≈2× soloQ，差在 ban/fearless/順序）；**可用性約束主宰、身份訊號皆弱**；meta 半衰期 8 patch、位移 pre-Worlds 小季前大、偵測＝模型 recall 掉幅(+0.49)。draft 偏離→勝負三度確認平。

## 現況（2026-06-15）

- 資料：**9648 場** KR Challenger+GM ranked solo（queue 420）→ `data/lol.db`
  - patches：16.10=2921, 16.11=2916, 16.9=1571, 16.12=1157, 16.8=733, 其餘少量（analyze 時 filter）
  - 13403 distinct players、172 champions、96480 對線列、cache 7.9G
- 4 把 Riot dev key 在 `.env`（`RIOT_API_KEY`..`KEY4`，**每 24h 過期，過期要換新的**）

## 已完成的發現

1. **對線強度 ⊥ 勝率**（`analyze_laning.py`）：lane bully（MF/Varus/Irelia）對線碾壓但勝率平庸；scaler（Vladimir/DrMundo/Pantheon）對線墊底卻高勝率。→ 對線只是基準，β（打團/後期/邊帶）是另一條軸。
2. **對線預測勝負**（`predict_winrate.py` / `lane_winrate.py`）：gd@10 AUC=0.78、gd@14=0.83；分路重要度 **打野 > 中路 > 下路 > 上路 > 輔助**（partial 係數與單變量 AUC 一致）；金錢曲線 +2k@10→87% 勝、−2k→18%；打野領先 500→~60%。
3. **選角幾乎不決定勝負**（`draft_vs_execution.py`）：純 draft AUC≈**0.50**（高端夠平衡）；execution ≫ draft；英雄身份在 N=1500 加不進預測（427 特徵過擬合 + gold@10 已吸收英雄價值）。是否為資料不足，待大 crawl 重測。
4. **選角預測 / meta 收斂**（`predict_picks.py`，9648 場，masked-completion）：預測 picks → top-1 **15%** / top-5 41%（context 只 +3pp ＝高端照 tier list 選、不繞 lobby）；有效英雄池 **BOT25 < JNG/SUP31 < MID48 < TOP51**（meta 是分路現象、跨 patch 穩定）；模型機率 vs 勝率全平 ~50% ＝ **off-meta 無勝率稅**（未控玩家；§5 已控玩家確認 → 見 finding #5）。交接見 [`HANDOFF-pick-prediction-2026-06-15.md`](HANDOFF-pick-prediction-2026-06-15.md)。
5. **off-meta 傷線不傷勝率、真軸是 familiarity**（`offmeta_causal.py`，96480 picks，控玩家 FE）：off-meta 系統性輸線（gd10 vs on-meta 差 ~50–100 金、每 familiarity 層都在＝英雄屬性）但幾乎不傷勝率；**熟練 off-meta one-trick 53.1% ＝ 熟練 on-meta 53.1%**，off-meta 唯一勝率稅是「不熟 × off-meta」交互（−2.9pp，扛得住 autofill）。within-player FE 把 RAW 傷線砍半（−61→−27 金）、player 勝率 covariate 係數僅 **0.04**（matchmaking 天花板再現）。＝ finding #1 在玩家層級重現 + **menu B 首動**。交接見 [`HANDOFF-offmeta-causal-2026-06-15.md`](HANDOFF-offmeta-causal-2026-06-15.md)。

## 檔案

| 檔 | 用途 |
|---|---|
| `config.py` | KR routing、queue、rate limit、多 key 載入 |
| `riot_client.py` | 多 key 輪替 + 限流 + 429 退避 + 磁碟快取（可續傳） |
| `build_dataset.py` | crawl：league→puuid→matchId→detail+timeline→算對線→SQLite |
| `db.py` | schema：matches / participants / laning |
| `analyze_laning.py` | 五路對線強度榜 |
| `predict_winrate.py` | 五路 gd → logistic 預測勝負、分路重要度 |
| `lane_winrate.py` | 單一分路 gd@10 → 勝率曲線 |
| `draft_vs_execution.py` | draft vs 早期戰況 的預測力分解 |
| `build_counter_matrix.py` | counter δ 矩陣（A vs B 對線優劣，Bayesian shrink；menu A）|
| `predict_picks.py` | 選角預測（masked-completion）+ meta 收斂度（有效英雄池）+ off-meta 預告（§4）|
| `offmeta_causal.py` | off-meta 對勝負的因果（控玩家 FE + familiarity + autofill 穩健；§5、menu B 首動）|
| `probe_sample.py` | 抓單場驗結構（gold-first 探針） |

## 跑 / 續傳

```bash
# 抓更多資料（4 key ~4×、可續傳、走快取；killed 也能直接重跑接上）
python3 build_dataset.py 8000

# 分析（不碰 API、秒級、可重複跑）
python3 analyze_laning.py 16.12 15      # [patch] [min_games]
python3 predict_winrate.py
python3 lane_winrate.py JUNGLE          # TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY
python3 draft_vs_execution.py
python3 predict_picks.py                # 選角預測率 + meta 收斂度（§4，16s）
python3 offmeta_causal.py               # off-meta 對勝負的因果（控玩家 FE，§5，1s）
```

環境：python3 + numpy/sklearn/scipy/pandas/matplotlib（已驗證可用）。

## 技術地雷（接手必讀）

- **Cloudflare 1010**：urllib 預設 UA 被擋 → `riot_client.py` 設瀏覽器 UA（curl 預設反而沒事）。
- **加密 ID 綁 key**：puuid 用「另一把」key 查會 `400 Exception decrypting`；matchId 不綁 key。所以收 id 單 key、抓 match 多 key 輪替。
- **口徑**：對位=同 teamPosition 敵方；diff = 自己−對手 的 totalGold/xp/(minionsKilled+jungleMinionsKilled) @ frame10/14。

---

## 先行研究與定位（survey 2026-06-15）

完整 survey 見 auto-memory `sr-prior-art-survey`。峽谷勝負預測是擁擠紅海，但所有嚴肅研究/工具的數字都在背書本專案的核心發現（draft⊥勝率、execution≫draft）。

**兩個地基洞見（比文獻清單更重要）：**

1. **matchmaking 天花板＝為什麼純 draft 撞 ~55%。** 遊戲按 MMR 把每場做成 ~50/50，**賽前訊號被刻意抹平**，~55% 是資訊論上限而非模型爛。要分清：
   - **賽前 predictor**（draft、MMR）→ 被 matchmaking 抹平 → 上限低（你的純 draft AUC≈0.50）。
   - **賽中 predictor**（gd@10、物件）→ execution 下游、沒被抹平 → 預測力高（你的 gd@10 AUC=0.78）。它不是「預測」，是在量比賽已跑出的結果。
   - 殘存賽前訊號活在**玩家身份殘差**（smurf/autofill/熟練度），不在 draft → 指向 menu B。
   - 推論：別追 draft（高端已被抹平），追 ① 賽中動態（D）② 玩家身份（B）。ARAM comp 預測 58% 之所以 > Rift draft 55%，是因 ARAM 英雄隨機發（不對稱真實）、Rift 高端鏡像最佳化（不對稱被抹平）——結構性差異。

2. **指標哲學：acc 不是重點，校準＋解析度才是。** acc 對 resolution 不敏感（永遠喊 50% 也有不錯 acc）。產品/模型指標 stack：
   - **log-loss**（主選型）＋ **Brier 分解**（Reliability−Resolution+Uncertainty；Resolution 直接回答「60% 隊比 50% 強多少」）＋ **reliability diagram / ECE**（標竿：LoLDraftAI ECE=0.0088）。
   - 每個勝率附 **Beta 可信區間**。90% CI 半寬 ≈ 0.82/√N（每格樣本）：每英雄需 ~1000 場（總 crawl ~17k）才到 ±2.6pp；否則 tier 必重疊 → 畫成**區間帶**別假裝嚴格排名。60% 勝率 ≈ +70 Elo。

**對標工具（分屬兩條軸）：**

| 工具 | 軸 | 方法 | 開源 | 準度 |
|---|---|---|---|---|
| **DraftGap** | draft (A/E) | 加總式對位 δ（Elo logit + empirical-Bayes prior）。**非 5v5** | ✅ MIT [vigovlugt/draftgap](https://github.com/vigovlugt/draftgap) | 54.66% / ECE 0.0199 |
| **LoLDraftAI** | draft (A/E) | **真 5v5 NN**（整副 draft、傷害組成/scaling/elo） | ✅ 部分 [Looyyd/loldraftai-monorepo-public](https://github.com/Looyyd/loldraftai-monorepo-public) | 55.88% / **ECE 0.0088**（校準標竿） |
| **ProjektZero** | player (B/C) | Elo+TrueSkill+EGPM ensemble（職業賽） | ✅ AGPL-3.0（停更 2021） | 63.66% / Brier 0.2255 |

> 紅海勿做：champ/matchup/synergy 勝率表（Lolalytics 有 Challenger filter）、加總式 draft 評估器、轉播 live WP（Riot×AWS XGBoost）。

**實際複現（2026-06-15，見 [`repro/RESULTS.md`](repro/RESULTS.md)）**：ProjektZero 忠實複現 ensemble **64.3%**（原 63.7%）、誠實 time-split 63.8%、bootstrap 90% CI **±0.9pp**；draft→win 在 KR 高端（9648 場）**52.5%**、公開 HF Challenger（10k）51.9%（均勉強贏擲硬幣，logloss 幾乎贏不過 base rate）。**player 強度 64% vs draft 52%** 坐實 matchmaking 天花板。

## 研究方向（各自可開一個新對話）

| 方向 | 內容 | 狀態 | 對標 / 可 cite |
|---|---|---|---|
| **A. Counter 矩陣 (δ)** | 英雄 vs 英雄的對線優劣，Bayesian shrink（prior=μ_A，= ARAM lift 同構）。回答「A 對 B 吃不吃虧」 | 直接接續；先把 crawl 續到 ~8k 補對位樣本 | DraftGap（加總式上限）；高端 draft⊥勝率＝賣點 |
| **B. 玩家身份 (ID 軸)** | 第三軸。用 PUUID 追個別玩家：招牌英雄池、smurf 偵測、穩定度/variance、player fixed-effect 控制球員強弱後再看英雄/對位 | **§5 首動**（`offmeta_causal.py`）：familiarity ≫ meta、off-meta 傷線不傷勝率；下一步 champ-FE 分離英雄屬性 + player latent skill(TrueSkill) | ProjektZero（Player-Elo/TrueSkill 主訊號）；⚠️ IEEE CoG 2021 玩家歷史 AUC0.97 是 leakage 教訓 |
| **C. 職業賽 (Oracle's Elixir)** | 另一條資料線（非 Riot API）。pro vs soloQ meta 分歧、職業限定/路人限定英雄、職業對線型態 | 全新資料源；下載 CSV 即用 | Oracle's Elixir（免費職業 CSV，@10/@15 diff 已算好）；ProjektZero |
| **D. 比賽動態建模** | live win-probability over time、翻盤分析（高端多少金錢差還救得回）、objective/tempo 特徵、scale-vs-lane 象限分類器 | 接續 finding 2 | Silva&Pappa 2018(RNN 時序)；Junior 2023(gold #1 特徵)；Riot×AWS(XGBoost 特徵集) |
| **E. SR tier list / comp 產品** | 各路 Bayesian tier list + team-comp synergy/lift + 靜態站（ARAM 風格移植到峽谷） | 產品向 | 高端勝率帶窄→**必附 CI 畫區間帶**；指標用 log-loss/Brier 分解非 acc |

> 提醒：dev key 24h 過期；要再 crawl 先確認 `.env` 的 key 還活著（`curl -H "X-Riot-Token: <key>" -H "User-Agent: Mozilla/5.0" https://kr.api.riotgames.com/lol/status/v4/platform-data` 回 200 才有效）。
