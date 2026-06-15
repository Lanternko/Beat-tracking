# rift-highelo-lab

KR 高端 soloQ（Challenger + GrandMaster）對線分析。ARAM-Mayhem-Database 的姊妹作，改用 Riot 官方 API。
三軸：**分路 / 對線 / 玩家身份(ID)**。

> 這份 README 是交接入口。新對話要接手：讀這裡 + auto-memory `sr-highelo-project`，挑下面一個「研究方向」即可。

## 現況（2026-06-15）

- 資料：**2532 場** KR Challenger+GM ranked solo（queue 420）→ `data/lol.db`
  - patches：16.11=1153, 16.12=679, 16.10=478, 16.9=150, 其餘少量（analyze 時 filter）
  - 5837 distinct players、172 champions、25320 對線列、cache 2.1G
- 4 把 Riot dev key 在 `.env`（`RIOT_API_KEY`..`KEY4`，**每 24h 過期，過期要換新的**）

## 已完成的發現

1. **對線強度 ⊥ 勝率**（`analyze_laning.py`）：lane bully（MF/Varus/Irelia）對線碾壓但勝率平庸；scaler（Vladimir/DrMundo/Pantheon）對線墊底卻高勝率。→ 對線只是基準，β（打團/後期/邊帶）是另一條軸。
2. **對線預測勝負**（`predict_winrate.py` / `lane_winrate.py`）：gd@10 AUC=0.78、gd@14=0.83；分路重要度 **打野 > 中路 > 下路 > 上路 > 輔助**（partial 係數與單變量 AUC 一致）；金錢曲線 +2k@10→87% 勝、−2k→18%；打野領先 500→~60%。
3. **選角幾乎不決定勝負**（`draft_vs_execution.py`）：純 draft AUC≈**0.50**（高端夠平衡）；execution ≫ draft；英雄身份在 N=1500 加不進預測（427 特徵過擬合 + gold@10 已吸收英雄價值）。是否為資料不足，待大 crawl 重測。

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
```

環境：python3 + numpy/sklearn/scipy/pandas/matplotlib（已驗證可用）。

## 技術地雷（接手必讀）

- **Cloudflare 1010**：urllib 預設 UA 被擋 → `riot_client.py` 設瀏覽器 UA（curl 預設反而沒事）。
- **加密 ID 綁 key**：puuid 用「另一把」key 查會 `400 Exception decrypting`；matchId 不綁 key。所以收 id 單 key、抓 match 多 key 輪替。
- **口徑**：對位=同 teamPosition 敵方；diff = 自己−對手 的 totalGold/xp/(minionsKilled+jungleMinionsKilled) @ frame10/14。

---

## 研究方向（各自可開一個新對話）

| 方向 | 內容 | 狀態 |
|---|---|---|
| **A. Counter 矩陣 (δ)** | 英雄 vs 英雄的對線優劣，Bayesian shrink（prior=μ_A，= ARAM lift 同構）。回答「A 對 B 吃不吃虧」 | 直接接續；先把 crawl 續到 ~8k 補對位樣本 |
| **B. 玩家身份 (ID 軸)** | 第三軸，未動。用 PUUID 追個別玩家：招牌英雄池、smurf 偵測、穩定度/variance、player fixed-effect 控制球員強弱後再看英雄/對位 | 全新；資料已有 5837 玩家（seed 玩家場次多） |
| **C. 職業賽 (Oracle's Elixir)** | 另一條資料線（非 Riot API）。pro vs soloQ meta 分歧、職業限定/路人限定英雄、職業對線型態 | 全新資料源；下載 CSV 即用 |
| **D. 比賽動態建模** | live win-probability over time、翻盤分析（高端多少金錢差還救得回）、objective/tempo 特徵、scale-vs-lane 象限分類器 | 接續 finding 2 |
| **E. SR tier list / comp 產品** | 各路 Bayesian tier list + team-comp synergy/lift + 靜態站（ARAM 風格移植到峽谷） | 產品向 |

> 提醒：dev key 24h 過期；要再 crawl 先確認 `.env` 的 key 還活著（`curl -H "X-Riot-Token: <key>" -H "User-Agent: Mozilla/5.0" https://kr.api.riotgames.com/lol/status/v4/platform-data` 回 200 才有效）。
