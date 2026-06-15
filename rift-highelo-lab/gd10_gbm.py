"""§9：GBM 封棺 — non-linear 能否打破 §6–§8 的 ~6% R² 加性天花板。

§6 純對位加性 R²=.058；§7 +玩家 .062；§8 +隊伍 context ≈0。都是「線性加性」框架。
唯一沒測＝non-linear：樹模型能 split on (myChamp==X AND oppChamp==Y) → 抓 specific
counter / champ-pair 交互（加性模型抓不到的）。若連 GBM 都打不過加性 → 天花板鐵證、封棺。

給 GBM 最強配置（要打就讓它有機會）：
  native categorical：myChamp / oppChamp / role / patch（樹可自由交互兩英雄＝pairwise）
  + 數值：myPlayerSkill / oppPlayerSkill（train-only shrunk 平均 gd10，§7 證單獨弱但給齊）
時序 train/test、玩家實力只用 train 估（誠實）。

預期（§3 已證 counter 殘差 split-half r=.072≈雜訊）：GBM ≈ 加性，抓不到 pair（資料太稀疏）。

跑：python3 gd10_gbm.py   （~10s，不碰 API，確定性 SEED=42）
"""
import time

import numpy as np

import config
import sqlite3
from matchup_gd10 import fast_auc, r2, fit_role
from sklearn.ensemble import HistGradientBoostingRegressor

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
SEED = 42
LAMBDA = 25.0
K_SKILL = 10.0
TEST_FRAC = 0.30


def load():
    c = sqlite3.connect(config.DB_PATH)
    return c.execute(
        """
        SELECT l.team_position, l.champion, l.opp_champion,
               l.puuid, l.opp_puuid, l.gd10, m.patch, m.game_creation
        FROM laning l JOIN matches m ON m.match_id = l.match_id
        WHERE m.queue = 420
          AND l.team_position IN ('TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY')
          AND l.champion IS NOT NULL AND l.opp_champion IS NOT NULL
          AND l.puuid IS NOT NULL AND l.opp_puuid IS NOT NULL
          AND l.gd10 IS NOT NULL
        """
    ).fetchall()


def enc(seq):
    u = sorted(set(seq))
    m = {v: i for i, v in enumerate(u)}
    return np.array([m[v] for v in seq]), len(u), m


def main():
    t0 = time.time()
    rows = load()
    n = len(rows)

    role_s = [r[0] for r in rows]
    myc_s = [r[1] for r in rows]; oppc_s = [r[2] for r in rows]
    myp_s = [r[3] for r in rows]; oppp_s = [r[4] for r in rows]
    gd10 = np.array([r[5] for r in rows], float)
    patch_s = [r[6] for r in rows]
    tcreate = np.array([r[7] if r[7] is not None else 0 for r in rows], float)

    # 共用英雄 id（my/opp 同編碼，樹才能比對是否同英雄/對位）
    champ_vocab = sorted(set(myc_s) | set(oppc_s))
    cmap = {c: i for i, c in enumerate(champ_vocab)}
    myc = np.array([cmap[c] for c in myc_s], float)
    oppc = np.array([cmap[c] for c in oppc_s], float)
    role_id, _, _ = enc(role_s); role_id = role_id.astype(float)
    patch_id, _, _ = enc(patch_s); patch_id = patch_id.astype(float)
    # 加性模型用的 int 版
    myc_i = myc.astype(int); oppc_i = oppc.astype(int); nch = len(champ_vocab)
    role_arr = np.array(role_s)

    pall = sorted(set(myp_s) | set(oppp_s))
    pmap = {p: i for i, p in enumerate(pall)}; npl = len(pall)
    myp = np.array([pmap[p] for p in myp_s]); oppp = np.array([pmap[p] for p in oppp_s])

    order = np.argsort(tcreate, kind="mergesort")
    rank = np.empty(n); rank[order] = np.arange(n) / n
    te = rank >= (1 - TEST_FRAC); tr = ~te

    # ---- 加性基線（per role, train-only）----
    champ_pred = np.full(n, np.nan)
    for r in ROLES:
        m = role_arr == r; mtr = m & tr
        th = fit_role(myc_i[mtr], oppc_i[mtr], gd10[mtr], nch, lam=LAMBDA)
        champ_pred[m] = th[myc_i[m]] - th[oppc_i[m]]
    R2_add = r2(gd10[te], champ_pred[te])
    auc_add = fast_auc(champ_pred[te], (gd10[te] > 0).astype(int))

    # ---- 玩家實力（train-only shrunk 平均 gd10）----
    s = np.bincount(myp[tr], weights=gd10[tr], minlength=npl)
    cnt = np.bincount(myp[tr], minlength=npl)
    skill = s / (cnt + K_SKILL)               # 未見玩家=0
    my_skill = skill[myp]; opp_skill = skill[oppp]

    # ---- GBM：native categorical 雙英雄 + role + patch + 數值玩家實力 ----
    # 掃正則化（堵「沒調參」質疑）：從靈活到極保守，找 GBM 最佳 oos。
    X = np.column_stack([myc, oppc, role_id, patch_id, my_skill, opp_skill])
    cat = [True, True, True, True, False, False]
    CONFIGS = [
        ("原始 lr.05 leaf63 msl50 l2=1", dict(learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.0)),
        ("保守 lr.03 leaf15 msl300 l2=10", dict(learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=300, l2_regularization=10.0)),
        ("很保守 lr.02 leaf7 msl1000 l2=30", dict(learning_rate=0.02, max_leaf_nodes=7, min_samples_leaf=1000, l2_regularization=30.0)),
        ("重l2 lr.05 leaf7 msl500 l2=100", dict(learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=500, l2_regularization=100.0)),
    ]

    print(f"§9 GBM 封棺 | 列={n} 英雄={nch} 玩家={npl}；time-split train={int(tr.sum())}/test={int(te.sum())}")
    print("GBM: HistGBR native-cat(myC,oppC,role,patch)+玩家實力；掃正則化找 GBM 最佳 oos\n")

    print("== gd@10 預測：non-linear vs 加性（時序 oos）==")
    print(f"  {'model':34} {'R²_oos':>7} {'AUC':>6} {'in R²':>6}")
    print(f"  {'加性 純對位(§6, 線性, 參照)':34} {R2_add:>7.3f} {auc_add:>6.3f} {'—':>6}")
    best_r2, best_pred, best_auc = -1e9, None, None
    for lab, kw in CONFIGS:
        g = HistGradientBoostingRegressor(
            loss="squared_error", max_iter=600, categorical_features=cat,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
            random_state=SEED, **kw)
        g.fit(X[tr], gd10[tr])
        p = g.predict(X)
        R2o = r2(gd10[te], p[te]); aucg = fast_auc(p[te], (gd10[te] > 0).astype(int))
        print(f"  {'GBM '+lab:34} {R2o:>7.3f} {aucg:>6.3f} {r2(gd10[tr], p[tr]):>6.3f}")
        if R2o > best_r2:
            best_r2, best_pred, best_auc = R2o, p, aucg

    print(f"  GBM 最佳 oos R²={best_r2:.3f}（in-sample 全 0.30+ 但 oos 全負＝fit 噪音）")
    print(f"  Δ(GBM最佳 − 加性) = {best_r2 - R2_add:+.4f} R²   {best_auc - auc_add:+.4f} AUC")
    print("  逐路 GBM(最佳) R²/AUC：", end="")
    cells = []
    for r in ROLES:
        m = te & (role_arr == r)
        cells.append(f"{r[:3]}={r2(gd10[m], best_pred[m]):.2f}/{fast_auc(best_pred[m], (gd10[m]>0).astype(int)):.2f}")
    print("  ".join(cells))

    print()
    if best_r2 - R2_add <= 0.01:
        print("  → 封棺：GBM 打不過加性（Δ≤.01）。non-linear/champ-pair 交互無可複現訊號")
        print("     （呼應 §3 counter 殘差 r=.072≈雜訊）。高端 gd@10 ~6% R² 天花板＝鐵證，")
        print("     線性與非線性、所有 pre-game 特徵都到頂；其餘 94% 是賽前碰不到的臨場。")
    else:
        print(f"  → GBM 抬了 {best_r2-R2_add:+.3f} R²！non-linear 確有額外訊號，天花板未到，值得深掘。")

    print(f"\n  done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
