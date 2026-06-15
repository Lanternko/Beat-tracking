"""§8：隊伍/跨路 context 能否提升 gd@10 預測（接 §6/§7 的最後一個槓桿）。

§6 純對位 R²=.059；§7 +玩家實力只到 .062（94% 不可測）。最後沒測的：
  pairwise 模型只看「我這路 vs 對位」。但 gd@10 可能受**全隊 draft**影響：
    - 打野壓力：我方打野對位領先 → 我這路被 gank 助攻 → 我 gd10↑？
    - 隊伍 draft 強弱：其餘四路的對位優勢會不會外溢到我這路？
  測：gd10 ~ own(我這路θ差) + jng(我方打野θ差) + rest(我方其餘四路θ差總和)
      vs 只有 own。看隊伍 context 加不加得動 oos R²/AUC。

θ＝§6 反對稱加性對線評分（train-only fit，逐路）。meta 係數 train fit、test eval（誠實）。

跑：python3 gd10_teamctx.py   （秒級，不碰 API，確定性 SEED=42）
"""
import time
from collections import defaultdict

import numpy as np

import config
import sqlite3
from matchup_gd10 import fast_auc, r2, fit_role

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
SEED = 42
LAMBDA = 25.0
TEST_FRAC = 0.30


def load():
    c = sqlite3.connect(config.DB_PATH)
    return c.execute(
        """
        SELECT l.team_position, l.champion, l.opp_champion, l.gd10,
               p.team_id, l.match_id, m.game_creation
        FROM laning l
        JOIN matches m ON m.match_id = l.match_id
        JOIN participants p ON p.match_id = l.match_id AND p.puuid = l.puuid
        WHERE m.queue = 420
          AND l.team_position IN ('TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY')
          AND l.champion IS NOT NULL AND l.opp_champion IS NOT NULL
          AND l.gd10 IS NOT NULL
        """
    ).fetchall()


def main():
    t0 = time.time()
    rows = load()
    n = len(rows)

    role = np.array([r[0] for r in rows])
    myc_s = [r[1] for r in rows]; oppc_s = [r[2] for r in rows]
    gd10 = np.array([r[3] for r in rows], float)
    team = np.array([r[4] for r in rows])
    mid = np.array([r[5] for r in rows])
    tcreate = np.array([r[6] if r[6] is not None else 0 for r in rows], float)

    uniq_c = sorted(set(myc_s) | set(oppc_s))
    cid = {c: i for i, c in enumerate(uniq_c)}; nch = len(uniq_c)
    myc = np.array([cid[c] for c in myc_s]); oppc = np.array([cid[c] for c in oppc_s])

    order = np.argsort(tcreate, kind="mergesort")
    rank = np.empty(n); rank[order] = np.arange(n) / n
    te = rank >= (1 - TEST_FRAC); tr = ~te

    # θ per role（train-only），算每列 own θ差（全列）
    champ_adv = np.zeros(n)
    for r in ROLES:
        mtr = (role == r) & tr
        th = fit_role(myc[mtr], oppc[mtr], gd10[mtr], nch, lam=LAMBDA)
        m = role == r
        champ_adv[m] = th[myc[m]] - th[oppc[m]]

    # 每 (match,team) 的 lane→own θ差
    grp = defaultdict(dict)
    for i in range(n):
        grp[(mid[i], team[i])][role[i]] = champ_adv[i]
    own = champ_adv
    jng = np.array([grp[(mid[i], team[i])].get("JUNGLE", 0.0) for i in range(n)])
    team_total = np.array([sum(grp[(mid[i], team[i])].values()) for i in range(n)])
    rest = team_total - own                      # 我方其餘四路（含打野；jng 另列出看單獨效果）
    rest_nojng = rest - np.where(role == "JUNGLE", 0.0, jng)   # 其餘且不含打野

    def fit_eval(feats, names):
        X = np.column_stack([np.ones(n)] + feats)
        coef, *_ = np.linalg.lstsq(X[tr], gd10[tr], rcond=None)
        pred = X @ coef
        R2 = r2(gd10[te], pred[te])
        AUC = fast_auc(pred[te], (gd10[te] > 0).astype(int))
        cstr = "  ".join(f"{nm}={coef[j+1]:+.2f}" for j, nm in enumerate(names))
        return R2, AUC, cstr

    print(f"§8 隊伍/跨路 context → gd@10 | 列={n} match={len(set(mid.tolist()))}")
    print(f"time-split train={int(tr.sum())}/test={int(te.sum())}；θ=§6 對位評分(train-only)")
    print("係數單位：gd10(金)/單位 θ；own≈1 表示對位評分尺度對；jng/rest 顯著>0 才是跨路外溢\n")

    print("== gd@10 預測：加隊伍 context vs 只有對位（時序 oos）==")
    print(f"  {'features':28} {'R²_oos':>7} {'AUC':>6}   coef")
    for feats, names, lab in [
        ([own], ["own"], "own（純對位＝§6）"),
        ([own, jng], ["own", "jng"], "own + 打野θ差"),
        ([own, rest_nojng], ["own", "rest"], "own + 其餘四路(不含打野)"),
        ([own, jng, rest_nojng], ["own", "jng", "rest"], "own + 打野 + 其餘"),
    ]:
        R2, AUC, cstr = fit_eval(feats, names)
        print(f"  {lab:28} {R2:>7.3f} {AUC:>6.3f}   {cstr}")

    # 非打野列單獨看（打野壓力對 lane 的外溢，排除 jng==own 的打野列污染）
    nonj = (role != "JUNGLE")
    Xo = np.column_stack([np.ones(n), own])
    Xj = np.column_stack([np.ones(n), own, jng])
    co, *_ = np.linalg.lstsq(Xo[tr & nonj], gd10[tr & nonj], rcond=None)
    cj, *_ = np.linalg.lstsq(Xj[tr & nonj], gd10[tr & nonj], rcond=None)
    R2o = r2(gd10[te & nonj], (Xo @ co)[te & nonj])
    R2j = r2(gd10[te & nonj], (Xj @ cj)[te & nonj])
    print(f"\n  只看非打野列（打野壓力外溢測試）：own R²={R2o:.3f} → +jng R²={R2j:.3f}"
          f"（Δ={R2j-R2o:+.4f}；jng 係數={cj[2]:+.2f} 金/θ）")
    print("  讀法：jng/rest 係數≈0 且 R² 不動 → 隊伍 context 對個別路 gd@10 無外溢，對線是孤島。")

    print(f"\n  done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
