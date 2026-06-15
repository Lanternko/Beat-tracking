"""§6：draft → gd@10 回歸（Kojek 原創 task #8）。menu A 強化版。

接全專案主軸「draft⊥勝率、對線⊥勝率（matchmaking 天花板）」。缺的拼圖：
  draft 到底決不決定「對線結果(gd@10)」本身？
  → 若會（選角決定 10 分鐘金幣差）但 gd@10 又⊥勝率，就有完整機制故事：
    英雄對位決定早期領先，但高端這領先換不到勝場。

為何 gd@10 是對的 target：賽中數字、沒被 matchmaking 壓平 → 留得住真訊號；
又賽前(看 draft)就能算 → pre-game 可行動（menu A：「這對位 +X 金 @10，CI[...]」）。

模型（每路一組，反對稱加性）：
  gd10 = θ[role][myChamp] − θ[role][oppChamp] + ε      （ridge 正則）
  預測值 = 我方英雄對線評分 − 對方英雄對線評分。資料含雙向 → 自然 sum-to-zero。

三段：
  A 對位加性模型：時序 train/test 的 R²(純對位解釋多少 gd10 變異) / AUC(sign) / MAE，逐路 + CI
  B counter 增量：specific pair 殘差 split-half 信度（對位之外還剩多少；對標 counter 對 win 弱）
  C 閉合迴路：predicted gd10 → win 的 AUC（純 draft 經對線能否預測勝；應≈.50）vs actual gd10 → win

跑：python3 matchup_gd10.py   （秒級，不碰 API，確定性 SEED=42）
"""
import sqlite3
import time
from collections import Counter, defaultdict

import numpy as np

import config

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
SEED = 42
NBOOT = 1000
LAMBDA = 25.0          # ridge：對稀有英雄收縮（≈每英雄需 ~25 場才脫離 0）
TEST_FRAC = 0.30       # 時序最後 30% 當 test（walk-forward）
MIN_PAIR = 8           # counter split-half：每對至少場次


def load():
    """每列 = 一個對線位（含雙向）：role, myChamp, oppChamp, gd10, xd10, cd10, gd14, win, match_id, t。"""
    c = sqlite3.connect(config.DB_PATH)
    rows = c.execute(
        """
        SELECT l.team_position, l.champion, l.opp_champion,
               l.gd10, l.xd10, l.cd10, l.gd14, l.win,
               l.match_id, m.game_creation
        FROM laning l JOIN matches m ON m.match_id = l.match_id
        WHERE m.queue = 420
          AND l.team_position IN ('TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY')
          AND l.champion IS NOT NULL AND l.opp_champion IS NOT NULL
          AND l.gd10 IS NOT NULL
        """
    ).fetchall()
    return rows


def fast_auc(score, label):
    """rank-based AUC（label 0/1）。無 sklearn。"""
    label = np.asarray(label)
    pos, neg = label == 1, label == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    # tie 平均 rank
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def fit_role(my, opp, y, nch, lam=LAMBDA):
    """反對稱加性 ridge：解 (XᵀX+λI)θ = Xᵀy，X 每列 +1@my −1@opp。不materialize X。"""
    # XᵀX：對角 = 該英雄出現次數(當 my 或 opp 都 +1)；off-diag[a,b] = −(a,b 同列共現次數)
    XtX = np.zeros((nch, nch))
    Xty = np.zeros(nch)
    np.add.at(XtX, (my, my), 1.0)
    np.add.at(XtX, (opp, opp), 1.0)
    np.add.at(XtX, (my, opp), -1.0)
    np.add.at(XtX, (opp, my), -1.0)
    np.add.at(Xty, my, y)
    np.add.at(Xty, opp, -y)
    XtX[np.diag_indices(nch)] += lam
    theta = np.linalg.solve(XtX, Xty)
    return theta


def predict(theta, my, opp):
    return theta[my] - theta[opp]


def r2(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rows = load()
    n = len(rows)

    role = np.array([r[0] for r in rows])
    my_s = [r[1] for r in rows]
    opp_s = [r[2] for r in rows]
    gd10 = np.array([r[3] for r in rows], float)
    xd10 = np.array([np.nan if r[4] is None else r[4] for r in rows], float)
    cd10 = np.array([np.nan if r[5] is None else r[5] for r in rows], float)
    gd14 = np.array([np.nan if r[6] is None else r[6] for r in rows], float)
    win = np.array([r[7] for r in rows], float)
    mid = np.array([r[8] for r in rows])
    tcreate = np.array([r[9] if r[9] is not None else 0 for r in rows], float)

    # 英雄 -> int（全域，跨路共用 id 沒差，每路各自 fit）
    uniq = sorted(set(my_s) | set(opp_s))
    cid = {c: i for i, c in enumerate(uniq)}
    nch = len(uniq)
    my = np.array([cid[c] for c in my_s])
    opp = np.array([cid[c] for c in opp_s])

    # 時序切：每路各自照 game_creation 排序，最後 TEST_FRAC 當 test（walk-forward）
    order = np.argsort(tcreate, kind="mergesort")
    rank = np.empty(n, float)
    rank[order] = np.arange(n) / n
    is_test = rank >= (1 - TEST_FRAC)

    print(f"§6 draft→gd@10  |  對線列={n}  英雄={nch}  match={len(set(mid.tolist()))}")
    print(f"gd10: mean={gd10.mean():+.0f} sd={gd10.std():.0f} 金（反對稱、~0 心）；"
          f"time-split train={int((~is_test).sum())} / test={int(is_test.sum())}")
    print(f"ridge λ={LAMBDA}；AUC(sign)=「對位能否預測誰對線領先」；R²=純對位解釋 gd10 變異占比\n")

    # ============ A) 對位加性模型：逐路 R²/AUC/MAE（時序 out-of-sample）============
    print("== A) 對位加性模型 gd10 = θ[my] − θ[opp]（時序 test，純 draft，賽前可算）==")
    print(f"  {'role':8} {'n_test':>7} {'R²_oos':>7} {'AUC':>6} {'MAE金':>7} {'baseMAE':>8}  {'R²_in':>6}")
    role_rows = {}
    overall_pred = np.full(n, np.nan)
    for r in ROLES:
        m = role == r
        tr = m & ~is_test
        te = m & is_test
        theta = fit_role(my[tr], opp[tr], gd10[tr], nch)
        yhat_te = predict(theta, my[te], opp[te])
        yhat_in = predict(theta, my[tr], opp[tr])
        overall_pred[te] = yhat_te
        R2o = r2(gd10[te], yhat_te)
        R2i = r2(gd10[tr], yhat_in)
        auc = fast_auc(yhat_te, (gd10[te] > 0).astype(int))
        mae = float(np.abs(gd10[te] - yhat_te).mean())
        base_mae = float(np.abs(gd10[te] - gd10[tr].mean()).mean())   # 預測常數(訓練均值≈0)
        role_rows[r] = (R2o, auc, mae)
        print(f"  {r:8} {int(te.sum()):>7} {R2o:>7.3f} {auc:>6.3f} {mae:>7.0f} {base_mae:>8.0f}  {R2i:>6.3f}")
    # 整體（pooled test）
    te_all = is_test & ~np.isnan(overall_pred)
    R2_all = r2(gd10[te_all], overall_pred[te_all])
    auc_all = fast_auc(overall_pred[te_all], (gd10[te_all] > 0).astype(int))

    # cluster(match) bootstrap CI on overall R² / AUC（test set）
    test_mids = mid[te_all]
    uniq_m = np.array(sorted(set(test_mids.tolist())))
    idx_by_m = {mm: np.where(test_mids == mm)[0] for mm in uniq_m}
    yv = gd10[te_all]; pv = overall_pred[te_all]
    lab = (yv > 0).astype(int)
    bR2, bAUC = np.empty(NBOOT), np.empty(NBOOT)
    for k in range(NBOOT):
        sel_m = uniq_m[rng.integers(0, len(uniq_m), len(uniq_m))]
        sel = np.concatenate([idx_by_m[mm] for mm in sel_m])
        bR2[k] = r2(yv[sel], pv[sel])
        bAUC[k] = fast_auc(pv[sel], lab[sel])
    ciR2 = np.percentile(bR2, [2.5, 97.5])
    ciAUC = np.percentile(bAUC, [2.5, 97.5])
    print(f"  {'OVERALL':8} {int(te_all.sum()):>7} {R2_all:>7.3f} {auc_all:>6.3f}")
    print(f"    R²  95%CI [{ciR2[0]:.3f}, {ciR2[1]:.3f}]   AUC 95%CI [{ciAUC[0]:.3f}, {ciAUC[1]:.3f}]  (cluster=match)")
    ranked = sorted(role_rows.items(), key=lambda kv: -kv[1][1])
    print("  逐路 AUC 排序：" + " > ".join(f"{r}({v[1]:.2f})" for r, v in ranked))
    print("  解讀：R²＝純選角對位能解釋多少『10 分鐘金幣差』；AUC＝能否賽前猜中誰對線領先。\n")

    # csd@10 / xd@10 / gd@14：同模型換 target（task 指名 csd@10；gd14 看領先持不持久）
    print("== A2) 換 target（同對位加性模型，時序 oos R² / AUC）==")
    for name, yv2, mask2 in [("cd10(csd)", cd10, ~np.isnan(cd10)),
                              ("xd10(xpd)", xd10, ~np.isnan(xd10)),
                              ("gd14", gd14, ~np.isnan(gd14))]:
        R2s, AUCs, ntot = [], [], 0
        predall = np.full(n, np.nan)
        for r in ROLES:
            m = (role == r) & mask2
            tr = m & ~is_test; te = m & is_test
            if tr.sum() < 50 or te.sum() < 50:
                continue
            th = fit_role(my[tr], opp[tr], yv2[tr], nch)
            predall[te] = predict(th, my[te], opp[te])
        te = is_test & mask2 & ~np.isnan(predall)
        R2v = r2(yv2[te], predall[te])
        aucv = fast_auc(predall[te], (yv2[te] > 0).astype(int))
        print(f"  {name:10} oos R²={R2v:>6.3f}  AUC={aucv:.3f}  (n_test={int(te.sum())})")
    print()

    # ============ B) counter 增量：specific pair 殘差的 split-half 信度 ============
    # 加性模型抓「英雄各自對線強度」。counter = 「A 專剋 B」＝對位殘差。
    # 若殘差是真訊號 → 把 test 對折，每對的平均殘差兩半應正相關；若 ≈0 → counter 是雜訊。
    print("== B) counter（對位殘差）有沒有真訊號：split-half 信度 ==")
    # 用全資料 fit 一次拿殘差（要的是殘差結構，不是 oos）
    resid = np.full(n, np.nan)
    for r in ROLES:
        m = role == r
        th = fit_role(my[m], opp[m], gd10[m], nch)
        resid[m] = gd10[m] - predict(th, my[m], opp[m])
    rng2 = np.random.default_rng(SEED)
    half = rng2.random(n) < 0.5
    pair = defaultdict(lambda: [[], []])   # (role,my,opp) -> [resid_h0, resid_h1]
    for i in range(n):
        pair[(role[i], my[i], opp[i])][int(half[i])].append(resid[i])
    a, bvals = [], []
    for key, (h0, h1) in pair.items():
        if len(h0) >= MIN_PAIR and len(h1) >= MIN_PAIR:
            a.append(np.mean(h0)); bvals.append(np.mean(h1))
    a, bvals = np.array(a), np.array(bvals)
    if len(a) >= 10:
        rcorr = float(np.corrcoef(a, bvals)[0, 1])
        print(f"  每對殘差 split-half 相關 r = {rcorr:+.3f}  (n_pair≥{MIN_PAIR}×2 = {len(a)})")
        print(f"  殘差 sd = {np.nanstd(resid):.0f} 金 vs gd10 sd = {gd10.std():.0f} 金"
              f"（加性已吃掉 {(1-np.nanvar(resid)/gd10.var())*100:.0f}% 變異）")
        print("  讀法：r≈0 → 『專剋』不可複現＝雜訊（呼應 counter 對 win 弱）；r 顯著正 → 對位確有 counter 結構。\n")
    else:
        print(f"  可用對位太少（n_pair={len(a)}）；資料量/場次門檻不足以測 counter。\n")

    # ============ C) 閉合迴路：對線優勢 → 勝負（actual vs pre-game predicted）============
    # actual gd10 → win：實際對線贏多少能預測贏這局（含臨場/confound）
    # predicted gd10 → win：純 draft 經對線通道能預測多少（賽前可算）。若≈.50 ⇒
    #   draft→gd10 有訊號、gd10 的『可被 draft 預測那塊』→win 沒訊號 = 對線⊥勝率 + draft⊥勝率 的機制。
    print("== C) 閉合迴路：對線優勢 → 勝負（per-lane→該玩家勝負）==")
    # 用時序 test、且有 overall_pred 的列
    te = is_test & ~np.isnan(overall_pred)
    auc_actual = fast_auc(gd10[te], win[te].astype(int))
    auc_pred = fast_auc(overall_pred[te], win[te].astype(int))
    print(f"  actual    gd10 → win  AUC = {auc_actual:.3f}  (真實對線結果；含臨場)")
    print(f"  predicted gd10 → win  AUC = {auc_pred:.3f}  (純 draft 對位；賽前可算)")
    print(f"  → 對位能預測 gd10（A 段），但其『可被 draft 預測的部分』對勝負 AUC={auc_pred:.3f}")
    print(f"     ↔ 實際對線結果對勝負 AUC={auc_actual:.3f}。差距＝對線→勝的訊號多在臨場、非選角。")
    # 逐路 predicted gd10 → win
    print("  逐路 predicted gd10 → win AUC：", end="")
    cells = []
    for r in ROLES:
        m = te & (role == r)
        cells.append(f"{r[:3]}={fast_auc(overall_pred[m], win[m].astype(int)):.2f}")
    print("  ".join(cells))

    print(f"\n  done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
