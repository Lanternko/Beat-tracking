"""§5：偏離 meta 對勝負的因果（控玩家強度）。menu B 玩家身份軸首次動工。

接 §4(predict_picks.py)：§4 發現「模型機率十分位 vs 勝率全平 ~50%」(未控玩家)。
§5 問因果：同一個玩家，當他選 off-meta 時，對線 / 勝率 / variance 變怎樣？

off-meta 度量(model-free，每個 pick 都有 → 撐得起 within-player 控制)：
  該英雄在 (role,patch) 的 pick rate。off_score = -log10(pick_rate)，越大＝越冷門。
  bin：core ≥5% / playable 1-5% / off-meta 0.3-1% / pocket <0.3%（皆為該 role,patch 內佔比）。

控玩家強度的三層（重點＝看 raw 的 off-meta 懲罰，控玩家後縮多少 / 消不消失）：
  1. RAW(不控)        —— off-meta bin vs 結果（§4 預告的 gd10 + variance 版）
  2. within-player FE —— 只取 ≥K 場的玩家、減掉該玩家均值，看「同一人 off-meta 的 Δ」
  3. player 勝率 covariate（leave-one-out，避免機械相關）
confound：one-trick / smurf —— off-meta picks 再按 familiarity(該英雄佔該玩家場次比)切。

三 outcome：gd@10(傷不傷線)、win(控玩家後還剩多少勝率稅)、variance(boom-or-bust)。

跑：python3 offmeta_causal.py   （秒級，不碰 API，確定性 SEED=42）
"""
import sqlite3
import time
from collections import Counter, defaultdict

import numpy as np

import config

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
SEED = 42
NBOOT = 1000
K_GAMES = 10           # within-player 分析的最低場次門檻
OFF_MAX = 0.01         # off-meta：pick_rate < 1%
ON_MIN = 0.05          # on-meta ：pick_rate >= 5%


def load():
    c = sqlite3.connect(config.DB_PATH)
    rows = c.execute(
        """
        SELECT p.match_id, p.puuid, p.champion, p.team_position, m.patch, p.win, l.gd10
        FROM participants p
        JOIN matches m ON m.match_id = p.match_id
        LEFT JOIN laning l ON l.match_id = p.match_id AND l.puuid = p.puuid
        WHERE m.queue = 420
          AND p.team_position IN ('TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY')
        """
    ).fetchall()
    return rows


def cluster_boot_ratio(A, B, rng, nboot=NBOOT):
    """Σa/Σb 的 cluster(player) bootstrap 95% CI。"""
    n = len(A)
    if n == 0:
        return float("nan"), float("nan")
    out = np.empty(nboot)
    for k in range(nboot):
        sel = rng.integers(0, n, n)
        out[k] = A[sel].sum() / B[sel].sum()
    return tuple(np.percentile(out, [2.5, 97.5]))


def within_slope(pid, x, y, valid, pl_n, rng, min_games=K_GAMES):
    """within-player(FE) 斜率 β：y = α_player + β·x。減掉玩家均值後 OLS。
    回傳 β、95%CI、貢獻玩家數。只用 valid 列、玩家 ≥min_games 場、且 x 有 within 變異。"""
    rows_by_p = defaultdict(list)
    idx = np.where(valid)[0]
    for i in idx:
        rows_by_p[pid[i]].append(i)
    A, B = [], []
    for p, ii in rows_by_p.items():
        if pl_n[p] < min_games or len(ii) < 2:
            continue
        xs = x[ii]
        xc = xs - xs.mean()
        bb = float((xc * xc).sum())
        if bb <= 1e-9:
            continue
        yc = y[ii] - y[ii].mean()
        A.append(float((xc * yc).sum()))
        B.append(bb)
    A, B = np.array(A), np.array(B)
    if len(A) == 0:
        return float("nan"), (float("nan"), float("nan")), 0
    beta = A.sum() / B.sum()
    return beta, cluster_boot_ratio(A, B, rng), len(A)


def pooled_slope(x, y, valid):
    """不控玩家的 pooled OLS 斜率（對照 within）。"""
    xs, ys = x[valid], y[valid]
    xc = xs - xs.mean()
    return float((xc * (ys - ys.mean())).sum() / (xc * xc).sum())


def within_contrast(pid, off_mask, on_mask, y, valid, pl_n, rng, min_games=K_GAMES):
    """同一玩家 off-meta 減 on-meta 的 Δ(outcome)。回傳 meanΔ、95%CI、玩家數。"""
    bucket = defaultdict(lambda: {"off": [], "on": []})
    for i in np.where(valid)[0]:
        if off_mask[i]:
            bucket[pid[i]]["off"].append(i)
        elif on_mask[i]:
            bucket[pid[i]]["on"].append(i)
    deltas = []
    for p, d in bucket.items():
        if pl_n[p] < min_games or not d["off"] or not d["on"]:
            continue
        deltas.append(y[d["off"]].mean() - y[d["on"]].mean())
    deltas = np.array(deltas)
    if len(deltas) == 0:
        return float("nan"), (float("nan"), float("nan")), 0
    boots = np.array([rng.choice(deltas, len(deltas), replace=True).mean()
                      for _ in range(NBOOT)])
    return deltas.mean(), tuple(np.percentile(boots, [2.5, 97.5])), len(deltas)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rows = load()
    n = len(rows)

    # ---- 解出每個 pick 的欄位 ----
    puuid = [r[1] for r in rows]
    champ = [r[2] for r in rows]
    role = [r[3] for r in rows]
    patch = [r[4] for r in rows]
    win = np.array([r[5] for r in rows], float)
    gd10 = np.array([np.nan if r[6] is None else r[6] for r in rows], float)
    has_gd = ~np.isnan(gd10)

    # puuid -> int id
    uniq_p = sorted(set(puuid))
    pid_of = {p: i for i, p in enumerate(uniq_p)}
    pid = np.array([pid_of[p] for p in puuid])
    pl_n = np.bincount(pid)                       # 每玩家場次
    npl = len(uniq_p)

    # ---- pick_rate：每英雄在 (role,patch) 的佔比 ----
    rp_total = Counter()
    rp_champ = defaultdict(Counter)
    for i in range(n):
        rp_total[(role[i], patch[i])] += 1
        rp_champ[(role[i], patch[i])][champ[i]] += 1
    prate = np.array([rp_champ[(role[i], patch[i])][champ[i]] / rp_total[(role[i], patch[i])]
                      for i in range(n)])
    off_score = -np.log10(prate)                  # 越大＝越冷門

    # ---- familiarity：該英雄佔該玩家總場次比例（one-trick 偵測）----
    pl_champ = defaultdict(Counter)
    for i in range(n):
        pl_champ[pid[i]][champ[i]] += 1
    famil = np.array([pl_champ[pid[i]][champ[i]] / pl_n[pid[i]] for i in range(n)])

    # ---- player 整體勝率（leave-one-out，當 covariate 用）----
    pl_wins = np.bincount(pid, weights=win)
    pl_wr_loo = (pl_wins[pid] - win) / np.maximum(pl_n[pid] - 1, 1)

    # ================= 診斷 =================
    print(f"§5 off-meta 因果分析  |  picks={n}  players={npl}  patches={sorted(set(patch))}")
    print(f"laning 覆蓋率(gd10 非空)：{has_gd.mean()*100:.1f}%  ({int(has_gd.sum())}/{n})")
    pct = np.percentile(pl_n, [50, 75, 90, 95, 99])
    print("每玩家場次分位 p50/75/90/95/99 = "
          + "/".join(f"{x:.0f}" for x in pct)
          + f"   ≥5:{int((pl_n>=5).sum())}  ≥10:{int((pl_n>=10).sum())}"
          + f"  ≥20:{int((pl_n>=20).sum())}  ≥50:{int((pl_n>=50).sum())}")
    print("注意：Challenger+GM 單 elo 帶；off-meta=該 role,patch 內 pick rate 低；gd10 正=對線領先。\n")

    # bin
    def binof(pr):
        if pr >= ON_MIN:
            return 0  # core
        if pr >= 0.01:
            return 1  # playable
        if pr >= 0.003:
            return 2  # off-meta
        return 3      # pocket
    b = np.array([binof(p) for p in prate])
    BNAME = ["core ≥5%", "playable 1-5%", "off-meta .3-1%", "pocket <.3%"]

    # ================= 1) RAW（不控玩家）=================
    print("== 1) RAW：off-meta 程度 vs 結果（不控玩家強度）==")
    print(f"  {'bin':16} {'n':>7} {'win%':>7} {'gd10':>8} {'sd(gd10)':>9}")
    for k in range(4):
        m = b == k
        mg = m & has_gd
        wr = win[m].mean() * 100
        g = gd10[mg].mean() if mg.sum() else float("nan")
        sd = gd10[mg].std() if mg.sum() else float("nan")
        print(f"  {BNAME[k]:16} {int(m.sum()):>7} {wr:>6.1f} {g:>8.0f} {sd:>9.0f}")
    print(f"  pooled 斜率(不控)：win~off_score = {pooled_slope(off_score, win, np.ones(n,bool))*100:+.2f} pp / 十倍冷門"
          f"；gd10~off_score = {pooled_slope(off_score, gd10, has_gd):+.0f} 金 / 十倍冷門\n")

    # ================= 2) within-player FE =================
    print(f"== 2) within-player FE（同一玩家、≥{K_GAMES} 場、減玩家均值）==")
    bw, ciw, nw = within_slope(pid, off_score, win, np.ones(n, bool), pl_n, rng)
    bg, cig, ng = within_slope(pid, off_score, gd10, has_gd, pl_n, rng)
    print(f"  win  ~ off_score : {bw*100:+.2f} pp / 十倍冷門   95%CI[{ciw[0]*100:+.2f},{ciw[1]*100:+.2f}]  (n_player={nw})")
    print(f"  gd10 ~ off_score : {bg:+.0f} 金 / 十倍冷門      95%CI[{cig[0]:+.0f},{cig[1]:+.0f}]  (n_player={ng})")
    print("  解讀：FE 斜率 vs RAW pooled 斜率——縮小＝玩家強弱在 confound off-meta。\n")

    # 離散對照：同一玩家 off-meta(<1%) 減 on-meta(>=5%)
    off_mask = prate < OFF_MAX
    on_mask = prate >= ON_MIN
    dw, cidw, ndw = within_contrast(pid, off_mask, on_mask, win, np.ones(n, bool), pl_n, rng)
    dg, cidg, ndg = within_contrast(pid, off_mask, on_mask, gd10, has_gd, pl_n, rng)
    print(f"  離散對照（同一玩家：off-meta<1% 減 on-meta≥5%，≥{K_GAMES} 場且兩邊都有）：")
    print(f"    Δwin  = {dw*100:+.2f} pp   95%CI[{cidw[0]*100:+.2f},{cidw[1]*100:+.2f}]  (n_player={ndw})")
    print(f"    Δgd10 = {dg:+.0f} 金     95%CI[{cidg[0]:+.0f},{cidg[1]:+.0f}]  (n_player={ndg})\n")

    # ================= 3) player 勝率 covariate =================
    # win ~ a + c1*off_score + c2*pl_wr_loo（看控掉玩家強度後 off_score 係數）
    Xd = np.column_stack([np.ones(n), off_score, pl_wr_loo])
    coef, *_ = np.linalg.lstsq(Xd, win, rcond=None)
    Xd0 = np.column_stack([np.ones(n), off_score])
    coef0, *_ = np.linalg.lstsq(Xd0, win, rcond=None)
    print("== 3) player 勝率 covariate（leave-one-out）==")
    print(f"  win ~ off_score          : {coef0[1]*100:+.2f} pp / 十倍冷門（不控）")
    print(f"  win ~ off_score + pl_wr  : {coef[1]*100:+.2f} pp / 十倍冷門（控玩家勝率）；pl_wr 係數={coef[2]:.2f}\n")

    # ================= 4) meta × familiarity：稅是 off-meta 還是「玩不熟」？=========
    # 關鍵 confound：off-meta 的勝率稅，是否其實是「玩不熟英雄」的稅（off-meta 只是
    # 比較常不熟）？→ 看 on-meta 的不熟 pick 會不會也掉。限 ≥K 場玩家讓 familiarity 有意義
    # （1-2 場玩家的 familiarity 恆=100%，是雜訊）。
    elig = pl_n[pid] >= K_GAMES
    fam_lab = ["unfam <10%", "mid 10-30%", "fam ≥30%"]
    fb = np.where(famil < 0.1, 0, np.where(famil < 0.3, 1, 2))
    meta_sel = [("on-meta ≥5%", prate >= ON_MIN), ("off-meta <1%", prate < OFF_MAX)]
    print(f"== 4) meta × familiarity 2×3（限 ≥{K_GAMES} 場玩家；win% (gd10) [n]）==")
    print("  問：off-meta 稅 vs 玩不熟稅。比『同 familiarity、跨 meta』的橫向差。")
    print(f"  {'':14} " + " ".join(f"{fl:>18}" for fl in fam_lab))
    for mlab, msel in meta_sel:
        cells = []
        for fk in range(3):
            m = elig & msel & (fb == fk)
            mg = m & has_gd
            if m.sum() == 0:
                cells.append(f"{'-':>18}"); continue
            wr = win[m].mean() * 100
            g = gd10[mg].mean() if mg.sum() else float("nan")
            cells.append(f"{wr:>5.1f} ({g:>+4.0f})[{int(m.sum()):>4}]")
        print(f"  {mlab:14} " + " ".join(cells))
    # 中介路徑：off-meta 是否比較常「不熟」
    for mlab, msel in meta_sel:
        m = elig & msel
        p_unfam = (famil[m] < 0.1).mean() * 100
        print(f"  P(unfam<10% | {mlab}) = {p_unfam:.0f}%   mean familiarity = {famil[m].mean():.2f}")
    print("  讀法：若 on-meta 的 unfam 也掉到 ~45% → 是『玩不熟』稅、非 off-meta 稅；")
    print("        off-meta 只是更常落在 unfam（中介），controlling familiarity 後 meta 主效應消失。\n")

    # ---- 4b) autofill 穩健性：unfam off-meta 的低勝率是否其實是 autofill？----
    pl_role = defaultdict(Counter)
    for i in range(n):
        pl_role[pid[i]][role[i]] += 1
    modal = {p: c.most_common(1)[0][0] for p, c in pl_role.items()}
    autofill = np.array([role[i] != modal[pid[i]] for i in range(n)])
    print(f"== 4b) autofill 穩健性（autofill≈該場 role≠該玩家眾數 role，限 ≥{K_GAMES} 場）==")
    base = elig & (prate < OFF_MAX) & (fb == 0)   # unfam off-meta 那格
    for af, lab in [(False, "非 autofill"), (True, "autofill  ")]:
        m = base & (autofill == af)
        mg = m & has_gd
        if m.sum() == 0:
            continue
        print(f"  unfam off-meta × {lab}: win {win[m].mean()*100:>5.1f}%  gd10 {gd10[mg].mean():>+5.0f}  n={int(m.sum())}")
    me = elig
    print(f"  參考：≥{K_GAMES}場玩家 autofill 率={autofill[me].mean()*100:.0f}%，"
          f"autofill 勝率={win[me & autofill].mean()*100:.1f}% vs 主位={win[me & ~autofill].mean()*100:.1f}%\n")

    # ================= 5) variance（boom-or-bust）=================
    print("\n== 5) variance：off-meta 是否更 boom-or-bust ==")
    print(f"  {'bin':16} {'sd(gd10)':>9} {'sd(win)':>8}")
    for k in range(4):
        m = b == k
        mg = m & has_gd
        sdg = gd10[mg].std() if mg.sum() else float("nan")
        sdw = win[m].std()
        print(f"  {BNAME[k]:16} {sdg:>9.0f} {sdw:>8.3f}")

    print(f"\n  done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
