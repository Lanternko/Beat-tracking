"""menu B（玩家身份軸）step 1：熟練紅利＝因果(mastery) 還是 survivorship(選擇)？

§5 發現 familiarity ≫ meta（熟練 off-meta one-trick 勝率 53.1% ＝ 熟練 on-meta），
但 familiarity→勝率的梯度有 survivorship confound：「你只 main 你會贏的英雄」。
本檔用巢狀 fixed-effect 把 selection 從 treatment 拆開：

  win / gd10 ~ familiarity，逐步加控制：
    RAW(pooled) → within-player(控玩家強度) → within-champ(控英雄強度) → two-way(都控)
  ★ 決定性＝ **within-player familiarity 斜率**：同一玩家在「自己招牌」vs「偶爾玩的」之間
    贏比較多嗎？survive → 熟練是真因果（net of 玩家強度）；歸零 → §5 梯度純 between-player 選擇。
  ★★ 最純＝ **學習曲線**：同一(玩家,英雄)隨 crawl 窗內時間變多場，gd10/win 有沒有爬升
     （同人同角、連英雄都固定 → 連 champ 選擇都不是 confound）。資料窗薄、預期樣本少，誠實報。

familiarity = 該英雄佔該玩家(crawl 窗)場次比；限 ≥10 場玩家（讓 familiarity 有意義）。
跑：python3 player_identity.py   （秒級，不碰 API，確定性 SEED=42）
"""
import sqlite3
import time
from collections import Counter, defaultdict

import numpy as np

import config

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
SEED = 42
NBOOT = 1000
K_GAMES = 10


def load():
    c = sqlite3.connect(config.DB_PATH)
    return c.execute(
        """
        SELECT p.match_id, p.puuid, p.champion, p.team_position, p.win, l.gd10, m.game_creation
        FROM participants p
        JOIN matches m ON m.match_id = p.match_id
        LEFT JOIN laning l ON l.match_id = p.match_id AND l.puuid = p.puuid
        WHERE m.queue = 420
          AND p.team_position IN ('TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY')
        """
    ).fetchall()


def cluster_boot_ratio(A, B, rng, nboot=NBOOT):
    n = len(A)
    if n == 0:
        return float("nan"), float("nan")
    out = np.empty(nboot)
    for k in range(nboot):
        sel = rng.integers(0, n, n)
        out[k] = A[sel].sum() / B[sel].sum()
    return tuple(np.percentile(out, [2.5, 97.5]))


def within_player_slope(pid, x, y, valid, pl_n, rng, min_games=K_GAMES):
    """within-player(FE) 斜率：減掉玩家均值後 OLS。回傳 β、95%CI（cluster-bootstrap）、玩家數。"""
    rows_by_p = defaultdict(list)
    for i in np.where(valid)[0]:
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
        A.append(float((xc * (y[ii] - y[ii].mean())).sum()))
        B.append(bb)
    A, B = np.array(A), np.array(B)
    if len(A) == 0:
        return float("nan"), (float("nan"), float("nan")), 0
    return A.sum() / B.sum(), cluster_boot_ratio(A, B, rng), len(A)


def fe_slope(x, y, groups, valid, iters=50):
    """逐組 demean（一組＝one-way、兩組＝two-way 交替投影）後的斜率。point estimate。"""
    idx = np.where(valid)[0]
    gs = [g[idx] for g in groups]

    def demean(v):
        v = v.astype(float).copy()
        for _ in range(iters):
            for g in gs:
                s = np.bincount(g, weights=v)
                c = np.bincount(g, minlength=s.shape[0])
                v = v - (s / np.maximum(c, 1))[g]
            if len(gs) == 1:
                break
        return v

    xt, yt = demean(x[idx]), demean(y[idx])
    d = (xt * xt).sum()
    return float((xt * yt).sum() / d) if d > 1e-9 else float("nan")


def pooled_slope(x, y, valid):
    xs, ys = x[valid], y[valid]
    xc = xs - xs.mean()
    return float((xc * (ys - ys.mean())).sum() / (xc * xc).sum())


def within_contrast(pid, a_mask, b_mask, y, valid, pl_n, rng, min_games=K_GAMES):
    """同一玩家 a 減 b 的 Δ(outcome)。回傳 meanΔ、95%CI、玩家數。"""
    bucket = defaultdict(lambda: {"a": [], "b": []})
    for i in np.where(valid)[0]:
        if a_mask[i]:
            bucket[pid[i]]["a"].append(i)
        elif b_mask[i]:
            bucket[pid[i]]["b"].append(i)
    deltas = []
    for p, d in bucket.items():
        if pl_n[p] < min_games or not d["a"] or not d["b"]:
            continue
        deltas.append(y[d["a"]].mean() - y[d["b"]].mean())
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

    puuid = [r[1] for r in rows]
    champ = [r[2] for r in rows]
    win = np.array([r[4] for r in rows], float)
    gd10 = np.array([np.nan if r[5] is None else r[5] for r in rows], float)
    gc = np.array([r[6] for r in rows], float)
    has_gd = ~np.isnan(gd10)

    uniq_p = sorted(set(puuid))
    pid_of = {p: i for i, p in enumerate(uniq_p)}
    pid = np.array([pid_of[p] for p in puuid])
    pl_n = np.bincount(pid)
    uniq_c = sorted(set(champ))
    cid_of = {c: i for i, c in enumerate(uniq_c)}
    cid = np.array([cid_of[c] for c in champ])

    pl_champ = defaultdict(Counter)
    for i in range(n):
        pl_champ[pid[i]][champ[i]] += 1
    famil = np.array([pl_champ[pid[i]][champ[i]] / pl_n[pid[i]] for i in range(n)])

    elig = pl_n[pid] >= K_GAMES

    print(f"menu B step1：熟練紅利＝mastery 還是 survivorship  |  picks={n}  players={len(uniq_p)}")
    print(f"  限 ≥{K_GAMES} 場玩家：{int((pl_n>=K_GAMES).sum())} 人、{int(elig.sum())} picks；familiarity 在此子集 mean={famil[elig].mean():.2f}")
    print("  注意：Challenger+GM 單 elo 帶；familiarity=該英雄佔該玩家 crawl 窗場次比；gd10 正=領先。\n")

    # ============ 1) familiarity → outcome：巢狀控制的斜率軌跡 ============
    print("== 1) familiarity → 結果：逐步加控制，看斜率縮不縮（每 +0.1 familiarity）==")
    print("  selection 若主導 → 加 player-FE 後斜率塌；mastery 若真 → within-player 仍 > 0。")
    print(f"  {'控制':22} {'Δwin pp':>10} {'Δgd10 金':>10}")
    # RAW
    sw = pooled_slope(famil, win, elig) * 10
    sg = pooled_slope(famil, gd10, elig & has_gd) * 0.1
    print(f"  {'RAW (pooled)':22} {sw:>+10.2f} {sg:>+10.0f}")
    # within-player（★ 決定性，含 CI）
    bw, ciw, nw = within_player_slope(pid, famil, win, elig, pl_n, rng)
    bg, cig, ng = within_player_slope(pid, famil, gd10, elig & has_gd, pl_n, rng)
    print(f"  {'+ within-player ★':22} {bw*10:>+10.2f} {bg*0.1:>+10.0f}")
    print(f"  {'   95%CI':22} [{ciw[0]*10:+.2f},{ciw[1]*10:+.2f}]  [{cig[0]*0.1:+.0f},{cig[1]*0.1:+.0f}]  (n_player={nw})")
    # within-champ
    cw = fe_slope(famil, win, [cid], elig) * 10
    cg = fe_slope(famil, gd10, [cid], elig & has_gd) * 0.1
    print(f"  {'+ within-champ':22} {cw:>+10.2f} {cg:>+10.0f}")
    # two-way
    tw = fe_slope(famil, win, [pid, cid], elig) * 10
    tg = fe_slope(famil, gd10, [pid, cid], elig & has_gd) * 0.1
    print(f"  {'+ two-way (玩家+英雄)':20} {tw:>+10.2f} {tg:>+10.0f}\n")

    # ============ 2) 離散 headline：同一玩家「招牌 vs 偶爾玩」 ============
    main_m = famil >= 0.3
    dab_m = famil <= 0.1
    dw, cidw, ndw = within_contrast(pid, main_m, dab_m, win, elig, pl_n, rng)
    dgd, cidg, ndg = within_contrast(pid, main_m, dab_m, gd10, elig & has_gd, pl_n, rng)
    print("== 2) headline（同一玩家：招牌 famil≥0.3 減 偶爾玩 famil≤0.1，≥10 場且兩邊都有）==")
    print(f"  Δwin  = {dw*100:+.2f} pp   95%CI[{cidw[0]*100:+.2f},{cidw[1]*100:+.2f}]  (n_player={ndw})")
    print(f"  Δgd10 = {dgd:+.0f} 金     95%CI[{cidg[0]:+.0f},{cidg[1]:+.0f}]  (n_player={ndg})")
    print("  正且 CI 排除 0 ＝ 熟練紅利在玩家內成立（不是純選擇）。\n")

    # ============ 3) 最純：學習曲線（同一玩家同一英雄、隨時間變多場）============
    pc = defaultdict(list)
    for i in np.where(elig)[0]:
        pc[(pid[i], cid[i])].append(i)
    dw_lc, dg_lc, npair = [], [], 0
    for (_p, _c), ii in pc.items():
        if len(ii) < 4:
            continue
        ii = sorted(ii, key=lambda j: gc[j])
        h = len(ii) // 2
        early, late = ii[:h], ii[h:]
        npair += 1
        dw_lc.append(win[late].mean() - win[early].mean())
        if all(has_gd[j] for j in ii):
            dg_lc.append(gd10[late].mean() - gd10[early].mean())
    print(f"== 3) 學習曲線（同(玩家,英雄) ≥4 場、crawl 窗內後半 減 前半）==")
    if npair >= 20:
        dw_lc = np.array(dw_lc)
        bw = np.array([rng.choice(dw_lc, len(dw_lc), replace=True).mean() for _ in range(NBOOT)])
        lo, hi = np.percentile(bw, [2.5, 97.5])
        line = f"  Δwin(後−前) = {dw_lc.mean()*100:+.2f} pp  95%CI[{lo*100:+.2f},{hi*100:+.2f}]"
        if dg_lc:
            dg_lc = np.array(dg_lc)
            line += f"   Δgd10 = {dg_lc.mean():+.0f} 金"
        print(line + f"  (n_pair={npair})")
        print("  正 ＝ 同人同角越打越好（最乾淨的 mastery 證據）；窗薄樣本少時僅參考。")
    else:
        print(f"  n_pair={npair} < 20，crawl 窗太薄（median 7 場/人）撐不起學習曲線；parked，待加玩家歷史。")

    print(f"\n  done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
