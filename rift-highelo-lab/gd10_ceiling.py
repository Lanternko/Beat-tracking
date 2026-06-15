"""§7：gd@10 預測準確率天花板 — champ vs player 分解 + win-flow（接 §6）。

§6([matchup_gd10.py]) 故意只用純對位 → oos R²=0.058。gd@10 母體 sd 867 金，
剩 94% 變異最大的可預測槓桿＝「玩家對線實力」（§6 完全沒用）。本檔：
  1. 提升準確率：champ-additive  →  +player skill（2-pass backfit、shrunk、train-only）
  2. 變異分解：高端對線多少是「選誰」vs「誰在玩」vs 純不可測
  3. win-flow（接 §6 punchline）：§6 證 champ 那塊對線預測力→勝負≈0(AUC.495)。
     那 player 那塊呢？player 對線實力 → win 是 ≈.50（連實力都⊥勝）還是 >.50（唯一流向勝負的對線成分）？

雙因子加性（backfit 近似 joint ridge）：
  gd10 = θ[role][myChamp] − θ[role][oppChamp] + φ[myPlayer] − φ[oppPlayer] + ε
  θ：每英雄單體對線強度（§6）。φ：玩家對線實力（champ 之外、shrunk-mean of 殘差）。
誠實：θ/φ 只用 train（時序前 70%）估，test（後 30%）從不餵自己；冷啟動玩家 φ=0。

跑：python3 gd10_ceiling.py   （秒級，不碰 API，確定性 SEED=42）
"""
import time

import numpy as np

import config
import sqlite3
from matchup_gd10 import fast_auc, r2, fit_role

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
SEED = 42
NBOOT = 1000
LAMBDA = 25.0          # champ ridge（同 §6）
K_SHRINK = 10.0        # player 殘差 shrink：需 ~10 場才脫離 0
N_PASS = 3             # backfit 來回次數
TEST_FRAC = 0.30


def load():
    c = sqlite3.connect(config.DB_PATH)
    return c.execute(
        """
        SELECT l.team_position, l.champion, l.opp_champion,
               l.puuid, l.opp_puuid, l.gd10, l.win,
               l.match_id, m.game_creation
        FROM laning l JOIN matches m ON m.match_id = l.match_id
        WHERE m.queue = 420
          AND l.team_position IN ('TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY')
          AND l.champion IS NOT NULL AND l.opp_champion IS NOT NULL
          AND l.puuid IS NOT NULL AND l.opp_puuid IS NOT NULL
          AND l.gd10 IS NOT NULL
        """
    ).fetchall()


def shrunk(idx, vals, n, K):
    """每 id 的 shrunk 均值：Σvals / (count + K)；未出現的 id = 0。"""
    s = np.bincount(idx, weights=vals, minlength=n)
    c = np.bincount(idx, minlength=n)
    return s / (c + K)


def backfit(myc, oppc, myp, oppp, y, nch, npl, lam, K, npass):
    """2-way 加性 backfit：回傳 θ(champ), φ(player)。只吃 train 列。"""
    theta = fit_role(myc, oppc, y, nch, lam=lam)
    phi = np.zeros(npl)
    for _ in range(npass):
        champ_resid = y - (theta[myc] - theta[oppc])      # 扣掉 champ 後給 player
        phi = shrunk(myp, champ_resid, npl, K)
        y_adj = y - (phi[myp] - phi[oppp])                # 扣掉 player 後重估 champ
        theta = fit_role(myc, oppc, y_adj, nch, lam=lam)
    return theta, phi


def boot_metric(pv, yv, mids, rng, fn):
    """cluster(match) bootstrap → (lo, hi) of fn(y, pred)。"""
    uniq = np.array(sorted(set(mids.tolist())))
    idx_by = {m: np.where(mids == m)[0] for m in uniq}
    out = np.empty(NBOOT)
    for k in range(NBOOT):
        sel_m = uniq[rng.integers(0, len(uniq), len(uniq))]
        sel = np.concatenate([idx_by[m] for m in sel_m])
        out[k] = fn(yv[sel], pv[sel])
    return tuple(np.percentile(out, [2.5, 97.5]))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rows = load()
    n = len(rows)

    role = np.array([r[0] for r in rows])
    myc_s = [r[1] for r in rows]
    oppc_s = [r[2] for r in rows]
    myp_s = [r[3] for r in rows]
    oppp_s = [r[4] for r in rows]
    gd10 = np.array([r[5] for r in rows], float)
    win = np.array([r[6] for r in rows], float)
    mid = np.array([r[7] for r in rows])
    tcreate = np.array([r[8] if r[8] is not None else 0 for r in rows], float)

    uniq_c = sorted(set(myc_s) | set(oppc_s))
    cid = {c: i for i, c in enumerate(uniq_c)}
    nch = len(uniq_c)
    myc = np.array([cid[c] for c in myc_s]); oppc = np.array([cid[c] for c in oppc_s])

    uniq_p = sorted(set(myp_s) | set(oppp_s))
    pid = {p: i for i, p in enumerate(uniq_p)}
    npl = len(uniq_p)
    myp = np.array([pid[p] for p in myp_s]); oppp = np.array([pid[p] for p in oppp_s])

    order = np.argsort(tcreate, kind="mergesort")
    rank = np.empty(n); rank[order] = np.arange(n) / n
    te = rank >= (1 - TEST_FRAC)
    tr = ~te

    # test 玩家在 train 的場次（coverage）
    train_pc = np.bincount(myp[tr], minlength=npl)
    cov5 = (train_pc[myp[te]] >= 5).mean()
    cov10 = (train_pc[myp[te]] >= 10).mean()

    print(f"§7 gd@10 準確率天花板 | 列={n} 英雄={nch} 玩家={npl} match={len(set(mid.tolist()))}")
    print(f"time-split train={int(tr.sum())}/test={int(te.sum())}；K_shrink={K_SHRINK} backfit×{N_PASS}")
    print(f"test 玩家 train 覆蓋：≥5 場 {cov5*100:.0f}% / ≥10 場 {cov10*100:.0f}%"
          f"（其餘冷啟動→φ=0，限制 player 能幫多少）\n")

    # ===== 逐路 fit、組 oos 預測（champ-only / champ+player）=====
    champ_pred = np.full(n, np.nan)
    full_pred = np.full(n, np.nan)
    player_pred = np.full(n, np.nan)
    for r in ROLES:
        m = role == r
        mtr = m & tr; mte = m & te
        theta, phi = backfit(myc[mtr], oppc[mtr], myp[mtr], oppp[mtr], gd10[mtr],
                             nch, npl, LAMBDA, K_SHRINK, N_PASS)
        champ_pred[mte] = theta[myc[mte]] - theta[oppc[mte]]
        player_pred[mte] = phi[myp[mte]] - phi[oppp[mte]]
        full_pred[mte] = champ_pred[mte] + player_pred[mte]

    v = te & ~np.isnan(full_pred)
    yv, mv = gd10[v], mid[v]
    sign = (yv > 0).astype(int)

    # ===== A) 準確率階梯 =====
    print("== A) 準確率階梯（時序 oos，pooled test）==")
    print(f"  {'model':22} {'R²_oos':>7} {'AUC':>6} {'MAE金':>7}")
    rows_out = []
    for name, pv in [("M0 champ-additive(§6)", champ_pred[v]),
                     ("M1 +player skill", full_pred[v]),
                     ("  player-only", player_pred[v])]:
        R2 = r2(yv, pv); AUC = fast_auc(pv, sign); MAE = float(np.abs(yv - pv).mean())
        print(f"  {name:22} {R2:>7.3f} {AUC:>6.3f} {MAE:>7.0f}")
        rows_out.append((name, R2, AUC))
    ci_r2 = boot_metric(full_pred[v], yv, mv, np.random.default_rng(SEED), r2)
    ci_auc = boot_metric(full_pred[v], sign.astype(float), mv,
                         np.random.default_rng(SEED), lambda y, p: fast_auc(p, y.astype(int)))
    print(f"  M1 95%CI: R²[{ci_r2[0]:.3f},{ci_r2[1]:.3f}]  AUC[{ci_auc[0]:.3f},{ci_auc[1]:.3f}] (cluster=match)")
    # 逐路 M1
    print("  逐路 M1(+player) R²/AUC：", end="")
    cells = []
    for r in ROLES:
        mm = v & (role == r)
        cells.append(f"{r[:3]}={r2(gd10[mm],full_pred[mm]):.2f}/{fast_auc(full_pred[mm],(gd10[mm]>0).astype(int)):.2f}")
    print("  ".join(cells))

    # ===== B) 變異分解 =====
    R2c = r2(yv, champ_pred[v]); R2f = r2(yv, full_pred[v])
    print(f"\n== B) gd@10 變異分解（oos，可預測天花板）==")
    print(f"  選誰(champ)      : {R2c*100:>4.1f}%")
    print(f"  誰在玩(+player)  : {(R2f-R2c)*100:>4.1f}%  (增量)")
    print(f"  純不可測(execution/variance/雜訊): {(1-R2f)*100:>4.1f}%")
    print(f"  → 高端對線可預測部分共 {R2f*100:.1f}%；player 把 champ 的 {R2c*100:.1f}% 推到 {R2f*100:.1f}%"
          f"（×{R2f/R2c:.1f}）")

    # ===== C) win-flow：哪塊對線預測力流向勝負 =====
    print(f"\n== C) win-flow：對線預測力的哪一塊流向勝負（per-lane→該玩家勝負）==")
    wv = win[v].astype(int)
    auc_champ = fast_auc(champ_pred[v], wv)
    auc_player = fast_auc(player_pred[v], wv)
    auc_full = fast_auc(full_pred[v], wv)
    auc_actual = fast_auc(yv, wv)
    ci_pw = boot_metric(player_pred[v], win[v], mv, np.random.default_rng(SEED),
                        lambda y, p: fast_auc(p, y.astype(int)))
    print(f"  champ-component  → win AUC = {auc_champ:.3f}   (純選角；§6 已證≈.50)")
    print(f"  player-component → win AUC = {auc_player:.3f}   95%CI[{ci_pw[0]:.3f},{ci_pw[1]:.3f}]")
    print(f"  full predicted   → win AUC = {auc_full:.3f}")
    print(f"  actual gd@10     → win AUC = {auc_actual:.3f}   (真實對線結果，含臨場)")
    if ci_pw[0] > 0.5:
        print(f"  → player 對線實力 *確實* 流向勝負（CI 過 .50）＝唯一從對線通到勝負的成分；")
        print(f"     champ 那塊死路。對線→勝的訊號＝玩家實力 ∩ 對線，不是選角。")
    else:
        print(f"  → 連 player 對線實力都≈⊥勝負（CI 含 .50）＝對線⊥勝率升級：")
        print(f"     不論選角或玩家對線實力，對線優勢都換不到勝（matchmaking 壓平一切）。")

    print(f"\n  done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
