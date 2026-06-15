"""選角預測 / meta 收斂度（§4，goal 樞紐）。

任務：masked-completion —— 遮住 10 個 pick 中的一個，用其餘 9 個（英雄+分路+敵我）
      + 目標分路 + patch 預測被遮的英雄。Riot soloQ 資料沒有 pick 順序、沒有 ban
      （和 LoLDraftAI 同樣的限制 → 同樣用隨機遮罩），所以是「補全」不是「逐手預測」。

三個量測（難度遞增，每個都是「picks 有多可預測」的尺）：
  1. 邊際 baseline  P(champ | role, patch)        —— 純 meta 集中度（最熱門）
  2. 情境模型 (LR)  P(champ | role, patch, 其餘9) —— 加上 synergy / counter / comp
  meta 收斂度：每 (role, patch) 的 perplexity = 2^H = 「有效英雄池大小」，跨 patch 追蹤。

輸出也順手把「off-meta」操作化（情境模型給實際選角的機率，低=偏離 meta），
為 §5（偏離 meta 對勝負的影響）鋪路。

跑：python3 predict_picks.py        （秒~分鐘級，不碰 API）
"""
import sqlite3
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

import config

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
ROLE_IX = {r: i for i, r in enumerate(ROLES)}
SEED = 42
MIN_PATCH_GAMES = 500  # 跨 patch 表只看樣本夠的版本


def load():
    """回傳 games = [(patch, {(team,pos): (champ, win)})]，以及 patch->match_id。"""
    c = sqlite3.connect(config.DB_PATH)
    patch_of = dict(c.execute("SELECT match_id, patch FROM matches").fetchall())
    rows = c.execute(
        "SELECT match_id, team_id, team_position, champion, win FROM participants"
    ).fetchall()
    by_match = {}
    for mid, team, pos, champ, win in rows:
        by_match.setdefault(mid, {})[(team, pos)] = (champ, win)
    games = []
    for mid, slots in by_match.items():
        if len(slots) != 10:
            continue
        teams = {100: set(), 200: set()}
        for (team, pos) in slots:
            teams.setdefault(team, set()).add(pos)
        if not all(teams.get(t) == set(ROLES) for t in (100, 200)):
            continue
        games.append((patch_of.get(mid, "?"), mid, slots))
    return games


def entropy_bits(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def meta_convergence(games):
    """純資料（model-free）：每 role、每 (role,patch) 的有效英雄池 = 2^H。"""
    # role -> patch -> champ -> count
    tally = {r: {} for r in ROLES}
    pooled = {r: {} for r in ROLES}
    for patch, _mid, slots in games:
        for (_team, pos), (champ, _w) in slots.items():
            pooled[pos][champ] = pooled[pos].get(champ, 0) + 1
            d = tally[pos].setdefault(patch, {})
            d[champ] = d.get(champ, 0) + 1

    print("== meta 收斂度（全資料，model-free）==")
    print("每路有效英雄池 = 2^entropy；越小＝meta 越收斂。最熱門%＝該路最常選英雄佔比。\n")
    print(f"  {'role':8} {'picks':>6} {'distinct':>9} {'eff.pool':>9} {'top1%':>7}  top-3 most picked")
    for r in ROLES:
        ch = pooled[r]
        counts = np.array(sorted(ch.values(), reverse=True), float)
        eff = 2 ** entropy_bits(counts)
        top3 = sorted(ch.items(), key=lambda kv: -kv[1])[:3]
        top1 = top3[0][1] / counts.sum() * 100
        names = ", ".join(f"{n}({v})" for n, v in top3)
        print(f"  {r:8} {int(counts.sum()):>6} {len(ch):>9} {eff:>9.1f} {top1:>6.1f}%  {names}")

    # 跨 patch 演變（只看樣本夠的 patch）
    patch_games = {}
    for patch, _m, _s in games:
        patch_games[patch] = patch_games.get(patch, 0) + 1
    patches = sorted([p for p, n in patch_games.items() if n >= MIN_PATCH_GAMES])
    print(f"\n== 跨 patch 的有效英雄池（每路 2^H；patch≥{MIN_PATCH_GAMES} 場）==")
    print(f"  {'role':8} " + " ".join(f"{p:>7}" for p in patches))
    for r in ROLES:
        cells = []
        for p in patches:
            d = tally[r].get(p, {})
            if d:
                counts = np.array(list(d.values()), float)
                cells.append(f"{2 ** entropy_bits(counts):>7.1f}")
            else:
                cells.append(f"{'-':>7}")
        print(f"  {r:8} " + " ".join(cells))
    print("  （數字↓＝該版本該路選擇更集中／meta 更收斂）\n")


def build_examples(games, champ_ix, patch_ix):
    """每場 → 10 個 masked-completion 樣本。回傳 X(csr), y(champ idx), role, win, group。"""
    nchamp, npatch = len(champ_ix), len(patch_ix)
    ctx_dim = nchamp * 5 * 2          # (champ, role, ally/enemy)
    role_off = ctx_dim                # 目標分路 one-hot
    patch_off = ctx_dim + 5           # patch one-hot
    dim = ctx_dim + 5 + npatch

    rows, cols, data = [], [], []
    y, y_role, y_win, group = [], [], [], []
    ri = 0
    for patch, mid, slots in games:
        pix = patch_ix[patch]
        items = list(slots.items())  # [((team,pos),(champ,win)), ...] 長度 10
        for ti in range(10):
            (t_team, t_pos), (t_champ, t_win) = items[ti]
            for si in range(10):
                if si == ti:
                    continue
                (s_team, s_pos), (s_champ, _w) = items[si]
                ally = 1 if s_team == t_team else 0
                col = (champ_ix[s_champ] * 5 + ROLE_IX[s_pos]) * 2 + ally
                rows.append(ri); cols.append(col); data.append(1.0)
            rows.append(ri); cols.append(role_off + ROLE_IX[t_pos]); data.append(1.0)
            rows.append(ri); cols.append(patch_off + pix); data.append(1.0)
            y.append(champ_ix[t_champ]); y_role.append(ROLE_IX[t_pos])
            y_win.append(t_win); group.append(mid)
            ri += 1
    X = csr_matrix((data, (rows, cols)), shape=(ri, dim))
    return X, np.array(y), np.array(y_role), np.array(y_win), np.array(group)


def topk_hits(rank_idx, y_true, k):
    """rank_idx[i] = 第 i 列由高到低的 champ idx；回傳 hit@k 布林。"""
    return np.array([yt in rank_idx[i, :k] for i, yt in enumerate(y_true)])


def main():
    t0 = time.time()
    games = load()
    champs = sorted({c for _p, _m, s in games for (c, _w) in s.values()})
    champ_ix = {c: i for i, c in enumerate(champs)}
    patches = sorted({p for p, _m, _s in games})
    patch_ix = {p: i for i, p in enumerate(patches)}
    print(f"games={len(games)}  champions={len(champs)}  patches={patches}")
    print("注意：masked-completion（無 pick 順序）、無 ban、Challenger+GM 單一 elo 帶。\n")

    meta_convergence(games)

    X, y, y_role, y_win, group = build_examples(games, champ_ix, patch_ix)
    # 依「比賽」切 train/test（避免同場 9 個共享英雄洩漏）
    rng = np.random.default_rng(SEED)
    uniq = np.array(sorted(set(group.tolist())))
    rng.shuffle(uniq)
    test_mids = set(uniq[: len(uniq) // 5].tolist())
    te = np.array([g in test_mids for g in group])
    tr = ~te
    print(f"masked 樣本：{len(y)}（train {tr.sum()} / test {te.sum()}，依 match 切）\n")

    # patch idx per row（給邊際 baseline 用）
    patch_col = X[:, -len(patches):].toarray().argmax(1)

    # ---- 1) 邊際 baseline：P(champ | role, patch)，train 統計 ----
    from collections import defaultdict
    freq_rp = defaultdict(lambda: np.zeros(len(champs)))  # (role,patch)
    freq_r = defaultdict(lambda: np.zeros(len(champs)))    # role fallback
    idx_tr = np.where(tr)[0]
    for i in idx_tr:
        freq_rp[(y_role[i], patch_col[i])][y[i]] += 1
        freq_r[y_role[i]][y[i]] += 1
    rank_rp = {k: np.argsort(-v) for k, v in freq_rp.items()}
    rank_r = {k: np.argsort(-v) for k, v in freq_r.items()}
    idx_te = np.where(te)[0]
    base_rank = np.zeros((len(idx_te), len(champs)), int)
    for j, i in enumerate(idx_te):
        key = (y_role[i], patch_col[i])
        base_rank[j] = rank_rp[key] if key in rank_rp else rank_r[y_role[i]]
    yte = y[idx_te]

    # ---- 2) 情境模型：multinomial logistic regression ----
    print("訓練情境模型（multinomial LR, saga）…", flush=True)
    clf = LogisticRegression(solver="saga", C=3.0, max_iter=200, tol=1e-3)
    clf.fit(X[tr], y[tr])
    proba = clf.predict_proba(X[te])
    classes = clf.classes_
    order = np.argsort(-proba, axis=1)
    lr_rank = classes[order]                      # 映回 champ idx
    p_true = np.array([                            # 模型給「實際選角」的機率（off-meta 用）
        proba[j, np.where(classes == yt)[0][0]] if yt in classes else 0.0
        for j, yt in enumerate(yte)
    ])

    # ---- 報告：overall + per role 的 top-1/3/5 ----
    def report(rank, yt, mask=None):
        if mask is None:
            mask = np.ones(len(yt), bool)
        return tuple(topk_hits(rank[mask], yt[mask], k).mean() * 100 for k in (1, 3, 5))

    roles_te = y_role[idx_te]
    print("\n== 選角預測率（test，hit@k %）==")
    print(f"  {'':16} {'top-1':>7} {'top-3':>7} {'top-5':>7}")
    b = report(base_rank, yte); m = report(lr_rank, yte)
    print(f"  {'邊際 baseline':16} {b[0]:>6.1f} {b[1]:>6.1f} {b[2]:>6.1f}")
    print(f"  {'情境模型 (LR)':14} {m[0]:>6.1f} {m[1]:>6.1f} {m[2]:>6.1f}")
    print(f"  {'context 的 lift':14} {m[0]-b[0]:>+6.1f} {m[1]-b[1]:>+6.1f} {m[2]-b[2]:>+6.1f}")
    print(f"  （隨機猜＝{100/len(champs):.2f}% top-1）\n")

    print("  per role（情境模型）：")
    print(f"  {'role':8} {'eff.pool':>9} {'top-1':>7} {'top-3':>7} {'top-5':>7}")
    for r in ROLES:
        msk = roles_te == ROLE_IX[r]
        mr = report(lr_rank, yte, msk)
        # 該路 test 有效池
        cnt = np.bincount(yte[msk], minlength=len(champs)).astype(float)
        eff = 2 ** entropy_bits(cnt[cnt > 0])
        print(f"  {r:8} {eff:>9.1f} {mr[0]:>6.1f} {mr[1]:>6.1f} {mr[2]:>6.1f}")

    # bootstrap 90% CI（overall top-1, 情境模型）
    hit1 = topk_hits(lr_rank, yte, 1)
    boot = [rng.choice(hit1, len(hit1), replace=True).mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [5, 95]) * 100
    print(f"\n  情境模型 top-1 = {hit1.mean()*100:.1f}%  90% CI [{lo:.1f}, {hi:.1f}]")

    # ---- §5 預告：off-meta（低 p_true）的原始勝率（未控玩家強度）----
    wins = y_win[idx_te]
    dec = np.clip((p_true * 10).astype(int), 0, 9)  # 機率十分位
    print("\n== §5 預告：模型給「實際選角」的機率 vs 該 pick 的勝率（未控玩家強度！）==")
    print("  低機率＝off-meta（模型覺得意外）。confound：冷門常是 one-trick/smurf 在玩。")
    print(f"  {'P(pick)十分位':14} {'n':>6} {'勝率%':>7}")
    for d in range(10):
        msk = dec == d
        if msk.sum() == 0:
            continue
        lab = f"{d/10:.1f}-{(d+1)/10:.1f}"
        print(f"  {lab:14} {int(msk.sum()):>6} {wins[msk].mean()*100:>6.1f}")

    # 最「意外」的幾個 pick（off-meta 範例）
    print("\n  最意外的 8 個 test picks（off-meta 範例；模型本來想選誰）：")
    sur = np.argsort(p_true)[:8]
    for j in sur:
        r = ROLES[roles_te[j]]
        actual = champs[yte[j]]
        modeltop = champs[lr_rank[j, 0]]
        wl = "W" if wins[j] else "L"
        print(f"    {r:8} 實際={actual:14} 模型#1={modeltop:14} p={p_true[j]:.3f}  [{wl}]")

    print(f"\n  done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
