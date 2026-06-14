"""
Beat This! Activation + Genre-Aware DBN (Experiment D)
=======================================================
研究問題：DBN 對 Transformer 的傷害，是因為架構本質的不匹配，
還是因為 DBN 參數不夠彈性？

實驗設計：
  1. 取得 Beat This! 的 frame-level activation（非 beat positions）
  2. 對每種曲風套用 Exp #6 找出的最佳 DBN 參數
  3. 若 F1 回升（相比 87.07%），說明 genre-aware 的彈性參數能減輕傷害
  4. 若 F1 仍低於 88.25%，確認了 DBN 的根本性不匹配

Usage:
    python beat_this_genre_dbn.py
"""
import os, sys, json, glob
import numpy as np
from scipy.interpolate import interp1d
import madmom

try:
    from beat_this.inference import Audio2Frames
    import librosa
    HAS_BEAT_THIS = True
except ImportError:
    HAS_BEAT_THIS = False

AUDIO_DIR = './GTZAN/'
ANNOT_DIR = './GTZAN_annotations/beats'
FPS_BT = 50
FPS_DBN = 100

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# 搜索空間（同 dbn_per_genre.py）
TRANSITION_LAMBDAS  = [50, 100, 150, 200, 300]
OBSERVATION_LAMBDAS = [8, 16, 24, 32]


def load_annotations(annot_path):
    data = np.loadtxt(annot_path)
    return data[:, 0] if data.ndim > 1 else data


def genre_of(file_id):
    return file_id.rsplit('_', 1)[0]


def resample_activation(act, src_fps, tgt_fps, tgt_length):
    src_times = np.arange(len(act)) / src_fps
    tgt_times = np.arange(tgt_length) / tgt_fps
    tgt_times = np.clip(tgt_times, 0, src_times[-1])
    f = interp1d(src_times, act, kind='linear', fill_value='extrapolate')
    return f(tgt_times)


def evaluate_params(cache_subset, transition_lambda, observation_lambda):
    dbn = madmom.features.beats.DBNBeatTrackingProcessor(
        fps=FPS_DBN,
        transition_lambda=transition_lambda,
        observation_lambda=observation_lambda,
    )
    evals = []
    for file_id, (act, targets) in cache_subset.items():
        try:
            detections = dbn(act)
            evals.append(madmom.evaluation.beats.BeatEvaluation(detections, targets))
        except Exception:
            pass
    if not evals:
        return 0.0
    return madmom.evaluation.beats.BeatMeanEvaluation(evals).fmeasure


def main():
    if not HAS_BEAT_THIS:
        print("beat_this is required.")
        sys.exit(1)

    print("=" * 65)
    print("Beat This! Activation + Genre-Aware DBN")
    print("=" * 65)

    # ── Step 1: 取得 Beat This! activation ──
    print("\nStep 1: Computing Beat This! activations...")
    try:
        bt_proc = Audio2Frames(checkpoint_path='final0', device='cuda')
    except Exception:
        bt_proc = Audio2Frames(checkpoint_path='final0', device='cpu')

    cache = {}  # file_id → (bt_act_at_100fps, targets)
    audio_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "**/*.wav"), recursive=True))

    for i, audio_path in enumerate(audio_files):
        file_id = os.path.basename(audio_path).replace('.wav', '').replace('.', '_')
        annot_path = os.path.join(ANNOT_DIR, f"gtzan_{file_id}.beats")
        if not os.path.exists(annot_path):
            continue

        try:
            audio, sr = librosa.load(audio_path, sr=22050)
            bt_out = bt_proc(audio, sr)
            # Audio2Frames returns (beat_logits_tensor, downbeat_logits_tensor)
            bt_beat_logits = bt_out[0].detach().cpu().numpy().astype(np.float64)
            bt_act = 1.0 / (1.0 + np.exp(-bt_beat_logits))

            # 需要一個目標長度：用音訊長度推算 100fps 的 frame 數
            n_frames_100fps = int(len(audio) / sr * FPS_DBN) + 1
            bt_act_resampled = resample_activation(bt_act, FPS_BT, FPS_DBN, n_frames_100fps)

            targets = load_annotations(annot_path)
            cache[file_id] = (bt_act_resampled, targets)

        except Exception as e:
            print(f"  Error {file_id}: {e}")

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(audio_files)} loaded...")

    print(f"  Loaded {len(cache)} files.\n")

    # ── Step 2: Default DBN on Beat This! activation ──
    print("Step 2: Beat This! activation + Default DBN (tl=100, ol=16)...")
    default_f1 = evaluate_params(cache, 100, 16)
    print(f"  F1 = {default_f1*100:.2f}%  (compare: Beat This! dbn=True → 87.07%)\n")

    # ── Step 3: Per-genre grid search on Beat This! activation ──
    print("Step 3: Per-genre DBN parameter search on Beat This! activation...")
    best_params = {}

    for genre in GENRES:
        subset = {k: v for k, v in cache.items() if genre_of(k) == genre}
        if not subset:
            continue

        best = (0.0, 100, 16)
        for tl in TRANSITION_LAMBDAS:
            for ol in OBSERVATION_LAMBDAS:
                f1 = evaluate_params(subset, tl, ol)
                if f1 > best[0]:
                    best = (f1, tl, ol)

        best_params[genre] = best
        print(f"  {genre:<12}  tl={best[1]:>3}, ol={best[2]:>2}  →  F1={best[0]*100:.2f}%")

    # ── Step 4: 用各曲風最佳參數生成 predictions ──
    print("\nStep 4: Generating predictions with per-genre best params...")
    predictions = {}
    per_genre_evals = {g: [] for g in GENRES}

    for file_id, (act, targets) in cache.items():
        genre = genre_of(file_id)
        _, tl, ol = best_params.get(genre, (0, 100, 16))
        dbn = madmom.features.beats.DBNBeatTrackingProcessor(
            fps=FPS_DBN, transition_lambda=tl, observation_lambda=ol)
        try:
            detections = dbn(act)
            predictions[file_id] = detections.tolist()
            ev = madmom.evaluation.beats.BeatEvaluation(detections, targets)
            per_genre_evals[genre].append(ev.fmeasure)
        except Exception:
            pass

    # 計算整體 F1
    all_evals = []
    for file_id, dets in predictions.items():
        annot_path = os.path.join(ANNOT_DIR, f"gtzan_{file_id}.beats")
        if os.path.exists(annot_path):
            targets = load_annotations(annot_path)
            all_evals.append(madmom.evaluation.beats.BeatEvaluation(
                np.array(dets), targets))

    genre_aware_f1 = madmom.evaluation.beats.BeatMeanEvaluation(all_evals).fmeasure

    # ── Step 5: Summary ──
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"\n{'Method':<45} {'F1 (%)'}")
    print("-" * 55)
    print(f"{'Beat This! (peak picking, no DBN)':<45} 88.25%")
    print(f"{'Beat This! + Default DBN (Exp #4)':<45} 87.07%")
    print(f"{'Beat This! act + Default DBN (this script)':<45} {default_f1*100:.2f}%")
    print(f"{'Beat This! act + Genre-Aware DBN (oracle)':<45} {genre_aware_f1*100:.2f}%")

    print(f"\nPer-genre breakdown:")
    print(f"{'Genre':<12} {'F1 (%)':<10} {'tl':<6} {'ol'}")
    print("-" * 35)
    for genre in GENRES:
        if genre in best_params and per_genre_evals[genre]:
            f1_val = np.mean(per_genre_evals[genre])
            _, tl, ol = best_params[genre]
            print(f"{genre:<12} {f1_val*100:<10.2f} {tl:<6} {ol}")

    with open("beat_this_genre_dbn_predictions.json", "w") as f:
        json.dump(predictions, f)
    print(f"\nSaved: beat_this_genre_dbn_predictions.json")

    # 關鍵研究問題的回答
    print("\n" + "=" * 65)
    print("RESEARCH QUESTION ANSWER:")
    if genre_aware_f1 > default_f1:
        delta = (genre_aware_f1 - default_f1) * 100
        print(f"  Genre-aware DBN 將 Beat This!+DBN 的 F1 提升了 {delta:.2f}%")
        if genre_aware_f1 * 100 > 88.25:
            print(f"  → 超越了 Beat This! 單獨使用的 88.25%！")
            print(f"  → DBN 的傷害主要來自參數不夠彈性，而非架構不匹配。")
        else:
            print(f"  → 但仍低於 Beat This! 單獨使用的 88.25%")
            print(f"  → 參數彈性能減輕傷害，但 architectural mismatch 仍是主因。")
    else:
        print(f"  Genre-aware DBN 未能提升 Beat This!+DBN 的效能。")
        print(f"  → 確認了 architectural mismatch 是 DBN 傷害 Transformer 的根本原因。")


if __name__ == "__main__":
    main()
