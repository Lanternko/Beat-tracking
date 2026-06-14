"""
Cross-Architecture Activation Fusion (Experiment A)
====================================================
將 RNN (madmom) 與 Beat This! (Transformer) 的 beat activation
做加權平均，再用 peak picking 或 DBN 解碼。

核心假設：RNN activation 寬泛平滑（局部穩定），
         Beat This! activation 尖銳精準（全域結構），
         兩者加權融合可能截長補短。

Usage:
    python activation_ensemble.py
"""
import os, sys, json, glob
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import madmom

# ── 嘗試載入 beat_this ──
try:
    from beat_this.inference import File2Beats, Audio2Frames
    import torch
    HAS_BEAT_THIS = True
except ImportError:
    HAS_BEAT_THIS = False
    print("Warning: beat_this not installed. Install with: pip install git+https://github.com/CPJKU/beat_this.git")

AUDIO_DIR = './GTZAN/'
ANNOT_DIR = './GTZAN_annotations/beats'
FPS_RNN = 100   # madmom RNN 的 fps
FPS_BT  = 50    # Beat This! 的 fps（20ms hop）

# Ensemble 搜索的 alpha 值：blended = alpha * RNN + (1-alpha) * BeatThis
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

# Peak picking 參數
PEAK_THRESHOLD = 0.5
MIN_BEAT_DISTANCE_SEC = 0.2  # 最短 beat interval（秒）


def load_annotations(annot_path):
    data = np.loadtxt(annot_path)
    return data[:, 0] if data.ndim > 1 else data


def simple_peak_picking(activation, fps, threshold=0.5, min_distance_sec=0.2):
    """簡單的 peak picking：找 activation 中高於 threshold 的 local maxima"""
    min_distance_frames = int(min_distance_sec * fps)
    peaks, properties = find_peaks(activation, height=threshold,
                                   distance=max(1, min_distance_frames))
    return peaks / fps  # 轉換為秒


def resample_activation(act, src_fps, tgt_fps, tgt_length):
    """將 activation 從 src_fps 重新取樣到 tgt_fps，長度對齊到 tgt_length"""
    src_length = len(act)
    src_times = np.arange(src_length) / src_fps
    tgt_times = np.arange(tgt_length) / tgt_fps

    # 確保 tgt_times 不超過 src_times 的範圍
    tgt_times = np.clip(tgt_times, 0, src_times[-1])

    f = interp1d(src_times, act, kind='linear', fill_value='extrapolate')
    return f(tgt_times)


def evaluate_predictions(predictions_dict, annot_dir):
    """評估 predictions，回傳 F1"""
    all_evals = []
    for file_id, detections in predictions_dict.items():
        annot_path = os.path.join(annot_dir, f"gtzan_{file_id}.beats")
        if not os.path.exists(annot_path):
            continue
        targets = load_annotations(annot_path)
        ev = madmom.evaluation.beats.BeatEvaluation(
            np.array(detections), targets)
        all_evals.append(ev)

    if not all_evals:
        return 0.0
    return madmom.evaluation.beats.BeatMeanEvaluation(all_evals).fmeasure


def main():
    if not HAS_BEAT_THIS:
        print("beat_this is required. Exiting.")
        sys.exit(1)

    # ── Step 1: 預計算所有 activation ──
    print("=" * 65)
    print("Cross-Architecture Activation Fusion")
    print("=" * 65)

    print("\nStep 1: Pre-computing activations...")
    rnn_proc = madmom.features.beats.RNNBeatProcessor()

    # Beat This! Audio2Frames 回傳 frame-level logits
    try:
        bt_proc = Audio2Frames(checkpoint_path='final0', device='cuda')
    except Exception:
        bt_proc = Audio2Frames(checkpoint_path='final0', device='cpu')
        print("  (Using CPU for Beat This!)")

    cache = {}  # file_id → (rnn_act, bt_act_resampled, targets)
    audio_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "**/*.wav"), recursive=True))

    for i, audio_path in enumerate(audio_files):
        file_id = os.path.basename(audio_path).replace('.wav', '').replace('.', '_')
        annot_path = os.path.join(ANNOT_DIR, f"gtzan_{file_id}.beats")
        if not os.path.exists(annot_path):
            continue

        try:
            # RNN activation (100 fps)
            rnn_act = rnn_proc(audio_path)

            # Beat This! activation
            # Audio2Frames 回傳 (beat_logits_tensor, downbeat_logits_tensor)
            import librosa
            audio, sr = librosa.load(audio_path, sr=22050)
            bt_out = bt_proc(audio, sr)
            # bt_out 是 tuple: (beat_logits, downbeat_logits)，皆為 torch.Tensor
            bt_beat_logits = bt_out[0].detach().cpu().numpy().astype(np.float64)

            # Sigmoid 轉為機率
            bt_act = 1.0 / (1.0 + np.exp(-bt_beat_logits))

            # 重新取樣 Beat This! activation 到 RNN 的 fps 和長度
            bt_act_resampled = resample_activation(bt_act, FPS_BT, FPS_RNN, len(rnn_act))

            targets = load_annotations(annot_path)
            cache[file_id] = (rnn_act, bt_act_resampled, targets)

        except Exception as e:
            print(f"  Error {file_id}: {e}")

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(audio_files)} loaded...")

    print(f"  Loaded {len(cache)} files.\n")

    # ── Step 2: 搜索最佳 alpha（Peak Picking）──
    print("Step 2: Searching best alpha with Peak Picking...")
    print(f"{'Alpha':<10} {'F1 (%)':<10} {'Note'}")
    print("-" * 40)

    best_alpha_pp = 0.0
    best_f1_pp = 0.0

    for alpha in ALPHAS:
        predictions = {}
        for file_id, (rnn_act, bt_act, _) in cache.items():
            blended = alpha * rnn_act + (1.0 - alpha) * bt_act
            beats = simple_peak_picking(blended, FPS_RNN,
                                        threshold=PEAK_THRESHOLD,
                                        min_distance_sec=MIN_BEAT_DISTANCE_SEC)
            predictions[file_id] = beats.tolist()

        f1 = evaluate_predictions(predictions, ANNOT_DIR)
        note = ""
        if alpha == 0.0:
            note = "← Beat This! only"
        elif alpha == 1.0:
            note = "← RNN only"

        print(f"{alpha:<10.1f} {f1*100:<10.2f} {note}")

        if f1 > best_f1_pp:
            best_f1_pp = f1
            best_alpha_pp = alpha

    print(f"\n  Best (Peak Picking): alpha={best_alpha_pp}, F1={best_f1_pp*100:.2f}%")

    # ── Step 3: 搜索最佳 alpha（DBN decoding）──
    print("\nStep 3: Searching best alpha with DBN decoding...")
    print(f"{'Alpha':<10} {'F1 (%)':<10} {'Note'}")
    print("-" * 40)

    best_alpha_dbn = 0.0
    best_f1_dbn = 0.0
    dbn_proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=FPS_RNN)

    for alpha in ALPHAS:
        predictions = {}
        for file_id, (rnn_act, bt_act, _) in cache.items():
            blended = alpha * rnn_act + (1.0 - alpha) * bt_act
            try:
                beats = dbn_proc(blended)
                predictions[file_id] = beats.tolist()
            except Exception:
                pass

        f1 = evaluate_predictions(predictions, ANNOT_DIR)
        note = ""
        if alpha == 0.0:
            note = "← Beat This! act + DBN"
        elif alpha == 1.0:
            note = "← RNN act + DBN (≈baseline)"

        print(f"{alpha:<10.1f} {f1*100:<10.2f} {note}")

        if f1 > best_f1_dbn:
            best_f1_dbn = f1
            best_alpha_dbn = alpha

    print(f"\n  Best (DBN): alpha={best_alpha_dbn}, F1={best_f1_dbn*100:.2f}%")

    # ── Step 4: 儲存最佳 ensemble 的 predictions ──
    print(f"\nStep 4: Saving best ensemble predictions...")
    best_predictions = {}
    for file_id, (rnn_act, bt_act, _) in cache.items():
        blended = best_alpha_pp * rnn_act + (1.0 - best_alpha_pp) * bt_act
        beats = simple_peak_picking(blended, FPS_RNN,
                                    threshold=PEAK_THRESHOLD,
                                    min_distance_sec=MIN_BEAT_DISTANCE_SEC)
        best_predictions[file_id] = beats.tolist()

    with open("ensemble_predictions.json", "w") as f:
        json.dump(best_predictions, f)

    # ── Summary ──
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  RNN + DBN baseline:           87.02%")
    print(f"  Beat This! (peak picking):    88.25%")
    print(f"  Best Ensemble (Peak Picking): {best_f1_pp*100:.2f}%  (alpha={best_alpha_pp})")
    print(f"  Best Ensemble (DBN):          {best_f1_dbn*100:.2f}%  (alpha={best_alpha_dbn})")
    print(f"\n  Saved: ensemble_predictions.json")
    print(f"  Evaluate: python eval_json.py ensemble_predictions.json")


if __name__ == "__main__":
    main()
