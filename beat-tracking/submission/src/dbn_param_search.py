"""
DBN Hyperparameter Search for Beat Tracking
實驗目的：系統性測試 DBNBeatTrackingProcessor 的 transition_lambda 和
observation_lambda 參數組合，找出在 GTZAN 上最佳的設定。

Usage:
    python dbn_param_search.py
"""
import os, json
import glob
import numpy as np
import madmom

AUDIO_DIR = './GTZAN/'
ANNOT_DIR = './GTZAN_annotations/beats'

def load_annotations(annot_path):
    data = np.loadtxt(annot_path)
    if data.ndim > 1:
        return data[:, 0]
    return data

def evaluate_params(activations_cache, transition_lambda, observation_lambda):
    """給定預先算好的 activation cache，用指定參數跑 DBN 並評估"""
    dbn_proc = madmom.features.beats.DBNBeatTrackingProcessor(
        fps=100,
        transition_lambda=transition_lambda,
        observation_lambda=observation_lambda
    )

    all_evals = []
    for file_id, (act, targets) in activations_cache.items():
        try:
            detections = dbn_proc(act)
            eval_obj = madmom.evaluation.beats.BeatEvaluation(detections, targets)
            all_evals.append(eval_obj)
        except Exception as e:
            print(f"  Error on {file_id}: {e}")

    if not all_evals:
        return 0.0
    mean_eval = madmom.evaluation.beats.BeatMeanEvaluation(all_evals)
    return mean_eval.fmeasure

def main():
    print("Step 1: Pre-computing RNN activations for all files (done once)...")
    rnn_proc = madmom.features.beats.RNNBeatProcessor()

    activations_cache = {}
    search_pattern = os.path.join(AUDIO_DIR, "**/*.wav")
    audio_files = sorted(glob.glob(search_pattern, recursive=True))

    for i, audio_path in enumerate(audio_files):
        file_id = os.path.basename(audio_path).replace('.wav', '').replace('.', '_')
        annot_path = os.path.join(ANNOT_DIR, f"gtzan_{file_id}.beats")
        if not os.path.exists(annot_path):
            continue
        try:
            act = rnn_proc(audio_path)
            targets = load_annotations(annot_path)
            activations_cache[file_id] = (act, targets)
        except Exception as e:
            print(f"  Error loading {file_id}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(audio_files)} files loaded...")

    print(f"Loaded {len(activations_cache)} files.\n")

    # Step 2: Grid search over DBN parameters
    transition_lambdas = [50, 100, 150, 200, 300]
    observation_lambdas = [8, 16, 24, 32]

    print("Step 2: Grid search over DBN parameters...")
    print(f"{'transition_lambda':<20} {'observation_lambda':<20} {'F1-Measure':<12}")
    print("-" * 55)

    results = []
    for tl in transition_lambdas:
        for ol in observation_lambdas:
            f1 = evaluate_params(activations_cache, tl, ol)
            results.append((f1, tl, ol))
            marker = " ← baseline" if tl == 100 and ol == 16 else ""
            print(f"{tl:<20} {ol:<20} {f1:.4f}{marker}")

    # Summary
    results.sort(reverse=True)
    print("\n" + "=" * 55)
    print("Top 3 configurations:")
    for f1, tl, ol in results[:3]:
        print(f"  transition_lambda={tl}, observation_lambda={ol}  →  F1={f1:.4f}")

    best_f1, best_tl, best_ol = results[0]
    print(f"\nBest: transition_lambda={best_tl}, observation_lambda={best_ol}")
    print(f"Best F1: {best_f1:.4f}")

    # Save best predictions
    print(f"\nGenerating predictions with best parameters...")
    dbn_best = madmom.features.beats.DBNBeatTrackingProcessor(
        fps=100,
        transition_lambda=best_tl,
        observation_lambda=best_ol
    )
    best_results = {}
    for file_id, (act, _) in activations_cache.items():
        detections = dbn_best(act)
        best_results[file_id] = detections.tolist()

    with open("dbn_tuned_predictions.json", "w") as f:
        json.dump(best_results, f)
    print("Saved to dbn_tuned_predictions.json")
    print("Evaluate with: python eval_json.py dbn_tuned_predictions.json")

if __name__ == "__main__":
    main()
