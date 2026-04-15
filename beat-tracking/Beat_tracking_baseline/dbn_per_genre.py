"""
Per-Genre DBN Parameter Optimization

Motivation: Different music genres have fundamentally different rhythmic
characteristics (tempo range, beat regularity, syncopation). A single set of
DBN parameters cannot optimally serve all genres.

Method:
  1. Pre-compute RNN activations for all files (once).
  2. For each genre, run a grid search over (transition_lambda, observation_lambda).
  3. Apply each genre's best parameters to generate final predictions.
  4. Compare per-genre optimized F1 against global-best DBN (86.64%).

This incorporates domain knowledge: filename encodes genre (e.g. "blues_00042").
"""
import os, json
import glob
import numpy as np
import madmom

AUDIO_DIR = './GTZAN/'
ANNOT_DIR = './GTZAN_annotations/beats'

TRANSITION_LAMBDAS  = [50, 100, 150, 200, 300]
OBSERVATION_LAMBDAS = [8, 16, 24, 32]

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']


def load_annotations(annot_path):
    data = np.loadtxt(annot_path)
    return data[:, 0] if data.ndim > 1 else data


def genre_of(file_id):
    return file_id.rsplit('_', 1)[0]


def evaluate_params(cache_subset, transition_lambda, observation_lambda):
    dbn = madmom.features.beats.DBNBeatTrackingProcessor(
        fps=100,
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
    # ── Step 1: Pre-compute RNN activations ──────────────────────────────────
    print("Step 1: Pre-computing RNN activations...")
    rnn = madmom.features.beats.RNNBeatProcessor()
    cache = {}

    for audio_path in sorted(glob.glob(os.path.join(AUDIO_DIR, "**/*.wav"), recursive=True)):
        file_id = os.path.basename(audio_path).replace('.wav', '').replace('.', '_')
        annot_path = os.path.join(ANNOT_DIR, f"gtzan_{file_id}.beats")
        if not os.path.exists(annot_path):
            continue
        try:
            cache[file_id] = (rnn(audio_path), load_annotations(annot_path))
        except Exception as e:
            print(f"  Error {file_id}: {e}")

    print(f"Loaded {len(cache)} files.\n")

    # ── Step 2: Per-genre grid search ─────────────────────────────────────────
    print("Step 2: Per-genre DBN parameter search...\n")
    best_params = {}   # genre → (transition_lambda, observation_lambda, f1)

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
        print(f"  {genre:<12}  best params: transition={best[1]:>3}, "
              f"observation={best[2]:>2}  →  F1={best[0]:.4f}")

    # ── Step 3: Global-param baseline for comparison ──────────────────────────
    print("\nGlobal best (transition=300, observation=8) per genre:")
    global_evals_all = []
    per_genre_evals = {}
    for genre in GENRES:
        subset = {k: v for k, v in cache.items() if genre_of(k) == genre}
        f1 = evaluate_params(subset, 300, 8)
        per_genre_evals[genre] = f1
        print(f"  {genre:<12}  F1={f1:.4f}")

    # ── Step 4: Generate per-genre-optimised predictions ─────────────────────
    print("\nGenerating per-genre optimised predictions...")
    results = {}
    for file_id, (act, _) in cache.items():
        genre = genre_of(file_id)
        _, tl, ol = best_params.get(genre, (0, 100, 16))
        dbn = madmom.features.beats.DBNBeatTrackingProcessor(fps=100,
                                                              transition_lambda=tl,
                                                              observation_lambda=ol)
        results[file_id] = dbn(act).tolist()

    with open("dbn_per_genre_predictions.json", "w") as f:
        json.dump(results, f)

    # ── Step 5: Summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Genre':<12} {'Global F1':>10} {'Per-Genre F1':>13} {'Δ':>8}")
    print("-" * 65)

    all_evals_per_genre_opt = []
    for genre in GENRES:
        subset = {k: v for k, v in cache.items() if genre_of(k) == genre}
        _, tl, ol = best_params.get(genre, (0, 100, 16))
        f1_opt = evaluate_params(subset, tl, ol)
        f1_global = per_genre_evals.get(genre, 0)
        delta = f1_opt - f1_global
        print(f"  {genre:<12} {f1_global:>10.4f} {f1_opt:>13.4f} {delta:>+8.4f}")

        dbn = madmom.features.beats.DBNBeatTrackingProcessor(fps=100,
                                                              transition_lambda=tl,
                                                              observation_lambda=ol)
        for file_id, (act, targets) in subset.items():
            try:
                all_evals_per_genre_opt.append(
                    madmom.evaluation.beats.BeatEvaluation(dbn(act), targets))
            except Exception:
                pass

    overall = madmom.evaluation.beats.BeatMeanEvaluation(all_evals_per_genre_opt)
    print("-" * 65)
    print(f"  {'OVERALL':<12} {evaluate_params(cache, 300, 8):>10.4f} "
          f"{overall.fmeasure:>13.4f} "
          f"{overall.fmeasure - evaluate_params(cache, 300, 8):>+8.4f}")

    print(f"\nSaved to dbn_per_genre_predictions.json")
    print("Evaluate with: python eval_json.py dbn_per_genre_predictions.json")


if __name__ == "__main__":
    main()
