"""
Per-Genre F1 Breakdown Analysis (Experiment C)
================================================
載入各方法的 prediction JSON，計算每種曲風的 F1。
輸出一張大表，用於 Discussion 章節。

Usage:
    python per_genre_analysis.py predictions1.json predictions2.json ...
    
    # 例：比較 baseline 與 Beat This!
    python per_genre_analysis.py baseline.json predictions.json
"""
import os, sys, json
import numpy as np
import madmom

ANNOT_DIR = './GTZAN_annotations/beats'
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']


def load_annotations(annot_path):
    data = np.loadtxt(annot_path)
    return data[:, 0] if data.ndim > 1 else data


def genre_of(file_id):
    return file_id.rsplit('_', 1)[0]


def compute_per_genre_f1(json_path):
    """回傳 dict: genre → mean F1"""
    with open(json_path) as f:
        predictions = json.load(f)

    genre_evals = {g: [] for g in GENRES}

    for file_id, detections in predictions.items():
        annot_path = os.path.join(ANNOT_DIR, f"gtzan_{file_id}.beats")
        if not os.path.exists(annot_path):
            continue

        genre = genre_of(file_id)
        if genre not in genre_evals:
            continue

        targets = load_annotations(annot_path)
        ev = madmom.evaluation.beats.BeatEvaluation(
            np.array(detections), targets)
        genre_evals[genre].append(ev.fmeasure)

    result = {}
    for genre in GENRES:
        if genre_evals[genre]:
            result[genre] = np.mean(genre_evals[genre])
        else:
            result[genre] = 0.0

    # 整體 F1
    all_evals = []
    for file_id, detections in predictions.items():
        annot_path = os.path.join(ANNOT_DIR, f"gtzan_{file_id}.beats")
        if not os.path.exists(annot_path):
            continue
        targets = load_annotations(annot_path)
        all_evals.append(madmom.evaluation.beats.BeatEvaluation(
            np.array(detections), targets))

    if all_evals:
        result['OVERALL'] = madmom.evaluation.beats.BeatMeanEvaluation(all_evals).fmeasure
    else:
        result['OVERALL'] = 0.0

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python per_genre_analysis.py pred1.json [pred2.json ...]")
        print("Example: python per_genre_analysis.py baseline.json predictions.json")
        sys.exit(1)

    json_files = sys.argv[1:]
    results = {}

    for jf in json_files:
        name = os.path.basename(jf).replace('.json', '')
        print(f"Evaluating: {jf} ...")
        results[name] = compute_per_genre_f1(jf)

    # 輸出大表
    methods = list(results.keys())
    header = f"{'Genre':<12}" + "".join(f"{m:<18}" for m in methods) + "  Best Method"
    print("\n" + "=" * len(header))
    print("PER-GENRE F1 BREAKDOWN (%)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for genre in GENRES + ['OVERALL']:
        row = f"{genre:<12}"
        values = []
        for m in methods:
            val = results[m].get(genre, 0) * 100
            values.append((val, m))
            row += f"{val:<18.2f}"

        best_val, best_method = max(values, key=lambda x: x[0])
        row += f"  {best_method}"

        if genre == 'OVERALL':
            print("-" * len(header))
        print(row)

    # 找出各方法最強/最弱的曲風
    print("\n" + "-" * 50)
    for m in methods:
        genre_scores = [(results[m].get(g, 0), g) for g in GENRES]
        genre_scores.sort(reverse=True)
        best_genre = genre_scores[0]
        worst_genre = genre_scores[-1]
        print(f"\n{m}:")
        print(f"  Best:  {best_genre[1]} ({best_genre[0]*100:.2f}%)")
        print(f"  Worst: {worst_genre[1]} ({worst_genre[0]*100:.2f}%)")
        spread = (best_genre[0] - worst_genre[0]) * 100
        print(f"  Spread: {spread:.2f}%")


if __name__ == "__main__":
    main()
