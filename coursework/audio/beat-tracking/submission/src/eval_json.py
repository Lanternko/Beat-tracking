import os, sys, json
import glob
import numpy as np
import madmom

ANNOT_DIR = './GTZAN_annotations/beats'

def load_annotations(annot_path):
    """
    Loads beat annotations. 
    Handles files that may have [timestamp, beat_index] or just [timestamp].
    """
    data = np.loadtxt(annot_path)
    if data.ndim > 1:
        return data[:, 0]  # Take the first column (timestamps)
    return data

def evaluate_json(json_path):
    with open(json_path) as json_data:
        results = json.load(json_data)

    all_evals = []

    for file_id in results.keys():
        annot_path = os.path.join(ANNOT_DIR, f"gtzan_{file_id}.beats")

        detections = results[file_id]
        targets = load_annotations(annot_path)

        eval_obj = madmom.evaluation.beats.BeatEvaluation(detections, targets)
        all_evals.append(eval_obj)
        # print(f"{file_id:<25} | {eval_obj.fmeasure:<10.3f} | {eval_obj.cemgil:<10.3f}")

    if all_evals:
        mean_eval = madmom.evaluation.beats.BeatMeanEvaluation(all_evals)
        print("-" * 70)
        print("FINAL MEAN RESULTS:")
        print(f"F-Measure: {mean_eval.fmeasure:.4f}")
        print(f"Cemgil:    {mean_eval.cemgil:.4f}")
        print(f"P-Score:   {mean_eval.pscore:.4f}")


if __name__ == "__main__":
    json_path = sys.argv[1]
    evaluate_json(json_path)