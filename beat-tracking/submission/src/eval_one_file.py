import os
import glob
import numpy as np
import madmom

def load_annotations(annot_path):
    """
    Loads beat annotations. 
    Handles files that may have [timestamp, beat_index] or just [timestamp].
    """
    data = np.loadtxt(annot_path)
    if data.ndim > 1:
        return data[:, 0]  # Take the first column (timestamps)
    return data

def run_evaluation(audio_path, annotation_path):
    all_evals = []
    # 1. Initialize Madmom Processors
    # RNNBeatProcessor: Generates beat activation functions
    # DBNBeatTrackingProcessor: Decodes activations into discrete beat times
    rnn_proc = madmom.features.beats.RNNBeatProcessor()
    dbn_proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)

    activations = rnn_proc(audio_path)
    # This returns the decoded beat timestamps in seconds
    detections = dbn_proc(activations)
    
    # 4. Load Ground Truth
    targets = load_annotations(annotation_path)

    # 5. Evaluate using Madmom's native evaluation class
    # This calculates F-Measure, Precision, Recall, Cemgil, etc.
    eval_obj = madmom.evaluation.beats.BeatEvaluation(detections, targets)
    all_evals.append(eval_obj)

    
    print ("Prediction:")
    print (detections)
    print ("Groundtruth annotation:")
    print (targets)

    print(f"{'Audio path':<25} | {'F-Measure':<10} | {'Cemgil':<10}")
    print("-" * 70)
    print(f"{audio_path:<25} | {eval_obj.fmeasure:<10.3f} | {eval_obj.cemgil:<10.3f}")

    mean_eval = madmom.evaluation.beats.BeatMeanEvaluation(all_evals)
    print("-" * 70)
    print("FINAL MEAN RESULTS:")
    print(f"F-Measure: {mean_eval.fmeasure:.4f}")
    print(f"Cemgil:    {mean_eval.cemgil:.4f}")
    print(f"P-Score:   {mean_eval.pscore:.4f}")

if __name__ == "__main__":
    audio_path = "./GTZAN/blues/blues.00000.wav"
    annotation_path = "./GTZAN_annotations/beats/gtzan_blues_00000.beats"
    run_evaluation(audio_path, annotation_path)