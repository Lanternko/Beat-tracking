import os
import glob
import numpy as np
import madmom

AUDIO_DIR = './GTZAN/'
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

def run_evaluation():
    # 1. Initialize Madmom Processors
    # RNNBeatProcessor: Generates beat activation functions
    # DBNBeatTrackingProcessor: Decodes activations into discrete beat times
    rnn_proc = madmom.features.beats.RNNBeatProcessor()
    dbn_proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    
    all_evals = []

    # 2. Identify all audio files in GTZAN (recursively)
    search_pattern = os.path.join(AUDIO_DIR, "**/*.wav")
    audio_files = sorted(glob.glob(search_pattern, recursive=True))

    print(f"{'Track ID':<25} | {'F-Measure':<10} | {'Cemgil':<10} | {'Information'}")
    print("-" * 70)

    for audio_path in audio_files:
        file_id = os.path.basename(audio_path).replace('.wav', '').replace('.', '_')
        
        # Match annotation file from the ISMIR2019 repo
        annot_path = os.path.join(ANNOT_DIR, f"gtzan_{file_id}.beats")
        
        if not os.path.exists(annot_path):
            continue

        try:
            # 3. Process Audio
            # This returns the beat activations (probabilities over time)
            activations = rnn_proc(audio_path)
            # This returns the decoded beat timestamps in seconds
            detections = dbn_proc(activations)
            
            # 4. Load Ground Truth
            targets = load_annotations(annot_path)

            # 5. Evaluate using Madmom's native evaluation class
            # This calculates F-Measure, Precision, Recall, Cemgil, etc.
            eval_obj = madmom.evaluation.beats.BeatEvaluation(detections, targets)
            all_evals.append(eval_obj)

            print(f"{file_id:<25} | {eval_obj.fmeasure:<10.3f} | {eval_obj.cemgil:<10.3f}")

        except Exception as e:
            print(f"Error processing {file_id}: {e}")

    if all_evals:
        mean_eval = madmom.evaluation.beats.BeatMeanEvaluation(all_evals)
        
        print("-" * 70)
        print("FINAL MEAN RESULTS:")
        print(f"F-Measure: {mean_eval.fmeasure:.4f}")
        print(f"Cemgil:    {mean_eval.cemgil:.4f}")
        print(f"P-Score:   {mean_eval.pscore:.4f}")

if __name__ == "__main__":
    run_evaluation()