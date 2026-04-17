import os
import glob
import numpy as np
import madmom
import time
from multiprocessing import Pool, cpu_count

AUDIO_DIR = './GTZAN/'
ANNOT_DIR = './GTZAN_annotations/beats'
RNN_PROC = None
DBN_PROC = None

def load_annotations(annot_path):
    """
    Loads beat annotations. 
    Handles files that may have [timestamp, beat_index] or just [timestamp].
    """
    data = np.loadtxt(annot_path)
    if data.ndim > 1:
        return data[:, 0]  # Take the first column (timestamps)
    return data


def _init_worker():
    """Initialize Madmom processors once per worker process."""
    global RNN_PROC, DBN_PROC
    RNN_PROC = madmom.features.beats.RNNBeatProcessor()
    DBN_PROC = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)


def _process_track(audio_path):
    file_id = os.path.basename(audio_path).replace('.wav', '').replace('.', '_')
    annot_path = os.path.join(ANNOT_DIR, f"gtzan_{file_id}.beats")

    if not os.path.exists(annot_path):
        return ("skip", file_id, None, None, None)

    try:
        activations = RNN_PROC(audio_path)
        detections = DBN_PROC(activations)
        targets = load_annotations(annot_path)
        return ("ok", file_id, detections, targets, None)
    except Exception as e:
        return ("error", file_id, None, None, str(e))

def run_evaluation():
    print ("Start evaluation time:", time.time())
    all_evals = []

    # 2. Identify all audio files in GTZAN (recursively)
    search_pattern = os.path.join(AUDIO_DIR, "**/*.wav")
    audio_files = sorted(glob.glob(search_pattern, recursive=True))

    print(f"{'Track ID':<25} | {'F-Measure':<10} | {'Cemgil':<10} | {'Information'}")
    print("-" * 70)

    workers = 4
    chunksize = max(1, len(audio_files) // (workers * 4)) if audio_files else 1

    with Pool(processes=workers, initializer=_init_worker) as pool:
        for status, file_id, detections, targets, err in pool.imap_unordered(
            _process_track, audio_files, chunksize=chunksize
        ):
            if status == "skip":
                continue
            if status == "error":
                print(f"Error processing {file_id}: {err}")
                continue

            # Evaluate in main process and keep the original output format.
            eval_obj = madmom.evaluation.beats.BeatEvaluation(detections, targets)
            all_evals.append(eval_obj)
            print(f"{file_id:<25} | {eval_obj.fmeasure:<10.3f} | {eval_obj.cemgil:<10.3f}")

    if all_evals:
        mean_eval = madmom.evaluation.beats.BeatMeanEvaluation(all_evals)
        
        print("-" * 70)
        print("FINAL MEAN RESULTS:")
        print(f"F-Measure: {mean_eval.fmeasure:.4f}")
        print(f"Cemgil:    {mean_eval.cemgil:.4f}")
        print(f"P-Score:   {mean_eval.pscore:.4f}")

    print ("End evaluation time:", time.time())

if __name__ == "__main__":
    run_evaluation()