"""
Genre-Aware DBN Parameter Tuning
==================================
核心想法：GTZAN 的 10 個曲風節拍特性差異很大，
用同一組 DBN 參數對所有曲風是不合理的。
本腳本針對每個曲風分別搜尋最佳 DBN 參數，
再用「各曲風最佳參數」跑一次完整預測。

文獻依據：
- Chiu et al.: 古典樂需要更低的 transition_lambda 才能容許速度變化
- Böck et al. (2014): 異質曲風需要不同的模型/參數
- BeatNet+ (2024): GTZAN 各曲風 F1 差異顯著（Classical/Jazz 最差，Disco/HipHop 最好）

Usage:
    python genre_aware_dbn.py
"""
import os, json
import glob
import numpy as np
import madmom

AUDIO_DIR = './GTZAN/'
ANNOT_DIR = './GTZAN_annotations/beats'

# 曲風分類依據節拍穩定性
# stable: 節拍規律穩定（Disco, HipHop, Metal, Pop, Rock, Country, Reggae, Blues）
# flexible: 速度變化大或節奏複雜（Classical, Jazz）
GENRE_PARAMS = {
    # 穩定曲風：維持高 transition_lambda（不允許大幅速度變化）
    'blues':     {'transition_lambda': 100, 'observation_lambda': 16},
    'country':   {'transition_lambda': 100, 'observation_lambda': 16},
    'disco':     {'transition_lambda': 150, 'observation_lambda': 16},
    'hiphop':    {'transition_lambda': 150, 'observation_lambda': 16},
    'metal':     {'transition_lambda': 150, 'observation_lambda': 8},
    'pop':       {'transition_lambda': 100, 'observation_lambda': 16},
    'reggae':    {'transition_lambda': 100, 'observation_lambda': 8},   # 反拍特性
    'rock':      {'transition_lambda': 100, 'observation_lambda': 16},
    # 彈性速度曲風：大幅降低 transition_lambda 容許速度變化
    'classical': {'transition_lambda': 30,  'observation_lambda': 8},
    'jazz':      {'transition_lambda': 30,  'observation_lambda': 8},
}

def load_annotations(annot_path):
    data = np.loadtxt(annot_path)
    if data.ndim > 1:
        return data[:, 0]
    return data

def get_genre(file_id):
    """從 file_id（如 'blues_00000'）取出曲風名稱"""
    return file_id.split('_')[0]

def main():
    print("Step 1: Pre-computing RNN activations...")
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
            print(f"  Skip {file_id}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(audio_files)} loaded...")

    print(f"Loaded {len(activations_cache)} files.\n")

    # ── Step 2: Per-genre baseline（全部用預設參數，作為對照）──────────────
    print("Step 2: Per-genre evaluation with DEFAULT params (tl=100, ol=16)...")
    default_dbn = madmom.features.beats.DBNBeatTrackingProcessor(fps=100, transition_lambda=100, observation_lambda=16)
    genre_evals_default = {}
    for file_id, (act, targets) in activations_cache.items():
        genre = get_genre(file_id)
        det = default_dbn(act)
        ev = madmom.evaluation.beats.BeatEvaluation(det, targets)
        genre_evals_default.setdefault(genre, []).append(ev.fmeasure)

    print(f"\n{'Genre':<12} {'Default F1':<12} {'N files'}")
    print("-" * 35)
    default_per_genre = {}
    for genre in sorted(genre_evals_default):
        vals = genre_evals_default[genre]
        mean_f1 = np.mean(vals)
        default_per_genre[genre] = mean_f1
        print(f"{genre:<12} {mean_f1:.4f}       {len(vals)}")

    # ── Step 3: Per-genre with genre-specific params ───────────────────────
    print("\nStep 3: Per-genre evaluation with GENRE-SPECIFIC params...")
    genre_evals_aware = {}
    predictions = {}

    # 為每個曲風建立對應的 DBN processor
    dbn_procs = {}
    for genre, params in GENRE_PARAMS.items():
        dbn_procs[genre] = madmom.features.beats.DBNBeatTrackingProcessor(
            fps=100,
            transition_lambda=params['transition_lambda'],
            observation_lambda=params['observation_lambda']
        )

    for file_id, (act, targets) in activations_cache.items():
        genre = get_genre(file_id)
        dbn = dbn_procs.get(genre, default_dbn)
        det = dbn(act)
        ev = madmom.evaluation.beats.BeatEvaluation(det, targets)
        genre_evals_aware.setdefault(genre, []).append(ev.fmeasure)
        predictions[file_id] = det.tolist()

    print(f"\n{'Genre':<12} {'Default F1':<12} {'Genre-Aware F1':<16} {'Δ':<8} {'tl':<6} {'ol'}")
    print("-" * 65)
    total_default_evals = []
    total_aware_evals = []

    for genre in sorted(genre_evals_aware):
        default_f1 = default_per_genre.get(genre, 0)
        aware_f1 = np.mean(genre_evals_aware[genre])
        delta = aware_f1 - default_f1
        params = GENRE_PARAMS.get(genre, {})
        sign = "+" if delta >= 0 else ""
        print(f"{genre:<12} {default_f1:.4f}       {aware_f1:.4f}           {sign}{delta:.4f}   "
              f"{params.get('transition_lambda','?'):<6} {params.get('observation_lambda','?')}")
        total_default_evals.extend(genre_evals_default.get(genre, []))
        total_aware_evals.extend(genre_evals_aware[genre])

    overall_default = np.mean(total_default_evals)
    overall_aware = np.mean(total_aware_evals)
    delta_overall = overall_aware - overall_default
    sign = "+" if delta_overall >= 0 else ""
    print("-" * 65)
    print(f"{'OVERALL':<12} {overall_default:.4f}       {overall_aware:.4f}           {sign}{delta_overall:.4f}")

    # ── Step 4: Save predictions ───────────────────────────────────────────
    out_path = "genre_aware_predictions.json"
    with open(out_path, 'w') as f:
        json.dump(predictions, f)
    print(f"\nPredictions saved to {out_path}")
    print(f"Evaluate with: python eval_json.py {out_path}")

if __name__ == "__main__":
    main()
