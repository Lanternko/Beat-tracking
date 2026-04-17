# Beat Tracking on GTZAN — Homework 1 Submission

Final F1-Measure on GTZAN: **88.72%** (Beat This! single model).
Best ensemble: **88.75%** (Beat This! 3-checkpoint majority vote).
Target threshold: > 87.5% — passed by **+1.22%**.

---

## Quick Reproduction

The grader's expected workflow is to run the main script and then evaluate
the predictions against the GTZAN annotations. The dataset layout follows
the homework specification: audio under `./GTZAN/<genre>/<file>.wav` and
annotations under `./GTZAN_annotations/beats/gtzan_<genre>_<XXXXX>.beats`.

```bash
# 1. Install dependencies (Ubuntu 24.04, Python 3.12)
pip install -r requirements.txt

# 2. (Optional) verify CUDA is visible
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 3. Reproduce the main result
python generate_prediction_json.py predictions.json
python eval_json.py predictions.json
# → F1-Measure: 0.8872
```

A pre-computed `predictions.json` is included in this submission so the
grader can verify the score without re-running the (~5-minute on GPU,
~1-hour on CPU) inference.

---

## Environment

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04 LTS |
| Python | 3.12 |
| CUDA | 12.x (Beat This! auto-falls back to CPU if unavailable) |
| GPU | RTX 5090 (development); any CUDA GPU with ≥ 4 GB VRAM works |

Inside `requirements.txt`:
- `numpy<2.0` is **mandatory** — madmom 0.17-dev pickle files cannot
  be loaded under NumPy 2.x (`dtype(align=0)` deprecation breaks
  pickle.load).
- `madmom` and `beat-this` are installed directly from GitHub because
  no PyPI release supports Python 3.12 yet.

If `pip install` from git fails, run them manually:

```bash
pip install "numpy<2.0" scipy cython
pip install git+https://github.com/CPJKU/madmom.git
pip install torch torchaudio einops soxr rotary-embedding-torch
pip install git+https://github.com/CPJKU/beat_this.git
```

---

## File Manifest

### Main submission

| File | Description |
|------|-------------|
| `generate_prediction_json.py` | **Main script** — runs Beat This! (final0) on GTZAN/, writes `predictions.json`. |
| `predictions.json` | Pre-computed Beat This! predictions, **F1 = 88.72%**. |
| `eval_json.py` | Evaluates a predictions JSON against GTZAN_annotations. |
| `requirements.txt` | Python dependencies. |

### Baseline reproduction

| File | Description |
|------|-------------|
| `eval_gtzan.py` | Original baseline (RNN + DBN), single-process. |
| `eval_gtzan_multiprocess.py` | Same as above, multi-process. |
| `eval_one_file.py` | Run baseline on a single audio file (debugging). |
| `baseline.json` | RNN + DBN baseline predictions (F1 = 87.02%). |

### Experiments supporting the report (8 methods total)

| Script | Method | F1 |
|--------|--------|----|
| `beat_crf.py` | RNN + CRFBeatDetectionProcessor | 86.43% |
| `dbn_param_search.py` | RNN + DBN, global grid search | 86.64% |
| `dbn_per_genre.py` | RNN + DBN, per-genre grid search | **88.42%** |
| `genre_aware_dbn.py` | RNN + DBN, literature-driven per-genre params | 87.22% |
| `beat_this_dbn.py` | Beat This! with DBN post-processing (`dbn=True`) | 87.56% |
| `beat_this_checkpoint_ensemble.py` | Beat This! 3-checkpoint majority vote | **88.75%** |
| `beat_this_genre_dbn.py` | Beat This! activation + per-genre DBN (**novel**) | **88.59%** |
| `generate_prediction_json.py` | Beat This! single model (`final0`, no DBN) | **88.72%** |
| `generate_prediction_json_ensemble.py` | RNN + TCN ensemble (incomplete — see Notes) | — |

### Per-genre prediction JSONs

All eight per-method prediction JSONs are included so the grader can
re-run any per-genre comparison without re-computing activations.

| File | Source method |
|------|---------------|
| `predictions.json` | Beat This! |
| `baseline.json` | RNN + DBN baseline |
| `beat_this_dbn_predictions.json` | Beat This! + DBN |
| `beat_this_genre_dbn_predictions.json` | Beat This! + Per-genre DBN |
| `checkpoint_ensemble_predictions.json` | Beat This! Ensemble |
| `crf_predictions.json` | RNN + CRF |
| `dbn_per_genre_predictions.json` | RNN + Per-genre DBN |
| `genre_aware_predictions.json` | RNN + Literature DBN |

### Analysis tool

| File | Description |
|------|-------------|
| `per_genre_analysis.py` | Per-genre F1 breakdown across multiple prediction JSONs. |
| `dbn_per_genre_results.txt` | Captured stdout of `dbn_per_genre.py` for the report. |

### Reproducing the per-genre comparison table (Table 2 in report)

```bash
python per_genre_analysis.py \
    baseline.json crf_predictions.json \
    dbn_per_genre_predictions.json genre_aware_predictions.json \
    predictions.json beat_this_dbn_predictions.json \
    beat_this_genre_dbn_predictions.json checkpoint_ensemble_predictions.json
```

This reads only the JSONs and `GTZAN_annotations/beats/`, so it runs in
under one minute.

---

## Notes for the Grader

1. **`generate_prediction_json.py` requires CUDA** for tractable runtime.
   The script defaults to `device='cuda'`. On CPU-only systems the same
   code works but takes ~1 hour instead of ~5 minutes; flip the device
   argument in `File2Beats(...)` to `'cpu'`.

2. **The first run downloads the Beat This! checkpoint** (~50 MB) to
   `~/.cache/torch/hub/checkpoints/beat_this-final0.ckpt`.

3. **One audio file is corrupt in the public GTZAN distribution**:
   `jazz/jazz.00054.wav`. All scripts handle this with a try/except and
   skip it; expected total file count is 998 (out of 1000).

4. **TCN ensemble was attempted but abandoned**: the
   `generate_prediction_json_ensemble.py` script calls
   `madmom.features.beats.TCNBeatProcessor`, but the TCN model files are
   not bundled in madmom's GitHub release (only BLSTM/LSTM models for
   `beats/2015/` and `beats/2016/` are present). This is documented in
   the report's Discussion section.

5. **Reported F1 numbers in the report (998 files)** were computed via
   `per_genre_analysis.py` which uses madmom's `BeatMeanEvaluation`
   over the full set, identical to what `eval_json.py` does.

---

## Acknowledgements

- `madmom` library: Böck, Krebs, Schedl, Widmer et al., CPJKU.
- `beat_this` library: Böck, Davies, Knees (ISMIR 2024), CPJKU.
- GTZAN dataset: Tzanetakis & Cook, 2002.
