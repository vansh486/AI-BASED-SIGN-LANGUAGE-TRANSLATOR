# AI-Based Sign Language Translator

This repository currently implements Phase 0: a reproducible isolated-sign recognition baseline. It accepts a short clip, extracts 258 MediaPipe pose-and-hand features per frame, and predicts one sign class.

It is not yet a continuous sign translator. It does not use CTC, a real ST-GCN, gloss-to-English translation, an API, or a client application. Those are later phases and must be evaluated separately.

## Phase 0 contract

- Input: one isolated sign recorded in a short clip.
- Features: 33 pose landmarks with visibility, plus 21 left-hand and 21 right-hand landmarks.
- Feature dimension: 258 values per frame.
- Sequence policy: every clip is temporally resampled to 90 frames. No clip tail is silently truncated.
- Model: landmark feature encoder, positional Transformer, attention pooling, and classification head.
- Evaluation: deterministic class-stratified train/validation/test splits with accuracy, macro F1, per-class metrics, and a confusion matrix.
- Model artifact: a versioned checkpoint containing model settings, classes, sequence length, metrics, and weights.

## Setup

Use Python 3.10 or later. From the repository root:

```powershell
.\venv\Scripts\python -m pip install -r requirements.txt
```

## Dataset requirements

Processed samples must follow this structure:

```text
data/processed/
  hello/hello_0.npy
  help/help_0.npy
  no/no_0.npy
  no_sign/no_sign_0.npy
```

Each `.npy` file must have shape `[frames, 258]`, contain finite values, and have at least one frame. Training fails early if a sequence is malformed or if a class has fewer than three samples.

For meaningful results, collect at least 40 to 50 clips for every sign and for `no_sign`. `no_sign` should include neutral posture, hands entering/exiting frame, movement that is not a target sign, and varied backgrounds. The current processed dataset has 15 clips per target class and no `no_sign` class, so it is useful only for a smoke-test baseline, not a reliable live system.

## Workflow

Record 50 new clips per class without overwriting existing video files:

```powershell
.\venv\Scripts\python -m src.data.record_videos
```

Extract landmarks. Existing processed files are skipped unless `--overwrite` is passed:

```powershell
.\venv\Scripts\python -m src.data.preprocess
```

Train the baseline. The best validation checkpoint is saved atomically, then evaluated once on the held-out test set:

```powershell
.\venv\Scripts\python -m src.training.train
```

Read the evaluation report at `models/isolated_sign_classifier.metrics.json`. Do not report training accuracy as model performance; use the held-out test macro F1 and confusion matrix.

Run webcam inference with the matching checkpoint:

```powershell
.\venv\Scripts\python -m src.inference.predict --threshold 0.75
```

Inference only emits a sign above the confidence threshold. It averages the last five probability vectors to reduce flicker. A trained `no_sign` class is still required for reliable rejection.

Run automated checks:

```powershell
.\venv\Scripts\python -m unittest discover -s tests -v
```

## Evaluation discipline

Keep the test set untouched until model choices are final. With videos captured in one session, random clip splits can overestimate performance because lighting, signer, clothing, and background leak across splits. Before claiming generalization, create splits by recording session or signer and test under unseen conditions.

## Next phases

1. Expand and augment landmark data only after the Phase 0 metrics are trustworthy.
2. Train a real graph model or improve the isolated-sign model if the baseline is insufficient.
3. Move to continuous signing only after collecting continuous sequences with time-aligned gloss labels; then CTC becomes appropriate.
4. Add gloss-to-text, FastAPI, a client, profiling, ONNX export, and deployment after the recognition contract is stable.
