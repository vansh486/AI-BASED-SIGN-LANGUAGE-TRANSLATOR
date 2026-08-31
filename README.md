# AI-Based Sign Language Translator

This repository implements a reproducible Phase 0 isolated-sign baseline and the Phase 1 data-collection and landmark-augmentation tooling. It accepts a short clip, extracts 258 MediaPipe pose-and-hand features per frame, and predicts one sign class.

It is not yet a continuous sign translator. It does not use CTC, a real ST-GCN, gloss-to-English translation, an API, or a client application. Those are later phases and must be evaluated separately.

## Phase 0 contract

- Input: one isolated sign recorded in a short clip.
- Features: 33 pose landmarks with visibility, plus 21 left-hand and 21 right-hand landmarks.
- Feature dimension: 258 values per frame.
- Sequence policy: every clip is temporally resampled to 90 frames. No clip tail is silently truncated.
- Model: landmark feature encoder, positional Transformer, attention pooling, and classification head.
- Evaluation: deterministic class-stratified train/validation/test splits with accuracy, macro F1, per-class metrics, and a confusion matrix.
- Model artifact: a versioned checkpoint containing model settings, classes, sequence length, metrics, and weights.

## Phase 1: data collection and landmark augmentation

Phase 1 code is ready, but the current dataset is not ready: it has 15 clips for each of five classes, no `no_sign` class, and no collection provenance. Do not retrain or claim Phase 1 results until the audit passes.

### Collection standard

- Collect 50 clips per class, including `no_sign`; 40 is the enforced minimum.
- Use at least five sessions with a small batch per class in each session, instead of recording all 50 under one condition.
- Vary lighting, camera distance, clothing, signer speed, camera angle, and ideally signer identity between sessions.
- `no_sign` clips must include neutral posture, ordinary movement, hands entering/exiting the frame, and non-target gestures. Do not perform any target sign.
- Every raw clip is recorded in `data/metadata/collection_manifest.json` with its session and conditions. Preprocessing creates `data/metadata/processed_manifest.json`, linking each `.npy` sample back to its source video.

Example: capture ten clips of every class for one documented session. Repeat this command for five varied sessions, changing the condition values each time.

```powershell
.\venv\Scripts\python -m src.data.record_videos --samples-this-session 10 --signer-id signer_01 --session-id session_01 --lighting bright_indoor --distance medium --clothing dark_long_sleeve --speed normal --camera-angle frontal
```

When resuming collection, use the default target mode to fill any class that still has fewer than 50 raw clips:

```powershell
.\venv\Scripts\python -m src.data.record_videos --signer-id signer_02 --session-id session_06 --lighting low_indoor --distance far --clothing light_short_sleeve --speed fast --camera-angle slight_left
```

Extract landmarks and preserve the provenance manifest:

```powershell
.\venv\Scripts\python -m src.data.preprocess
```

Audit the processed dataset before training. `--strict` returns a non-zero status until every class, including `no_sign`, meets the sample and provenance requirements.

```powershell
.\venv\Scripts\python -m src.data.audit --strict
```

### Augmentation policy

Augmentation runs only for the training split and creates fresh variants each epoch. It never writes duplicate `.npy` files and does not touch validation or test samples. The default policy applies conservative image-plane rotation, body-centered scale jitter, coordinate noise, frame replacement for dropped frames, and temporal rate warping. Pose visibility and absent-hand zero values are preserved. Horizontal flipping is deliberately excluded because it is not reliably label-preserving for sign language.

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

For meaningful results, collect at least 40 to 50 clips for every sign and for `no_sign`. The current processed dataset has 15 clips per target class and no `no_sign` class, so it is useful only for a smoke-test baseline, not a reliable live system.

## Workflow

Train the baseline after the Phase 1 audit passes. Landmark augmentation is enabled by default and its exact configuration is stored inside the checkpoint. The best validation checkpoint is saved atomically, then evaluated once on the held-out test set:

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
2. Train a real graph model or improve the isolated-sign model if the augmented baseline is insufficient.
3. Move to continuous signing only after collecting continuous sequences with time-aligned gloss labels; then CTC becomes appropriate.
4. Add gloss-to-text, FastAPI, a client, profiling, ONNX export, and deployment after the recognition contract is stable.
