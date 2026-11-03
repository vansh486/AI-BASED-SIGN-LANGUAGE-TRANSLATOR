# AI-Based Sign Language Translator

## Project overview

This project is being developed toward a two-way sign-language communication system. The first working stage recognizes **one isolated sign at a time** from a webcam. Continuous translation, speech processing, and the 3D avatar are future stages.

The current recognition labels are determined by the folders in `data/processed`. The sample data in this repository has five labels: `hello`, `help`, `no`, `thank_you`, and `yes`.

## Proposed full architecture

```text
SIGN TO SPEECH
Camera -> OpenCV -> MediaPipe landmarks -> ST-GCN -> Transformer
       -> Gloss sequence -> NLP -> English text -> Text-to-speech

SPEECH TO SIGN
Voice -> Speech-to-text -> NLP -> Sign glosses
      -> Motion generation -> 3D motion -> Three.js avatar
```

Only the isolated-sign recognition portion is currently implemented. This README distinguishes working code from planned blocks so the project status is clear.

## Implemented recognition pipeline

```text
Webcam -> OpenCV -> MediaPipe pose and hand landmarks
       -> ST-GCN -> temporal Transformer -> sign class prediction
```

1. OpenCV reads webcam frames.
2. MediaPipe Holistic extracts 33 pose landmarks and 21 landmarks for each hand.
3. Each frame is represented by 258 values. The dataset loader resamples clips to 90 frames.
4. The ST-GCN treats the body and hands as a graph of 75 joints. It aggregates connected-joint information and applies temporal convolutions to learn movement over time.
5. A Transformer models longer-range frame relationships, and the classifier predicts one label.
6. Webcam inference averages recent probabilities and displays predictions above a confidence threshold.

The older frame-wise landmark encoder is retained as a baseline for comparison. Training defaults to the real graph model with `--model-type stgcn_transformer`; the previous compatibility wrapper has been replaced by graph and temporal convolution layers.

## Project status and limitations

- Implemented: video collection tools, MediaPipe landmark extraction, preprocessing, data validation/resampling, training-only augmentation, a frame-wise baseline, the ST-GCN + Transformer classifier, evaluation metrics, and webcam inference.
- Not yet implemented: continuous gloss recognition, gloss-to-English NLP, speech-to-text, text-to-speech, motion generation, Three.js avatar, a working API, and a web/mobile client. The current API entry point and NLP service are empty.
- The checked-in sample data has 15 clips for each of five classes (75 clips total), no `no_sign` class, and no recorded multi-session provenance. It is suitable for development tests, not for claiming general-purpose performance.
- The existing `models/isolated_sign_classifier.metrics.json` report is from the earlier baseline: 66.7% accuracy and 0.651 macro F1 on 15 test samples. It is **not** a result for the new ST-GCN model. Train the graph model and evaluate it before reporting its performance.
- Training supports clip-, session-, and signer-disjoint splits. The current checked-in data has no provenance manifest, so only the clip split can currently be used with it. For new documented recordings, prefer a session- or signer-based split; every class must appear in at least three groups.
- Model checkpoints, raw videos, and collection metadata are ignored by Git. A fresh clone may not include local datasets or trained models.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/data/record_videos.py` | Records labeled videos and documents collection conditions. |
| `src/data/landmarks.py` | Extracts pose and hand landmarks with MediaPipe. |
| `src/data/preprocess.py` | Converts raw videos to landmark NumPy sequences. |
| `src/data/loader.py` | Validates, loads, resamples, and splits landmark sequences. |
| `src/data/augment.py` | Applies geometric and temporal augmentation to training data. |
| `src/data/audit.py` | Checks dataset size, metadata, and recording variation. |
| `src/models/stgcn.py` | Defines the 75-joint graph and spatial-temporal graph convolution. |
| `src/models/hybrid.py` | Defines the baseline and ST-GCN + Transformer classifiers. |
| `src/models/transformer.py` | Encodes frame order with a temporal Transformer. |
| `src/training/` | Trains classifiers and reports held-out metrics. |
| `src/inference/predict.py` | Runs real-time webcam predictions from a saved checkpoint. |
| `src/utils/` | Saves and validates checkpoints and metadata. |
| `tests/` | Tests the data pipeline, augmentation, and models. |
| `api/`, `frontend/`, `notebooks/` | Future application and experimentation areas; currently incomplete. |

## Setup

Use Python 3.10 or later. Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Collect and prepare data

Record a documented batch of ten videos per class in one session. The recorder's default class list includes `no_sign`.

```powershell
.\.venv\Scripts\python.exe -m src.data.record_videos --samples-this-session 10 --signer-id signer_01 --session-id session_01 --lighting bright_indoor --distance medium --clothing casual --speed normal --camera-angle frontal
```

Repeat across different sessions and, where possible, different signers and conditions. Include neutral posture and non-target movement in `no_sign` recordings.

Extract landmarks and audit the result:

```powershell
.\.venv\Scripts\python.exe -m src.data.preprocess
.\.venv\Scripts\python.exe -m src.data.audit --strict
```

The strict audit requires at least 40 samples per class, a `no_sign` class, at least five sessions per class, variation in collection conditions, and provenance records. The current sample dataset does not pass this audit. Collect data before relying on model results.

Each processed `.npy` sequence has shape `[frames, 258]`, with finite values and at least one frame. Training resamples each sequence to the configured length.

## Train and evaluate

The default command trains the ST-GCN + Transformer model using session-disjoint splits and saves it separately from the baseline:

```powershell
.\.venv\Scripts\python.exe -m src.training.train
```

Default graph-model checkpoint: `models/stgcn_transformer.pt`. Its report is saved beside it as `models/stgcn_transformer.metrics.json`.

Session-based splitting is the default and uses `data/metadata/processed_manifest.json` to keep every session entirely within one split. Use `--split-by signer` to keep signer identities disjoint instead. Each class must have examples across at least three distinct selected groups. If using the old sample data without provenance, pass `--split-by clip`; its result is not a signer/session generalization estimate.

To train the original frame-wise baseline for comparison:

```powershell
.\.venv\Scripts\python.exe -m src.training.train --model-type baseline
```

The baseline checkpoint is `models/isolated_sign_classifier.pt`. You can override either output path with `--checkpoint`. Compare test macro F1, per-class results, and confusion matrices using the same data split; do not compare training accuracy.

## Run webcam recognition

After training the graph model:

```powershell
.\.venv\Scripts\python.exe -m src.inference.predict --threshold 0.75
```

The default checkpoint is `models/stgcn_transformer.pt`. To use another checkpoint:

```powershell
.\.venv\Scripts\python.exe -m src.inference.predict --checkpoint models/isolated_sign_classifier.pt
```

Press `q` to close the webcam window. A trained `no_sign` class is needed for the model to learn a background/no-target category; the confidence threshold alone is not a substitute for that data.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Next development stages

1. Collect a larger, diverse isolated-sign dataset, including `no_sign`, and keep signer/session metadata.
2. Train and compare the ST-GCN and baseline with signer-held-out evaluation; tune using validation data and keep the final test group untouched.
3. Collect continuous signing videos with time-aligned gloss labels before developing continuous gloss recognition.
4. Build and evaluate gloss-to-English and speech-to-gloss stages. These require representative examples and a defined gloss vocabulary.
5. Create or source properly licensed sign-motion assets, map glosses to avatar motions, and then implement the Three.js avatar.
6. Add the API and user-facing web/mobile application after the recognition and translation components have stable interfaces.
