# AI-Based Sign Language Translator

This project is a working prototype for recognizing **one sign at a time**. It has a React website, a FastAPI backend, MediaPipe landmark extraction, and a trained ST-GCN + Transformer classifier.

## A simple explanation for a project presentation

> “Our project recognizes an isolated sign from a short video. OpenCV reads the video, MediaPipe extracts body and hand points, and an ST-GCN plus Transformer model classifies the movement. The result is shown as a word and can be spoken by the browser. We have two separate recognition models: a five-sign model trained on our local examples, which is the default for live camera and uploaded videos, and an optional 50-label INCLUDE-50 model for uploaded videos. The reverse screen can look up supported words as sign glosses.”

## What the system does

| Part | Implemented behavior |
| --- | --- |
| Sign recognition | Predicts one label from a short video; it does not continuously translate a stream of signs into a sentence. |
| Live camera | The website records a short camera clip and sends it to the custom five-label model. |
| Uploaded video | Choose either the custom five-label model (default, recommended for local signs) or INCLUDE-50. |
| Landmark processing | OpenCV reads frames; MediaPipe Holistic extracts pose and hand landmarks. |
| Classifier | The default model uses a spatial-temporal graph convolution network (ST-GCN), a temporal Transformer, and a classifier head. |
| English and audio | The API formats the predicted label as a readable English word. The browser's built-in speech synthesis can speak that word when requested. This is not sentence translation. |
| Speech-to-sign screen | Browser dictation can fill the text box; the user submits it to a fixed word/phrase-to-gloss lookup. Unsupported words are reported. |
| 3D avatar | A procedural Three.js character can idle, rotate, and reset. It does **not** perform sign-language animations. |

## Architecture and data flow

### Sign video to a spoken word

```text
Camera capture or uploaded video
        │
        ▼
React frontend ── video file ──> FastAPI
                                  │
                                  ├─ OpenCV reads frames
                                  ├─ MediaPipe extracts pose + hand landmarks
                                  ├─ resample sequence to the checkpoint length
                                  ├─ ST-GCN encodes connected joints and movement
                                  ├─ Transformer models longer temporal relationships
                                  └─ classifier returns one sign label
        ▲
        └── gloss + confidence + English display word
                │
                └── browser speech synthesis (when the user presses Speak)
```

### Text or dictation to a gloss preview

text
Typed text ─┐
            ├─> browser dictation/text -> FastAPI vocabulary lookup -> gloss chips
Voice input ┘                                                     └-> idle 3D avatar


The browser's voice recognition only fills the text field; the user still presses **Prepare signs**. The API matches known words and phrases, lists unmatched words, and returns an empty `motions` list. There is no sentence-aware translation model or avatar motion playback.

### What one model input contains

MediaPipe landmarks are stored as a sequence with shape `[frames, 258]`:

- 33 pose landmarks × 4 values (`x`, `y`, `z`, visibility) = 132 values.
- 21 left-hand landmarks × 3 values = 63 values.
- 21 right-hand landmarks × 3 values = 63 values.
- Total: 258 values per frame and 75 body/hand joints.

The loader validates each sequence and resamples it to 90 frames by default. The graph representation adds a presence value for each hand joint, then uses the pose and finger connections to aggregate neighboring joints. Training-only augmentation can apply small geometric changes, landmark noise, frame drops, and temporal warping. The graph encoder includes temporal convolutions; a Transformer then models relationships across the sequence.

## The two datasets and models

The datasets are **not merged**. Each checkpoint has its own labels and is selected for a different workflow.

| Workflow | Training data | Labels | Checkpoint expected by the API |
| --- | --- | ---: | --- |
| Live camera | Local recorded dataset | 5: `hello`, `help`, `no`, `thank_you`, `yes` | `models/stgcn_transformer.pt` |
| Uploaded video (default) | Local recorded dataset | 5 | `models/stgcn_transformer.pt` |
| Uploaded video (optional) | INCLUDE-50 | 50 | `models/include50_stgcn_transformer.pt` |

The API loads both checkpoints when it starts. Therefore, both checkpoint files must be available unless the checkpoint configuration is changed. Each model can only predict a label from its own vocabulary.

The local processed data currently contains 75 sequences (15 examples for each of the five labels). INCLUDE-50's local processed set has 943 indexed sequences: 675 train, 77 validation, and 191 test, using its recorded official split. These datasets, source videos, manifests, and model checkpoints are local machine assets; most are ignored by Git. A new clone does not necessarily contain them.

### Existing evaluation results

| Model | Reported evaluation | Important caveat |
| --- | --- | --- |
| Local five-sign ST-GCN + Transformer | 80% accuracy and 79.8% macro-F1 on 15 held-out clips | Small clip-level test. It is not signer-independent and does not establish performance on unseen people. |
| INCLUDE-50 ST-GCN + Transformer | 89.5% accuracy and 85.8% macro-F1 on its official 191-clip test split | Different dataset, vocabulary, and evaluation split from the local model. |
| Earlier frame-wise baseline | A saved report records 66.7% accuracy and 0.651 macro-F1 on 15 test clips | Older result; it is not the graph model result. Compare models only when they use the same data and split. |

These are project-specific reported measurements, not directly comparable scores. They do not prove that the system is more accurate than another research project.

## Research background and honest comparison

The design uses established ideas rather than claiming a new recognition algorithm:

1. **ST-GCN:** Yan, Xiong, and Lin introduced spatial-temporal graph convolution for skeleton-based action recognition. This project applies the general idea of representing a body skeleton as connected joints and modeling both spatial and temporal structure to MediaPipe pose/hand landmarks. It is a project implementation, not a reproduction of their full experiments. [Paper: Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition (AAAI 2018)](https://arxiv.org/abs/1801.07455).
2. **Transformer:** The temporal encoder follows the attention-based sequence-modeling approach introduced by Vaswani et al. [Paper: Attention Is All You Need (NeurIPS 2017)](https://arxiv.org/abs/1706.03762).
3. **Sign-language recognition context:** Published work evaluates different datasets, signing tasks, label vocabularies, and metrics. The review by Adaloglou et al. is useful background on those differences. [A Comprehensive Study on Deep Learning-Based Methods for Sign Language Recognition, IEEE Transactions on Multimedia 24 (2022), 1750–1762](https://doi.org/10.1109/TMM.2021.3070438).
4. **Landmark extraction:** MediaPipe provides the pose/hand landmark tracking used by this project; it supplies features, not sign meanings. [MediaPipe Holistic documentation](https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker).

### What this project contributes as an engineering prototype

- A complete path from labeled videos to landmarks, model training/evaluation, API inference, and a browser UI.
- A graph-based pose/hand classifier combined with a temporal Transformer.
- Separate checkpoint routing for a small locally recorded vocabulary and INCLUDE-50.
- A browser speech output for the recognized single word and an explicitly limited text-to-gloss preview.

These are useful integration and project contributions, but **they are not evidence of research superiority or algorithmic novelty**. The reported scores use different datasets and splits, and the local test set is very small. A fair comparison would require the same dataset, identical train/validation/test protocol, comparable preprocessing, documented baseline implementations, repeated seeds, per-class results, and signer-disjoint evaluation. A research paper should also report limitations and avoid generalizing beyond the tested signers and labels.

## Project layout and file-by-file guide

### Root and data

| Path | Purpose |
| --- | --- |
| `README.md` | This guide: project status, data/model flow, setup, research context, limitations, and file directory. |
| `requirements.txt` | Python dependencies for MediaPipe, OpenCV, PyTorch, FastAPI, and notebook tools. |
| `.gitignore` | Excludes virtual environments, raw/private datasets, metadata, model checkpoints, generated frontend files, and notebook checkpoints. |
| `data/processed/<label>/<sample>.npy` | Tracked local examples. Each file stores one variable-length `[frames, 258]` landmark sequence for a sign label. |
| `data/raw_videos/<label>/` | Local source recordings used to create the custom processed samples; ignored by Git. |
| `data/metadata/` | Local collection and preprocessing manifests; used for auditing and signer/session-aware splits; ignored by Git. |
| `data/include50/` | Local INCLUDE-50 index, videos, processed sequences, and official-split manifest; ignored by Git. |
| `models/` | Local checkpoints and training metric reports expected by the API; model files are not tracked. |

### Backend API

| File | Purpose |
| --- | --- |
| `api/main.py` | Creates the FastAPI app, loads both models on startup, configures CORS, and defines health, model-info, video prediction, text-to-gloss, and WebSocket routes. |
| `api/services/recognition.py` | Loads a recognition checkpoint, extracts landmarks from uploaded/live frames, validates media limits, predicts labels, and selects configured model paths/devices. |
| `api/services/nlp.py` | Formats one gloss as an English word and maps a limited set of text words/phrases to supported glosses. It is a lookup helper, not a full NLP translator. |
| `api/__init__.py` | Marks `api` as a Python package. |
| `api/services/__init__.py` | Marks API services as a Python package. |

### Data preparation

| File | Purpose |
| --- | --- |
| `src/data/record_videos.py` | Captures labeled camera clips and records signer/session and collection-condition metadata in a manifest. |
| `src/data/landmarks.py` | Uses MediaPipe Holistic to extract the 258 pose/hand values per frame and optionally draw landmarks. |
| `src/data/preprocess.py` | Converts locally recorded videos into atomic `.npy` landmark sequences and a provenance manifest. |
| `src/data/import_include50.py` | Validates the INCLUDE-50 index/video paths, extracts landmark sequences, and preserves official split labels. |
| `src/data/loader.py` | Validates and loads landmark sequences, resamples time, creates clip/group/official splits, and supports transformed datasets. |
| `src/data/augment.py` | Applies optional landmark-space geometry and temporal augmentation to training sequences only. |
| `src/data/audit.py` | Checks local dataset size, provenance, no-sign class, session count, and observed collection-condition diversity. |

### Models, training, and inference

| File | Purpose |
| --- | --- |
| `src/models/stgcn.py` | Builds the 75-node pose/hand graph, converts flat landmarks to graph tensors, and implements graph plus temporal convolution blocks. |
| `src/models/transformer.py` | Implements positional encoding and the temporal Transformer encoder. |
| `src/models/hybrid.py` | Implements both classifiers, attention pooling, and the model factory. `STGCNTransformerClassifier` is the default graph model; `IsolatedSignClassifier` is the frame-wise baseline. |
| `src/models/graph.py` | Empty placeholder; it currently contains no implementation and is not imported by the active pipeline. |
| `src/models/baseline.py` | Empty placeholder; the actual frame-wise baseline is implemented in `src/models/hybrid.py`. |
| `src/training/train.py` | Seeds training, creates data splits/loaders, trains a selected classifier, saves the best validation checkpoint, and reports held-out test metrics. |
| `src/training/evaluate.py` | Computes loss, accuracy, macro-F1, per-class precision/recall/F1, and confusion matrix. |
| `src/inference/predict.py` | Runs standalone webcam inference with probability smoothing and a confidence threshold. |
| `src/utils/checkpoints.py` | Saves checkpoints atomically and validates checkpoint schema and architecture metadata when loading. |
| `src/utils/io.py` | Provides atomic JSON save and JSON-object loading helpers for manifests/reports. |
| `src/utils/__init__.py` | Marks shared utilities as a Python package. |

### Frontend

| File | Purpose |
| --- | --- |
| `frontend/web/index.html` | Browser HTML shell and application title/metadata. |
| `frontend/web/package.json` | Frontend scripts and React, Three.js, TypeScript, and Vite dependencies. |
| `frontend/web/package-lock.json` | Exact npm dependency resolution for repeatable installs. |
| `frontend/web/.env.example` | Example API base URL setting. |
| `frontend/web/vite.config.ts` | Configures Vite and its React plugin. |
| `frontend/web/tsconfig.json` | Strict TypeScript compiler settings for the frontend source. |
| `frontend/web/tsconfig.node.json` | TypeScript project settings for Vite's Node-side configuration. |
| `frontend/web/src/main.tsx` | React entry point; mounts the app and imports styles. |
| `frontend/web/src/App.tsx` | Main UI and workflows: camera recording, uploaded-video prediction, dictation, text-to-gloss requests, result display, and browser speech synthesis. |
| `frontend/web/src/api.ts` | Typed REST/WebSocket helpers and request/response shapes for the backend. |
| `frontend/web/src/AvatarViewer.tsx` | Procedural Three.js avatar viewer with idle movement, pointer rotation, and reset controls; contains no sign animations. |
| `frontend/web/src/styles.css` | Layout, responsive styling, colors, and component/3D-viewer styles. |
| `frontend/web/src/vite-env.d.ts` | Vite client type declarations for TypeScript. |

### Tests and notebooks

| Path | Purpose/status |
| --- | --- |
| `tests/test_data_pipeline.py` | Tests sequence resampling, dataset loading, split behavior, provenance, and INCLUDE-50 index validation. |
| `tests/test_augment.py` | Tests deterministic augmentation, output shape/type, and preservation of missing hand landmarks. |
| `tests/test_model.py` | Tests graph topology/conversion and forward/backward behavior of both classifiers. |
| `tests/test_recognition.py` | Tests gloss formatting, text-to-gloss lookup, model configuration, and recognition edge cases. |
| `notebooks/01_data_exploration.ipynb` | Empty file; no runnable exploration notebook is currently included. |
| `notebooks/02_landmark_extraction.ipynb` | Empty file; no runnable extraction notebook is currently included. |
| `notebooks/03_baseline.ipynb` | Empty file; no runnable baseline notebook is currently included. |
| `notebooks/04_model_training.ipynb` | Empty file; no runnable training notebook is currently included. |

The active data/training pipeline is implemented in Python modules under `src/`; the notebooks are not part of the working pipeline.

## Setup and run

### 1. Install Python dependencies

Run from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Start the API

The default API configuration expects both trained checkpoints under `models/`. If they are not present, the API reports a startup error; a fresh clone must be supplied with the local checkpoint files or have them trained first.

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Default checkpoint paths:

- `models/stgcn_transformer.pt` for the custom five-sign model.
- `models/include50_stgcn_transformer.pt` for INCLUDE-50.

Optional PowerShell configuration:

```powershell
$env:SIGN_CUSTOM_MODEL_CHECKPOINT = "models/stgcn_transformer.pt"
$env:SIGN_INCLUDE50_MODEL_CHECKPOINT = "models/include50_stgcn_transformer.pt"
$env:SIGN_MODEL_DEVICE = "auto" # or "cpu" / "cuda"
```

The API accepts the older `SIGN_MODEL_CHECKPOINT` setting as a shared fallback when profile-specific settings are unset. The default frontend origins include Vite's usual `http://localhost:5173` and `http://127.0.0.1:5173` plus fallback port `5174`. If Vite selects another port, set `SIGN_FRONTEND_ORIGINS` to a comma-separated origin list before starting the API. Interactive API documentation is at `http://127.0.0.1:8000/docs`.

### 3. Start the website

Open a second PowerShell terminal:

```powershell
Set-Location frontend\web
npm ci
npm run dev
```

Open the localhost URL printed by Vite. To use a different backend, copy `.env.example` to `.env.local`, set `VITE_API_BASE_URL`, and restart Vite. Camera access requires browser permission and a secure context such as localhost or HTTPS.

Build and type-check the frontend with:

```powershell
npm run build
```

## API routes

| Method and route | What it does |
| --- | --- |
| `GET /api/v1/health` | Confirms the recognition models are available. |
| `GET /api/v1/model` | Reports model profiles, architectures, labels, and available features. |
| `POST /api/v1/predict-video` | Accepts MP4, MOV, or WebM, up to 25 MB and 500 frames. Use multipart fields `file`, `model` (`custom` or `include50`, defaults to `custom`), and optional `mirror_horizontal` (`true` for horizontally mirrored selfie recordings). |
| `POST /api/v1/speech-to-sign` | Accepts JSON text and returns known glosses, unmatched words, and an empty motion list. |
| `WS /ws/recognize` | Optional WebSocket client path for custom-model live recognition. Send `{"type":"capture"}` and binary JPEG frames; the API processes 20 frames. The React camera currently uses the REST video endpoint instead. |

The video endpoint extracts landmarks on the backend, selects the requested checkpoint, and responds with a gloss, confidence score, and top alternatives. The uploaded-video selector defaults to the custom model because the local webcam clips and five-sign vocabulary match that model; choose INCLUDE-50 only for clips from that dataset/domain. Some selfie-camera recordings are horizontally mirrored. If the clip visibly appears mirrored and the prediction is wrong, enable the mirror option to flip frames before landmark extraction. Confidence is a model score, not a guarantee of correctness. A no-hand input is reported rather than automatically assigned a sign.

## Collect, prepare, train, and test

### Custom videos

The recording tool defaults to five labels plus `no_sign`. Example: record ten examples per class for one documented signer/session:

```powershell
.\.venv\Scripts\python.exe -m src.data.record_videos --samples-this-session 10 --signer-id signer_01 --session-id session_01 --lighting bright_indoor --distance medium --clothing casual --speed normal --camera-angle frontal
```

Extract landmarks and audit the dataset:

```powershell
.\.venv\Scripts\python.exe -m src.data.preprocess
.\.venv\Scripts\python.exe -m src.data.audit --strict
```

The strict audit requires at least 40 samples per class, a `no_sign` class, five sessions per class, varied collection conditions, and complete provenance. The currently checked-in five-class dataset does not satisfy this research-ready collection requirement.

### INCLUDE-50 videos

With the dataset and index CSV placed under `data/include50/`, import videos and retain the source split:

```powershell
.\.venv\Scripts\python.exe -m src.data.import_include50
```

The importer writes to `data/include50/processed/` and `data/include50/processed_manifest.json`. A smoke test can use `--limit 3`; a partial import must not be used for final training or evaluation.

### Train the models

Local five-sign model (the current local samples lack provenance, so use a clip-level split):

```powershell
.\.venv\Scripts\python.exe -m src.training.train --data-dir data/processed --split-by clip --checkpoint models/stgcn_transformer.pt
```

INCLUDE-50 model (use the source dataset's official split):

```powershell
.\.venv\Scripts\python.exe -m src.training.train --data-dir data/include50/processed --processed-manifest data/include50/processed_manifest.json --split-by official --checkpoint models/include50_stgcn_transformer.pt
```

Training defaults to `stgcn_transformer`; pass `--model-type baseline` to train the frame-wise baseline. The trainer saves the best validation checkpoint and a `.metrics.json` report beside it. The test split should be reserved for final evaluation rather than model tuning. For future custom recordings, prefer `--split-by signer` or `--split-by session`; these require enough documented groups for every class.

Standalone Python webcam demo:

```powershell
.\.venv\Scripts\python.exe -m src.inference.predict --checkpoint models/stgcn_transformer.pt
```

Press `q` to exit.

Run the Python tests from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Limitations and next research steps

1. Record more custom examples from multiple signers, sessions, lighting conditions, distances, and camera angles; include a well-defined `no_sign` class.
2. Use signer-held-out evaluation for claims about unseen people. Report per-class metrics, confusion matrices, and multiple-run variability.
3. Compare the ST-GCN + Transformer against the included frame-wise baseline on the **same data, preprocessing, split, and evaluation protocol**.
4. For continuous translation, collect continuous signing with time-aligned gloss annotations and develop/test a sentence-level recognition and translation model.
5. Replace the fixed word lookup with an appropriately evaluated language/gloss model; connect licensed, reviewed sign motions to avatar animation before claiming sign generation.
6. Add broader integration tests and deployment configuration after the model/data workflow is stable.

## Questions a supervisor may ask

**Is this a proper ST-GCN and Transformer, or simple code?** The active model has a graph over pose/hand landmarks, learnable graph edge weighting, spatial aggregation, temporal graph convolutions, a Transformer encoder, attention pooling, and a classifier head. It is implemented in this repository rather than using an off-the-shelf pretrained sign-language model.

**Are both datasets used in one model?** No. They produce separate checkpoints and label spaces; the live camera uses the custom model, and uploaded files default to custom with INCLUDE-50 available as a selectable alternative.

**Does the reverse direction translate speech into signs?** Not fully. Dictation is a browser feature, and the API only looks up supported words/phrases as gloss labels. There is no grammar-aware translation or signing avatar animation.

**Can we say our model is better than published research?** Not from the current results. A defensible comparison requires a shared benchmark and evaluation protocol. Current results are evidence of prototype performance only.

**What is the next most important step?** Expand and document signer-diverse data, then evaluate on signers excluded from training before making real-world accuracy claims.
