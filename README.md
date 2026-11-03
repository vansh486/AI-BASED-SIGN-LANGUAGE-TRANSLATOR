# AI-Based Sign Language Translator

> **Start here:** this README explains what works today, how data moves through the project, what each important file does, and what is still needed before the full translator is complete.

## Explain the project to your mentor

### Start with these sections

1. Read the short presentation below to introduce the project.
2. Use [What works now](#what-works-now) and [How a video becomes a model input](#how-a-video-becomes-a-model-input) to explain the implementation.
3. Use [Project status and limitations](#project-status-and-limitations) and [Next development stages](#next-development-stages) to describe remaining work.
4. Use [Repository map: what each part is for](#repository-map-what-each-part-is-for) if your mentor asks about a file.

### Short presentation

> “I am building a sign-language translator with a React website and a Python recognition backend. The implemented part recognizes one isolated sign at a time. We extract body and hand landmarks from video with MediaPipe, then an ST-GCN and Transformer classify the movement. We have two separately trained models: our five-sign model is used for live camera captures, and INCLUDE-50 is used for uploaded videos. The models are not merged. The website sends video to a FastAPI service; speech-to-sign, sentence-level translation, and the 3D avatar are future work.”

### What works now

| Area | Current status |
| --- | --- |
| Sign vocabulary | Two separate vocabularies: our five local signs and INCLUDE-50's 50 labels. |
| Training | ST-GCN + Transformer checkpoints have been trained and evaluated for both datasets. |
| Website and API | React website calls FastAPI for predictions and displays the selected model's result. |
| Camera path | The website records a short camera clip on **Capture sign** and sends it to the custom five-sign model. |
| Uploaded-video path | The user selects a recorded video and clicks **Analyze clip**; the API uses INCLUDE-50. |
| Not complete yet | Continuous signing, backend English translation, a connected speech-to-text-to-sign pipeline, automatic speech output from a recognized sign, and the animated avatar. The website has a speech-to-sign screen and optional browser dictation, but those are not yet connected to a working translation backend. |

### What “using both datasets” means

Both trained models can be loaded by the same API at the same time. The app routes each request to one model:

```text
Live camera capture ──> custom model ──> hello / help / no / thank_you / yes
Uploaded video      ──> INCLUDE-50  ──> one of its 50 labels
```

The datasets and vocabularies are **not combined into one model**. A model can only return labels it was trained on. In particular, the custom model cannot recognize the other 45 INCLUDE-50 signs.

## Full project architecture

This is the intended two-way translator architecture. Only the isolated-sign recognition part described below is implemented end to end.

### Intended two-way pipeline

```text
SIGN TO SPEECH
Camera -> OpenCV -> MediaPipe landmarks -> ST-GCN -> Transformer
       -> Gloss sequence -> NLP -> English text -> Text-to-speech

SPEECH TO SIGN
Voice -> Speech-to-text -> NLP -> Sign glosses
      -> Motion generation -> 3D motion -> Three.js avatar
```

Implemented today: isolated-sign recognition, a web frontend, and a REST API. Sentence translation and speech-to-sign remain planned stages.

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
6. The standalone Python webcam demo smooths recent probabilities and applies a confidence threshold. In the website, a user clicks **Capture sign** to record a short clip; the API reports when no hands were detected instead of forcing a class.

The older frame-wise landmark encoder is retained as a baseline for comparison. Training defaults to the graph model with `--model-type stgcn_transformer`. Graph construction and graph convolution are implemented in `src/models/stgcn.py`; `src/models/graph.py` is currently an empty placeholder.

## Data summary (easy to share)

**In one sentence:** we have two separate video datasets. We turn each video into a sequence of MediaPipe body-and-hand landmarks, then train the sign classifier to predict the video’s label. The React website calls the FastAPI recognition backend; it does not train the model. The two models are selected by workflow, not blended together.

### The two datasets we have

| Dataset | What is in it now | Where it lives | How we use it |
| --- | --- | --- | --- |
| Our recorded sample dataset | 75 short clips: 15 each for `hello`, `help`, `no`, `thank_you`, and `yes`. The 75 processed landmark files are present too. | Videos: `data/raw_videos/`; landmarks: `data/processed/` | Small development dataset. Since these recordings currently have no saved signer/session provenance manifest, use a clip-level split for a basic test; it cannot measure how well the model works for a new signer. |
| INCLUDE-50 | 943 indexed clips in 50 labels. All 943 have been converted to landmark sequences. Its official split is preserved: 675 train, 77 validation, 191 test. | Download/index: `data/include50/`; landmarks: `data/include50/processed/`; split records: `data/include50/processed_manifest.json` | Larger dataset for training and held-out evaluation with INCLUDE-50’s official split. It is processed separately from our recordings. |

These counts were checked against the local files and manifests on **November 3, 2026**. The 75 processed local sample sequences are in the repository. Raw recordings, the INCLUDE-50 archive, its processed sequences, and trained model checkpoints are local/ignored files; a fresh clone needs those files supplied separately to run the API or repeat training.

### How a video becomes a model input

```text
Labeled video
     │
     ▼
Read video frames
     │
     ▼
MediaPipe Holistic finds body pose + left hand + right hand
     │
     ▼
258 numbers per frame ── saved as one .npy sequence per video
     │
     ▼
Validate and resample each clip to 90 frames for training
     │
     ├── training clips: augment slightly, then learn
     ├── validation clips: select the best training checkpoint
     └── test clips: final held-out evaluation
     │
     ▼
ST-GCN + Transformer learns to predict one sign label
```

Each frame has 258 values: 33 pose landmarks (x, y, z, visibility) plus 21 left-hand and 21 right-hand landmarks (x, y, z). That is 75 body/hand points in total. A `.npy` file keeps all frames for one clip; the loader checks its shape and values, then resamples the clip to 90 frames so clips of different lengths fit the model. Augmentation is applied to training data only, not validation or test data.

### How data moves through the running website

```text
Camera button or uploaded video in React
        │  video file + selected model profile
        ▼
FastAPI POST /api/v1/predict-video
        │  OpenCV reads frames; MediaPipe extracts 258 values per frame
        ▼
Landmark sequence → selected checkpoint → top label and alternatives
        │
        ▼
JSON response → React displays gloss and model score
```

The camera and file-upload controls use the same video endpoint but send different model profiles: `custom` for the camera and `include50` for an uploaded file. The API extracts landmarks at prediction time; it does not train from a video sent by the website. It stores the upload temporarily for decoding and removes that temporary file after processing. Training is a separate offline process that reads processed `.npy` sequences and writes checkpoint and metrics files.

### How we train with both datasets

**Each checkpoint is trained on one dataset at a time.** The API can load both checkpoints together and the website routes each type of input to the appropriate model, but the training sets and label vocabularies are not automatically combined. A true combined model would require a deliberately unified label map and a new split/provenance plan.

Train on our 75 locally recorded clips with a simple clip-level split:

```powershell
.\.venv\Scripts\python.exe -m src.training.train --data-dir data/processed --split-by clip --checkpoint models/stgcn_transformer.pt
```

Train on INCLUDE-50 with its official train/validation/test assignments:

```powershell
.\.venv\Scripts\python.exe -m src.training.train --data-dir data/include50/processed --processed-manifest data/include50/processed_manifest.json --split-by official --checkpoint models/include50_stgcn_transformer.pt
```

The graph model is the default. Training uses the training partition to learn, checks validation performance to keep the best checkpoint, and reports final metrics on the held-out test partition. Checkpoints and metrics are saved under `models/`; the INCLUDE-50 test split must not be used to tune the model.

**Current status:** all 943 INCLUDE-50 clips have processed files and official split entries. Training completed with the best validation checkpoint at epoch 60. Its held-out test report shows **89.5% accuracy and 85.8% macro F1 on 191 clips**. This measures the official INCLUDE-50 test split; it does not establish accuracy on our local signer or live camera recordings.

### How a user will use the system

```text
User makes a sign in front of a camera
     ▼
Camera frames → MediaPipe landmarks → trained classifier → predicted sign label
     ▼
Later planned stages: gloss/text → English translation → speech
```

The standalone Python webcam demo runs the camera and classifier directly. The React website records a short camera clip and sends it to FastAPI using the custom model; a video selected from disk is sent using INCLUDE-50. The speech-to-sign endpoint explicitly reports that the feature is not implemented yet. Full sentence translation, a connected speech-to-text → gloss → avatar pipeline, and generated speech from recognized signs are future stages.

## Project status and limitations

- Implemented: video collection tools, MediaPipe landmark extraction, preprocessing, data validation/resampling, training-only augmentation, a frame-wise baseline, the ST-GCN + Transformer classifier, evaluation metrics, a standalone webcam demo, FastAPI health/model/video endpoints, a custom-model WebSocket, and React integration with separate live-camera and uploaded-video model routing.
- Not yet implemented: continuous gloss recognition, gloss-to-English NLP, speech-to-text service, text-to-speech integration, motion generation, and Three.js avatar. The speech-to-sign API returns HTTP 501 until those stages are built.
- The checked-in sample data has 15 clips for each of five classes (75 clips total), no `no_sign` class, and no recorded multi-session provenance. It is suitable for development tests, not for claiming general-purpose performance.
- The existing `models/isolated_sign_classifier.metrics.json` report is from the earlier baseline: 66.7% accuracy and 0.651 macro F1 on 15 test samples. It is **not** a result for the ST-GCN model. The INCLUDE-50 ST-GCN report is separate and its score applies only to the official INCLUDE-50 test split.
- The custom five-sign checkpoint (`models/stgcn_transformer.pt`) has **80% accuracy and 79.8% macro F1 on 15 held-out clips**. This is a small clip-level test, not a signer-held-out test, and it does not guarantee performance on unseen signers.
- The INCLUDE-50 checkpoint (`models/include50_stgcn_transformer.pt`) has **89.5% accuracy and 85.8% macro F1 on its official 191-clip test split**. This score is not a measure of how the model performs on our own recordings.
- Training supports clip-, session-, and signer-disjoint splits. The current checked-in data has no provenance manifest, so only the clip split can currently be used with it. For new documented recordings, prefer a session- or signer-based split; every class must appear in at least three groups.
- Model checkpoints, raw videos, and collection metadata are ignored by Git. A fresh clone may not include local datasets or trained models.

## Repository map: what each part is for

| Path | Purpose |
| --- | --- |
| `README.md` | Project explanation, data flow, setup commands, status, limitations, and next steps. |
| `requirements.txt` | Python packages needed for landmark extraction, model training, and the API. |
| `.gitignore` | Keeps local environments, raw/private datasets, generated reports, and model checkpoints out of Git. |
| `data/raw_videos/<label>/` | Local source videos organized in folders named for their sign labels. |
| `data/processed/<label>/*.npy` | Processed local examples: one 258-feature landmark sequence per video. |
| `data/metadata/` | Collection and processed-data manifests; needed for provenance-aware audit and grouped splits. |
| `data/include50/` | Local INCLUDE-50 index, source videos, processed sequences, and official-split manifest. |
| `models/*.pt` | Locally saved trained checkpoints. The API needs both the custom and INCLUDE-50 checkpoint files. |
| `models/*.metrics.json` | Evaluation reports saved alongside trained checkpoints. |
| `src/data/record_videos.py` | Records labeled videos and documents collection conditions. |
| `src/data/landmarks.py` | Extracts pose and hand landmarks with MediaPipe. |
| `src/data/preprocess.py` | Converts our raw local videos into landmark `.npy` sequences and records provenance. |
| `src/data/import_include50.py` | Converts INCLUDE-50 source videos and preserves their official split assignments. |
| `src/data/loader.py` | Validates, loads, resamples, and splits landmark sequences for training/evaluation. |
| `src/data/augment.py` | Applies geometric and timing variations to training examples only. |
| `src/data/audit.py` | Checks sample counts, `no_sign`, provenance, sessions, and data variation. |
| `src/models/graph.py` | Empty placeholder; graph construction currently lives in `src/models/stgcn.py`. |
| `src/models/stgcn.py` | Implements spatial-temporal graph-convolution blocks and the frame-wise baseline encoder. |
| `src/models/transformer.py` | Models temporal relationships between frames. |
| `src/models/hybrid.py` | Combines encoders, Transformer, pooling, and classifier heads. |
| `src/models/baseline.py` | Keeps the original non-graph classifier for comparison. |
| `src/training/train.py` | Trains a selected model, chooses the best validation checkpoint, and reports test metrics. |
| `src/training/evaluate.py` | Calculates accuracy, macro F1, per-class scores, and confusion matrices. |
| `src/inference/predict.py` | Runs standalone real-time webcam recognition from one checkpoint. |
| `src/utils/checkpoints.py` | Saves and validates model checkpoint files. |
| `src/utils/io.py` | Shared JSON and filesystem helpers. |
| `api/main.py` | FastAPI startup, health/model routes, video prediction, and WebSocket endpoint. |
| `api/services/recognition.py` | Loads both checkpoints, extracts input-video landmarks, and runs predictions. |
| `api/services/nlp.py` | Empty placeholder; gloss-to-English NLP is not implemented yet. |
| `frontend/web/src/App.tsx` | React screens and user actions for camera, uploaded video, predictions, and speech-to-sign UI. |
| `frontend/web/src/api.ts` | Typed REST/WebSocket client and model-selection request fields. |
| `frontend/web/src/styles.css` | Website layout, components, and responsive styling. |
| `frontend/web/src/main.tsx` | React browser entry point. |
| `frontend/web/src/vite-env.d.ts` | TypeScript declarations for Vite environment variables. |
| `frontend/web/index.html`, `vite.config.ts`, `tsconfig*.json` | Frontend HTML, development/build configuration, and TypeScript settings. |
| `frontend/web/package.json`, `package-lock.json` | JavaScript dependencies and reproducible package versions. |
| `frontend/web/.env.example` | Example frontend setting for the API address. |
| `frontend/avatar/` | Reserved for future 3D avatar assets and motion files. |
| `tests/test_data_pipeline.py` | Tests sequence loading, validation, provenance, and data splitting. |
| `tests/test_augment.py` | Tests training-data augmentation. |
| `tests/test_model.py` | Tests model shapes, graph behavior, and classifiers. |
| `tests/test_recognition.py` | Tests model selection and no-hand prediction handling. |
| `notebooks/01_data_exploration.ipynb` | Early dataset exploration. |
| `notebooks/02_landmark_extraction.ipynb` | Landmark-extraction experiments. |
| `notebooks/03_baseline.ipynb` | Early baseline experiments. |
| `notebooks/04_model_training.ipynb` | Model-training experiments. |

## Setup

Use Python 3.10 or later. Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the backend API

The API loads both ST-GCN + Transformer checkpoints at startup, so both files must exist. Live camera captures use the model trained on this project's five locally recorded signs (`hello`, `help`, `no`, `thank_you`, `yes`); uploaded video clips use the separate 50-label INCLUDE-50 model. From the repository root, start it after installing the Python requirements:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The default checkpoints are `models/stgcn_transformer.pt` and `models/include50_stgcn_transformer.pt`. To change them or select another compute device, set these before launching:

```powershell
$env:SIGN_CUSTOM_MODEL_CHECKPOINT = "models/stgcn_transformer.pt"
$env:SIGN_INCLUDE50_MODEL_CHECKPOINT = "models/include50_stgcn_transformer.pt"
$env:SIGN_MODEL_DEVICE = "auto" # or "cpu" / "cuda"
```

The older `SIGN_MODEL_CHECKPOINT` setting remains supported as an override for both profiles when the profile-specific settings are unset.

The frontend's default API address is `http://127.0.0.1:8000`. The API allows the local Vite origins `http://localhost:5173` and `http://127.0.0.1:5173`; set `SIGN_FRONTEND_ORIGINS` to a comma-separated origin list if your frontend runs elsewhere. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Run the web frontend

The React + TypeScript frontend lives in `frontend\web`. Keep the API running in its terminal, then open a second PowerShell terminal at the repository root:

```powershell
Set-Location frontend\web
npm install
npm run dev
```

Open the local URL printed by Vite. To point the browser at a backend hosted somewhere other than `http://127.0.0.1:8000`, copy `.env.example` to `.env.local` in `frontend/web`, set `VITE_API_BASE_URL`, and restart Vite. Build and type-check the frontend from its directory with:

```powershell
Set-Location frontend\web
npm run build
```

The UI includes sign-to-speech and speech-to-sign screens, a backend connection banner, and a retry action. In sign-to-speech, users can start the camera and click **Capture sign** to record a three-second clip, or select a local video and click **Analyze clip**. The first path selects the custom model; the uploaded-video path selects INCLUDE-50. The browser asks for camera permission; webcam use requires localhost or HTTPS. In speech-to-sign, users can type a sentence or try browser dictation with English (India), English (US), or Hindi selected. Dictation depends on browser Web Speech API support; neither path currently produces a sign translation because that backend stage is not implemented. The avatar view is an illustrative placeholder.

### API endpoints

Implemented FastAPI REST endpoints:

| Method and path | Request | Example response / behavior |
| --- | --- | --- |
| `GET /api/v1/health` | None | `{"status":"ok"}` if model startup succeeded |
| `GET /api/v1/model` | None | Both model profiles, architectures, devices, label lists, and feature availability |
| `POST /api/v1/predict-video` | `multipart/form-data`, file field `file`, `model` field `custom` or `include50`; MP4/MOV/WebM, max 25 MB and 500 frames | Prediction, top alternatives, and the selected model profile |
| `POST /api/v1/speech-to-sign` | JSON `{"text":"Hello"}` | HTTP 501 with a message that speech-to-sign is not implemented yet |

For camera recognition, the frontend previews the camera and records a three-second clip when **Capture sign** is clicked, then sends the original video with `model=custom`. This preserves the camera's captured frame rate and uses the five-label model. Uploaded recorded clips are sent with `model=include50`, selecting INCLUDE-50's 50-label vocabulary. The API extracts MediaPipe landmarks and returns the gloss, selected model, model score, and top alternatives; analysis may take a few seconds. To try another sign, click **Capture sign** again. The API reports when no hand landmarks were found rather than forcing a label. The React camera currently uses REST, not the WebSocket. Other clients can use `/ws/recognize`, which accepts binary JPEG frames and a `{"type":"capture"}` command, then classifies the next 20 frames using the custom five-label model. Both models recognize isolated labels, not continuous sentence-level signing. Each model chooses among its own labels when landmarks are present; confidence is not a guarantee that a prediction is correct.

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

## Import INCLUDE-50

The INCLUDE-50 files are kept under the ignored `data/include50/` directory. The downloaded CSV and extracted video paths have been checked: the index describes 943 clips across 50 labels with official train/validation/test assignments (675/77/191), and every indexed video was present. The videos must be converted to this project's 258-value MediaPipe format before training; importing is resumable and writes into a separate `data/include50/processed/` directory.

Run the importer from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.data.import_include50
```

By default, it reads `data/include50/include-50_dataset.csv` and videos from `data/include50/include-50_dataset/content/include_50_dataset`, saves landmark sequences to `data/include50/processed`, and records each original label and official split in `data/include50/processed_manifest.json`. It does not modify the existing `data/processed` samples. A short import can be tested with `--limit 3`, but do not train on a partial import.

Once all videos are imported, train using the dataset's provided split and a separate model checkpoint:

```powershell
.\.venv\Scripts\python.exe -m src.training.train --data-dir data/include50/processed --processed-manifest data/include50/processed_manifest.json --split-by official --checkpoint models/include50_stgcn_transformer.pt
```

This converts the dataset to the current model input. It does not make the 50-sign dataset directly equivalent to the five locally recorded signs: the label vocabulary differs, and the official test split omits two classes. Keep the official split intact when reporting INCLUDE-50 results.

## Train and evaluate

For the current 75-example local dataset, use a clip-level split because it has no signer/session manifest:

```powershell
.\.venv\Scripts\python.exe -m src.training.train --data-dir data/processed --split-by clip --checkpoint models/stgcn_transformer.pt
```

Default graph-model checkpoint: `models/stgcn_transformer.pt`. Its report is saved beside it as `models/stgcn_transformer.metrics.json`.

Session-based splitting is the default and uses `data/metadata/processed_manifest.json` to keep every session entirely within one split. Use `--split-by signer` to keep signer identities disjoint instead. Each class must have examples across at least three distinct selected groups. The currently available 75-clip sample dataset has no provenance manifest, so specify `--split-by clip` for that dataset; its result is not a signer/session generalization estimate.

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

1. Collect more local signers/sessions, include a `no_sign` class, and preserve recording provenance.
2. Evaluate the INCLUDE-50 and local models with signer-held-out data before claiming performance on new users.
3. Collect continuous signing clips with time-aligned glosses before developing sentence-level recognition.
4. Build and evaluate speech-to-text, gloss-to-English, and text-to-speech services.
5. Create or source licensed sign-motion assets, connect glosses to motion, and implement the Three.js avatar.
6. Add end-to-end tests, reconnect/error handling, and deployment configuration for the API and frontend.

## Questions your mentor may ask

**What does one model input contain?** A video is converted into frames of 258 MediaPipe values: 33 pose landmarks and 21 landmarks for each hand. The loader resamples a clip to 90 frames. The graph model represents those points as 75 connected joints.

**Why use an ST-GCN and a Transformer?** The graph convolution learns relationships between connected body/hand joints. The Transformer learns how those landmarks change across time. The classifier returns one isolated-sign label.

**Are the datasets merged?** No. Each is used to train its own checkpoint. The API can keep both loaded and routes webcam captures to the five-sign model and uploaded videos to INCLUDE-50.

**What do the accuracy numbers mean?** The local model scored 80% on 15 held-out clips; INCLUDE-50 scored 89.5% on its official 191-clip test split. Neither result proves performance on unseen signers. A confidence score is not the same as accuracy.

**What is the most important next step?** Record many more examples across signers, sessions, and camera conditions, include `no_sign`, and evaluate with signer-held-out data. Then build and test the sentence, speech, and avatar stages.
