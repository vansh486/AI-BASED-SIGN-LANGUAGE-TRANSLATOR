"""FastAPI application for isolated sign recognition."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from api.services.recognition import (
    MAX_JPEG_BYTES,
    MIN_LIVE_FRAMES,
    InvalidMediaError,
    MediaTooLargeError,
    RecognitionEngine,
    configured_checkpoints,
    configured_device,
    configured_frontend_origins,
    new_live_state,
)
from src.data.landmarks import LandmarkExtractor


MAX_VIDEO_UPLOAD_BYTES = 25 * 1024 * 1024


class SpeechToSignRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = configured_device()
    app.state.recognition = {
        profile: await run_in_threadpool(RecognitionEngine, checkpoint, device)
        for profile, checkpoint in configured_checkpoints().items()
    }
    yield


app = FastAPI(
    title="Sign Language Translator API",
    description="Isolated-sign recognition using MediaPipe and ST-GCN + Transformer.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_frontend_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def get_engines() -> dict[str, RecognitionEngine]:
    engines = getattr(app.state, "recognition", None)
    if not isinstance(engines, dict) or not engines:
        raise HTTPException(
            status_code=503,
            detail="Recognition model is not loaded. Restart the API and check its startup logs.",
        )
    return engines


def get_engine(profile: str = "custom") -> RecognitionEngine:
    engine = get_engines().get(profile)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"The {profile} recognition model is not loaded.",
        )
    return engine


@app.get("/api/v1/health")
async def health() -> dict[str, str]:
    get_engines()
    return {"status": "ok"}


@app.get("/api/v1/model")
async def model_info() -> dict:
    engines = get_engines()
    return {
        **engines["custom"].model_info,
        "models": {profile: engine.model_info for profile, engine in engines.items()},
    }


@app.post("/api/v1/predict-video")
async def predict_video(
    file: UploadFile = File(...),
    model: Literal["custom", "include50"] = Form("include50"),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov", ".webm"}:
        await file.close()
        raise HTTPException(
            status_code=415, detail="Upload a video in MP4, MOV, or WebM format."
        )

    content = await file.read(MAX_VIDEO_UPLOAD_BYTES + 1)
    await file.close()
    if len(content) > MAX_VIDEO_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail="Video uploads must be no larger than 25 MB."
        )
    try:
        engine = get_engine(model)
        result = await run_in_threadpool(engine.predict_video, content, suffix)
    except MediaTooLargeError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except InvalidMediaError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    result["english_text"] = None
    result["model"] = model
    result["model_name"] = (
        "Custom dataset (5 signs)" if model == "custom" else "INCLUDE-50 (50 signs)"
    )
    return result


@app.post("/api/v1/speech-to-sign")
async def speech_to_sign(request: SpeechToSignRequest) -> None:
    raise HTTPException(
        status_code=501,
        detail=(
            "Speech-to-sign is not implemented yet. Speech recognition, "
            "gloss generation, and avatar motion are planned stages."
        ),
    )


@app.websocket("/ws/recognize")
async def recognize(websocket: WebSocket) -> None:
    origin = websocket.headers.get("origin")
    if origin and origin not in configured_frontend_origins():
        await websocket.close(code=1008, reason="WebSocket origin is not allowed.")
        return

    try:
        engine = get_engine("custom")
    except HTTPException:
        await websocket.close(code=1013, reason="Recognition model is not loaded.")
        return

    await websocket.accept()
    frame_buffer = new_live_state()
    extractor = LandmarkExtractor()
    capture_active = False
    try:
        await websocket.send_json(
            {
                "type": "status",
                "status": "connected",
                "message": "Camera ready. Position yourself, then click Capture sign.",
                "frames_required": MIN_LIVE_FRAMES,
            }
        )
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            text = message.get("text")
            if text is not None:
                try:
                    command = json.loads(text)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"type": "error", "detail": "Send the capture command as valid JSON."}
                    )
                    continue
                if not isinstance(command, dict) or command.get("type") != "capture":
                    await websocket.send_json(
                        {
                            "type": "error",
                            "detail": 'The supported WebSocket command is {"type":"capture"}.',
                        }
                    )
                    continue
                if capture_active:
                    await websocket.send_json(
                        {"type": "error", "detail": "A sign capture is already in progress."}
                    )
                    continue
                frame_buffer.clear()
                capture_active = True
                await websocket.send_json(
                    {
                        "type": "status",
                        "status": "capturing",
                        "message": (
                            f"Capturing the next {MIN_LIVE_FRAMES} frames. "
                            "Perform one sign now and hold your hands in view."
                        ),
                        "frames_required": MIN_LIVE_FRAMES,
                    }
                )
                continue

            data = message.get("bytes")
            if data is None:
                await websocket.send_json(
                    {
                        "type": "error",
                        "detail": "Send camera frames as binary JPEG messages.",
                    }
                )
                continue
            if len(data) > MAX_JPEG_BYTES:
                await websocket.send_json(
                    {"type": "error", "detail": "Image frames must be no larger than 1 MB."}
                )
                await websocket.close(code=1009, reason="Image frame too large.")
                break

            if not capture_active:
                continue

            try:
                frame = await run_in_threadpool(engine.decode_jpeg_frame, data)
                landmarks = await run_in_threadpool(
                    engine.extract_frame, frame, extractor
                )
            except MediaTooLargeError as error:
                await websocket.send_json({"type": "error", "detail": str(error)})
                await websocket.close(code=1009, reason="Image dimensions too large.")
                break
            except InvalidMediaError as error:
                await websocket.send_json({"type": "error", "detail": str(error)})
                continue

            frame_buffer.append(landmarks)
            if len(frame_buffer) < MIN_LIVE_FRAMES:
                await websocket.send_json(
                    {
                        "type": "status",
                        "status": "warming_up",
                        "message": (
                            f"Collecting movement: {len(frame_buffer)}/"
                            f"{MIN_LIVE_FRAMES} frames."
                        ),
                        "frames_received": len(frame_buffer),
                        "frames_required": MIN_LIVE_FRAMES,
                    }
                )
                continue
            sequence = np.asarray(frame_buffer, dtype=np.float32)
            prediction = await run_in_threadpool(engine.predict_landmarks, sequence)
            capture_active = False
            await websocket.send_json(
                {
                    "type": "prediction",
                    **prediction,
                    "frames_received": len(frame_buffer),
                    "message": "Capture complete. Click Capture sign to try another sign.",
                }
            )
    except WebSocketDisconnect:
        pass
    finally:
        extractor.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)