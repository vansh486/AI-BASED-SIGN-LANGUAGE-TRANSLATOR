export type ModelInfo = {
  model_name?: string;
  status?: string;
  labels?: string[];
  supported_labels?: string[];
  features?: {
    live_sign_recognition?: boolean;
    video_prediction?: boolean;
    speech_to_sign?: boolean;
    english_translation?: boolean;
    avatar_motion?: boolean;
  };
  [key: string]: unknown;
};

export type RecognitionMessage = {
  type?: "prediction" | "status" | "error";
  status?: string;
  gloss?: string | null;
  label?: string;
  confidence?: number;
  english_text?: string | null;
  top_predictions?: Array<{ gloss: string; confidence: number }>;
  frames_received?: number;
  frames_required?: number;
  recognized?: boolean;
  detail?: string;
  message?: string;
};

export type VideoPrediction = {
  gloss: string | null;
  confidence: number;
  model?: "custom" | "include50";
  model_name?: string;
  recognized?: boolean;
  message?: string;
  english_text?: string | null;
  top_predictions?: Array<{ gloss: string; confidence: number }>;
};

export type SpeechToSignResult = {
  recognized_text?: string;
  glosses: string[];
  motions?: Array<{ gloss: string; animation_id: string }>;
  message?: string;
};

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

export function getWebSocketUrl(path = "/ws/recognize"): string {
  const url = new URL(API_BASE_URL);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = path;
  url.search = "";
  url.hash = "";
  return url.toString();
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new Error("Cannot reach the API. Check that the backend is running.");
  }

  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as
      | { detail?: string; message?: string }
      | null;
    throw new Error(
      body?.detail ||
        body?.message ||
        `The API request failed (${response.status}).`,
    );
  }
  return (await response.json()) as T;
}

export async function getHealth(): Promise<{ status: string }> {
  return request("/api/v1/health");
}

export async function getModelInfo(): Promise<ModelInfo> {
  return request("/api/v1/model");
}

export async function predictVideo(
  file: File,
  model: "custom" | "include50",
): Promise<VideoPrediction> {
  const form = new FormData();
  form.append("file", file);
  form.append("model", model);
  return request("/api/v1/predict-video", { method: "POST", body: form });
}

export async function translateSpeechToSign(
  text: string,
): Promise<SpeechToSignResult> {
  return request("/api/v1/speech-to-sign", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
}

export function openRecognitionSocket(
  onMessage: (message: RecognitionMessage) => void,
  onError: (message: string) => void,
  onClose: () => void,
): WebSocket {
  const socket = new WebSocket(getWebSocketUrl());
  socket.binaryType = "blob";
  socket.onmessage = (event: MessageEvent<string | Blob>) => {
    if (typeof event.data !== "string") {
      onError("The recognition server must respond with JSON messages.");
      return;
    }
    try {
      const message = JSON.parse(event.data) as RecognitionMessage;
      if (message.type === "error") {
        onError(message.detail || message.message || "Recognition failed.");
      } else {
        onMessage(message);
      }
    } catch {
      onError("Received an invalid response from the recognition server.");
    }
  };
  socket.onerror = () => onError("WebSocket connection failed.");
  socket.onclose = onClose;
  return socket;
}

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}
