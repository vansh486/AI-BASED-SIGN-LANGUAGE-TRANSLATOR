import { useCallback, useEffect, useRef, useState } from "react";
import {
  getApiBaseUrl,
  getHealth,
  getModelInfo,
  predictVideo,
  translateSpeechToSign,
  type ModelInfo,
  type RecognitionMessage,
  type SpeechToSignResult,
  type VideoPrediction,
} from "./api";

type Direction = "sign" | "speech";
type ApiStatus = "checking" | "online" | "offline";

type SpeechRecognitionResult = {
  transcript: string;
  isFinal?: boolean;
};

type SpeechRecognitionLike = {
  lang: string;
  interimResults: boolean;
  onresult: ((event: {
    resultIndex: number;
    results: ArrayLike<ArrayLike<SpeechRecognitionResult>>;
  }) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type BrowserSpeechWindow = Window & {
  SpeechRecognition?: new () => SpeechRecognitionLike;
  webkitSpeechRecognition?: new () => SpeechRecognitionLike;
};

function Icon({
  name,
  size = 20,
}: {
  name: string;
  size?: number;
}) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true as const,
  };

  switch (name) {
    case "sign":
      return (
        <svg {...common}>
          <path d="M8 11V5a1.5 1.5 0 0 1 3 0v5-7a1.5 1.5 0 0 1 3 0v7-5a1.5 1.5 0 0 1 3 0v6-3a1.5 1.5 0 0 1 3 0v7c0 4-2.6 7-6.5 7h-1c-2.5 0-3.8-.9-5.3-2.7L4 15a1.7 1.7 0 0 1 2.5-2.2L9 15" />
        </svg>
      );
    case "sound":
      return (
        <svg {...common}>
          <path d="M11 5 6 9H3v6h3l5 4V5Z" />
          <path d="M15.5 8.5a5 5 0 0 1 0 7M18.5 5.5a9 9 0 0 1 0 13" />
        </svg>
      );
    case "mic":
      return (
        <svg {...common}>
          <rect x="9" y="2" width="6" height="12" rx="3" />
          <path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3m-4 0h8" />
        </svg>
      );
    case "upload":
      return (
        <svg {...common}>
          <path d="M12 16V4m-5 5 5-5 5 5M4 20h16" />
        </svg>
      );
    case "camera":
      return (
        <svg {...common}>
          <path d="m15 10 5-3v10l-5-3" />
          <rect x="3" y="6" width="12" height="12" rx="2" />
        </svg>
      );
    case "arrow":
      return (
        <svg {...common}>
          <path d="M5 12h14m-6-6 6 6-6 6" />
        </svg>
      );
    case "spark":
      return (
        <svg {...common}>
          <path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2L12 3Z" />
          <path d="m19 16 .8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16Z" />
        </svg>
      );
    case "check":
      return (
        <svg {...common}>
          <path d="m5 12 4 4L19 6" />
        </svg>
      );
    case "play":
      return (
        <svg {...common} fill="currentColor" stroke="none">
          <path d="m8 5 11 7-11 7V5Z" />
        </svg>
      );
    case "stop":
      return (
        <svg {...common} fill="currentColor" stroke="none">
          <rect x="6" y="6" width="12" height="12" rx="2" />
        </svg>
      );
    case "close":
      return (
        <svg {...common}>
          <path d="m18 6-12 12M6 6l12 12" />
        </svg>
      );
    default:
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="9" />
          <path d="M12 8v4m0 4h.01" />
        </svg>
      );
  }
}

function formatName(name: string): string {
  return name.replaceAll("_", " ");
}

function App() {
  const [direction, setDirection] = useState<Direction>("sign");
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [cameraError, setCameraError] = useState("");
  const [recognitionError, setRecognitionError] = useState("");
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isStartingCamera, setIsStartingCamera] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [streamMessage, setStreamMessage] = useState("");
  const [latestPrediction, setLatestPrediction] =
    useState<RecognitionMessage | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState("");
  const [videoResult, setVideoResult] = useState<VideoPrediction | null>(null);
  const [isVideoLoading, setIsVideoLoading] = useState(false);
  const [speechText, setSpeechText] = useState("");
  const [speechLanguage, setSpeechLanguage] = useState("en-IN");
  const [isListening, setIsListening] = useState(false);
  const [speechResult, setSpeechResult] =
    useState<SpeechToSignResult | null>(null);
  const [speechError, setSpeechError] = useState("");
  const [isSpeechLoading, setIsSpeechLoading] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const captureTimerRef = useRef<number | null>(null);
  const speechRecognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const speechBaseTextRef = useRef("");

  const checkBackend = useCallback(async () => {
    setApiStatus("checking");
    try {
      const [health, model] = await Promise.all([getHealth(), getModelInfo()]);
      if (health.status !== "ok" && health.status !== "healthy") {
        throw new Error("Backend health check did not report a healthy status.");
      }
      setModelInfo(model);
      setApiStatus("online");
    } catch {
      setApiStatus("offline");
      setModelInfo(null);
    }
  }, []);

  useEffect(() => {
    void checkBackend();
    const intervalId = window.setInterval(() => void checkBackend(), 30_000);
    return () => window.clearInterval(intervalId);
  }, [checkBackend]);

  useEffect(() => {
    if (!videoFile) {
      setVideoPreviewUrl("");
      return;
    }
    const url = URL.createObjectURL(videoFile);
    setVideoPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [videoFile]);

  const stopRecognition = useCallback(() => {
    if (captureTimerRef.current !== null) {
      window.clearTimeout(captureTimerRef.current);
      captureTimerRef.current = null;
    }
    if (recorderRef.current) {
      recorderRef.current.ondataavailable = null;
      recorderRef.current.onstop = null;
      if (recorderRef.current.state !== "inactive") recorderRef.current.stop();
      recorderRef.current = null;
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsCameraOn(false);
    setIsStartingCamera(false);
    setIsConnected(false);
    setIsCapturing(false);
  }, []);

  useEffect(
    () => () => {
      if (captureTimerRef.current !== null) {
        window.clearTimeout(captureTimerRef.current);
      }
      if (recorderRef.current?.state !== "inactive") recorderRef.current?.stop();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      speechRecognitionRef.current?.stop();
    },
    [],
  );

  const startRecognition = async () => {
    setCameraError("");
    setRecognitionError("");
    setStreamMessage("Opening the camera and connecting to recognition...");
    setLatestPrediction(null);
    setVideoResult(null);
    setIsCapturing(false);
    setIsStartingCamera(true);
    if (apiStatus !== "online") {
      setRecognitionError("Start the backend API before connecting live recognition.");
      setIsStartingCamera(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsCameraOn(true);
      setIsConnected(true);
      setStreamMessage("Camera ready. Click Capture sign, then perform one sign.");
    } catch (error) {
      stopRecognition();
      const message =
        error instanceof Error ? error.message : "Could not start the webcam.";
      setCameraError(
        message.includes("Permission")
          ? "Camera permission was denied. Allow camera access in your browser settings."
          : message,
      );
    } finally {
      setIsStartingCamera(false);
    }
  };

  const captureSign = () => {
    const stream = streamRef.current;
    if (!stream || !isConnected) {
      setRecognitionError("Start the camera before capturing a sign.");
      return;
    }
    if (typeof MediaRecorder === "undefined") {
      setRecognitionError("This browser cannot record a camera clip. Try a recent version of Chrome or Edge.");
      return;
    }
    const mimeType = [
      "video/webm;codecs=vp8",
      "video/webm",
      "video/mp4",
    ].find((candidate) => MediaRecorder.isTypeSupported(candidate));
    let recorder: MediaRecorder;
    try {
      recorder = new MediaRecorder(
        stream,
        mimeType ? { mimeType } : undefined,
      );
    } catch (error) {
      setRecognitionError(
        error instanceof Error ? `Could not record camera video: ${error.message}` : "Could not record camera video.",
      );
      return;
    }
    const chunks: Blob[] = [];
    recorderRef.current = recorder;
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };
    recorder.onerror = () => {
      setIsCapturing(false);
      setRecognitionError("Camera recording failed. Please try again.");
    };
    recorder.onstop = () => {
      if (recorderRef.current === recorder) recorderRef.current = null;
      const clip = new Blob(chunks, { type: recorder.mimeType || mimeType || "video/webm" });
      if (clip.size === 0) {
        setIsCapturing(false);
        setRecognitionError("The camera did not produce a video clip. Please try again.");
        return;
      }
      const extension = clip.type.includes("mp4") ? "mp4" : "webm";
      const file = new File([clip], `live-sign-${Date.now()}.${extension}`, {
        type: clip.type,
      });
      setIsVideoLoading(true);
      setStreamMessage("Analyzing the recorded sign clip. This may take a few seconds...");
      void predictVideo(file, "custom")
        .then((result) => {
          setLatestPrediction(null);
          setVideoResult(result);
          setStreamMessage("Analysis complete. Click Capture sign to try another sign.");
        })
        .catch((error: unknown) => {
          setRecognitionError(
            error instanceof Error ? error.message : "Could not analyze the camera clip.",
          );
          setStreamMessage("The camera clip could not be analyzed.");
        })
        .finally(() => {
          setIsCapturing(false);
          setIsVideoLoading(false);
        });
    };

    setRecognitionError("");
    setLatestPrediction(null);
    setVideoResult(null);
    setIsCapturing(true);
    setStreamMessage("Recording for 3 seconds. Perform one sign now and keep your hands in view.");
    try {
      recorder.start(250);
      captureTimerRef.current = window.setTimeout(() => {
        captureTimerRef.current = null;
        if (recorder.state === "recording") recorder.stop();
      }, 3000);
    } catch (error) {
      recorderRef.current = null;
      setIsCapturing(false);
      setRecognitionError(
        error instanceof Error ? `Could not start camera recording: ${error.message}` : "Could not start camera recording.",
      );
    }
  };

  const handleVideoTest = async () => {
    if (!videoFile) return;
    setIsVideoLoading(true);
    setRecognitionError("");
    setVideoResult(null);
    try {
      setVideoResult(await predictVideo(videoFile, "include50"));
    } catch (error) {
      setRecognitionError(
        error instanceof Error ? error.message : "Video prediction failed.",
      );
    } finally {
      setIsVideoLoading(false);
    }
  };

  const startSpeechRecognition = () => {
    setSpeechError("");
    const speechWindow = window as BrowserSpeechWindow;
    const Recognition =
      speechWindow.SpeechRecognition || speechWindow.webkitSpeechRecognition;
    if (!Recognition) {
      setSpeechError(
        "Voice input is not supported in this browser. Type your sentence instead.",
      );
      return;
    }
    const recognition = new Recognition();
    recognition.lang = speechLanguage;
    recognition.interimResults = true;
    recognition.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((result) => result[0]?.transcript ?? "")
        .join(" ");
      setSpeechText(
        [speechBaseTextRef.current, transcript].filter(Boolean).join(" "),
      );
    };
    recognition.onerror = (event) => {
      setSpeechError(`Voice recognition failed: ${event.error}.`);
    };
    recognition.onend = () => setIsListening(false);
    speechRecognitionRef.current = recognition;
    speechBaseTextRef.current = speechText.trim();
    setIsListening(true);
    recognition.start();
  };

  const stopSpeechRecognition = () => {
    speechRecognitionRef.current?.stop();
    speechRecognitionRef.current = null;
    setIsListening(false);
  };

  const chooseDirection = (nextDirection: Direction) => {
    stopRecognition();
    if (isListening) stopSpeechRecognition();
    setDirection(nextDirection);
  };

  const chooseVideo = (file: File | null) => {
    if (file && file.size > 25 * 1024 * 1024) {
      setRecognitionError("Choose a video smaller than 25 MB.");
      return;
    }
    setVideoFile(file);
    setVideoResult(null);
    setRecognitionError("");
  };

  const handleSpeechToSign = async () => {
    const text = speechText.trim();
    if (!text) {
      setSpeechError("Enter or speak a sentence first.");
      return;
    }
    setSpeechError("");
    setSpeechResult(null);
    setIsSpeechLoading(true);
    try {
      setSpeechResult(await translateSpeechToSign(text));
    } catch (error) {
      setSpeechError(
        error instanceof Error ? error.message : "Speech-to-sign request failed.",
      );
    } finally {
      setIsSpeechLoading(false);
    }
  };

  const speakText = (text: string) => {
    if (!("speechSynthesis" in window)) {
      setRecognitionError("Speech playback is not supported in this browser.");
      return;
    }
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  };

  const predictionLabel =
    latestPrediction?.gloss || latestPrediction?.label || "—";
  const supportedLabels =
    modelInfo?.supported_labels || modelInfo?.labels || [];
  const alternativePredictions =
    (videoResult?.top_predictions || latestPrediction?.top_predictions)?.slice(1) ??
    [];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" aria-label="Signa home">
          <span className="brand-mark">
            <Icon name="sign" size={23} />
          </span>
          <span className="brand-name">signa<span>.</span></span>
        </a>

        <div className="side-label">WORKSPACE</div>
        <nav className="side-nav" aria-label="Main navigation">
          <button
            className={`nav-item ${direction === "sign" ? "active" : ""}`}
            aria-label="Sign to speech"
            title="Sign to speech"
            onClick={() => chooseDirection("sign")}
          >
            <Icon name="sign" />
            <span>Sign to speech</span>
            <span className="nav-arrow">↗</span>
          </button>
          <button
            className={`nav-item ${direction === "speech" ? "active" : ""}`}
            aria-label="Speech to sign"
            title="Speech to sign"
            onClick={() => chooseDirection("speech")}
          >
            <Icon name="sound" />
            <span>Speech to sign</span>
            <span className="nav-arrow">↗</span>
          </button>
        </nav>

        <div className="sidebar-spacer" />

        <div className="connection-card">
          <div className="connection-top">
            <span className={`status-dot ${apiStatus}`} />
            <span>
              {apiStatus === "online"
                ? "Backend connected"
                : apiStatus === "checking"
                  ? "Checking backend"
                  : "Backend offline"}
            </span>
          </div>
          <div className="connection-url">{getApiBaseUrl()}</div>
          <button className="text-button" onClick={() => void checkBackend()}>
            Check connection <Icon name="arrow" size={14} />
          </button>
        </div>

        <div className="sidebar-footer">
          <span className="footer-orbit">✳</span>
          <div>
            <strong>Built for better connection</strong>
            <span>Recognition prototype · v0.1</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span>{" "}
            {direction === "sign" ? "Sign to speech" : "Speech to sign"}
          </div>
          <div className="topbar-right">
            <span className="prototype-badge">
              <span /> PROTOTYPE
            </span>
            <div className="avatar-initials" title="Local workspace">S</div>
          </div>
        </header>

        <div className="page-wrap">
          <section className="page-heading">
            <div>
              <div className="eyebrow"><Icon name="spark" size={15} /> TWO-WAY COMMUNICATION</div>
              <h1>Words beyond words.</h1>
              <p className="page-subtitle">
                {direction === "sign"
                  ? "Turn a sign into a voice. Start with a sign, and we’ll help carry it across."
                  : "Turn a voice into a sign. Speak or type a message to prepare it for signing."}
              </p>
            </div>
            <div className="today-pill"><span className="today-icon">✳</span> ISL · Indian Sign Language</div>
          </section>

          <div className="direction-switch" role="tablist" aria-label="Translation direction">
            <button
              role="tab"
              aria-selected={direction === "sign"}
              className={direction === "sign" ? "selected" : ""}
              onClick={() => chooseDirection("sign")}
            >
              <Icon name="sign" size={18} /> Sign <span>to</span> Speech
            </button>
            <button
              role="tab"
              aria-selected={direction === "speech"}
              className={direction === "speech" ? "selected" : ""}
              onClick={() => chooseDirection("speech")}
            >
              <Icon name="sound" size={18} /> Speech <span>to</span> Sign
            </button>
          </div>

          <div className={`backend-banner ${apiStatus}`} role="status" aria-live="polite">
            <span className={`status-dot ${apiStatus}`} />
            <div>
              <strong>
                {apiStatus === "online"
                  ? "Recognition service connected"
                  : apiStatus === "checking"
                    ? "Checking the recognition service"
                    : "Frontend preview is ready"}
              </strong>
              <span>
                {apiStatus === "online"
                  ? "You can use the live camera and connected API features."
                  : apiStatus === "checking"
                    ? "Checking the configured API address."
                    : "Camera recognition and translations need the backend API, which is not connected yet."}
              </span>
            </div>
            {apiStatus !== "online" && (
              <button
                className="text-button"
                onClick={() => void checkBackend()}
                disabled={apiStatus === "checking"}
              >
                {apiStatus === "checking" ? "Checking..." : "Retry connection"}
                <Icon name="arrow" size={14} />
              </button>
            )}
          </div>

          {direction === "sign" ? (
            <div className="workspace-grid">
              <section className="panel camera-panel">
                <div className="panel-heading">
                  <div>
                    <div className="section-kicker">01 / INPUT</div>
                    <h2>Show your sign</h2>
                  </div>
                  <div className={`live-badge ${isCameraOn ? "recording" : ""}`}>
                    <span /> {isCameraOn ? "LIVE" : "CAMERA OFF"}
                  </div>
                </div>
                <div className={`camera-stage ${isCameraOn ? "camera-active" : ""}`}>
                  <video ref={videoRef} className="camera-video" muted playsInline />
                  {!isCameraOn && (
                    <div className="camera-placeholder">
                      <div className="camera-illustration">
                        <div className="camera-ring"><Icon name="camera" size={31} /></div>
                        <span className="sparkle sparkle-one">✳</span>
                        <span className="sparkle sparkle-two">✦</span>
                      </div>
                      <strong>Your camera, your conversation.</strong>
                      <span>Start the camera, click Capture sign, then perform one sign for 2–3 seconds.</span>
                    </div>
                  )}
                  {isCameraOn && (
                    <div className="camera-overlay">
                      <span><i /> CAMERA ACTIVE</span>
                      <span className="frame-counter">{isConnected ? "CAMERA READY" : "CONNECTING..."}</span>
                    </div>
                  )}
                </div>
                {isCameraOn && streamMessage && (
                  <p className="stream-feedback" role="status" aria-live="polite">
                    {streamMessage}
                  </p>
                )}
                {cameraError && <p className="inline-error">{cameraError}</p>}
                {recognitionError && <p className="inline-error">{recognitionError}</p>}
                <div className="camera-actions">
                  <button
                    className={isCameraOn ? "button button-stop" : "button button-primary"}
                    onClick={() => (isCameraOn ? stopRecognition() : void startRecognition())}
                    disabled={isStartingCamera || (!isCameraOn && apiStatus !== "online")}
                  >
                    <Icon name={isCameraOn ? "stop" : "play"} size={17} />
                    {isStartingCamera ? "Starting camera..." : isCameraOn ? "Stop camera" : "Start camera"}
                  </button>
                  {isCameraOn && (
                    <button
                      className="button button-primary"
                      onClick={captureSign}
                      disabled={!isConnected || isCapturing || isVideoLoading}
                    >
                      {isCapturing ? "Recording sign..." : isVideoLoading ? "Analyzing sign..." : "Capture sign"}
                    </button>
                  )}
                  <span className="privacy-note"><Icon name="info" size={14} /> A 3-second camera clip is sent when captured</span>
                </div>
                <div className="panel-divider" />
                <div className="upload-row">
                  <div className="upload-copy">
                    <span className="upload-icon"><Icon name="upload" size={17} /></span>
                    <div><strong>Prefer a recorded clip?</strong><span>Preview it here; analyze sends it to the backend</span></div>
                  </div>
                  <label className="button button-outline file-button">
                    {videoFile ? "Change clip" : "Choose video"}
                    <input
                      type="file"
                      accept="video/mp4,video/quicktime,video/webm"
                      onChange={(event) => {
                        chooseVideo(event.target.files?.[0] ?? null);
                        event.currentTarget.value = "";
                      }}
                    />
                  </label>
                </div>
                {videoFile && (
                  <div className="clip-preview">
                    <video src={videoPreviewUrl} controls playsInline aria-label={`Preview of ${videoFile.name}`} />
                    <div className="selected-file">
                      <span title={videoFile.name}>{videoFile.name} · {(videoFile.size / (1024 * 1024)).toFixed(1)} MB</span>
                      <button aria-label="Remove selected video" onClick={() => chooseVideo(null)}><Icon name="close" size={15} /></button>
                      <button className="button button-small" onClick={() => void handleVideoTest()} disabled={isVideoLoading || apiStatus !== "online"}>
                        {isVideoLoading ? "Analyzing..." : "Analyze clip"}
                      </button>
                    </div>
                  </div>
                )}
              </section>

              <section className="panel result-panel">
                <div className="panel-heading">
                  <div>
                    <div className="section-kicker">02 / OUTPUT</div>
                    <h2>Your translation</h2>
                  </div>
                  <span className="result-spark"><Icon name="spark" size={18} /></span>
                </div>
                <div className="result-content">
                  <div className="result-label">RECOGNIZED GLOSS</div>
                  {videoResult?.model_name && (
                    <div className="result-label">{videoResult.model_name}</div>
                  )}
                  <div className={`prediction-word ${predictionLabel === "—" ? "muted" : ""}`}>
                    {videoResult?.recognized === false
                      ? "No clear sign detected"
                      : formatName(videoResult?.gloss || predictionLabel)}
                  </div>
                  <div className="confidence-line">
                    <span>Model score (not accuracy)</span>
                    <strong>
                      {videoResult
                        ? `${Math.round(videoResult.confidence * 100)}%`
                        : latestPrediction?.confidence !== undefined
                          ? `${Math.round(latestPrediction.confidence * 100)}%`
                          : "—"}
                    </strong>
                  </div>
                  {videoResult?.recognized === false && videoResult.message && (
                    <p className="recognition-hint" role="status">
                      {videoResult.message}
                    </p>
                  )}
                  <div className="confidence-track"><span style={{ width: `${Math.min(100, Math.max(0, (videoResult?.confidence ?? latestPrediction?.confidence ?? 0) * 100))}%` }} /></div>
                  {alternativePredictions.length > 0 && (
                    <div className="alternative-predictions">
                      <span>OTHER POSSIBILITIES</span>
                      <div>
                        {alternativePredictions.map((item) => (
                          <span className="alternative-chip" key={item.gloss}>
                            {formatName(item.gloss)}
                            <small>{Math.round(item.confidence * 100)}%</small>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  <div className="translation-card">
                    <div className="translation-title"><span>ENGLISH TEXT</span><span className="coming-label">NLP STAGE</span></div>
                    <p>{videoResult?.english_text || latestPrediction?.english_text || "English translation will appear when the NLP service is connected."}</p>
                    <button
                      className="speak-button"
                      onClick={() => {
                        const text = videoResult?.english_text || latestPrediction?.english_text;
                        if (text) speakText(text);
                      }}
                      disabled={!videoResult?.english_text && !latestPrediction?.english_text}
                    >
                      <Icon name="sound" size={16} /> Speak translation
                    </button>
                  </div>
                </div>
                <div className="result-footnote"><span className="info-dot">i</span> Live camera uses your five-sign custom model; uploaded clips use INCLUDE-50. Confidence is a model score, not a guarantee of accuracy.</div>
              </section>

              <section className="panel process-panel">
                <div className="panel-heading compact-heading">
                  <div><div className="section-kicker">THE PROCESS</div><h2>From movement to meaning</h2></div>
                  <span className="process-caption">SIGN → SPEECH</span>
                </div>
                <div className="pipeline">
                  {[
                    ["01", "Camera", "Video input"],
                    ["02", "MediaPipe", "Body landmarks"],
                    ["03", "ST-GCN + Transformer", "Sign recognition"],
                    ["04", "Gloss → NLP → Speech", "Translation stages"],
                  ].map(([number, title, caption], index) => (
                    <div className="pipeline-step" key={number}>
                      <span className={`step-number ${index < 3 ? "step-ready" : "step-planned"}`}>{number}</span>
                      <div><strong>{title}</strong><span>{caption}</span></div>
                      {index < 3 && <span className="step-check"><Icon name="check" size={14} /></span>}
                      {index < 3 && <span className="step-connector" />}
                    </div>
                  ))}
                </div>
              </section>

              <section className="panel model-panel">
                <div className="section-kicker">MODEL STATUS</div>
                <div className="model-status-row">
                  <span className={`model-status-dot ${apiStatus === "online" ? "ready" : ""}`} />
                  <strong>{apiStatus === "online" ? "Two recognition models ready" : "Waiting for backend"}</strong>
                </div>
                <p>{apiStatus === "online" ? "Custom model for live camera; INCLUDE-50 model for uploaded videos." : "The website is ready. Start FastAPI when model training is complete."}</p>
                {supportedLabels.length > 0 && (
                  <div className="label-chips">
                    {supportedLabels.slice(0, 8).map((label) => <span key={label}>{formatName(label)}</span>)}
                    {supportedLabels.length > 8 && <span>+{supportedLabels.length - 8}</span>}
                  </div>
                )}
                {apiStatus !== "online" && <button className="text-button model-refresh" onClick={() => void checkBackend()}>Retry connection <Icon name="arrow" size={14} /></button>}
              </section>
            </div>
          ) : (
            <div className="speech-grid">
              <section className="panel speech-input-panel">
                <div className="panel-heading">
                  <div><div className="section-kicker">01 / INPUT</div><h2>Say what you mean</h2></div>
                  <span className="speech-icon"><Icon name="mic" size={20} /></span>
                </div>
                <p className="panel-description">
                  Speak or type a short sentence.{" "}
                  {apiStatus === "online"
                    ? "The recognition API is connected, but speech-to-sign is a planned stage and is not available yet."
                    : "Speech-to-sign needs the backend and is not available yet. You can still try the example phrases or dictate text."}
                </p>
                <div className="example-row">
                  <span>TRY AN EXAMPLE</span>
                  {["Hello", "Thank you", "How are you?"].map((example) => (
                    <button
                      className="example-chip"
                      key={example}
                      onClick={() => {
                        setSpeechText(example);
                        setSpeechError("");
                        setSpeechResult(null);
                      }}
                    >
                      {example}
                    </button>
                  ))}
                </div>
                <label className="field-label" htmlFor="speech-text">YOUR MESSAGE</label>
                <textarea id="speech-text" value={speechText} onChange={(event) => setSpeechText(event.target.value)} placeholder="Type a message here, or use the microphone..." rows={5} />
                <div className="input-meta">
                  <label htmlFor="dictation-language">Dictation language</label>
                  <select
                    id="dictation-language"
                    value={speechLanguage}
                    onChange={(event) => setSpeechLanguage(event.target.value)}
                    disabled={isListening}
                  >
                    <option value="en-IN">English (India)</option>
                    <option value="en-US">English (US)</option>
                    <option value="hi-IN">Hindi</option>
                  </select>
                  <span>{speechText.length} characters</span>
                  {speechText && (
                    <button
                      className="clear-text"
                      onClick={() => {
                        setSpeechText("");
                        setSpeechResult(null);
                        setSpeechError("");
                      }}
                    >
                      Clear
                    </button>
                  )}
                </div>
                <div className="speech-controls">
                  <button className={isListening ? "button button-stop" : "button button-outline"} onClick={() => (isListening ? stopSpeechRecognition() : startSpeechRecognition())}>
                    <Icon name="mic" size={17} />                     {isListening ? "Stop listening" : "Use microphone"}
                  </button>
                  <button className="button button-primary" disabled={isSpeechLoading || apiStatus !== "online" || modelInfo?.features?.speech_to_sign !== true || !speechText.trim()} onClick={() => void handleSpeechToSign()}>
                    {isSpeechLoading ? "Preparing..." : "Prepare signs"} <Icon name="arrow" size={17} />
                  </button>
                </div>
                {speechError && <p className="inline-error">{speechError}</p>}
              </section>
              <section className="panel motion-panel">
                <div className="panel-heading">
                  <div><div className="section-kicker">02 / OUTPUT</div><h2>Sign sequence</h2></div>
                  <span className="result-spark"><Icon name="spark" size={18} /></span>
                </div>
                {speechResult ? (
                  <div className="motion-result">
                    <div className="avatar-placeholder"><span className="avatar-orbit orbit-a" /><span className="avatar-orbit orbit-b" /><div className="avatar-figure"><span className="figure-head" /><span className="figure-body" /><span className="figure-arm arm-left" /><span className="figure-arm arm-right" /></div><div className="avatar-label">3D AVATAR VIEW</div></div>
                    <div className="gloss-heading"><span>GLOSS SEQUENCE</span><span>{speechResult.glosses.length} SIGNS</span></div>
                    <div className="gloss-list">{speechResult.glosses.map((gloss, index) => <span className="gloss-chip" key={`${gloss}-${index}`}><small>{String(index + 1).padStart(2, "0")}</small>{formatName(gloss)}</span>)}</div>
                    <p className="motion-note">The avatar area is a preview placeholder until sign motion assets are connected.</p>
                  </div>
                ) : (
                  <div className="motion-empty">
                    <div className="avatar-placeholder empty-avatar"><span className="avatar-orbit orbit-a" /><span className="avatar-orbit orbit-b" /><div className="avatar-figure"><span className="figure-head" /><span className="figure-body" /><span className="figure-arm arm-left" /><span className="figure-arm arm-right" /></div><div className="avatar-label">AVATAR PREVIEW</div></div>
                    <strong>Your sign sequence will appear here</strong>
                    <span>Enter a message and choose “Prepare signs” to begin.</span>
                  </div>
                )}
                {speechResult?.message && <p className="motion-note">{speechResult.message}</p>}
              </section>
              <section className="panel process-panel speech-process">
                <div className="panel-heading compact-heading">
                  <div><div className="section-kicker">THE PROCESS</div><h2>From voice to movement</h2></div>
                  <span className="process-caption">SPEECH → SIGN</span>
                </div>
                <div className="pipeline">
                  {[
                    ["01", "Voice / text", "Message input"],
                    ["02", "STT + NLP", "Text understanding"],
                    ["03", "Sign gloss", "Sign sequence"],
                    ["04", "Motion → Avatar", "3D sign output"],
                  ].map(([number, title, caption], index) => (
                    <div className="pipeline-step" key={number}>
                      <span className={`step-number ${index < 2 ? "step-ready" : "step-planned"}`}>{number}</span>
                      <div><strong>{title}</strong><span>{caption}</span></div>
                      {index < 2 && <span className="step-check"><Icon name="check" size={14} /></span>}
                      {index < 3 && <span className="step-connector" />}
                    </div>
                  ))}
                </div>
              </section>
            </div>
          )}

          <footer className="page-footer">
            <span>Signa <b>·</b> An accessible communication prototype</span>
            <span>Recognition is currently limited to isolated signs.</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

export default App;
