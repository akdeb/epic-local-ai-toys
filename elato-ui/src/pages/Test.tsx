import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { useActiveUser } from "../state/ActiveUserContext";

const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://127.0.0.1:8000";

type VoiceMsg =
  | { type: "transcript"; text: string; is_final: boolean }
  | { type: "token"; content: string }
  | { type: "audio"; content: string; sentence?: string }
  | { type: "pause_mic" }
  | { type: "resume_mic" }
  | { type: "done"; full_response: string }
  | { type: "error"; message: string };

export const TestPage = () => {
  const { activeUser } = useActiveUser();

  const [status, setStatus] = useState<string>("connecting");
  const [error, setError] = useState<string | null>(null);
  const [voice, setVoice] = useState<string>("dave");
  const [systemPrompt, setSystemPrompt] = useState<string>("You are a helpful voice assistant. Be concise.");
  const [characterName, setCharacterName] = useState<string>("—");

  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const isRecordingRef = useRef(false);
  const isPausedRef = useRef(false);

  const [micLevel, setMicLevel] = useState<number>(0);
  const lastLevelAtRef = useRef<number>(0);
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);

  const wsRef = useRef<WebSocket | null>(null);
  const connectNonceRef = useRef(0);
  const connectTimeoutRef = useRef<number | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  const audioQueueRef = useRef<string[]>([]);
  const isPlayingRef = useRef(false);

  const wsUrl = useMemo(() => {
    const u = new URL(API_BASE);
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
    u.pathname = "/ws/voice";
    return u.toString();
  }, []);

  const playNextAudio = async () => {
    if (isPlayingRef.current) return;
    const next = audioQueueRef.current.shift();
    if (!next) return;

    isPlayingRef.current = true;
    setIsSpeaking(true);
    try {
      const audio = new Audio(`data:audio/wav;base64,${next}`);
      await audio.play();
      audio.onended = () => {
        isPlayingRef.current = false;
        setIsSpeaking(false);
        playNextAudio();
      };
    } catch {
      isPlayingRef.current = false;
      setIsSpeaking(false);
    }
  };

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const ps = await api.getPersonalities(true);
        const selectedId = activeUser?.current_personality_id;
        const selected = ps.find((p: any) => p.id === selectedId);
        if (!cancelled && selected) {
          setCharacterName(selected.name || "—");
          setVoice(selected.voice_id || "dave");
          setSystemPrompt(selected.prompt || "You are a helpful voice assistant. Be concise.");
        }
      } catch {
        // ignore
      }
    };

    load();
  }, [activeUser?.current_personality_id]);

  const stopRecording = () => {
    setIsRecording(false);
    isRecordingRef.current = false;
    isPausedRef.current = false;
    setIsPaused(false);

    try {
      processorRef.current?.disconnect();
    } catch {
      // ignore
    }
    try {
      sourceRef.current?.disconnect();
    } catch {
      // ignore
    }
    try {
      audioCtxRef.current?.close();
    } catch {
      // ignore
    }
    audioCtxRef.current = null;
    processorRef.current = null;
    sourceRef.current = null;

    try {
      mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    } catch {
      // ignore
    }
    mediaStreamRef.current = null;
  };

  const startRecording = async () => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setError("Voice WebSocket is not connected");
      return;
    }

    setError(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Microphone access requires a secure context (HTTPS) or localhost");
      return;
    }

    const mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    mediaStreamRef.current = mediaStream;

    const audioCtx = new AudioContext();
    audioCtxRef.current = audioCtx;

    const source = audioCtx.createMediaStreamSource(mediaStream);
    sourceRef.current = source;

    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;

    const actualRate = audioCtx.sampleRate;

    processor.onaudioprocess = (e) => {
      const socket = wsRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN) return;
      if (!isRecordingRef.current) return;
      if (isPausedRef.current) return;

      const input = e.inputBuffer.getChannelData(0);

      let sumSq = 0;
      for (let i = 0; i < input.length; i++) {
        const v = input[i] || 0;
        sumSq += v * v;
      }
      const rms = Math.sqrt(sumSq / Math.max(1, input.length));
      const now = performance.now();
      if (now - lastLevelAtRef.current > 80) {
        lastLevelAtRef.current = now;
        setMicLevel(Math.min(1, rms * 6));
      }

      const ratio = 24000 / actualRate;
      const outputLen = Math.round(input.length * ratio);
      const resampled = new Float32Array(outputLen);
      for (let i = 0; i < outputLen; i++) {
        const srcIdx = i / ratio;
        const idx0 = Math.floor(srcIdx);
        const idx1 = Math.min(idx0 + 1, input.length - 1);
        const frac = srcIdx - idx0;
        resampled[i] = input[idx0] * (1 - frac) + input[idx1] * frac;
      }

      const pcm = new Int16Array(resampled.length);
      for (let i = 0; i < resampled.length; i++) {
        pcm[i] = Math.max(-32768, Math.min(32767, resampled[i] * 32768));
      }

      socket.send(pcm.buffer);
    };

    source.connect(processor);
    processor.connect(audioCtx.destination);
    setIsRecording(true);
    isRecordingRef.current = true;
  };

  const toggleRecording = async () => {
    if (isRecording) {
      stopRecording();
    } else {
      await startRecording();
    }
  };

  const connectWs = () => {
    connectNonceRef.current += 1;
    const nonce = connectNonceRef.current;

    try {
      wsRef.current?.close();
    } catch {
      // ignore
    }

    if (connectTimeoutRef.current) {
      window.clearTimeout(connectTimeoutRef.current);
      connectTimeoutRef.current = null;
    }

    setError(null);
    setStatus("connecting");

    let ws: WebSocket;
    try {
      ws = new WebSocket(wsUrl);
    } catch (e: any) {
      setStatus("error");
      setError(e?.message || `Failed to create WebSocket: ${wsUrl}`);
      return;
    }

    wsRef.current = ws;

    connectTimeoutRef.current = window.setTimeout(() => {
      if (nonce !== connectNonceRef.current) return;
      if (ws.readyState === WebSocket.OPEN) return;
      try {
        ws.close();
      } catch {
        // ignore
      }
      setStatus("error");
      setError(`Can't connect to voice server (${wsUrl}). Is the Python sidecar running on port 8000?`);
    }, 6000);

    ws.onopen = () => {
      if (nonce !== connectNonceRef.current) return;
      if (connectTimeoutRef.current) {
        window.clearTimeout(connectTimeoutRef.current);
        connectTimeoutRef.current = null;
      }
      setStatus("connected");
      try {
        ws.send(JSON.stringify({ voice, system_prompt: systemPrompt }));
      } catch {
        // ignore
      }
    };

    ws.onclose = () => {
      if (nonce !== connectNonceRef.current) return;
      if (connectTimeoutRef.current) {
        window.clearTimeout(connectTimeoutRef.current);
        connectTimeoutRef.current = null;
      }
      setStatus("disconnected");
      stopRecording();
    };

    ws.onerror = () => {
      if (nonce !== connectNonceRef.current) return;
      if (connectTimeoutRef.current) {
        window.clearTimeout(connectTimeoutRef.current);
        connectTimeoutRef.current = null;
      }
      setStatus("error");
      setError(`Voice WebSocket error (${wsUrl}).`);
      stopRecording();
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as VoiceMsg;

        if (msg.type === "audio") {
          if (msg.content) {
            audioQueueRef.current.push(msg.content);
            playNextAudio();
          }
        } else if (msg.type === "pause_mic") {
          isPausedRef.current = true;
          setIsPaused(true);
          setMicLevel(0);
        } else if (msg.type === "resume_mic") {
          isPausedRef.current = false;
          setIsPaused(false);
        } else if (msg.type === "error") {
          setError(msg.message || "Unknown error");
        }
      } catch {
        // ignore
      }
    };
  };

  useEffect(() => {
    connectWs();

    return () => {
      if (connectTimeoutRef.current) {
        window.clearTimeout(connectTimeoutRef.current);
        connectTimeoutRef.current = null;
      }
      try {
        wsRef.current?.close();
      } catch {
        // ignore
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wsUrl]);

  useEffect(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    try {
      ws.send(JSON.stringify({ voice, system_prompt: systemPrompt }));
    } catch {
      // ignore
    }
  }, [voice, systemPrompt]);

  const statusDotClass =
    status === "connected" ? "bg-[#00c853]" : status === "error" ? "bg-red-500" : "bg-[#ffd400]";

  const orbScale = useMemo(() => {
    const base = isRecording ? 1.03 : 1;
    const mic = isRecording && !isPaused ? micLevel * 0.18 : 0;
    const speak = isSpeaking ? 0.08 : 0;
    return base + mic + speak;
  }, [isRecording, isPaused, micLevel, isSpeaking]);

  return (
    <div>
      <div className="flex justify-between items-start mb-8">
        <div>
          <h2 className="text-3xl font-black">TEST</h2>
          <div className="mt-2 font-mono text-xs text-gray-600">
            Character: <span className="font-bold text-black">{characterName}</span>
          </div>
          <div className="mt-1 font-mono text-xs text-gray-600 inline-flex items-center gap-2">
            <span className={`w-2.5 h-2.5 rounded-full border border-black ${statusDotClass}`} />
            <span className="capitalize">{status}</span>
            {isRecording && (
              <span className="text-gray-500">
                • {isPaused ? "paused" : "listening"}
              </span>
            )}
          </div>
          {error && <div className="mt-3 font-mono text-xs text-red-700">{error}</div>}
        </div>
      </div>

      <div className="flex flex-col items-center justify-center py-16">
        <button
          type="button"
          className="rounded-full border-2 border-black shadow-[0_14px_30px_rgba(0,0,0,0.18)] bg-[#9b5cff] hover:shadow-[0_18px_40px_rgba(0,0,0,0.20)] transition-shadow"
          onClick={() => {
            if (status !== "connected") {
              connectWs();
              return;
            }
            void toggleRecording();
          }}
          aria-label={isRecording ? "Stop microphone" : "Start microphone"}
          style={{
            width: 148,
            height: 148,
            transform: `scale(${orbScale})`,
            transition: "transform 80ms linear",
            opacity: status === "connected" ? 1 : 0.7,
          }}
        />

        <div className="mt-8 font-mono text-xs text-gray-600 text-center">
          {status === "connecting" && "Connecting…"}
          {status === "error" && "Click to retry"}
          {status === "disconnected" && "Click to reconnect"}
          {status === "connected" && (isRecording ? "Click to stop" : "Click to start")}
        </div>
      </div>
    </div>
  );
};
