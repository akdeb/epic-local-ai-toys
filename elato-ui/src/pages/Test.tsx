import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { useActiveUser } from "../state/ActiveUserContext";

const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://127.0.0.1:8000";

type VoiceMsg =
  | { type: "transcription"; text: string }
  | { type: "response"; text: string }
  | { type: "audio"; data: string }
  | { type: "audio_end" }
  | { type: "error"; message: string };

export const TestPage = () => {
  const { activeUser } = useActiveUser();

  const [status, setStatus] = useState<string>("disconnected");
  const [error, setError] = useState<string | null>(null);
  const [voice, setVoice] = useState<string>("dave");
  const [systemPrompt, setSystemPrompt] = useState<string>("You are a helpful voice assistant. Be concise.");
  const [characterName, setCharacterName] = useState<string>("—");
  const [configReady, setConfigReady] = useState<boolean>(false);

  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const isRecordingRef = useRef(false);
  const isPausedRef = useRef(false);

  const vadSilenceFramesRef = useRef(0);
  const vadIsSpeechActiveRef = useRef(false);
  const autoStartedMicRef = useRef(false);

  const awaitingResumeRef = useRef(false);

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

  const ttsSampleRate = 24000;
  const ttsAudioCtxRef = useRef<AudioContext | null>(null);
  const ttsPcmQueueRef = useRef<Int16Array[]>([]);
  const ttsScriptNodeRef = useRef<ScriptProcessorNode | null>(null);
  const ttsPlaybackActiveRef = useRef(false);

  const base64EncodeBytes = (bytes: Uint8Array) => {
    // Avoid stack overflow from String.fromCharCode(...bigArray)
    const CHUNK = 0x8000;
    let binary = "";
    for (let i = 0; i < bytes.length; i += CHUNK) {
      const slice = bytes.subarray(i, Math.min(i + CHUNK, bytes.length));
      binary += String.fromCharCode(...slice);
    }
    return window.btoa(binary);
  };

  // Continuous PCM streaming playback (like pyaudio speaker_stream.write)
  const ttsCurrentChunkRef = useRef<Int16Array | null>(null);
  const ttsCurrentChunkOffsetRef = useRef(0);

  const startTtsPlayback = () => {
    if (ttsPlaybackActiveRef.current) return;

    const ctx = new AudioContext({ sampleRate: ttsSampleRate });
    ttsAudioCtxRef.current = ctx;

    if (ctx.state === "suspended") {
      void ctx.resume();
    }

    // ScriptProcessorNode pulls samples continuously
    const bufferSize = 4096;
    const scriptNode = ctx.createScriptProcessor(bufferSize, 1, 1);
    ttsScriptNodeRef.current = scriptNode;

    scriptNode.onaudioprocess = (e) => {
      const output = e.outputBuffer.getChannelData(0);
      let outIdx = 0;

      while (outIdx < output.length) {
        // Get current chunk or fetch next
        if (!ttsCurrentChunkRef.current || ttsCurrentChunkOffsetRef.current >= ttsCurrentChunkRef.current.length) {
          const next = ttsPcmQueueRef.current.shift();
          if (!next) {
            // No more data - fill rest with silence
            while (outIdx < output.length) {
              output[outIdx++] = 0;
            }
            // Check if we should stop
            if (awaitingResumeRef.current && ttsPcmQueueRef.current.length === 0) {
              // Schedule stop after this buffer plays
              setTimeout(() => {
                stopTtsPlayback();
                awaitingResumeRef.current = false;
                resumeMic();
              }, (bufferSize / ttsSampleRate) * 1000 + 50);
            }
            break;
          }
          ttsCurrentChunkRef.current = next;
          ttsCurrentChunkOffsetRef.current = 0;
        }

        // Copy samples from current chunk
        const chunk = ttsCurrentChunkRef.current!;
        const remaining = chunk.length - ttsCurrentChunkOffsetRef.current;
        const needed = output.length - outIdx;
        const toCopy = Math.min(remaining, needed);

        for (let i = 0; i < toCopy; i++) {
          output[outIdx++] = chunk[ttsCurrentChunkOffsetRef.current++] / 32768;
        }
      }
    };

    scriptNode.connect(ctx.destination);
    ttsPlaybackActiveRef.current = true;
    setIsSpeaking(true);
  };

  const stopTtsPlayback = () => {
    if (ttsScriptNodeRef.current) {
      try {
        ttsScriptNodeRef.current.disconnect();
      } catch { /* ignore */ }
      ttsScriptNodeRef.current = null;
    }
    if (ttsAudioCtxRef.current) {
      try {
        ttsAudioCtxRef.current.close();
      } catch { /* ignore */ }
      ttsAudioCtxRef.current = null;
    }
    ttsCurrentChunkRef.current = null;
    ttsCurrentChunkOffsetRef.current = 0;
    ttsPcmQueueRef.current = [];
    ttsPlaybackActiveRef.current = false;
    setIsSpeaking(false);
  };

  const enqueueTtsChunk = (base64Pcm: string) => {
    // Decode base64 to Int16Array
    const binary = window.atob(base64Pcm);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const int16 = new Int16Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 2));

    ttsPcmQueueRef.current.push(int16);

    // Start playback if not already running
    if (!ttsPlaybackActiveRef.current) {
      startTtsPlayback();
    }
  };

  const wsUrl = useMemo(() => {
    // Force 127.0.0.1 to avoid IPv6 localhost issues
    const base = API_BASE.replace("localhost", "127.0.0.1");
    const u = new URL(base);
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
    u.pathname = "/ws";
    return u.toString();
  }, []);

  const resumeMic = () => {
    isPausedRef.current = false;
    setIsPaused(false);
    setMicLevel(0);
  };

  // Note: TTS playback is handled by schedulePcmChunk() using WebAudio.

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        if (!cancelled) setConfigReady(false);
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
      } finally {
        if (!cancelled) setConfigReady(true);
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

  const disconnectWs = () => {
    if (connectTimeoutRef.current) {
      window.clearTimeout(connectTimeoutRef.current);
      connectTimeoutRef.current = null;
    }
    stopRecording();
    try {
      wsRef.current?.close(1000);
    } catch {
      // ignore
    }
    wsRef.current = null;
    setStatus("disconnected");
    setIsSpeaking(false);
    stopTtsPlayback();
    awaitingResumeRef.current = false;
    autoStartedMicRef.current = false;
    resumeMic();
  };

  const startRecording = async () => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setError("Voice WebSocket is not connected");
      return;
    }

    setError(null);

    // We treat recording as "mic armed". If we are waiting for playback to finish, do not start.
    if (isPausedRef.current) {
      setError("Mic is paused while the assistant is speaking");
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Microphone access requires a secure context (HTTPS) or localhost");
      return;
    }

    let mediaStream: MediaStream;
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
    } catch (e: any) {
      setError(e?.message || "Failed to access microphone");
      return;
    }
    mediaStreamRef.current = mediaStream;

    const audioCtx = new AudioContext();
    audioCtxRef.current = audioCtx;

    const source = audioCtx.createMediaStreamSource(mediaStream);
    sourceRef.current = source;

    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;

    const actualRate = audioCtx.sampleRate;

    // VAD State
    vadSilenceFramesRef.current = 0;
    vadIsSpeechActiveRef.current = false;
    const SILENCE_THRESHOLD_FRAMES = 10; // ~1 second with buffer of 4096 @ 44.1k (~92ms)
    const RMS_THRESHOLD = 0.015;

    // Accumulate bytes while speech is active (and trailing silence)
    let utteranceBytes: number[] = [];

    processor.onaudioprocess = (e) => {
      const socket = wsRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN) return;
      if (!isRecordingRef.current) return;
      if (isPausedRef.current) return;

      const input = e.inputBuffer.getChannelData(0);

      // 1. Calculate RMS
      let sumSq = 0;
      for (let i = 0; i < input.length; i++) {
        const v = input[i] || 0;
        sumSq += v * v;
      }
      const rms = Math.sqrt(sumSq / Math.max(1, input.length));
      
      // Update UI level
      const now = performance.now();
      if (now - lastLevelAtRef.current > 80) {
        lastLevelAtRef.current = now;
        setMicLevel(Math.min(1, rms * 6));
      }

      // 2. VAD Logic
      if (rms > RMS_THRESHOLD) {
        vadSilenceFramesRef.current = 0;
        if (!vadIsSpeechActiveRef.current) {
          vadIsSpeechActiveRef.current = true;
        }
      } else {
        if (vadIsSpeechActiveRef.current) {
          vadSilenceFramesRef.current++;
        }
      }

      // 3. Resample to 16kHz (Backend Whisper expectation)
      const ratio = 16000 / actualRate;
      const outputLen = Math.round(input.length * ratio);
      const resampled = new Float32Array(outputLen);
      for (let i = 0; i < outputLen; i++) {
        const srcIdx = i / ratio;
        const idx0 = Math.floor(srcIdx);
        const idx1 = Math.min(idx0 + 1, input.length - 1);
        const frac = srcIdx - idx0;
        resampled[i] = input[idx0] * (1 - frac) + input[idx1] * frac;
      }

      // 4. Convert to Int16
      const pcm = new Int16Array(resampled.length);
      for (let i = 0; i < resampled.length; i++) {
        pcm[i] = Math.max(-32768, Math.min(32767, resampled[i] * 32768));
      }

      // Only buffer/send while speech is active or while we are counting trailing silence.
      if (vadIsSpeechActiveRef.current) {
        const bytes = new Uint8Array(pcm.buffer);
        for (let i = 0; i < bytes.length; i++) utteranceBytes.push(bytes[i]);
      }

      // 6. Check for End of Speech
      if (vadIsSpeechActiveRef.current && vadSilenceFramesRef.current > SILENCE_THRESHOLD_FRAMES) {
        // Pause mic until we finish assistant playback
        isPausedRef.current = true;
        setIsPaused(true);
        setMicLevel(0);

        // Send buffered utterance once
        const base64Data = base64EncodeBytes(new Uint8Array(utteranceBytes));
        utteranceBytes = [];
        try {
          socket.send(JSON.stringify({ type: "audio", data: base64Data }));
          socket.send(JSON.stringify({ type: "end_of_speech" }));
        } catch {
          // ignore
        }
        
        // Reset VAD
        vadIsSpeechActiveRef.current = false;
        vadSilenceFramesRef.current = 0;
      }
    };

    source.connect(processor);
    processor.connect(audioCtx.destination);
    setIsRecording(true);
    isRecordingRef.current = true;
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
        ws.send(JSON.stringify({ type: "config", voice, system_prompt: systemPrompt }));
      } catch {
        // ignore
      }

      // Auto-start mic after WS connect (single-button UX)
      if (!autoStartedMicRef.current) {
        autoStartedMicRef.current = true;
        void startRecording();
      }
    };

    ws.onclose = (ev) => {
      if (nonce !== connectNonceRef.current) return;
      if (connectTimeoutRef.current) {
        window.clearTimeout(connectTimeoutRef.current);
        connectTimeoutRef.current = null;
      }
      setStatus("disconnected");
      if (ev?.code && ev.code !== 1000) {
        setError(`Voice WebSocket closed (code=${ev.code}${ev.reason ? `, reason=${ev.reason}` : ""}).`);
      }
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
          if (msg.data) {
            enqueueTtsChunk(msg.data);
          }
        } else if (msg.type === "audio_end") {
          // Server has finished sending audio. Resume mic only after playback queue drains.
          awaitingResumeRef.current = true;
          // The ScriptProcessor will detect empty queue and trigger resume
          if (!ttsPlaybackActiveRef.current && ttsPcmQueueRef.current.length === 0) {
            awaitingResumeRef.current = false;
            resumeMic();
          }
        } else if (msg.type === "transcription") {
          // no-op for now (could render)
        } else if (msg.type === "response") {
          // no-op for now (could render)
        } else if (msg.type === "error") {
          setError(msg.message || "Unknown error");
        }
      } catch {
        // ignore
      }
    };
  };

  useEffect(() => {
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
  }, [wsUrl, configReady]);

  useEffect(() => {
    if (!configReady) return;
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    try {
      ws.send(JSON.stringify({ type: "config", voice, system_prompt: systemPrompt }));
    } catch {
      // ignore
    }
  }, [voice, systemPrompt, configReady]);

  const statusDotClass =
    status === "connected" ? "bg-[#00c853]" : status === "error" ? "bg-red-500" : "bg-[#ffd400]";

  const micStatusLabel = useMemo(() => {
    if (!isRecording) return null;
    if (isPaused) return "paused";
    if (isSpeaking) return "paused";
    return "listening";
  }, [isRecording, isPaused, isSpeaking]);

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
                • {micStatusLabel}
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

            // Connected: this button is STOP (disconnect everything)
            disconnectWs();
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
          {status === "error" && "Click to start"}
          {status === "disconnected" && "Click to start"}
          {status === "connected" && "Click to stop"}
        </div>
      </div>
    </div>
  );
};
