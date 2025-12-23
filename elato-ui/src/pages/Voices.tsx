import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Plus, X } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import { PersonalityModal } from "../components/PersonalityModal";
import { VoiceActionButtons } from "../components/VoiceActionButtons";
import { useVoicePlayback } from "../hooks/useVoicePlayback";

export const VoicesPage = () => {
  const navigate = useNavigate();
  const [voices, setVoices] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloadingVoiceId, setDownloadingVoiceId] = useState<string | null>(null);
  const [downloadedVoiceIds, setDownloadedVoiceIds] = useState<Set<string>>(new Set());
  const [audioSrcByVoiceId, setAudioSrcByVoiceId] = useState<Record<string, string>>({});
  const [searchParams] = useSearchParams();

  const toTimestamp = (v: any) => {
    if (v == null) return 0;
    if (typeof v === "number") return Number.isFinite(v) ? v : 0;
    if (typeof v === "string") {
      const asNum = Number(v);
      if (Number.isFinite(asNum)) return asNum;
      const ms = Date.parse(v);
      if (Number.isFinite(ms)) return Math.floor(ms / 1000);
    }
    return 0;
  };

  const [createVoiceOpen, setCreateVoiceOpen] = useState(false);
  const [cloneName, setCloneName] = useState("");
  const [cloneDesc, setCloneDesc] = useState("");
  const [cloneFile, setCloneFile] = useState<File | null>(null);
  const [clonePreviewUrl, setClonePreviewUrl] = useState<string | null>(null);
  const [creatingVoice, setCreatingVoice] = useState(false);
  const [recording, setRecording] = useState(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordChunksRef = useRef<Blob[]>([]);
  const recordStopTimeoutRef = useRef<number | null>(null);

  const [createPersonalityOpen, setCreatePersonalityOpen] = useState(false);
  const [createPersonalityVoiceId, setCreatePersonalityVoiceId] = useState<string | null>(null);
  const [createPersonalityVoiceName, setCreatePersonalityVoiceName] = useState<string | null>(null);

  const sortedVoices = useMemo(() => {
    const arr = Array.isArray(voices) ? voices.slice() : [];
    arr.sort((a, b) => {
      const aT = toTimestamp(a?.created_at);
      const bT = toTimestamp(b?.created_at);
      if (aT !== bT) return bT - aT;
      return 0;
    });
    return arr;
  }, [voices]);

  const encodeWav16 = (samples: Float32Array, sampleRate: number) => {
    const numChannels = 1;
    const bytesPerSample = 2;
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = samples.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    const writeStr = (off: number, s: string) => {
      for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
    };

    writeStr(0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);
    writeStr(36, "data");
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }

    return buffer;
  };

  const decodeToWavFile = async (blob: Blob) => {
    const buf = await blob.arrayBuffer();
    const ctx = new AudioContext();
    const audio = await ctx.decodeAudioData(buf.slice(0));
    const input = audio.getChannelData(0);

    const targetRate = 16000;
    const ratio = targetRate / audio.sampleRate;
    const outLen = Math.max(1, Math.round(input.length * ratio));
    const resampled = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const src = i / ratio;
      const idx0 = Math.floor(src);
      const idx1 = Math.min(idx0 + 1, input.length - 1);
      const frac = src - idx0;
      resampled[i] = input[idx0] * (1 - frac) + input[idx1] * frac;
    }

    await ctx.close();

    const wav = encodeWav16(resampled, targetRate);
    return new File([wav], "recording.wav", { type: "audio/wav" });
  };

  const stopRecorder = async () => {
    if (recordStopTimeoutRef.current) {
      window.clearTimeout(recordStopTimeoutRef.current);
      recordStopTimeoutRef.current = null;
    }
    const rec = recorderRef.current;
    if (!rec) return;
    try {
      if (rec.state !== "inactive") rec.stop();
    } catch {
      // ignore
    }
  };

  const startRecord10s = async () => {
    if (recording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordChunksRef.current = [];
      const rec = new MediaRecorder(stream);
      recorderRef.current = rec;
      setRecording(true);

      rec.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) recordChunksRef.current.push(e.data);
      };

      rec.onstop = async () => {
        try {
          stream.getTracks().forEach((t) => t.stop());
        } catch {
          // ignore
        }

        setRecording(false);
        recorderRef.current = null;
        const blob = new Blob(recordChunksRef.current, { type: rec.mimeType || "audio/webm" });
        recordChunksRef.current = [];

        try {
          const wavFile = await decodeToWavFile(blob);
          setCloneFile(wavFile);
          if (clonePreviewUrl) URL.revokeObjectURL(clonePreviewUrl);
          setClonePreviewUrl(URL.createObjectURL(wavFile));
        } catch (e) {
          console.error(e);
          setError("Failed to process recording");
        }
      };

      rec.start();
      recordStopTimeoutRef.current = window.setTimeout(() => {
        void stopRecorder();
      }, 10_000);
    } catch (e: any) {
      setRecording(false);
      setError(e?.message || "Failed to start recording");
    }
  };

  const selectedVoiceId = useMemo(() => {
    const v = searchParams.get("voice_id");
    return v ? String(v) : null;
  }, [searchParams]);

  const selectedRef = useRef<HTMLDivElement | null>(null);
  const { playingVoiceId, isPaused, toggle: toggleVoice } = useVoicePlayback(async (voiceId) => {
    let src = audioSrcByVoiceId[voiceId];
    if (!src) {
      const b64 = await invoke<string | null>("read_voice_base64", { voiceId });
      if (!b64) return null;
      src = `data:audio/wav;base64,${b64}`;
      setAudioSrcByVoiceId((prev) => ({ ...prev, [voiceId]: src! }));
    }
    return src;
  });

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        setError(null);
        const data = await api.getVoices();
        if (!cancelled) setVoices(Array.isArray(data) ? data : []);
      } catch (e: any) {
        if (!cancelled) setError(e?.message || "Failed to load voices");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    const loadAll = async () => {
      await load();
      try {
        const ids = await invoke<string[]>("list_downloaded_voices");
        if (!cancelled) setDownloadedVoiceIds(new Set(Array.isArray(ids) ? ids : []));
      } catch {
        if (!cancelled) setDownloadedVoiceIds(new Set());
      }
    };

    loadAll();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (clonePreviewUrl) URL.revokeObjectURL(clonePreviewUrl);
    };
  }, [clonePreviewUrl]);

  useEffect(() => {
    if (createVoiceOpen) return;
    void stopRecorder();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [createVoiceOpen]);

  useEffect(() => {
    return () => {
      void stopRecorder();
      if (recordStopTimeoutRef.current) {
        window.clearTimeout(recordStopTimeoutRef.current);
        recordStopTimeoutRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fileToBase64 = async (file: File): Promise<string> => {
    const buf = await file.arrayBuffer();
    let binary = "";
    const bytes = new Uint8Array(buf);
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      const chunk = bytes.subarray(i, i + chunkSize);
      binary += String.fromCharCode(...chunk);
    }
    return btoa(binary);
  };

  const slugify = (s: string) =>
    s
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/(^-|-$)/g, "");

  const createVoiceClone = async () => {
    if (!cloneFile) return;
    if (!cloneName.trim()) {
      setError("Name is required");
      return;
    }
    setCreatingVoice(true);
    try {
      setError(null);
      const uuid = globalThis.crypto?.randomUUID ? globalThis.crypto.randomUUID() : String(Date.now());
      const voiceId = `${slugify(cloneName)}-${uuid}`;
      const b64 = await fileToBase64(cloneFile);
      await invoke<string>("save_voice_wav_base64", { voiceId, base64Wav: b64 });
      await api.createVoice({ voice_id: voiceId, voice_name: cloneName.trim(), voice_description: cloneDesc.trim() || null });

      const data = await api.getVoices();
      setVoices(Array.isArray(data) ? data : []);
      setDownloadedVoiceIds((prev) => {
        const next = new Set(prev);
        next.add(voiceId);
        return next;
      });

      setCreateVoiceOpen(false);
      setCloneName("");
      setCloneDesc("");
      setCloneFile(null);
      if (clonePreviewUrl) URL.revokeObjectURL(clonePreviewUrl);
      setClonePreviewUrl(null);
    } catch (e: any) {
      const msg = typeof e === "string" ? e : e?.message ? String(e.message) : String(e);
      setError(msg || "Failed to create voice");
    } finally {
      setCreatingVoice(false);
    }
  };

  useEffect(() => {
    if (!selectedVoiceId) return;
    if (!selectedRef.current) return;
    selectedRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [selectedVoiceId, voices.length]);

  const downloadVoice = async (voiceId: string) => {
    setDownloadingVoiceId(voiceId);
    try {
      await invoke<string>("download_voice", { voiceId });
      setDownloadedVoiceIds((prev) => {
        const next = new Set(prev);
        next.add(voiceId);
        return next;
      });
    } catch (e: any) {
      console.error("download_voice failed", e);
      const msg =
        typeof e === "string"
          ? e
          : e?.message
            ? String(e.message)
            : e?.toString
              ? String(e.toString())
              : "Failed to download voice";
      setError(msg);
    } finally {
      setDownloadingVoiceId(null);
    }
  };

  const togglePlay = async (voiceId: string) => {
    if (!downloadedVoiceIds.has(voiceId)) return;
    try {
      await toggleVoice(voiceId);
    } catch (e) {
      console.error("toggleVoice failed", e);
    }
  };

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-3xl font-black">VOICES</h2>
        <button type="button" className="retro-btn" onClick={() => setCreateVoiceOpen(true)}>       
          +  Create    
        </button>
      </div>

      <PersonalityModal
        open={createPersonalityOpen}
        mode="create"
        createVoiceId={createPersonalityVoiceId}
        createVoiceName={createPersonalityVoiceName}
        onClose={() => setCreatePersonalityOpen(false)}
        onSuccess={async () => {
          setCreatePersonalityOpen(false);
          navigate('/');
        }}
      />

      {createVoiceOpen && (
        <div className="fixed inset-0 z-50 backdrop-blur-sm flex items-center justify-center p-6">
          <button
            type="button"
            aria-label="Close"
            className="absolute inset-0 bg-black/40"
            onClick={() => {
              void stopRecorder();
              setCreateVoiceOpen(false);
            }}
          />
          <div className="relative w-full max-w-xl retro-card">
            <div className="flex items-start justify-between gap-4 mb-4">
              <div className="text-xl font-black uppercase">Create Voice Clone</div>
              <button
                type="button"
                className="retro-icon-btn"
                onClick={() => {
                  void stopRecorder();
                  setCreateVoiceOpen(false);
                }}
                aria-label="Close"
              >
                <X />
              </button>
            </div>

            <div className="space-y-4">
              <div className="font-mono text-xs text-gray-700">
                Upload or record a clean sample. Best results with: 1. quiet room, 2. steady volume, 3. no background music, 4. 10-12s seconds is great
              </div>

              <div>
                <label className="block font-bold mb-2 uppercase text-sm">Audio Sample (WAV)</label>
                <input
                  type="file"
                  accept="audio/wav"
                  onChange={(e) => {
                    const f = e.target.files?.[0] || null;
                    setCloneFile(f);
                    if (clonePreviewUrl) URL.revokeObjectURL(clonePreviewUrl);
                    setClonePreviewUrl(f ? URL.createObjectURL(f) : null);
                  }}
                />
                <div className="mt-3 flex gap-2">
                  <button
                    type="button"
                    className="retro-btn"
                    onClick={startRecord10s}
                    disabled={recording}
                  >
                    {recording ? "Recording…" : "Record 10s"}
                  </button>
                  <button
                    type="button"
                    className="retro-btn retro-btn-outline"
                    onClick={() => stopRecorder()}
                    disabled={!recording}
                  >
                    Stop
                  </button>
                </div>
              </div>

              {clonePreviewUrl && (
                <div>
                  <label className="block font-bold mb-2 uppercase text-sm">Preview</label>
                  <audio controls src={clonePreviewUrl} className="w-full" />
                </div>
              )}

              <div>
                <label className="block font-bold mb-2 uppercase text-sm">Name</label>
                <input className="retro-input" value={cloneName} onChange={(e) => setCloneName(e.target.value)} placeholder="Winnie the Pooh" />
              </div>

              <div>
                <label className="block font-bold mb-2 uppercase text-sm">Short Description</label>
                <input className="retro-input" value={cloneDesc} onChange={(e) => setCloneDesc(e.target.value)} placeholder="Bear-like cartoon voice" />
              </div>

              <div className="flex justify-end">
                <button type="button" className="retro-btn" onClick={createVoiceClone} disabled={creatingVoice || !cloneFile || !cloneName.trim()}>
                  {creatingVoice ? "Saving…" : "Submit"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {loading && <div className="retro-card font-mono text-sm mb-4">Loading…</div>}
      {error && <div className="retro-card font-mono text-sm mb-4">{error}</div>}

      {!loading && !error && voices.length === 0 && (
        <div className="retro-card font-mono text-sm mb-4">No voices found.</div>
      )}

      <div className="flex flex-col gap-3">
        {sortedVoices.map((v) => (
          <div
            key={v.voice_id}
            ref={selectedVoiceId === v.voice_id ? selectedRef : null}
            className={`retro-card relative ${selectedVoiceId === v.voice_id ? "retro-selected" : ""}`}
          >
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <h3 className="text-lg font-black inline-flex items-center gap-2">
                  <span>{v.voice_name || v.voice_id}</span>
                  {/* <span
                    className="inline-flex items-center justify-center align-middle"
                    title={String(v.gender).toLowerCase() === "female" ? "female voice" : "male voice"}
                  >
                    <span className="text-sm leading-none relative top-[1px]">
                      {String(v.gender).toLowerCase() === "female" ? "🙋‍♀️" : "🙋‍♂️"}
                    </span>
                  </span> */}
                </h3>
                
                <p className="text-gray-600 text-sm font-medium mt-1">
                  {v.voice_description ? v.voice_description : "—"}
                </p>
              </div>

              <div className="shrink-0 pt-1">
                <div className="flex flex-col items-end gap-2">
                  <button
                    type="button"
                    className="retro-btn"
                    onClick={() => {
                      setCreatePersonalityVoiceId(String(v.voice_id));
                      setCreatePersonalityVoiceName(String(v.voice_name || v.voice_id));
                      setCreatePersonalityOpen(true);
                    }}
                  >
                    <span className="inline-flex items-center gap-2">
                    <Plus size={16} />
                    </span>
                  </button>
                  <VoiceActionButtons
                    voiceId={String(v.voice_id)}
                    isDownloaded={downloadedVoiceIds.has(String(v.voice_id))}
                    downloadingVoiceId={downloadingVoiceId}
                    onDownload={(id) => downloadVoice(id)}
                    onTogglePlay={(id) => togglePlay(id)}
                    isPlaying={playingVoiceId === String(v.voice_id)}
                    isPaused={isPaused}
                  />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
