import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { useSearchParams } from "react-router-dom";
import { Download, Loader2, Play } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";

export const VoicesPage = () => {
  const [voices, setVoices] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloadingVoiceId, setDownloadingVoiceId] = useState<string | null>(null);
  const [downloadedVoiceIds, setDownloadedVoiceIds] = useState<Set<string>>(new Set());
  const [audioSrcByVoiceId, setAudioSrcByVoiceId] = useState<Record<string, string>>({});
  const [searchParams] = useSearchParams();

  const sortedVoices = useMemo(() => {
    const arr = Array.isArray(voices) ? voices.slice() : [];
    arr.sort((a, b) => {
      const aId = String(a?.voice_id || "");
      const bId = String(b?.voice_id || "");
      const aDownloaded = downloadedVoiceIds.has(aId);
      const bDownloaded = downloadedVoiceIds.has(bId);
      if (aDownloaded !== bDownloaded) return aDownloaded ? -1 : 1;

      const aName = String(a?.voice_name || aId);
      const bName = String(b?.voice_name || bId);
      return aName.localeCompare(bName);
    });
    return arr;
  }, [voices, downloadedVoiceIds]);

  const selectedVoiceId = useMemo(() => {
    const v = searchParams.get("voice_id");
    return v ? String(v) : null;
  }, [searchParams]);

  const selectedRef = useRef<HTMLDivElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

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

  const playVoice = async (voiceId: string) => {
    if (!downloadedVoiceIds.has(voiceId)) return;
    try {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
      }

      let src = audioSrcByVoiceId[voiceId];
      if (!src) {
        const b64 = await invoke<string | null>("read_voice_base64", { voiceId });
        if (!b64) return;
        src = `data:audio/wav;base64,${b64}`;
        setAudioSrcByVoiceId((prev) => ({ ...prev, [voiceId]: src! }));
      }

      const audio = new Audio(src);
      audioRef.current = audio;
      await audio.play();
    } catch (e) {
      console.error("playVoice failed", e);
    }
  };

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
      }
    };
  }, []);

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-3xl font-black">VOICES</h2>
      </div>

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
                {downloadedVoiceIds.has(v.voice_id) ? (
                  <button
                    type="button"
                    className="retro-btn"
                    onClick={() => playVoice(v.voice_id)}
                    title={`Play ${v.voice_id}.wav`}
                  >
                    <span className="inline-flex items-center gap-2">
                      <Play fill="currentColor" size={16} />
                    </span>
                  </button>
                ) : (
                  <button
                    type="button"
                    className="retro-btn"
                    onClick={() => downloadVoice(v.voice_id)}
                    disabled={downloadingVoiceId === v.voice_id}
                    title={`Download ${v.voice_id}.wav`}
                  >
                    <span className="inline-flex items-center gap-2">
                      {downloadingVoiceId === v.voice_id ? (
                        <Loader2 size={16} className="animate-spin" />
                      ) : (
                        <Download size={16} />
                      )}
                    </span>
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
