import { useEffect, useMemo, useState, type MouseEvent } from 'react';
import { api } from '../api';
import { Pencil, Trash2 } from 'lucide-react';
import { useActiveUser } from '../state/ActiveUserContext';
import { PersonalityModal, PersonalityForModal } from '../components/PersonalityModal';
import { Link } from 'react-router-dom';
import { invoke } from '@tauri-apps/api/core';
import { VoiceActionButtons } from '../components/VoiceActionButtons';
import { useVoicePlayback } from '../hooks/useVoicePlayback';

export const Personalities = () => {
  const [personalities, setPersonalities] = useState<any[]>([]);
  const [voices, setVoices] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloadedVoiceIds, setDownloadedVoiceIds] = useState<Set<string>>(new Set());
  const [downloadingVoiceId, setDownloadingVoiceId] = useState<string | null>(null);
  const [audioSrcByVoiceId, setAudioSrcByVoiceId] = useState<Record<string, string>>({});

  const { playingVoiceId, isPaused, toggle: toggleVoice } = useVoicePlayback(async (voiceId) => {
    let src = audioSrcByVoiceId[voiceId];
    if (!src) {
      const b64 = await invoke<string | null>('read_voice_base64', { voiceId });
      if (!b64) return null;
      src = `data:audio/wav;base64,${b64}`;
      setAudioSrcByVoiceId((prev) => ({ ...prev, [voiceId]: src! }));
    }
    return src;
  });
  
  // Modal state
  const [modalOpen, setModalOpen] = useState(false);
  const [modalMode, setModalMode] = useState<'create' | 'edit'>('create');
  const [selectedPersonality, setSelectedPersonality] = useState<PersonalityForModal | null>(null);

  const { activeUserId, activeUser, refreshUsers } = useActiveUser();

  const toTimestamp = (v: any) => {
    if (v == null) return 0;
    if (typeof v === 'number') return Number.isFinite(v) ? v : 0;
    if (typeof v === 'string') {
      const asNum = Number(v);
      if (Number.isFinite(asNum)) return asNum;
      const ms = Date.parse(v);
      if (Number.isFinite(ms)) return Math.floor(ms / 1000);
    }
    return 0;
  };

  const load = async () => {
    try {
      setError(null);
      const data = await api.getPersonalities(false);
      setPersonalities(data);
    } catch (e: any) {
      setError(e?.message || 'Failed to load personalities');
    } finally {
      setLoading(false);
    }
  };

  const sortedPersonalities = useMemo(() => {
    const arr = Array.isArray(personalities) ? personalities.slice() : [];
    arr.sort((a, b) => {
      const aT = toTimestamp(a?.created_at);
      const bT = toTimestamp(b?.created_at);
      if (aT !== bT) return bT - aT;
      return 0;
    });
    return arr;
  }, [personalities]);

  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    let cancelled = false;
    const loadDownloaded = async () => {
      try {
        const ids = await invoke<string[]>('list_downloaded_voices');
        if (!cancelled) setDownloadedVoiceIds(new Set(Array.isArray(ids) ? ids : []));
      } catch {
        if (!cancelled) setDownloadedVoiceIds(new Set());
      }
    };
    loadDownloaded();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const loadVoices = async () => {
      try {
        const data = await api.getVoices();
        if (!cancelled) setVoices(Array.isArray(data) ? data : []);
      } catch {
        if (!cancelled) setVoices([]);
      }
    };
    loadVoices();
    return () => {
      cancelled = true;
    };
  }, []);

  const voiceById = useMemo(() => {
    const m = new Map<string, any>();
    for (const v of voices) {
      if (v?.voice_id) m.set(String(v.voice_id), v);
    }
    return m;
  }, [voices]);

  const downloadVoice = async (voiceId: string) => {
    setDownloadingVoiceId(voiceId);
    try {
      await invoke<string>('download_voice', { voiceId });
      setDownloadedVoiceIds((prev) => {
        const next = new Set(prev);
        next.add(voiceId);
        return next;
      });
    } catch (e: any) {
      console.error('download_voice failed', e);
      const msg = typeof e === 'string' ? e : e?.message ? String(e.message) : String(e);
      setError(msg || 'Failed to download voice');
    } finally {
      setDownloadingVoiceId(null);
    }
  };

  const togglePlay = async (voiceId: string) => {
    if (!downloadedVoiceIds.has(voiceId)) return;
    try {
      await toggleVoice(voiceId);
    } catch (e) {
      console.error('toggleVoice failed', e);
    }
  };

  const assignToActiveUser = async (personalityId: string) => {
    if (!activeUserId) {
      setError('Select an active user first');
      return;
    }
    try {
      setError(null);
      await api.updateUser(activeUserId, { current_personality_id: personalityId });
      await refreshUsers();
      try {
        await api.setAppMode('chat');
      } catch {
        // non-blocking
      }
    } catch (e: any) {
      setError(e?.message || 'Failed to assign personality');
    }
  };

  const deletePersonality = async (p: any, e: MouseEvent) => {
    e.stopPropagation();
    if (p?.is_global) return;
    const ok = window.confirm('Delete this personality? This will also delete its sessions and conversations.');
    if (!ok) return;
    try {
      setError(null);
      await api.deletePersonality(p.id);
      await load();
    } catch (err: any) {
      setError(err?.message || 'Failed to delete personality');
    }
  };

  const handleCreate = () => {
    setModalMode('create');
    setSelectedPersonality(null);
    setModalOpen(true);
  };

  const handleEdit = (p: any, e: React.MouseEvent) => {
    e.stopPropagation();
    setModalMode('edit');
    setSelectedPersonality(p);
    setModalOpen(true);
  };

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-3xl font-black">PERSONALITIES</h2>
        <button 
          className="retro-btn"
          onClick={handleCreate}
        >
          + Create
        </button>
      </div>

      <PersonalityModal 
        open={modalOpen}
        mode={modalMode}
        personality={selectedPersonality}
        onClose={() => setModalOpen(false)}
        onSuccess={async () => {
          await load();
        }}
      />

      {loading && (
        <div className="retro-card font-mono text-sm">Loading…</div>
      )}
      {error && (
        <div className="retro-card font-mono text-sm">{error}</div>
      )}
      {!loading && !error && personalities.length === 0 && (
        <div className="retro-card font-mono text-sm">No personalities found.</div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {sortedPersonalities.map((p) => (
          <div
            key={p.id}
            role="button"
            tabIndex={0}
            onClick={() => assignToActiveUser(p.id)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') assignToActiveUser(p.id);
            }}
            className={`retro-card relative group text-left cursor-pointer transition-shadow flex flex-col ${activeUser?.current_personality_id === p.id ? 'retro-selected' : 'retro-not-selected'}`}
          >
            <div className="absolute top-4 right-4 flex items-center gap-2">
              {!p.is_global && (
                <button
                  type="button"
                  className="retro-icon-btn"
                  aria-label="Edit personality"
                  onClick={(e) => handleEdit(p, e)}
                >
                  <Pencil size={16} />
                </button>
              )}

              {!p.is_global && (
                <button
                  type="button"
                  className="retro-icon-btn"
                  aria-label="Delete personality"
                  onClick={(e) => deletePersonality(p, e)}
                  title="Delete"
                >
                  <Trash2 size={16} />
                </button>
              )}
            </div>

            <h3 className="text-xl font-bold mb-2 pr-14">{p.name}</h3>
            <p className="text-gray-600 mb-4 text-sm font-medium border-l-4 border-gray-300 pl-2">
              "{p.short_description}"
            </p>

            <div className="mt-auto border-t-2 border-black pt-3 hover:scale-103 transition-transform">
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-bold uppercase">Voice</div>
                  <Link
                    to={`/voices?voice_id=${encodeURIComponent(p.voice_id)}`}
                    onClick={(e) => e.stopPropagation()}
                    className="block font-bold truncate"
                    title="View voice"
                  >
                    {voiceById.get(p.voice_id)?.voice_name || p.voice_id}
                  </Link>
                </div>

                <div className="shrink-0">
                  <VoiceActionButtons
                    voiceId={String(p.voice_id)}
                    isDownloaded={downloadedVoiceIds.has(String(p.voice_id))}
                    downloadingVoiceId={downloadingVoiceId}
                    onDownload={(id) => downloadVoice(id)}
                    onTogglePlay={(id) => togglePlay(id)}
                    isPlaying={playingVoiceId === String(p.voice_id)}
                    isPaused={isPaused}
                    stopPropagation
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
