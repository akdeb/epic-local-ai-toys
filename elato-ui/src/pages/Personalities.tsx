import { useEffect, useMemo, useState } from 'react';
import { api } from '../api';
import { Download, Eye, EyeOff, Pencil } from 'lucide-react';
import { useActiveUser } from '../state/ActiveUserContext';
import { PersonalityModal, PersonalityForModal } from '../components/PersonalityModal';
import { Link } from 'react-router-dom';

export const Personalities = () => {
  const [personalities, setPersonalities] = useState<any[]>([]);
  const [voices, setVoices] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showHidden, setShowHidden] = useState(false);
  
  // Modal state
  const [modalOpen, setModalOpen] = useState(false);
  const [modalMode, setModalMode] = useState<'create' | 'edit'>('create');
  const [selectedPersonality, setSelectedPersonality] = useState<PersonalityForModal | null>(null);

  const { activeUserId, activeUser, refreshUsers } = useActiveUser();

  const load = async () => {
    try {
      setError(null);
      const data = await api.getPersonalities(true);
      setPersonalities(data);
    } catch (e: any) {
      setError(e?.message || 'Failed to load personalities');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
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

  const visible = personalities.filter((p) => p.is_visible);
  const hidden = personalities.filter((p) => !p.is_visible);

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

  const toggleVisibility = async (p: any) => {
    try {
      setError(null);
      await api.updatePersonality(p.id, { is_visible: !p.is_visible });
      load();
    } catch (e: any) {
      setError(e?.message || 'Failed to update visibility');
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
        {visible.map((p) => (
          <div
            key={p.id}
            role="button"
            tabIndex={0}
            onClick={() => assignToActiveUser(p.id)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') assignToActiveUser(p.id);
            }}
            className={`retro-card relative group text-left cursor-pointer transition-shadow ${activeUser?.current_personality_id === p.id ? 'retro-selected' : ''}`}
          >
            <div className="absolute top-4 right-4 flex items-center gap-2">
              {/* Edit button for non-global personalities */}
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
              
              <button
                type="button"
                className="retro-icon-btn"
                aria-label="Hide personality"
                onClick={(e) => {
                  e.stopPropagation();
                  toggleVisibility(p);
                }}
              >
                <EyeOff size={16} />
              </button>

            </div>
            <div className="absolute bottom-4 right-4 flex items-center gap-2">
              {/* Edit button for non-global personalities */}
              {!p.is_global && (
                <button
                  type="button"
                  className="retro-icon-btn"
                  aria-label="Edit personality"
                  onClick={(e) => handleEdit(p, e)}
                >
                  <Download size={16} />
                </button>
              )}
            </div>
            <h3 className="text-xl font-bold mb-2">{p.name}</h3>
            <p className="text-gray-600 mb-4 text-sm font-medium border-l-4 border-gray-300 pl-2">
              "{p.short_description}"
            </p>
            <div className="mb-4">
              <Link
                to={`/voices?voice_id=${encodeURIComponent(p.voice_id)}`}
                onClick={(e) => e.stopPropagation()}
                className="inline-block px-2 py-1 border border-black text-xs font-bold lowercase"
                title="View voice"
              >
                {voiceById.get(p.voice_id)?.voice_name || p.voice_id}
              </Link>
            </div>
            <div className="flex flex-wrap gap-2">
              {p.tags.map((tag: string) => (
                <span key={tag} className="px-2 py-1 border border-black text-xs font-bold lowercase">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-8">
        <button
          type="button"
          className="retro-btn retro-btn-outline px-4 py-2 text-sm"
          onClick={() => setShowHidden((v) => !v)}
        >
          {showHidden ? 'Hide Hidden Personalities' : `Show Hidden Personalities (${hidden.length})`}
        </button>
        {showHidden && (
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-6">
            {hidden.map((p) => (
              <div key={p.id} className="retro-card relative">
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
                  <button
                    type="button"
                    className="retro-icon-btn"
                    aria-label="Unhide personality"
                    onClick={() => toggleVisibility(p)}
                  >
                    <Eye size={16} />
                  </button>
                </div>
                <h3 className="text-xl font-bold mb-2">{p.name}</h3>
                <p className="text-gray-600 mb-4 text-sm font-medium border-l-4 border-gray-300 pl-2">
                  "{p.short_description}"
                </p>
                <div className="mb-4">
                  <Link
                    to={`/voices?voice_id=${encodeURIComponent(p.voice_id)}`}
                    className="inline-block px-2 py-1 border border-black text-xs font-bold lowercase"
                    title="View voice"
                  >
                    {voiceById.get(p.voice_id)?.voice_name || p.voice_id}
                  </Link>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

    </div>
  );
};
