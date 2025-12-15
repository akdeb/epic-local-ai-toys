import { useEffect, useState } from 'react';
import { api } from '../api';
import { Eye, EyeOff } from 'lucide-react';
import { useActiveUser } from '../state/ActiveUserContext';

export const Personalities = () => {
  const [personalities, setPersonalities] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showHidden, setShowHidden] = useState(false);
  const { activeUserId, activeUser, refreshUsers } = useActiveUser();

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        setError(null);
        const data = await api.getPersonalities(true);
        if (!cancelled) setPersonalities(data);
      } catch (e: any) {
        if (!cancelled) setError(e?.message || 'Failed to load personalities');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, []);

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
        // non-blocking: selection should still work even if app-mode endpoint is unavailable
      }
    } catch (e: any) {
      setError(e?.message || 'Failed to assign personality');
    }
  };

  const toggleVisibility = async (p: any) => {
    try {
      setError(null);
      await api.updatePersonality(p.id, { is_visible: !p.is_visible });
      const data = await api.getPersonalities(true);
      setPersonalities(data);
    } catch (e: any) {
      setError(e?.message || 'Failed to update visibility');
    }
  };

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-3xl font-black">PERSONALITIES</h2>
        <button className="retro-btn">
          + Create
        </button>
      </div>

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
              <div className="bg-black text-white px-2 py-1 text-xs font-bold uppercase">
                {p.voice_id}
              </div>
            </div>
            <h3 className="text-xl font-bold mb-2">{p.name}</h3>
            <p className="text-gray-600 mb-4 text-sm font-medium border-l-4 border-gray-300 pl-2">
              "{p.short_description}"
            </p>
            <div className="flex flex-wrap gap-2">
              {p.tags.map((tag: string) => (
                <span key={tag} className="px-2 py-1 bg-[#fff3b0] border border-black text-xs font-bold uppercase">
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
          className="retro-btn px-4 py-2 text-sm"
          onClick={() => setShowHidden((v) => !v)}
        >
          {showHidden ? 'Hide Hidden Personalities' : `Show Hidden Personalities (${hidden.length})`}
        </button>
        {showHidden && (
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-6">
            {hidden.map((p) => (
              <div key={p.id} className="retro-card relative">
                <div className="absolute top-4 right-4 flex items-center gap-2">
                  <button
                    type="button"
                    className="retro-icon-btn"
                    aria-label="Unhide personality"
                    onClick={() => toggleVisibility(p)}
                  >
                    <Eye size={16} />
                  </button>
                  <div className="bg-black text-white px-2 py-1 text-xs font-bold uppercase">
                    {p.voice_id}
                  </div>
                </div>
                <h3 className="text-xl font-bold mb-2">{p.name}</h3>
                <p className="text-gray-600 mb-4 text-sm font-medium border-l-4 border-gray-300 pl-2">
                  "{p.short_description}"
                </p>
              </div>
            ))}
          </div>
        )}
      </div>

    </div>
  );
};
