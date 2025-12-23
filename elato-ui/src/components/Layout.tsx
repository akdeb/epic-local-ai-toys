import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { useActiveUser } from '../state/ActiveUserContext';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import { useEffect, useState } from 'react';
import { Bot, ShieldCheck } from 'lucide-react';

export const Layout = () => {
  const { activeUser } = useActiveUser();
  const navigate = useNavigate();
  const [activePersonalityName, setActivePersonalityName] = useState<string | null>(null);
  const [deviceConnected, setDeviceConnected] = useState<boolean>(false);
  const [deviceSessionId, setDeviceSessionId] = useState<string | null>(null);

  const statusLabel = deviceConnected ? 'Chat in progress' : 'Ready to connect';
  const statusDotClass = deviceConnected ? 'bg-[#00c853]' : 'bg-[#ffd400]';
  const statusTextClass = deviceConnected ? 'text-green-800' : 'text-yellow-900';

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const ds = { connected: false, session_id: null };
        
        // await api.getDeviceStatus().catch(() => ({ connected: false, session_id: null }));
        if (!cancelled) {
          setDeviceConnected(!!ds?.connected);
          setDeviceSessionId(ds?.session_id || null);
        }

        const selectedId = activeUser?.current_personality_id;
        if (!selectedId) {
          if (!cancelled) setActivePersonalityName(null);
          return;
        }

        const ps = await api.getPersonalities(true).catch(() => []);
        const selected = ps.find((p: any) => p.id === selectedId);
        if (!cancelled) setActivePersonalityName(selected?.name || null);
      } catch {
        // ignore
      }
    };

    load();
  }, [activeUser?.current_personality_id]);

  return (
    <div className="flex h-screen overflow-hidden bg-[#f6f0e6] retro-dots">
      <Sidebar />
      <main className="flex-1 min-h-0 p-8 pb-36 overflow-y-auto">
        <div className="max-w-4xl mx-auto">
          <Outlet />
        </div>

        {activeUser?.current_personality_id && (
          <div className="fixed bottom-0 left-64 right-0 pointer-events-none">
            <div className="max-w-4xl mx-auto px-8 pb-6 pointer-events-auto">
              <div className="bg-white border-2 border-black rounded-[24px] px-5 py-4 flex items-center justify-between shadow-[0_10px_24px_rgba(0,0,0,0.14)]">
                <div className="min-w-0">
                  <div className="flex items-center flex-row gap-3">
                    <div className="font-mono text-xs text-gray-500">Active</div>
                    <ShieldCheck size={16} className="text-gray-500" />
                  </div>
                  <div className="mt-0.5 flex items-center gap-3 min-w-0">
                    <div className="font-black text-base text-black truncate">{activePersonalityName || '—'}</div>
                    <div className="inline-flex items-center gap-2 font-mono text-[11px] shrink-0">
                      <span className={`w-2 h-2 rounded-full border border-black ${statusDotClass}`} />
                      <span className={statusTextClass}>{statusLabel}</span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    className="retro-btn bg-white px-4 py-2 text-sm flex items-center gap-2"
                    onClick={() => navigate('/test')}
                  >
                  <Bot size={18} className="flex-shrink-0" />  Test
                  </button>
                  {deviceConnected && deviceSessionId && (
                    <button
                      type="button"
                      className="retro-btn bg-white px-4 py-2 text-sm"
                      onClick={() => navigate(`/conversations?session=${encodeURIComponent(deviceSessionId)}`)}
                    >
                      View
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};
