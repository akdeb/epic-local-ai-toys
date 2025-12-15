import { Link, useLocation } from 'react-router-dom';
import { Users, MessageSquare, Settings, Mic2, Lock } from 'lucide-react';
import clsx from 'clsx';
import { useActiveUser } from '../state/ActiveUserContext';
import { useEffect, useState } from 'react';
import { api } from '../api';

const NavItem = ({
  to,
  icon: Icon,
  label,
  trailingIcon: TrailingIcon,
}: {
  to: string;
  icon: any;
  label: string;
  trailingIcon?: any;
}) => {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <Link
      to={to}
      className={clsx(
        "flex items-center gap-3 px-4 py-3 border border-black rounded-[18px] transition-colors hover:bg-[#fff3b0]",
        isActive 
          ? "bg-[#ffd400] text-black" 
          : "bg-white"
      )}
    >
      <Icon size={20} />
      <span className="font-bold flex-1">{label}</span>
      {TrailingIcon && <TrailingIcon size={16} className="opacity-70" />}
    </Link>
  );
};

export const Sidebar = () => {
  const { users, activeUserId, activeUser, setActiveUserId } = useActiveUser();
  const [activePersonalityName, setActivePersonalityName] = useState<string | null>(null);
  const [deviceConnected, setDeviceConnected] = useState<boolean>(false);
  const [deviceSessionId, setDeviceSessionId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const ds = await api.getDeviceStatus().catch(() => ({ connected: false, session_id: null }));
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
    const id = window.setInterval(load, 1500);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [activeUser?.current_personality_id]);

  return (
    <div className="w-64 shrink-0 bg-[#f6f0e6] border-r-2 border-black p-4 flex flex-col gap-4 h-full overflow-y-auto overscroll-contain">
      <div className="mb-6 p-4 border-2 border-black bg-[#9b5cff] text-white rounded-[24px]">
        <h1 className="text-2xl font-black tracking-wider brand-font">ELATO</h1>
        <p className="text-xs font-mono opacity-90">Epic Local AI Toys</p>

        <div className="mt-4">
          <div className="text-[10px] font-bold uppercase tracking-wider opacity-90 mb-1">
            Active User
          </div>
          <select
            className="w-full px-3 py-2 bg-white text-black border-2 border-black rounded-[18px]"
            value={activeUserId || ''}
            onChange={(e) => setActiveUserId(e.target.value || null)}
          >
            {users.length === 0 && <option value="">No users</option>}
            {users.map((u) => (
              <option key={u.id} value={u.id}>
                {u.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      <nav className="flex flex-col gap-3">
        <NavItem to="/" icon={Mic2} label="Personalities" />
        <NavItem to="/conversations" icon={MessageSquare} label="Conversations" trailingIcon={Lock} />
        <NavItem to="/users" icon={Users} label="Users" />
        <NavItem to="/settings" icon={Settings} label="AI Settings" />
      </nav>
    </div>
  );
};
