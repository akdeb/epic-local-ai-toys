import { Link, useLocation } from 'react-router-dom';
import { Users, MessageSquare, Settings, Mic2 } from 'lucide-react';
import clsx from 'clsx';

const NavItem = ({ to, icon: Icon, label }: { to: string; icon: any; label: string }) => {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <Link
      to={to}
      className={clsx(
        "flex items-center gap-3 px-4 py-3 border-2 border-black transition-all hover:bg-[#eee8d5]",
        isActive 
          ? "bg-[#2aa198] text-white shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] translate-x-[2px] translate-y-[2px]" 
          : "bg-white shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[3px_3px_0px_0px_rgba(0,0,0,1)]"
      )}
    >
      <Icon size={20} />
      <span className="font-bold">{label}</span>
    </Link>
  );
};

export const Sidebar = () => {
  return (
    <div className="w-64 bg-[#fdf6e3] border-r-2 border-black p-4 flex flex-col gap-4 h-screen">
      <div className="mb-6 p-4 border-2 border-black bg-[#ff6b6b] text-white shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
        <h1 className="text-2xl font-black tracking-tighter">ELATO</h1>
        <p className="text-xs font-mono opacity-90">AI COMPANION v1.0</p>
      </div>

      <nav className="flex flex-col gap-3">
        <NavItem to="/" icon={Mic2} label="Personalities" />
        <NavItem to="/conversations" icon={MessageSquare} label="Conversations" />
        <NavItem to="/users" icon={Users} label="Users" />
        <NavItem to="/settings" icon={Settings} label="Settings" />
      </nav>

      <div className="mt-auto p-4 border-2 border-black bg-white shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-3 h-3 rounded-full bg-green-500 border border-black"></div>
          <span className="text-xs font-bold">SYSTEM ONLINE</span>
        </div>
        <div className="text-[10px] font-mono text-gray-500">
          MEM: 64KB OK<br/>
          CPU: ACTIVE
        </div>
      </div>
    </div>
  );
};
