import { useEffect, useState } from 'react';
import { api } from '../api';
import { User, Volume2 } from 'lucide-react';

export const UsersPage = () => {
  const [users, setUsers] = useState<any[]>([]);

  useEffect(() => {
    api.getUsers().then(setUsers);
  }, []);

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-3xl font-black">USERS</h2>
        <button className="retro-btn">
          + ADD USER
        </button>
      </div>

      <div className="grid grid-cols-1 gap-4">
        {users.map((u) => (
          <div key={u.id} className="retro-card flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-[#ff6b6b] border-2 border-black flex items-center justify-center shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]">
                <User className="text-white" size={24} />
              </div>
              <div>
                <h3 className="text-xl font-bold flex items-center gap-2">
                  {u.name}
                  <span className="text-xs bg-black text-white px-2 py-0.5 uppercase">
                    {u.user_type || 'family'}
                  </span>
                </h3>
                <div className="flex gap-4 text-sm text-gray-600 mt-1">
                  <span>Age: {u.age || 'N/A'}</span>
                  <span>•</span>
                  <span className="flex items-center gap-1">
                    <Volume2 size={14} />
                    {u.device_volume}%
                  </span>
                </div>
              </div>
            </div>
            
            <div className="flex flex-col items-end gap-2">
              <div className="text-xs font-bold uppercase tracking-wider text-gray-500">
                Interests
              </div>
              <div className="flex gap-1">
                {u.hobbies.slice(0, 3).map((hobby: string) => (
                  <span key={hobby} className="px-2 py-1 bg-[#fdf6e3] border border-black text-xs font-bold">
                    {hobby}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
