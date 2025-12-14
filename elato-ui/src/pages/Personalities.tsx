import { useEffect, useState } from 'react';
import { api } from '../api';
import { Mic } from 'lucide-react';

export const Personalities = () => {
  const [personalities, setPersonalities] = useState<any[]>([]);

  useEffect(() => {
    api.getPersonalities().then(setPersonalities);
  }, []);

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-3xl font-black">PERSONALITIES</h2>
        <button className="retro-btn">
          + NEW IDENTITY
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {personalities.map((p) => (
          <div key={p.id} className="retro-card relative group">
            <div className="absolute top-4 right-4 bg-black text-white px-2 py-1 text-xs font-bold uppercase">
              {p.voice_id}
            </div>
            <div className="w-12 h-12 bg-[#2aa198] rounded-full border-2 border-black flex items-center justify-center mb-4 shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]">
              <Mic className="text-white" size={24} />
            </div>
            <h3 className="text-xl font-bold mb-2">{p.name}</h3>
            <p className="text-gray-600 mb-4 text-sm font-medium border-l-4 border-gray-300 pl-2">
              "{p.short_description}"
            </p>
            <div className="flex flex-wrap gap-2">
              {p.tags.map((tag: string) => (
                <span key={tag} className="px-2 py-1 bg-[#eee8d5] border border-black text-xs font-bold uppercase">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
