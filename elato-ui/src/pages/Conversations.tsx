import { useEffect, useState } from 'react';
import { api } from '../api';
import { Bot, User } from 'lucide-react';

export const Conversations = () => {
  const [conversations, setConversations] = useState<any[]>([]);

  useEffect(() => {
    api.getConversations().then(setConversations);
  }, []);

  return (
    <div>
      <h2 className="text-3xl font-black mb-8">LOGS</h2>

      <div className="bg-white border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
        {conversations.map((c, i) => (
          <div 
            key={c.id} 
            className={`p-4 flex gap-4 ${i !== conversations.length - 1 ? 'border-b-2 border-black' : ''} ${c.role === 'ai' ? 'bg-[#fdf6e3]' : ''}`}
          >
            <div className={`w-8 h-8 shrink-0 border-2 border-black flex items-center justify-center shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] ${c.role === 'ai' ? 'bg-[#2aa198]' : 'bg-[#ff6b6b]'}`}>
              {c.role === 'ai' ? <Bot size={16} className="text-white" /> : <User size={16} className="text-white" />}
            </div>
            
            <div className="flex-1">
              <div className="flex justify-between items-start mb-1">
                <span className="font-bold uppercase text-xs tracking-wider">
                  {c.role === 'ai' ? 'SYSTEM' : 'OPERATOR'}
                </span>
                <span className="font-mono text-xs text-gray-500">
                  {new Date(c.timestamp * 1000).toLocaleString()}
                </span>
              </div>
              <p className="font-medium leading-relaxed">{c.transcript}</p>
            </div>
          </div>
        ))}
        {conversations.length === 0 && (
          <div className="p-8 text-center font-mono text-gray-500">
            NO DATA LOGGED
          </div>
        )}
      </div>
    </div>
  );
};
