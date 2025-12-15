export const Settings = () => {
  return (
    <div>
      <h2 className="text-3xl font-black mb-8">AI SETTINGS</h2>
      
      <div className="retro-card space-y-6">
        <div>
          <label className="block font-bold mb-2 uppercase text-sm">API Endpoint</label>
          <input 
            type="text" 
            value="http://localhost:8000" 
            readOnly
            className="retro-input font-mono bg-gray-100" 
          />
        </div>

        <div>
          <label className="block font-bold mb-2 uppercase text-sm">Theme</label>
          <div className="flex gap-4">
            <button className="retro-btn retro-btn-purple">RETRO</button>
            <button className="retro-btn bg-white text-black">CLEAN</button>
          </div>
        </div>

        <div className="pt-6 border-t-2 border-black">
          <h3 className="font-bold mb-4 uppercase">Device Status</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 border-2 border-black bg-[#f6f0e6]">
              <div className="text-xs text-gray-500 uppercase">Memory</div>
              <div className="text-xl font-black">64%</div>
            </div>
            <div className="p-4 border-2 border-black bg-[#f6f0e6]">
              <div className="text-xs text-gray-500 uppercase">Uptime</div>
              <div className="text-xl font-black">12h 4m</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
