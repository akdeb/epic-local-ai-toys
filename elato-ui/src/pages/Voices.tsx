import { useEffect, useState } from "react";
import { api } from "../api";

export const VoicesPage = () => {
  const [voices, setVoices] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        setError(null);
        const data = await api.getVoices();
        if (!cancelled) setVoices(Array.isArray(data) ? data : []);
      } catch (e: any) {
        if (!cancelled) setError(e?.message || "Failed to load voices");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-3xl font-black">VOICES</h2>
      </div>

      {loading && <div className="retro-card font-mono text-sm mb-4">Loading…</div>}
      {error && <div className="retro-card font-mono text-sm mb-4">{error}</div>}

      {!loading && !error && voices.length === 0 && (
        <div className="retro-card font-mono text-sm mb-4">No voices found.</div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {voices.map((v) => (
          <div key={v.voice_id} className="retro-card relative">
            <div className="absolute top-4 right-4 flex items-center gap-2">
              <div className="bg-black text-white px-2 py-1 text-xs font-bold uppercase">
                {v.voice_id}
              </div>
              {v.is_global && (
                <div className="bg-[#ffd400] text-black px-2 py-1 text-xs font-bold uppercase border border-black">
                  Global
                </div>
              )}
            </div>

            <h3 className="text-xl font-bold mb-2">{v.voice_name || v.voice_id}</h3>
            <p className="text-gray-600 mb-4 text-sm font-medium border-l-4 border-gray-300 pl-2">
              {v.voice_description ? `\"${v.voice_description}\"` : "—"}
            </p>
            <div className="flex flex-wrap gap-2">
              {v.gender && (
                <span className="px-2 py-1 border border-black text-xs font-bold lowercase">{v.gender}</span>
              )}
              {v.voice_src && (
                <span className="px-2 py-1 border border-black text-xs font-bold lowercase break-all">
                  {v.voice_src}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
