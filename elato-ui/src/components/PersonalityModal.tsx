import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { Modal } from "./Modal";
import { Brain, ArrowUp } from "lucide-react";

export type PersonalityForModal = {
  id: string;
  name: string;
  prompt: string;
  short_description: string;
  tags: string[];
  voice_id: string;
  is_visible: boolean;
};

type PersonalityModalProps = {
  open: boolean;
  mode: "create" | "edit";
  personality?: PersonalityForModal | null;
  onClose: () => void;
  onSuccess: () => Promise<void> | void;
};

export function PersonalityModal({ open, mode, personality, onClose, onSuccess }: PersonalityModalProps) {
  // Create mode state
  const [description, setDescription] = useState("");
  
  // Edit mode state
  const [name, setName] = useState("");
  const [prompt, setPrompt] = useState("");
  const [shortDescription, setShortDescription] = useState("");
  const [tags, setTags] = useState("");
  const [voiceId, setVoiceId] = useState("dave");
  
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const parsedTags = useMemo(() => {
    return tags
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  }, [tags]);

  const reset = () => {
    setDescription("");
    setName("");
    setPrompt("");
    setShortDescription("");
    setTags("");
    setVoiceId("dave");
    setError(null);
  };

  useEffect(() => {
    if (!open) return;

    if (mode === "edit") {
      if (!personality) {
        reset();
        return;
      }
      setName(personality.name || "");
      setPrompt(personality.prompt || "");
      setShortDescription(personality.short_description || "");
      setTags((personality.tags || []).join(", "));
      setVoiceId(personality.voice_id || "dave");
      setError(null);
    } else {
      reset();
    }
  }, [open, mode, personality?.id]);

  const submitCreate = async () => {
    if (!description.trim()) return;
    setSubmitting(true);
    setError(null);
    try {
      await api.generatePersonality(description.trim());
      await onSuccess();
      reset();
      onClose();
    } catch (e: any) {
      setError(e?.message || "Failed to generate personality");
    } finally {
      setSubmitting(false);
    }
  };

  const submitEdit = async () => {
    if (!name.trim()) {
      setError("Name is required");
      return;
    }
    if (!prompt.trim()) {
      setError("Prompt is required");
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      const payload = {
        name: name.trim(),
        prompt: prompt.trim(),
        short_description: shortDescription.trim(),
        tags: parsedTags,
        voice_id: voiceId,
      };

      if (personality) {
        await api.updatePersonality(personality.id, payload);
      }

      await onSuccess();
      reset();
      onClose();
    } catch (e: any) {
      setError(e?.message || "Failed to update personality");
    } finally {
      setSubmitting(false);
    }
  };

  if (mode === "create") {
    return (
      <Modal
        open={open}
        title={""}
        onClose={() => {
          reset();
          onClose();
        }}
      >
        <div className="space-y-6 text-center">
            {error && <div className="font-mono text-sm text-red-600 mb-4">{error}</div>}
            
            <div className="flex flex-col items-center gap-2 mb-6">
                <div className="p-3 bg-[#9b5cff] rounded-full border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
                    <Brain fill="white" className="w-8 h-8 text-white" />
                </div>
                <h3 className="font-black text-2xl uppercase mt-2">Create Your Character</h3>
            </div>

            <div className="relative w-full">
                <textarea
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="Describe the character you'd like to create..."
                    className="w-full min-h-[120px] p-4 pr-14 rounded-[20px] border-2 border-black resize-none text-lg bg-white focus:outline-none shadow-inner placeholder:text-gray-500"
                    onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            submitCreate();
                        }
                    }}
                />
                <button 
                    onClick={submitCreate}
                    disabled={submitting || !description.trim()}
                    className={`absolute bottom-3 right-3 w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed ${(!submitting && !!description.trim()) ? 'cursor-pointer bg-[#9b5cff] text-white border-black shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] hover:brightness-105 hover:scale-[1.03] hover:translate-x-[-1px] hover:translate-y-[-1px] hover:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] active:scale-[0.98] active:translate-x-[0px] active:translate-y-[0px] active:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black focus-visible:ring-offset-2 focus-visible:ring-offset-white' : 'bg-gray-200 text-gray-700 border-transparent hover:border-black'}`}
                >
                    {submitting ? (
                        <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                    ) : (
                        <ArrowUp className="w-5 h-5" />
                    )}
                </button>
            </div>
        </div>
      </Modal>
    );
  }

  return (
    <Modal
      open={open}
      title="Edit Personality"
      onClose={() => {
        reset();
        onClose();
      }}
    >
      <div className="space-y-4">
        {error && <div className="font-mono text-sm text-red-600">{error}</div>}

        <div>
          <label className="block font-bold mb-2 uppercase text-sm">Name</label>
          <input
            className="retro-input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Helpful Assistant"
          />
        </div>

        <div>
          <label className="block font-bold mb-2 uppercase text-sm">System Prompt</label>
          <textarea
            className="retro-input min-h-[100px]"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="You are a helpful AI assistant..."
          />
        </div>

        <div>
          <label className="block font-bold mb-2 uppercase text-sm">Short Description</label>
          <input
            className="retro-input"
            value={shortDescription}
            onChange={(e) => setShortDescription(e.target.value)}
            placeholder="e.g. A general purpose assistant"
          />
        </div>

        <div>
          <label className="block font-bold mb-2 uppercase text-sm">Voice ID</label>
          <select className="retro-input" value={voiceId} onChange={(e) => setVoiceId(e.target.value)}>
            <option value="dave">Dave</option>
            <option value="fin">Fin</option>
            <option value="sandra">Sandra</option>
            <option value="libby">Libby</option>
          </select>
        </div>

        <div>
          <label className="block font-bold mb-2 uppercase text-sm">Tags (comma separated)</label>
          <input
            className="retro-input"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            placeholder="e.g. helper, general, fun"
          />
        </div>

        <div className="flex justify-end">
          <button className="retro-btn" type="button" onClick={submitEdit} disabled={submitting}>
            {submitting ? "Saving…" : "Save"}
          </button>
        </div>
      </div>
    </Modal>
  );
}
