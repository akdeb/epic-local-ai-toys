import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { Modal } from "./Modal";

export type UserForModal = {
  id: string;
  name: string;
  age?: number | null;
  hobbies: string[];
  user_type?: string | null;
  device_volume?: number | null;
};

type UserModalProps = {
  open: boolean;
  mode: "create" | "edit";
  user?: UserForModal | null;
  onClose: () => void;
  onSuccess: () => Promise<void> | void;
};

export function UserModal({ open, mode, user, onClose, onSuccess }: UserModalProps) {
  const [name, setName] = useState("");
  const [age, setAge] = useState<string>("");
  const [hobbies, setHobbies] = useState("");
  const [userType, setUserType] = useState("family");
  const [deviceVolume, setDeviceVolume] = useState<string>("70");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const parsedHobbies = useMemo(() => {
    return hobbies
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  }, [hobbies]);

  const reset = () => {
    setName("");
    setAge("");
    setHobbies("");
    setUserType("family");
    setDeviceVolume("70");
    setError(null);
  };

  useEffect(() => {
    if (!open) return;

    if (mode === "edit") {
      if (!user) {
        reset();
        return;
      }
      setName(user.name || "");
      setAge(user.age != null ? String(user.age) : "");
      setHobbies((user.hobbies || []).join(", "));
      setUserType(user.user_type || "family");
      setDeviceVolume(user.device_volume != null ? String(user.device_volume) : "70");
      setError(null);
    } else {
      reset();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, mode, user?.id]);

  const submit = async () => {
    if (!name.trim()) {
      setError("Name is required");
      return;
    }

    if (mode === "edit" && !user) return;

    setSubmitting(true);
    setError(null);

    try {
      if (mode === "create") {
        await api.createUser({
          name: name.trim(),
          age: age ? Number(age) : null,
          hobbies: parsedHobbies,
          user_type: userType,
          device_volume: deviceVolume ? Number(deviceVolume) : 70,
        });
      } else {
        await api.updateUser(user!.id, {
          name: name.trim(),
          age: age ? Number(age) : null,
          hobbies: parsedHobbies,
          user_type: userType,
          device_volume: deviceVolume ? Number(deviceVolume) : 70,
        });
      }

      await onSuccess();
      reset();
      onClose();
    } catch (e: any) {
      setError(e?.message || (mode === "create" ? "Failed to create member" : "Failed to update member"));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal
      open={open}
      title={mode === "create" ? "Add Member" : "Edit Member"}
      onClose={() => {
        reset();
        onClose();
      }}
    >
      <div className="space-y-4">
        {error && <div className="font-mono text-sm">{error}</div>}

        <div>
          <label className="block font-bold mb-2 uppercase text-sm">Name</label>
          <input
            className="retro-input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder={mode === "create" ? "e.g. Akash" : undefined}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block font-bold mb-2 uppercase text-sm">Age</label>
            <input
              className="retro-input"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              placeholder={mode === "create" ? "e.g. 8" : undefined}
              inputMode="numeric"
            />
          </div>
        </div>

        <div>
          <label className="block font-bold mb-2 uppercase text-sm">Member Type</label>
          <select className="retro-input" value={userType} onChange={(e) => setUserType(e.target.value)}>
            <option value="family">family</option>
            <option value="friend">friend</option>
            <option value="guest">guest</option>
          </select>
        </div>

        <div>
          <label className="block font-bold mb-2 uppercase text-sm">Hobbies</label>
          <input
            className="retro-input"
            value={hobbies}
            onChange={(e) => setHobbies(e.target.value)}
            placeholder={mode === "create" ? "drawing, lego, dinosaurs" : undefined}
          />
        </div>

        <div className="flex justify-end">
          <button className="retro-btn" type="button" onClick={submit} disabled={submitting}>
            {mode === "create" ? (submitting ? "Creating…" : "Create Member") : submitting ? "Saving…" : "Save"}
          </button>
        </div>
      </div>
    </Modal>
  );
}
