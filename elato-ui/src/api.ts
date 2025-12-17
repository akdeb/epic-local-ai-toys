const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://127.0.0.1:8000";

const request = async (path: string, init?: RequestInit) => {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const contentType = res.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      try {
        const data: any = await res.json();
        const msg =
          (typeof data?.detail === "string" && data.detail) ||
          (typeof data?.message === "string" && data.message) ||
          (typeof data?.error === "string" && data.error) ||
          "";
        const err: any = new Error(msg || `Request failed: ${res.status}`);
        err.status = res.status;
        throw err;
      } catch (e: any) {
        const err: any = new Error(e?.message || `Request failed: ${res.status}`);
        err.status = res.status;
        throw err;
      }
    }

    const text = await res.text().catch(() => "");
    const err: any = new Error(text || `Request failed: ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
};

export const api = {
  health: async () => {
    return request(`/health`);
  },

  getActiveUser: async () => {
    return request(`/active-user`);
  },

  setActiveUser: async (userId: string | null) => {
    return request(`/active-user`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId }),
    });
  },

  getAppMode: async () => {
    return request(`/app-mode`);
  },

  setAppMode: async (mode: string) => {
    return request(`/app-mode`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
  },

  // Personalities
  getPersonalities: async (includeHidden = false) => {
    const qs = includeHidden ? `?include_hidden=true` : ``;
    return request(`/personalities${qs}`);
  },
  createPersonality: async (data: any) => {
    return request(`/personalities`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
  },

  updatePersonality: async (id: string, data: any) => {
    return request(`/personalities/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
  },

  // Users
  getUsers: async () => {
    return request(`/users`);
  },
  createUser: async (data: any) => {
    return request(`/users`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
  },

  updateUser: async (id: string, data: any) => {
    return request(`/users/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
  },

  // Conversations
  getConversations: async (limit = 50, offset = 0) => {
    return request(`/conversations?limit=${limit}&offset=${offset}`);
  },

  getConversationsBySession: async (sessionId: string) => {
    return request(`/conversations?session_id=${encodeURIComponent(sessionId)}`);
  },

  getSessions: async (limit = 50, offset = 0, userId?: string | null) => {
    const qs = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (userId) qs.set("user_id", userId);
    return request(`/sessions?${qs.toString()}`);
  },

  getDeviceStatus: async () => {
    return request(`/device-status`);
  },
};
