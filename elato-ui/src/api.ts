const API_BASE = "http://localhost:8000";

export const api = {
  // Personalities
  getPersonalities: async () => {
    const res = await fetch(`${API_BASE}/personalities`);
    return res.json();
  },
  createPersonality: async (data: any) => {
    const res = await fetch(`${API_BASE}/personalities`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  // Users
  getUsers: async () => {
    const res = await fetch(`${API_BASE}/users`);
    return res.json();
  },
  createUser: async (data: any) => {
    const res = await fetch(`${API_BASE}/users`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  // Conversations
  getConversations: async (limit = 50, offset = 0) => {
    const res = await fetch(`${API_BASE}/conversations?limit=${limit}&offset=${offset}`);
    return res.json();
  },
};
