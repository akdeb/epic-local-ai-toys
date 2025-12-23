import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Layout } from "./components/Layout";
import { Personalities } from "./pages/Personalities";
import { UsersPage } from "./pages/Users";
import { Conversations } from "./pages/Conversations";
import { Settings } from "./pages/Settings";
import { TestPage } from "./pages/Test";
import { ChatModePage } from "./pages/ChatMode";
import { SetupPage } from "./pages/Setup";
import { ModelSetupPage } from "./pages/ModelSetup";
import { VoicesPage } from "./pages/Voices";
import { api } from "./api";
import "./App.css";

function SetupGate() {
  const [checking, setChecking] = useState(true);
  const [needsSetup, setNeedsSetup] = useState(false);
  const [backendReady, setBackendReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const checkFirstLaunch = async () => {
      try {
        const isFirst = await invoke<boolean>("is_first_launch");
        if (!cancelled) setNeedsSetup(isFirst);
      } catch (e) {
        console.error("Failed to check first launch:", e);
        if (!cancelled) setNeedsSetup(true);
      } finally {
        if (!cancelled) setChecking(false);
      }
    };
    checkFirstLaunch();

    // If setup is complete, wait until the Python API server is reachable.
    // This restores the previous "Starting AI engine" loading state.
    const waitForBackend = async () => {
      while (!cancelled) {
        try {
          await api.health();
          if (!cancelled) setBackendReady(true);
          return;
        } catch {
          await new Promise((r) => setTimeout(r, 500));
        }
      }
    };
    waitForBackend();

    return () => {
      cancelled = true;
    };
  }, []);

  if (checking) {
    return (
      <div className="min-h-screen bg-[var(--color-retro-bg)] retro-dots flex items-center justify-center">
        <div className="text-center retro-card">
          <div className="text-2xl font-black mb-2 tracking-wider brand-font">ELATO</div>
          <div className="text-gray-500 font-mono">Loading...</div>
        </div>
      </div>
    );
  }

  if (needsSetup) {
    return <Navigate to="/setup" replace />;
  }

  if (!backendReady) {
    return (
      <div className="min-h-screen bg-[var(--color-retro-bg)] retro-dots flex items-center justify-center">
        <div className="text-center retro-card">
          <div className="text-2xl font-black mb-2 tracking-wider brand-font">ELATO</div>
          <div className="text-gray-500 font-mono">Starting AI engine...</div>
        </div>
      </div>
    );
  }

  return <Layout />;
}

import { ActiveUserProvider } from "./state/ActiveUserContext";

function App() {
  return (
    <ActiveUserProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/setup" element={<SetupPage />} />
          <Route path="/model-setup" element={<ModelSetupPage />} />

          <Route path="/" element={<SetupGate />}>
            <Route index element={<Personalities />} />
            <Route path="voices" element={<VoicesPage />} />
            <Route path="users" element={<UsersPage />} />
            <Route path="conversations" element={<Conversations />} />
            <Route path="test" element={<TestPage />} />
            <Route path="chat" element={<ChatModePage />} />
            <Route path="settings" element={<Settings />} />
          </Route>

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </ActiveUserProvider>
  );
}

export default App;
