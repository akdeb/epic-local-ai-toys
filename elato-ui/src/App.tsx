import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Personalities } from "./pages/Personalities";
import { UsersPage } from "./pages/Users";
import { Conversations } from "./pages/Conversations";
import { Settings } from "./pages/Settings";
import { TestPage } from "./pages/Test";
import { ChatModePage } from "./pages/ChatMode";
import "./App.css";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Personalities />} />
          <Route path="users" element={<UsersPage />} />
          <Route path="conversations" element={<Conversations />} />
          <Route path="test" element={<TestPage />} />
          <Route path="chat" element={<ChatModePage />} />
          <Route path="settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
