import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { SignalGeneratorPage } from "./pages/SignalGeneratorPage";

export function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/signal-generator" replace />} />
        <Route path="signal-generator" element={<SignalGeneratorPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
