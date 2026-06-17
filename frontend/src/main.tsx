import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { AppStateProvider } from "./context/AppStateContext";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <AppStateProvider>
      <App />
    </AppStateProvider>
  </React.StrictMode>
);
