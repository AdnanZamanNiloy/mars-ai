import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./theme.css";
import "./layout.css";
import "./components.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
