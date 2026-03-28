import { useState } from "react";
import axios from "axios";

/**
 * In `npm run dev`, use relative `/api` so Vite proxies to the backend.
 * `vite preview` has no proxy — set `VITE_API_URL=http://127.0.0.1:8000` in `.env.development` / `.env`.
 */
function apiBase() {
  const raw = import.meta.env.VITE_API_URL;
  if (raw && String(raw).trim()) {
    return `${String(raw).replace(/\/$/, "")}/api`;
  }
  if (import.meta.env.DEV) {
    return "/api";
  }
  return "http://127.0.0.1:8000/api";
}

const client = axios.create({
  timeout: 180_000,
  headers: { "Content-Type": "application/json" },
});

export function useShield() {
  const [status, setStatus] = useState("idle");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const applyShield = async (imageBase64, settings) => {
    setStatus("processing");
    setError(null);
    try {
      const res = await client.post(`${apiBase()}/shield`, {
        image: imageBase64,
        settings,
      });
      setResult(res.data);
      setStatus("done");
    } catch (err) {
      const d = err.response?.data;
      let msg =
        (typeof d?.error === "string" && d.error) ||
        (typeof d?.detail === "object" && d.detail?.error) ||
        (typeof d?.detail === "string" && d.detail) ||
        err.message ||
        "Something went wrong";

      const net =
        err.code === "ERR_NETWORK" ||
        err.message === "Network Error" ||
        (err.message || "").toLowerCase().includes("network");
      if (net) {
        msg =
          "Cannot reach the Face Shield API. Start the backend (uvicorn on port 8000), run the UI with `npm run dev` (not preview) so /api is proxied, or set VITE_API_URL=http://127.0.0.1:8000 in frontend/.env and allow that origin in CORS_ORIGINS.";
      }
      if (err.code === "ECONNABORTED") {
        msg = "Request timed out — the first run can be slow (TensorFlow / MediaPipe). Try again or lower image size.";
      }

      setError(msg);
      setStatus("error");
    }
  };

  return { applyShield, status, result, error };
}
