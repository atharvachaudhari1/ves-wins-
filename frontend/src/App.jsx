import { useRef, useState } from "react";
import { useShield } from "./hooks/useShield.js";
import ConfidenceTable from "./components/ConfidenceTable.jsx";

export default function App() {
  const [originalB64, setOriginalB64] = useState(null);
  const [settings, setSettings] = useState({
    layer1_intensity: 20,
    layer1_pattern: "gaussian",
    layer2_intensity: 0.08,
    layer2_band: "mid",
    layer3_intensity: 25,
  });
  const { applyShield, status, result, error } = useShield();
  const fileRef = useRef();

  const handleFile = (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => setOriginalB64(e.target.result);
    reader.readAsDataURL(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleShield = () => {
    if (originalB64) applyShield(originalB64, settings);
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0a0a0f",
        color: "#e8e8f0",
        fontFamily: "Space Mono, monospace",
        padding: "40px 20px",
        maxWidth: 1000,
        margin: "0 auto",
      }}
    >
      <h1
        style={{
          fontFamily: "'Syne', sans-serif",
          fontSize: 42,
          fontWeight: 800,
          marginBottom: 8,
        }}
      >
        Face<span style={{ color: "#00ff88" }}>Shield</span>
      </h1>
      <p style={{ fontSize: 11, color: "#555570", marginBottom: 40 }}>
        Triple-domain adversarial noise · Pixel · Frequency · Semantic
      </p>

      <div
        role="presentation"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        onClick={() => fileRef.current?.click()}
        style={{
          border: "2px dashed #1e1e2e",
          borderRadius: 16,
          height: 200,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          cursor: "pointer",
          marginBottom: 24,
          overflow: "hidden",
          background: originalB64 ? "transparent" : "#111118",
        }}
      >
        {originalB64 ? (
          <img
            src={originalB64}
            alt="Upload preview"
            style={{ height: "100%", objectFit: "contain" }}
          />
        ) : (
          <span style={{ color: "#555570", fontSize: 12 }}>
            Drop face photo here or click to upload
          </span>
        )}
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          style={{ display: "none" }}
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: 16,
          marginBottom: 24,
        }}
      >
        {[
          { label: "L1 Intensity", key: "layer1_intensity", min: 5, max: 60 },
          {
            label: "L2 Intensity",
            key: "layer2_intensity",
            min: 0.01,
            max: 0.3,
            step: 0.01,
          },
          { label: "L3 Intensity", key: "layer3_intensity", min: 5, max: 60 },
        ].map(({ label, key, min, max, step = 1 }) => (
          <div key={key}>
            <label
              style={{
                fontSize: 10,
                color: "#555570",
                letterSpacing: 1.5,
                textTransform: "uppercase",
                display: "block",
                marginBottom: 6,
              }}
            >
              {label}: <span style={{ color: "#00ff88" }}>{settings[key]}</span>
            </label>
            <input
              type="range"
              min={min}
              max={max}
              step={step}
              value={settings[key]}
              onChange={(e) =>
                setSettings((s) => ({
                  ...s,
                  [key]:
                    step === 0.01
                      ? parseFloat(e.target.value)
                      : parseInt(e.target.value, 10),
                }))
              }
              style={{ width: "100%" }}
            />
          </div>
        ))}
      </div>

      <button
        type="button"
        onClick={handleShield}
        disabled={!originalB64 || status === "processing"}
        style={{
          width: "100%",
          padding: "16px",
          fontSize: 13,
          fontFamily: "Space Mono, monospace",
          fontWeight: 700,
          letterSpacing: 2,
          textTransform: "uppercase",
          background: "#00ff88",
          color: "#000",
          border: "none",
          borderRadius: 12,
          cursor: "pointer",
          marginBottom: 24,
          opacity: !originalB64 || status === "processing" ? 0.4 : 1,
        }}
      >
        {status === "processing" ? "Applying 3-Layer Shield..." : "Apply Shield"}
      </button>

      {error && (
        <div
          style={{
            padding: 16,
            background: "#1a0a0f",
            border: "1px solid #ff3366",
            borderRadius: 12,
            color: "#ff3366",
            fontSize: 11,
            marginBottom: 24,
          }}
        >
          Error: {error}
        </div>
      )}

      {result && result.confidence_scores && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 24,
          }}
        >
          <div>
            <p
              style={{
                fontSize: 10,
                color: "#555570",
                letterSpacing: 2,
                textTransform: "uppercase",
                marginBottom: 12,
              }}
            >
              Shielded Output
            </p>
            <img
              src={result.shielded_image}
              alt="Shielded"
              style={{ width: "100%", borderRadius: 12 }}
            />
            <a href={result.shielded_image} download="faceshield_protected.png">
              <button
                type="button"
                style={{
                  width: "100%",
                  marginTop: 10,
                  padding: "10px",
                  fontFamily: "Space Mono, monospace",
                  fontSize: 10,
                  background: "transparent",
                  color: "#00ff88",
                  border: "1px solid #00ff88",
                  borderRadius: 8,
                  cursor: "pointer",
                  letterSpacing: 1,
                }}
              >
                Download Protected Image
              </button>
            </a>
          </div>
          <div>
            <ConfidenceTable scores={result.confidence_scores} />
          </div>
        </div>
      )}
    </div>
  );
}
