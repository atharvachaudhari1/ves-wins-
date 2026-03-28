import { useEffect, useState } from "react";

export default function ConfidenceTable({ scores }) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    setTimeout(() => setVisible(true), 100);
  }, [scores]);

  const rows = [
    { label: "Original", score: scores.original, color: "#ff3366" },
    { label: "+ Layer 1 (Pixel)", score: scores.after_layer1, color: "#ff8833" },
    { label: "+ Layer 2 (Frequency)", score: scores.after_layer2, color: "#ffcc00" },
    { label: "+ Layer 3 (Semantic)", score: scores.after_layer3, color: "#00ff88" },
  ];

  return (
    <div style={{ fontFamily: "Space Mono, monospace" }}>
      <p
        style={{
          fontSize: 10,
          letterSpacing: 2,
          color: "#555570",
          textTransform: "uppercase",
          marginBottom: 16,
        }}
      >
        DeepFace Confidence Scores
      </p>
      {rows.map((row, i) => (
        <div key={row.label} style={{ marginBottom: 14 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: 11,
              marginBottom: 5,
            }}
          >
            <span style={{ color: "#e8e8f0" }}>{row.label}</span>
            <span style={{ color: row.color, fontWeight: 700 }}>
              {Math.round(row.score)}%
            </span>
          </div>
          <div
            style={{
              height: 4,
              background: "#1e1e2e",
              borderRadius: 2,
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                width: visible ? `${Math.min(100, row.score)}%` : "0%",
                background: row.color,
                borderRadius: 2,
                transition: `width 0.8s ease ${i * 0.15}s`,
                boxShadow: `0 0 8px ${row.color}66`,
              }}
            />
          </div>
        </div>
      ))}
      <div
        style={{
          marginTop: 20,
          padding: "12px 16px",
          background: "#0a0a0f",
          borderRadius: 10,
          border: "1px solid #00ff88",
        }}
      >
        <span style={{ fontSize: 10, color: "#555570" }}>PROTECTION SCORE </span>
        <span style={{ fontSize: 20, fontWeight: 700, color: "#00ff88" }}>
          {Math.round(100 - scores.after_layer3)}%
        </span>
        <span style={{ fontSize: 10, color: "#555570" }}> REDUCTION</span>
      </div>
    </div>
  );
}
