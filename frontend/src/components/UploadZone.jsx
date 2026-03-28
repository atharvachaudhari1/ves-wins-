export function UploadZone({ fileName, onImageReady, onClear, disabled }) {
  return (
    <div className="mb-4 flex flex-wrap items-center gap-3">
      <label
        className={`block flex-1 cursor-pointer rounded-lg border border-dashed border-surface-border px-5 py-5 transition-colors hover:border-cyan-500/40 hover:bg-cyan-500/5 ${disabled ? "pointer-events-none opacity-50" : ""}`}
      >
        <input
          type="file"
          accept="image/*"
          className="sr-only"
          disabled={disabled}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (!f) return;
            const reader = new FileReader();
            reader.onload = () => {
              const dataUrl = String(reader.result || "");
              const b64 = dataUrl.includes(",") ? dataUrl.split(",", 2)[1] : dataUrl;
              const mime = f.type && f.type.startsWith("image/") ? f.type : "image/jpeg";
              onImageReady({ base64: b64, mime, name: f.name });
            };
            reader.readAsDataURL(f);
            e.target.value = "";
          }}
        />
        <span className="text-sm text-slate-500">
          {fileName || "Drop an image or click to choose"}
        </span>
      </label>
      {fileName && (
        <button
          type="button"
          disabled={disabled}
          className="rounded-lg border border-surface-border px-3 py-1.5 text-xs font-medium text-slate-500 hover:bg-slate-800 disabled:opacity-45"
          onClick={onClear}
        >
          Clear
        </button>
      )}
    </div>
  );
}
