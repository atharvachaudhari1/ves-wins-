function imageSrc(b64, mime) {
  if (!b64) return "";
  if (String(b64).startsWith("data:")) return b64;
  const type = mime || "image/png";
  return `data:${type};base64,${b64}`;
}

function Thumb({ b64, mime, label }) {
  if (!b64) {
    return (
      <div className="flex aspect-video min-h-[72px] flex-1 items-center justify-center rounded-md border border-surface-border bg-slate-950/80 text-[10px] uppercase tracking-wide text-slate-600">
        {label}
      </div>
    );
  }
  return (
    <div className="min-w-0 flex-1">
      <p className="mb-1 truncate text-center text-[10px] uppercase tracking-wide text-slate-500">
        {label}
      </p>
      <div className="overflow-hidden rounded-md border border-surface-border bg-slate-950">
        <img
          src={imageSrc(b64, mime)}
          alt={label}
          className="block max-h-28 w-full object-contain"
        />
      </div>
    </div>
  );
}

export function OutputPreview({
  originalImage,
  originalMime,
  shieldedImage,
  layerImages,
  status,
}) {
  const processing = status === "processing";

  return (
    <div className="space-y-4">
      <div>
        <h3 className="mb-2 text-xs font-bold uppercase tracking-widest text-slate-500">
          Shielded output
        </h3>
        {shieldedImage ? (
          <div className="overflow-hidden rounded-lg border border-surface-border bg-slate-950">
            <img
              src={imageSrc(shieldedImage, "image/png")}
              alt="Shielded"
              className="block w-full"
            />
          </div>
        ) : (
          <div className="flex min-h-[200px] items-center justify-center rounded-lg border border-surface-border bg-slate-950/60 text-sm text-slate-600">
            {processing ? "Processing…" : "Run shield to see output"}
          </div>
        )}
      </div>

      <div>
        <h3 className="mb-2 text-xs font-bold uppercase tracking-widest text-slate-500">
          Pipeline frames
        </h3>
        <div className="flex flex-wrap gap-2">
          <Thumb b64={originalImage} mime={originalMime} label="Original" />
          <Thumb b64={layerImages?.layer1} mime="image/png" label="Layer 1" />
          <Thumb b64={layerImages?.layer2} mime="image/png" label="Layer 2" />
          <Thumb b64={layerImages?.layer3} mime="image/png" label="Layer 3" />
        </div>
      </div>
    </div>
  );
}
