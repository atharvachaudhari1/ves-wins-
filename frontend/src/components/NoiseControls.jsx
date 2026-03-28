function SliderRow({ label, valueDisplay, children }) {
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-sm text-slate-500">
        <span>{label}</span>
        <span className="tabular-nums text-cyan-400">{valueDisplay}</span>
      </div>
      {children}
    </div>
  );
}

export function NoiseControls({ settings, onChange, disabled }) {
  const set = (patch) => onChange({ ...settings, ...patch });

  return (
    <div className={`space-y-4 ${disabled ? "pointer-events-none opacity-50" : ""}`}>
      <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500">
        Noise settings
      </h3>
      <div className="space-y-4">
        <SliderRow label="Layer 1 intensity" valueDisplay={settings.layer1_intensity}>
          <input
            type="range"
            min="0"
            max="80"
            step="1"
            className="w-full accent-cyan-500"
            value={settings.layer1_intensity}
            onChange={(e) => set({ layer1_intensity: Number(e.target.value) })}
          />
        </SliderRow>
        <div className="space-y-1.5">
          <div className="flex justify-between text-sm text-slate-500">
            <span>Layer 1 pattern</span>
            <span className="text-cyan-400">{settings.layer1_pattern}</span>
          </div>
          <select
            className="w-full rounded-lg border border-surface-border bg-slate-950 px-3 py-2 text-sm text-slate-200"
            value={settings.layer1_pattern}
            onChange={(e) => set({ layer1_pattern: e.target.value })}
          >
            <option value="gaussian">gaussian</option>
            <option value="structured">structured</option>
            <option value="checkerboard">checkerboard</option>
          </select>
        </div>
        <SliderRow
          label="Layer 2 intensity"
          valueDisplay={settings.layer2_intensity.toFixed(3)}
        >
          <input
            type="range"
            min="0.02"
            max="0.25"
            step="0.005"
            className="w-full accent-cyan-500"
            value={settings.layer2_intensity}
            onChange={(e) => set({ layer2_intensity: Number(e.target.value) })}
          />
        </SliderRow>
        <div className="space-y-1.5">
          <div className="flex justify-between text-sm text-slate-500">
            <span>Layer 2 band</span>
            <span className="text-cyan-400">{settings.layer2_band}</span>
          </div>
          <select
            className="w-full rounded-lg border border-surface-border bg-slate-950 px-3 py-2 text-sm text-slate-200"
            value={settings.layer2_band}
            onChange={(e) => set({ layer2_band: e.target.value })}
          >
            <option value="low">low</option>
            <option value="mid">mid</option>
            <option value="high">high</option>
          </select>
        </div>
        <SliderRow label="Layer 3 intensity" valueDisplay={settings.layer3_intensity}>
          <input
            type="range"
            min="0"
            max="80"
            step="1"
            className="w-full accent-cyan-500"
            value={settings.layer3_intensity}
            onChange={(e) => set({ layer3_intensity: Number(e.target.value) })}
          />
        </SliderRow>
      </div>
    </div>
  );
}
