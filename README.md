# Face Shield

Triple-domain perturbation: **pixel** noise, **FFT** frequency-domain noise, and **MediaPipe** landmark–weighted mask noise. Stack: **FastAPI** + **React 18 / Vite / Tailwind**.

**Repository:** [github.com/atharvachaudhari1/ves-wins-](https://github.com/atharvachaudhari1/ves-wins-)

## Configuration

- Copy **`.env.example`** at the repo root to **`backend/.env`** and adjust flags (for example `USE_DEEPFACE`, `CORS_ORIGINS`, `DEEPFACE_MODEL`). See comments in `.env.example`.
- For production builds, set **`VITE_API_URL`** in the frontend environment (see Deploy below).

## Run locally (two terminals)

From the **repository root** (folders `backend/` and `frontend/` are next to each other):

**Terminal 1 — backend**

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

On first run, create a venv if you like: `python -m venv .venv` then activate (Windows: `.venv\Scripts\activate`).

**Terminal 2 — frontend**

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). The dev server proxies **`/api`** and **`/health`** to `http://127.0.0.1:8000`.

Layer 3 may download `face_landmarker.task` into `%LOCALAPPDATA%\faceshield_cache\` on Windows.

## Deploy: Render (API) + Vercel (UI)

### 1. Render — backend

1. In [Render](https://render.com), create a **Web Service** from this repo (or use **Blueprint** with `render.yaml`).
2. **Root Directory:** `backend`
3. **Build command:** `pip install -r requirements.txt`
4. **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. **Environment** (dashboard or `render.yaml`):
   - `CORS_ORIGINS` — include your Vercel URL, e.g. `https://my-app.vercel.app` (comma-separate if you also use local dev URLs).
   - Optional: `USE_DEEPFACE=1`, `MAX_IMAGE_SIZE_MB`, `DEEPFACE_MODEL`, etc.
6. After deploy, note the public URL, e.g. `https://faceshield-api.onrender.com`.

**Note:** TensorFlow + DeepFace are large. Render **free** tier may run out of **RAM or disk** during install or at runtime; upgrade the instance if builds fail or the app is OOM-killed.

### 2. Vercel — frontend

1. Import the repo, set **Root Directory** to **`frontend`**.
2. **Environment variables** → add **`VITE_API_URL`** = your Render service origin only, e.g. `https://faceshield-api.onrender.com` (no `/api`, no trailing slash). Redeploy after saving — Vite bakes this in at **build** time.
3. Deploy. The UI calls `VITE_API_URL` + `/api/...`; CORS must allow your **`*.vercel.app`** origin (step 1).

### 3. Smoke test

- `GET https://<render-host>/health` → `{"status":"ok",...}`
- Open the Vercel site, upload an image, run shield — if the browser console shows CORS errors, fix **`CORS_ORIGINS`** on Render.

## Suggested build order (Cursor)

Work bottom-up so each step is testable before the next:

1. `backend/utils/image_utils.py` — foundation (decode / encode / validate)
2. `backend/services/layer1_pixel.py` — confirm pixel noise
3. `backend/services/layer2_frequency.py` — FFT layer (+ JPEG test in `tests/`)
4. `backend/services/layer3_semantic.py` — MediaPipe mask noise
5. `backend/services/noise_composer.py` — `compose_shield` / stages
6. `backend/services/deepface_probe.py` — `probe_all_layers` / confidence
7. `backend/routes/shield.py` — JSON + multipart routes
8. `backend/main.py` — app, CORS, `/api` router, `/health`
9. `frontend/src/App.jsx` + components — UI last

## API (summary)

| Method | Path | Purpose |
|--------|------|--------|
| `GET` | `/health` | `{ "status": "ok", "model": "<DeepFace model>" }` |
| `POST` | `/api/shield/` | JSON: `{ "image": "<base64>", "settings": { ... } }` |
| `POST` | `/api/shield/process` | Multipart upload + form fields |

Validation errors: **422** with `{ "error": "..." }`.

## DeepFace

Set `USE_DEEPFACE=1` in `.env` (see `.env.example`) for real `DeepFace.verify`. Model name: `DEEPFACE_MODEL` (default `Facenet`). Failures fall back to demo-style scores where applicable.

## Notebook

`notebooks/benchmark.ipynb` — stub for batch metrics.
