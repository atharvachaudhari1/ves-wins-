# Face Shield

Triple-domain perturbation: **pixel** noise, **FFT** frequency-domain noise, and **MediaPipe** landmark–weighted mask noise. Stack: **FastAPI** + **React 18 / Vite / Tailwind**.

## Run locally (two terminals)

From the **`faceshield`** folder (or use `faceshield/backend` and `faceshield/frontend` from the repo root):

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
