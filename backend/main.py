import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import config, settings
from core.logger import get_logger, setup_logging
from routes.shield import router as shield_router

setup_logging()
get_logger("faceshield")

app = FastAPI(title="FaceShield API", version="0.1.0")

_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
if not _origins:
    _origins = list(config.CORS_ORIGINS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(shield_router, prefix="/api")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errs = exc.errors()
    msg = errs[0].get("msg", "Validation error") if errs else "Validation error"
    return JSONResponse(status_code=422, content={"error": msg})


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": config.DEEPFACE_MODEL}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=int(settings.port),
        reload=True,
    )
