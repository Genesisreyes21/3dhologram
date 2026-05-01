"""FastAPI server for webcam-to-DECA-to-particle-point-cloud inference."""

from __future__ import annotations

import base64
import binascii
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from export_points import clamp_point_count, make_demo_face_point_cloud
from run_deca import DecaReconstructor, DecaUnavailableError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(
    title="DECA Hologram Prototype",
    description="Capture a webcam face, run released DECA inference, and return particle point clouds.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

deca = DecaReconstructor()


class CapturePayload(BaseModel):
    image: str = Field(..., description="A data URL or base64-encoded JPEG/PNG image.")
    point_count: int = Field(12000, ge=500, le=20000)
    face_landmarks: list[list[float]] | None = Field(
        default=None,
        description="Optional normalized ml5 faceMesh keypoints for better crop framing.",
    )
    face_metrics: dict[str, Any] | None = Field(
        default=None,
        description="Optional browser-side expression metrics derived from ml5 faceMesh.",
    )


@app.middleware("http")
async def disable_cache(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/api/health")
def health() -> dict[str, Any]:
    status = deca.status()
    return {
        "ok": True,
        "deca": status.__dict__,
        "setup_hint": [
            "Clone https://github.com/yfeng95/DECA outside this project.",
            "Run DECA's fetch_data.sh to download the released pretrained checkpoint.",
            "Set DECA_ROOT and DECA_MODEL_PATH before starting this server.",
        ],
    }


@app.get("/api/demo-points")
def demo_points(count: int = 7000) -> dict[str, Any]:
    point_count = clamp_point_count(count)
    points = make_demo_face_point_cloud(point_count)
    return {
        "points": points,
        "meta": {
            "source": "procedural-demo",
            "point_count": len(points),
            "note": "Visual test cloud only. Use /api/reconstruct with DECA for real faces.",
        },
    }


@app.post("/api/reconstruct")
async def reconstruct(request: Request) -> dict[str, Any]:
    image_bytes, point_count, face_landmarks, face_metrics = await _read_capture_request(request)
    point_count = clamp_point_count(point_count)

    with tempfile.TemporaryDirectory(prefix="deca_hologram_") as temp_dir:
        image_path = Path(temp_dir) / "capture.png"
        _write_normalized_image(image_bytes, image_path)

        try:
            return deca.reconstruct(
                image_path=image_path,
                point_count=point_count,
                face_landmarks=face_landmarks,
                face_metrics=face_metrics,
            )
        except DecaUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": str(exc),
                    "kind": "deca_unavailable",
                    "next_steps": [
                        "Install the official DECA repository and dependencies.",
                        "Download the released DECA model with fetch_data.sh.",
                        "Restart this server with DECA_ROOT and DECA_MODEL_PATH set.",
                    ],
                },
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": f"DECA reconstruction failed: {exc}",
                    "kind": "deca_runtime_error",
                },
            ) from exc


async def _read_capture_request(
    request: Request,
) -> tuple[bytes, int, list[list[float]] | None, dict[str, Any] | None]:
    """Accept either JSON data URLs or multipart file uploads."""

    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("image")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail="multipart form requires an image file")
        point_count = int(form.get("point_count", 7000))
        return await upload.read(), point_count, None, None

    try:
        payload = CapturePayload.model_validate(await request.json())
    except AttributeError:
        # Pydantic v1 fallback.
        payload = CapturePayload.parse_obj(await request.json())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON capture payload: {exc}") from exc

    return _decode_data_url(payload.image), payload.point_count, payload.face_landmarks, payload.face_metrics


def _decode_data_url(value: str) -> bytes:
    if "," in value and value.lower().startswith("data:"):
        _, encoded = value.split(",", 1)
    else:
        encoded = value

    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image must be a base64 data URL") from exc


def _write_normalized_image(raw_bytes: bytes, output_path: Path) -> None:
    try:
        with Image.open(BytesIO(raw_bytes)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.save(output_path, "PNG")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode image: {exc}") from exc


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/{asset_path:path}", include_in_schema=False)
def frontend_asset(asset_path: str):
    requested = FRONTEND_DIR / asset_path
    if requested.is_file():
        return FileResponse(requested)
    if asset_path.startswith("api/"):
        return JSONResponse(
            status_code=404,
            content={
                "detail": {
                    "message": f"API route not found: /{asset_path}",
                    "kind": "api_not_found",
                }
            },
        )
    if Path(asset_path).suffix:
        return JSONResponse(
            status_code=404,
            content={
                "detail": {
                    "message": f"Asset not found: /{asset_path}",
                    "kind": "asset_not_found",
                }
            },
        )
    return FileResponse(FRONTEND_DIR / "index.html")
