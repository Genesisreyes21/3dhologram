"""Thin DECA inference adapter used by the FastAPI server.

This module intentionally does not vendor DECA. Point ``DECA_ROOT`` at a local
clone of the official repository and ``DECA_MODEL_PATH`` at the released
``deca_model.tar`` checkpoint downloaded by DECA's ``fetch_data.sh`` script.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from export_points import load_obj_mesh, mesh_to_hologram_point_cloud


class DecaUnavailableError(RuntimeError):
    """Raised when the released DECA model or code is not configured."""


@dataclass
class DecaStatus:
    configured: bool
    loaded: bool
    root: str | None
    model_path: str | None
    device: str
    rasterizer: str
    detector: str
    message: str


class DecaReconstructor:
    """Lazy loader around the official DECA Python API."""

    def __init__(self) -> None:
        self.deca_root = _optional_path(os.getenv("DECA_ROOT"))
        self.model_path = _optional_path(os.getenv("DECA_MODEL_PATH"))
        self.device_setting = os.getenv("DECA_DEVICE", "auto")
        self.rasterizer_type = os.getenv("DECA_RASTERIZER", "standard")
        self.face_detector = os.getenv("DECA_FACE_DETECTOR", "fan")

        self._lock = threading.Lock()
        self._loaded = False
        self._deca: Any | None = None
        self._torch: Any | None = None
        self._device = "cpu"

    def status(self) -> DecaStatus:
        root = str(self.deca_root) if self.deca_root else None
        model_path = str(self._resolved_model_path()) if self._resolved_model_path() else None

        missing: list[str] = []
        if not self.deca_root:
            missing.append("set DECA_ROOT to the official DECA repository clone")
        elif not self.deca_root.exists():
            missing.append(f"DECA_ROOT does not exist: {self.deca_root}")
        elif not (self.deca_root / "decalib").exists():
            missing.append(f"DECA_ROOT does not look like a DECA repo: {self.deca_root}")

        resolved_model = self._resolved_model_path()
        if not resolved_model:
            missing.append("set DECA_MODEL_PATH or run DECA's fetch_data.sh")
        elif not resolved_model.exists():
            missing.append(f"DECA model checkpoint not found: {resolved_model}")
        elif not _looks_like_deca_checkpoint(resolved_model):
            missing.append(
                f"DECA checkpoint looks invalid or incomplete: {resolved_model}. "
                "It should be the real ~414 MB deca_model.tar, not a Google Drive HTML page."
            )

        message = "DECA is ready." if not missing else "; ".join(missing)
        return DecaStatus(
            configured=not missing,
            loaded=self._loaded,
            root=root,
            model_path=model_path,
            device=self._device if self._loaded else self.device_setting,
            rasterizer=self.rasterizer_type,
            detector=self.face_detector,
            message=message,
        )

    def reconstruct(
        self,
        image_path: Path,
        point_count: int,
        face_landmarks: list[list[float]] | None = None,
        face_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run DECA inference and return normalized point-cloud data."""

        self._ensure_loaded()
        assert self._deca is not None
        assert self._torch is not None

        images, crop_meta = _load_capture_tensor(
            image_path=image_path,
            torch_module=self._torch,
            device=self._device,
            crop_size=224,
            face_landmarks=face_landmarks,
        )

        with self._torch.no_grad():
            # We use the released model for inference only. Detail displacement
            # is disabled here because the hologram needs points, not textured
            # rendering, and coarse FLAME vertices are fast and stable.
            codedict = self._deca.encode(images, use_detail=False)
            opdict = self._deca.decode(
                codedict,
                rendering=False,
                vis_lmk=False,
                return_vis=False,
                use_detail=False,
            )

        vertices = opdict["verts"][0].detach().cpu().numpy()
        faces = self._deca.render.faces[0].detach().cpu().numpy()
        landmarks3d = opdict["landmarks3d_world"][0].detach().cpu().numpy()
        points = mesh_to_hologram_point_cloud(
            vertices,
            faces,
            landmarks3d=landmarks3d,
            target_count=point_count,
        )

        return {
            "points": points,
            "meta": {
                "source": "deca",
                "point_count": len(points),
                "mesh_vertices": int(vertices.shape[0]),
                "mesh_faces": int(faces.shape[0]),
                "device": self._device,
                "detector": crop_meta["detector"],
                "rasterizer": self.rasterizer_type,
                "crop": crop_meta,
                "expression": face_metrics or {},
                "feature_codes": {
                    "0": "surface",
                    "1": "eyes",
                    "2": "nose",
                    "3": "mouth",
                    "4": "jaw",
                    "5": "brows",
                },
            },
        }

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            status = self.status()
            if not status.configured:
                raise DecaUnavailableError(status.message)

            _patch_legacy_deca_dependencies()

            try:
                import torch

                if str(self.deca_root) not in sys.path:
                    sys.path.insert(0, str(self.deca_root))

                _patch_torch_load_for_deca(torch)

                from decalib.deca import DECA
                from decalib.utils.config import cfg as deca_cfg
                _patch_deca_renderer_for_pointcloud(DECA, torch)
            except Exception as exc:  # pragma: no cover - depends on local DECA install
                raise DecaUnavailableError(
                    "DECA dependencies are not importable. Install DECA's own "
                    f"requirements in this Python environment. Original error: {exc}"
                ) from exc

            if self.device_setting == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.device_setting

            deca_cfg.pretrained_modelpath = str(self._resolved_model_path())
            deca_cfg.rasterizer_type = self.rasterizer_type
            deca_cfg.model.use_tex = False
            deca_cfg.model.extract_tex = False

            self._torch = torch
            self._deca = DECA(config=deca_cfg, device=self._device)
            self._loaded = True

    def _resolved_model_path(self) -> Path | None:
        if self.model_path:
            return self.model_path
        if self.deca_root:
            return self.deca_root / "data" / "deca_model.tar"
        return None


def _optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _looks_like_deca_checkpoint(path: Path) -> bool:
    try:
        if path.stat().st_size < 100_000_000:
            return False
        with path.open("rb") as handle:
            head = handle.read(64).lstrip()
        return not head.lower().startswith(b"<!doctype html")
    except OSError:
        return False


def _patch_legacy_deca_dependencies() -> None:
    """Bridge old DECA/chumpy expectations on modern Python/NumPy.

    The released DECA code depends on chumpy, which predates Python 3.12 and
    NumPy 2.x. These aliases are enough for loading FLAME pickles during
    inference without editing the installed third-party package.
    """

    import inspect
    import numpy as np

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]

    legacy_numpy_aliases = {
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }

    for name, value in legacy_numpy_aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)

    try:
        import face_alignment

        landmarks = face_alignment.LandmarksType
        if not hasattr(landmarks, "_2D") and hasattr(landmarks, "TWO_D"):
            setattr(landmarks, "_2D", landmarks.TWO_D)
    except Exception:
        # The point-cloud preprocessing below avoids face_alignment entirely,
        # so a missing or incompatible detector should not block DECA loading.
        pass


def _patch_torch_load_for_deca(torch_module: Any) -> None:
    """Allow the official DECA checkpoint to load on PyTorch 2.6+.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
    DECA's released checkpoint is an older trusted artifact that expects the
    previous behavior, so we set the default only when DECA did not pass the
    argument explicitly.
    """

    if getattr(torch_module.load, "_deca_hologram_patched", False):
        return

    original_load = torch_module.load

    def load_with_deca_default(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        if not torch_module.cuda.is_available():
            kwargs.setdefault("map_location", torch_module.device("cpu"))
        return original_load(*args, **kwargs)

    load_with_deca_default._deca_hologram_patched = True  # type: ignore[attr-defined]
    torch_module.load = load_with_deca_default


def _patch_deca_renderer_for_pointcloud(deca_class: Any, torch_module: Any) -> None:
    """Skip DECA's CUDA/PyTorch3D renderer for point-cloud-only inference.

    The stock DECA initializer always calls ``_setup_renderer``. This app only
    needs mesh vertices and face topology, not rendered images, so this patch
    avoids C++/CUDA extension compilation on CPU-only Macs.
    """

    if getattr(deca_class, "_deca_hologram_renderer_patched", False):
        return

    class PointCloudRender(torch_module.nn.Module):
        def __init__(self, faces_tensor: Any) -> None:
            super().__init__()
            self.register_buffer("faces", faces_tensor)

    def setup_pointcloud_renderer(self: Any, model_cfg: Any) -> None:
        _, faces = load_obj_mesh(model_cfg.topology_path)
        faces_tensor = torch_module.as_tensor(faces, dtype=torch_module.long, device=self.device)[None, ...]
        self.render = PointCloudRender(faces_tensor).to(self.device)

    deca_class._setup_renderer = setup_pointcloud_renderer
    deca_class._deca_hologram_renderer_patched = True


def _load_capture_tensor(
    image_path: Path,
    torch_module: Any,
    device: str,
    crop_size: int = 224,
    face_landmarks: list[list[float]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Create DECA's ``[1, 3, H, W]`` input tensor without FAN downloads.

    DECA's demo uses ``face_alignment`` for landmarks. That package changes APIs
    across versions and may download detector weights on first run. For a webcam
    prototype, OpenCV Haar detection plus a centered fallback is more reliable.
    """

    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image)
    height, width = rgb.shape[:2]
    bbox = _bbox_from_face_landmarks(face_landmarks, width, height)
    detector_source = "ml5-facemesh" if bbox is not None else None

    if bbox is None:
        bbox = _detect_face_bbox(rgb)
        detector_source = "opencv-haar" if bbox is not None else None

    if bbox is None:
        detector = "center-crop"
        center_x = width * 0.5
        center_y = height * 0.5
        size = min(width, height) * 0.92
    else:
        detector = detector_source or "opencv-haar"
        x, y, face_w, face_h = bbox
        center_x = x + face_w * 0.5
        center_y = y + face_h * 0.5 + max(face_w, face_h) * 0.08
        size = max(face_w, face_h) * 1.55

    left = int(round(center_x - size * 0.5))
    top = int(round(center_y - size * 0.5))
    right = int(round(center_x + size * 0.5))
    bottom = int(round(center_y + size * 0.5))

    crop = image.crop((left, top, right, bottom)).resize((crop_size, crop_size), Image.Resampling.BILINEAR)
    array = np.asarray(crop).astype(np.float32) / 255.0
    tensor = torch_module.tensor(array.transpose(2, 0, 1)).float().to(device)[None, ...]

    return tensor, {
        "detector": detector,
        "bbox": [int(v) for v in bbox] if bbox is not None else None,
        "source_size": [int(width), int(height)],
        "crop_box": [left, top, right, bottom],
    }


def _bbox_from_face_landmarks(
    landmarks: list[list[float]] | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if not landmarks:
        return None

    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 20:
        return None

    points = points[:, :2]
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if len(points) < 20:
        return None

    # Frontend sends normalized points. Accept absolute points too for easier
    # debugging or future clients.
    if float(np.nanmax(points[:, 0])) <= 1.5 and float(np.nanmax(points[:, 1])) <= 1.5:
        points[:, 0] *= width
        points[:, 1] *= height

    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    box_w = max_x - min_x
    box_h = max_y - min_y

    if box_w < width * 0.08 or box_h < height * 0.08:
        return None

    return (
        int(np.clip(min_x, 0, width - 1)),
        int(np.clip(min_y, 0, height - 1)),
        int(np.clip(box_w, 1, width)),
        int(np.clip(box_h, 1, height)),
    )


def _detect_face_bbox(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    try:
        import cv2
    except Exception:
        return None

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(48, 48),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None

    x, y, width, height = max(faces, key=lambda item: item[2] * item[3])
    return int(x), int(y), int(width), int(height)
