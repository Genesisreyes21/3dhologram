"""Utilities for turning DECA meshes into browser-friendly point clouds.

The frontend intentionally renders particles rather than a heavy mesh. These
helpers sample a mesh surface, normalize the result, and return compact
``[x, y, z]`` arrays that p5.js can use as particle targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def clamp_point_count(value: int, minimum: int = 500, maximum: int = 20000) -> int:
    """Keep JSON payloads and p5 particle counts in a practical range."""

    return max(minimum, min(maximum, int(value)))


def load_obj_mesh(obj_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vertices and triangulated faces from a Wavefront OBJ file.

    This is useful if you choose to run DECA's ``save_obj`` flow and sample the
    written mesh instead of using the in-memory vertices returned by DECA.
    """

    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    with Path(obj_path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                # OBJ face entries can be v, v/vt, v//vn, or v/vt/vn. OBJ is
                # 1-indexed; negative indices are relative to the current list.
                polygon = [_parse_obj_index(token, len(vertices)) for token in parts[1:]]
                for i in range(1, len(polygon) - 1):
                    faces.append([polygon[0], polygon[i], polygon[i + 1]])

    if not vertices or not faces:
        raise ValueError(f"No usable mesh data found in OBJ: {obj_path}")

    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _parse_obj_index(token: str, vertex_count: int) -> int:
    raw = token.split("/")[0]
    index = int(raw)
    return index - 1 if index > 0 else vertex_count + index


def mesh_to_point_cloud(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_count: int = 7000,
    seed: int | None = 42,
) -> list[list[float]]:
    """Sample and normalize a mesh into a compact point cloud.

    DECA's coarse FLAME topology has enough vertices to preserve the eyes,
    nose, mouth, and jaw. We keep vertices where possible, then add area-based
    surface samples so the browser receives a fuller volumetric hologram.
    """

    target_count = clamp_point_count(target_count)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [N, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [M, 3]")

    rng = np.random.default_rng(seed)
    vertex_points = _select_anchor_vertices(vertices, target_count, rng)
    remaining = max(0, target_count - len(vertex_points))
    surface_points = sample_mesh_surface(vertices, faces, remaining, rng) if remaining else np.empty((0, 3))

    points = np.vstack([vertex_points, surface_points])
    rng.shuffle(points)
    normalized = normalize_points(points)
    return np.round(normalized, 6).tolist()


def mesh_to_hologram_point_cloud(
    vertices: np.ndarray,
    faces: np.ndarray,
    landmarks3d: np.ndarray | None = None,
    target_count: int = 7000,
    seed: int | None = 42,
) -> list[list[float]]:
    """Return a feature-aware point cloud for the hologram renderer.

    The FLAME mesh carries identity/expression in geometry, but a uniform point
    cloud can look too smooth. We keep mesh surface points, then add compact
    landmark-derived feature strokes so eyes, nose, mouth, brows, and jaw read
    clearly in a particle hologram.

    Feature code in the fourth value:
    ``0`` surface, ``1`` eyes, ``2`` nose, ``3`` mouth, ``4`` jaw, ``5`` brows.
    """

    target_count = clamp_point_count(target_count)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    rng = np.random.default_rng(seed)

    feature_budget = int(target_count * 0.34) if landmarks3d is not None else 0
    feature_points, feature_codes = _landmark_feature_points(landmarks3d, feature_budget, rng)
    feature_count = min(len(feature_points), feature_budget)

    if feature_count:
        feature_points = feature_points[:feature_count]
        feature_codes = feature_codes[:feature_count]

    surface_count = target_count - feature_count
    surface_points = _sample_mesh_points(vertices, faces, surface_count, rng)
    surface_codes = np.zeros(len(surface_points), dtype=np.float32)

    if feature_count:
        points = np.vstack([surface_points, feature_points])
        codes = np.concatenate([surface_codes, feature_codes.astype(np.float32)])
    else:
        points = surface_points
        codes = surface_codes

    order = rng.permutation(len(points))
    points = normalize_points(points[order])
    codes = codes[order]

    packed = np.column_stack([points, codes])
    return np.round(packed, 6).tolist()


def _select_anchor_vertices(vertices: np.ndarray, target_count: int, rng: np.random.Generator) -> np.ndarray:
    if len(vertices) <= target_count:
        return vertices.copy()

    # Keep a strong vertex skeleton when downsampling; pure surface sampling can
    # soften facial features that are important for a recognizable face.
    anchor_count = max(500, int(target_count * 0.65))
    indices = rng.choice(len(vertices), size=min(anchor_count, len(vertices)), replace=False)
    return vertices[indices].copy()


def _sample_mesh_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if target_count <= 0:
        return np.empty((0, 3), dtype=np.float32)

    vertex_points = _select_anchor_vertices(vertices, target_count, rng)
    remaining = max(0, target_count - len(vertex_points))
    surface_points = sample_mesh_surface(vertices, faces, remaining, rng) if remaining else np.empty((0, 3))
    return np.vstack([vertex_points, surface_points]).astype(np.float32)


def sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    sample_count: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Area-weighted random sampling on mesh triangles."""

    if sample_count <= 0:
        return np.empty((0, 3), dtype=np.float32)

    rng = rng or np.random.default_rng()
    triangles = vertices[faces]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    areas = np.linalg.norm(np.cross(edge_a, edge_b), axis=1) * 0.5

    total_area = float(np.sum(areas))
    if total_area <= 0:
        raise ValueError("mesh has no measurable triangle area")

    probabilities = areas / total_area
    chosen = rng.choice(len(triangles), size=sample_count, replace=True, p=probabilities)
    tri = triangles[chosen]

    # Uniform barycentric sampling inside a triangle.
    r1 = np.sqrt(rng.random(sample_count, dtype=np.float32))
    r2 = rng.random(sample_count, dtype=np.float32)
    a = 1.0 - r1
    b = r1 * (1.0 - r2)
    c = r1 * r2
    return (tri[:, 0] * a[:, None] + tri[:, 1] * b[:, None] + tri[:, 2] * c[:, None]).astype(np.float32)


def _landmark_feature_points(
    landmarks3d: np.ndarray | None,
    target_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if landmarks3d is None or target_count <= 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)

    landmarks = np.asarray(landmarks3d, dtype=np.float32)
    if landmarks.ndim != 2 or landmarks.shape[0] < 68 or landmarks.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)

    groups = [
        (1, [36, 37, 38, 39, 40, 41], True, 0.17),   # right eye
        (1, [42, 43, 44, 45, 46, 47], True, 0.17),   # left eye
        (2, [27, 28, 29, 30, 33, 34, 35, 33, 32, 31], False, 0.17),
        (3, list(range(48, 60)), True, 0.22),        # outer mouth
        (3, list(range(60, 68)), True, 0.12),        # inner mouth / expression
        (4, list(range(0, 17)), False, 0.08),
        (5, [17, 18, 19, 20, 21], False, 0.035),
        (5, [22, 23, 24, 25, 26], False, 0.035),
    ]

    extent = float(np.max(np.ptp(landmarks, axis=0)))
    jitter = max(extent * 0.0055, 1e-5)
    all_points: list[np.ndarray] = []
    all_codes: list[np.ndarray] = []

    for code, indices, closed, weight in groups:
        count = max(24, int(target_count * weight))
        sampled = _sample_landmark_polyline(landmarks[indices], count, closed, jitter, rng)
        all_points.append(sampled)
        all_codes.append(np.full(len(sampled), code, dtype=np.int32))

    points = np.vstack(all_points)
    codes = np.concatenate(all_codes)

    if len(points) < target_count:
        extra_indices = rng.choice(np.arange(68), size=target_count - len(points), replace=True)
        extra = landmarks[extra_indices] + rng.normal(0, jitter * 0.8, size=(len(extra_indices), 3))
        extra_codes = _landmark_codes(extra_indices)
        points = np.vstack([points, extra.astype(np.float32)])
        codes = np.concatenate([codes, extra_codes])
    elif len(points) > target_count:
        keep = rng.choice(len(points), size=target_count, replace=False)
        points = points[keep]
        codes = codes[keep]

    return points.astype(np.float32), codes.astype(np.int32)


def _sample_landmark_polyline(
    points: np.ndarray,
    count: int,
    closed: bool,
    jitter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(points) < 2:
        return np.repeat(points[:1], count, axis=0)

    end_points = np.vstack([points, points[:1]]) if closed else points
    starts = end_points[:-1]
    ends = end_points[1:]
    lengths = np.linalg.norm(ends - starts, axis=1)
    total = float(np.sum(lengths))
    probabilities = lengths / total if total > 0 else np.full(len(lengths), 1 / len(lengths))
    chosen = rng.choice(len(starts), size=count, replace=True, p=probabilities)
    t = rng.random(count, dtype=np.float32)[:, None]
    sampled = starts[chosen] * (1.0 - t) + ends[chosen] * t
    sampled += rng.normal(0, jitter, size=sampled.shape).astype(np.float32)
    return sampled.astype(np.float32)


def _landmark_codes(indices: np.ndarray) -> np.ndarray:
    codes = np.zeros(len(indices), dtype=np.int32)
    codes[np.isin(indices, np.arange(36, 48))] = 1
    codes[np.isin(indices, np.arange(27, 36))] = 2
    codes[np.isin(indices, np.arange(48, 68))] = 3
    codes[np.isin(indices, np.arange(0, 17))] = 4
    codes[np.isin(indices, np.arange(17, 27))] = 5
    return codes


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Center and scale points into a roughly ``[-1, 1]`` browser space."""

    points = np.asarray(points, dtype=np.float32)
    center = np.median(points, axis=0)
    centered = points - center
    extents = np.max(centered, axis=0) - np.min(centered, axis=0)
    scale = float(np.max(extents))

    if scale <= 1e-8:
        return centered

    normalized = centered / scale * 2.0

    # Keep the face vertically balanced and slightly taller, which reads better
    # as a projected bust in the p5 scene.
    normalized[:, 1] *= 1.08
    return normalized


def make_demo_face_point_cloud(target_count: int = 7000, seed: int | None = 7) -> list[list[float]]:
    """Procedural face-like point cloud for testing the hologram without DECA.

    This is not a DECA substitute. It exists so the UI, particle system, and
    projector look can be developed before the licensed DECA model is installed.
    """

    target_count = clamp_point_count(target_count)
    rng = np.random.default_rng(seed)
    groups: list[np.ndarray] = []

    shell_count = int(target_count * 0.64)
    feature_count = target_count - shell_count
    groups.append(_demo_head_shell(shell_count, rng))
    groups.append(_demo_eyes(max(240, feature_count // 4), rng))
    groups.append(_demo_nose(max(260, feature_count // 5), rng))
    groups.append(_demo_mouth(max(260, feature_count // 5), rng))
    groups.append(_demo_jaw(max(220, feature_count // 6), rng))

    points = np.vstack(groups)
    if len(points) > target_count:
        points = points[rng.choice(len(points), size=target_count, replace=False)]
    elif len(points) < target_count:
        extra = _demo_head_shell(target_count - len(points), rng)
        points = np.vstack([points, extra])

    normalized = normalize_points(points)
    return np.round(normalized, 6).tolist()


def _demo_head_shell(count: int, rng: np.random.Generator) -> np.ndarray:
    points: list[list[float]] = []
    while len(points) < count:
        x = rng.uniform(-0.72, 0.72)
        y = rng.uniform(-1.05, 0.95)
        mask = (x / 0.72) ** 2 + (y / 1.08) ** 2
        if mask > 1.0:
            continue
        z = np.sqrt(max(0.0, 1.0 - mask)) * 0.5
        z += rng.normal(0.0, 0.035)
        x += 0.035 * np.sin(y * 5.0)
        points.append([x, y, z])
    return np.asarray(points, dtype=np.float32)


def _demo_eyes(count: int, rng: np.random.Generator) -> np.ndarray:
    points: list[list[float]] = []
    for side in (-1.0, 1.0):
        for _ in range(count // 2):
            theta = rng.uniform(0, np.pi * 2)
            radius = rng.normal(0.11, 0.025)
            x = side * 0.28 + np.cos(theta) * radius * 1.35
            y = 0.18 + np.sin(theta) * radius * 0.45
            z = 0.46 + rng.normal(0.0, 0.025)
            points.append([x, y, z])
    return np.asarray(points, dtype=np.float32)


def _demo_nose(count: int, rng: np.random.Generator) -> np.ndarray:
    points: list[list[float]] = []
    for _ in range(count):
        y = rng.uniform(-0.22, 0.28)
        width = 0.035 + (0.22 - y) * 0.09
        x = rng.normal(0.0, max(0.025, width))
        z = 0.55 - abs(y - 0.02) * 0.35 + rng.normal(0.0, 0.025)
        points.append([x, y, z])
    return np.asarray(points, dtype=np.float32)


def _demo_mouth(count: int, rng: np.random.Generator) -> np.ndarray:
    points: list[list[float]] = []
    for _ in range(count):
        x = rng.uniform(-0.34, 0.34)
        curve = -0.46 - 0.08 * (x / 0.34) ** 2
        y = curve + rng.normal(0.0, 0.025)
        z = 0.43 + rng.normal(0.0, 0.025)
        points.append([x, y, z])
    return np.asarray(points, dtype=np.float32)


def _demo_jaw(count: int, rng: np.random.Generator) -> np.ndarray:
    points: list[list[float]] = []
    for _ in range(count):
        t = rng.uniform(-1.0, 1.0)
        x = t * 0.54
        y = -0.74 - 0.18 * (1.0 - abs(t))
        z = 0.31 + rng.normal(0.0, 0.025)
        points.append([x, y, z])
    return np.asarray(points, dtype=np.float32)
