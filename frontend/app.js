/* Webcam capture and backend API flow.

   JavaScript handles browser-only work: webcam access, capture, UI state, and
   sending the snapshot to FastAPI. Python handles DECA because the released
   model is a PyTorch project and should run server-side.
*/

const dom = {};
let lastCaptureDataUrl = "";
let latestFaceMesh = null;
let lastFaceMeshPayload = null;
let lastHairShellPayload = [];
let faceMeshModel = null;
let liveMeshMode = false;
let liveStreamActive = false;

document.addEventListener("DOMContentLoaded", () => {
  bindDom();
  bindControls();
  updateSliderReadouts();
  refreshHealth();
});

function bindDom() {
  dom.video = document.getElementById("webcam");
  dom.canvas = document.getElementById("capture-canvas");
  dom.cameraStatus = document.getElementById("camera-status");
  dom.faceMeshStatus = document.getElementById("facemesh-status");
  dom.startCamera = document.getElementById("start-camera");
  dom.captureFace = document.getElementById("capture-face");
  dom.generate = document.getElementById("generate-hologram");
  dom.snapshotPreview = document.getElementById("snapshot-preview");
  dom.snapshotLabel = document.getElementById("snapshot-label");
  dom.status = document.getElementById("status");
  dom.cloudMeta = document.getElementById("cloud-meta");
  dom.pointCount = document.getElementById("point-count");
  dom.pointCountValue = document.getElementById("point-count-value");
  dom.noiseStrength = document.getElementById("noise-strength");
  dom.noiseStrengthValue = document.getElementById("noise-strength-value");
  dom.rotationStrength = document.getElementById("rotation-strength");
  dom.rotationStrengthValue = document.getElementById("rotation-strength-value");
  dom.particleSize = document.getElementById("particle-size");
  dom.particleSizeValue = document.getElementById("particle-size-value");
  dom.expansion = document.getElementById("expansion");
  dom.expansionValue = document.getElementById("expansion-value");
  dom.movement = document.getElementById("movement");
  dom.movementValue = document.getElementById("movement-value");
  dom.brightness = document.getElementById("brightness");
  dom.brightnessValue = document.getElementById("brightness-value");
  dom.hue = document.getElementById("hue");
  dom.hueValue = document.getElementById("hue-value");
}

function bindControls() {
  dom.startCamera.addEventListener("click", startCamera);
  dom.captureFace.addEventListener("click", captureFrame);
  dom.generate.addEventListener("click", generateHologram);

  dom.pointCount.addEventListener("input", updateSliderReadouts);
  dom.noiseStrength.addEventListener("input", () => {
    updateSliderReadouts();
    window.hologramSketch?.setNoiseStrength(Number(dom.noiseStrength.value) / 100);
  });
  dom.rotationStrength.addEventListener("input", () => {
    updateSliderReadouts();
    window.hologramSketch?.setRotationStrength(Number(dom.rotationStrength.value) / 100);
  });

  dom.particleSize.addEventListener("input", () => {
    updateSliderReadouts();
    window.hologramSketch?.setParticleSize(Number(dom.particleSize.value) / 100);
  });
  dom.expansion.addEventListener("input", () => {
    updateSliderReadouts();
    window.hologramSketch?.setExpansion(Number(dom.expansion.value) / 100);
  });
  dom.movement.addEventListener("input", () => {
    updateSliderReadouts();
    window.hologramSketch?.setMovement(Number(dom.movement.value) / 100);
  });
  dom.brightness.addEventListener("input", () => {
    updateSliderReadouts();
    window.hologramSketch?.setBrightness(Number(dom.brightness.value) / 100);
  });
  dom.hue.addEventListener("input", () => {
    updateSliderReadouts();
    window.hologramSketch?.setHue(Number(dom.hue.value) / 100);
  });
}

async function startCamera() {
  setStatus("Requesting webcam access...");

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 960 },
      },
      audio: false,
    });

    dom.video.srcObject = stream;
    await dom.video.play();
    dom.captureFace.disabled = false;
    dom.cameraStatus.textContent = "Camera live. Center your face.";
    liveMeshMode = true;
    liveStreamActive = true;
    startFaceMeshTracking();
    setStatus("Camera live. Hologram will appear when your face is detected.");
    return true;
  } catch (error) {
    setStatus(`Could not start webcam: ${error.message}`, "error");
    dom.cameraStatus.textContent = "Camera blocked or unavailable";
    return false;
  }
}

function captureFrame() {
  const capture = captureCurrentFrame({ updatePreview: true, labelPrefix: "JPEG captured" });
  if (!capture) {
    return;
  }

  lastCaptureDataUrl = capture.dataUrl;
  lastFaceMeshPayload = capture.facePayload;
  lastHairShellPayload = capture.hairPayload;
  dom.generate.disabled = false;
  setStatus("Snapshot captured. Generate the DECA hologram when ready.");
}

async function generateHologram() {
  if (!lastCaptureDataUrl) {
    setStatus("Capture a webcam frame first.", "error");
    return;
  }

  const cameraOn = !!dom.video.srcObject;
  const ok = await requestReconstruction({
    dataUrl: lastCaptureDataUrl,
    facePayload: lastFaceMeshPayload,
    hairPayload: lastHairShellPayload,
    loadingMessage: "Sending snapshot to Python backend. DECA inference may take a moment...",
    successMessage: "Hologram loaded!",
    liveDriven: cameraOn,
  });

  if (ok && cameraOn) {
    liveStreamActive = false;
    liveMeshMode = true;
  }
}


function captureCurrentFrame({ updatePreview = true, labelPrefix = "JPEG captured" } = {}) {
  if (!dom.video.videoWidth || !dom.video.videoHeight) {
    setStatus("The camera is not ready yet. Wait a second and try again.", "error");
    return null;
  }

  dom.canvas.width = dom.video.videoWidth;
  dom.canvas.height = dom.video.videoHeight;
  const context = dom.canvas.getContext("2d");

  context.save();
  context.translate(dom.canvas.width, 0);
  context.scale(-1, 1);
  context.drawImage(dom.video, 0, 0, dom.canvas.width, dom.canvas.height);
  context.restore();

  const dataUrl = dom.canvas.toDataURL("image/jpeg", 0.92);
  const facePayload = buildFaceMeshPayload();
  const hairPayload = buildHairShellPayload(context, dom.canvas.width, dom.canvas.height, facePayload?.face_landmarks || null);

  if (updatePreview) {
    dom.snapshotPreview.src = dataUrl;
    const expressionLabel = facePayload?.face_metrics
      ? ` | mouth ${Math.round(facePayload.face_metrics.mouthOpen * 100)}%`
      : "";
    dom.snapshotLabel.textContent = `${dom.canvas.width} x ${dom.canvas.height} ${labelPrefix}${expressionLabel}`;
  }

  return { dataUrl, facePayload, hairPayload };
}

async function requestReconstruction({
  dataUrl,
  facePayload,
  hairPayload = [],
  loadingMessage,
  successMessage,
  liveDriven = false,
}) {
  setLoading(true, loadingMessage);

  try {
    const response = await fetch(`${apiBase()}/api/reconstruct`, {
      method: "POST",
      cache: "no-store",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        image: dataUrl,
        point_count: Number(dom.pointCount.value),
        face_landmarks: facePayload?.face_landmarks || null,
        face_metrics: facePayload?.face_metrics || null,
      }),
    });

    const payload = await readJsonResponse(response, "Reconstruction failed");
    if (!response.ok) {
      throw new ApiError(extractErrorMessage(payload, "Reconstruction failed"), payload);
    }

    payload.meta = payload.meta || {};
    payload.meta.liveDriven = liveDriven;
    if (liveDriven) {
      payload.meta.liveFeatureOverlay = true;
      payload.meta.mode = "live-camera";
    }

    loadPointCloudIntoSketch(payload, hairPayload);
    setStatus(successMessage);
    return true;
  } catch (error) {
    const hint = error.payload?.detail?.kind === "deca_unavailable"
      ? " Live hologram needs the DECA backend configured."
      : "";
    setStatus(`${error.message || error}.${hint}`, "error");
    return false;
  } finally {
    setLoading(false);
  }
}

async function startFaceMeshTracking() {
  if (!window.ml5?.faceMesh || faceMeshModel) {
    dom.faceMeshStatus.textContent = faceMeshModel ? "faceMesh tracking" : "faceMesh unavailable";
    return;
  }

  try {
    dom.faceMeshStatus.textContent = "faceMesh loading";
    const options = { maxFaces: 1, refineLandmarks: true, flipHorizontal: false };
    const maybeModel = window.ml5.faceMesh(options);
    faceMeshModel = typeof maybeModel?.then === "function" ? await maybeModel : maybeModel;
    faceMeshModel.detectStart(dom.video, gotFaceMeshResults);
    dom.faceMeshStatus.textContent = "faceMesh scanning";
  } catch (error) {
    console.warn("ml5 faceMesh could not start", error);
    dom.faceMeshStatus.textContent = "faceMesh optional";
  }
}

function gotFaceMeshResults(results) {
  latestFaceMesh = Array.isArray(results) && results.length ? results[0] : null;
  dom.faceMeshStatus.textContent = latestFaceMesh ? "faceMesh lock" : "faceMesh scanning";

  if (!liveMeshMode) {
    window.hologramSketch?.setLiveExpression(null);
    window.hologramSketch?.setLiveFeatureMap([]);
    return;
  }

  if (!latestFaceMesh) {
    if (window.hologramSketch?.isLoaded()) {
      setStatus("Live hologram running. Re-center your face for expression tracking.");
    }
    return;
  }

  const ml5Cloud = buildMl5PointCloud();
  if (!ml5Cloud) return;

  const livePayload = buildFaceMeshPayload();

  if (!window.hologramSketch?.isLoaded()) {
    dom.generate.disabled = false;
    const cloudMeta = { liveDriven: true, liveFeatureOverlay: true, mode: "live-ml5", source: "live-ml5" };
    window.hologramSketch?.loadPointCloud(ml5Cloud, cloudMeta);
    window.hologramSketch?.setLiveExpression(livePayload?.face_metrics || null);
    window.hologramSketch?.setLiveFeatureMap(buildLiveFeatureMap(livePayload?.face_landmarks || null));
    setStatus("Live hologram active. Expressions tracking in real time.");
    return;
  }

  window.hologramSketch?.setLiveExpression(livePayload?.face_metrics || null);
  window.hologramSketch?.setLiveFeatureMap(buildLiveFeatureMap(livePayload?.face_landmarks || null));
  if (liveStreamActive) {
    window.hologramSketch?.streamPointCloud(ml5Cloud, {
      liveDriven: true,
      liveFeatureOverlay: true,
      mode: "live-ml5",
      source: "live-ml5",
    });
  }
}

function buildFaceMeshPayload() {
  const keypoints = latestFaceMesh?.keypoints;
  if (!Array.isArray(keypoints) || keypoints.length < 20 || !dom.video.videoWidth || !dom.video.videoHeight) {
    return null;
  }

  const faceLandmarks = keypoints
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
    .map((point) => {
      const rawX = point.x <= 1.5 ? point.x : point.x / dom.video.videoWidth;
      const rawY = point.y <= 1.5 ? point.y : point.y / dom.video.videoHeight;
      // The submitted capture is mirrored, so mirror ml5's raw video landmarks.
      return [clamp01(1 - rawX), clamp01(rawY)];
    });

  if (faceLandmarks.length < 20) {
    return null;
  }

  return {
    face_landmarks: faceLandmarks,
    face_metrics: calculateExpressionMetrics(faceLandmarks),
  };
}

function calculateExpressionMetrics(points) {
  // MediaPipe/ml5 faceMesh landmark ratios. These are capture hints and visual
  // metadata only; DECA remains responsible for 3D reconstruction.
  const getPoint = (index) => points[index] || null;
  const midpoint = (...indices) => {
    const valid = indices.map(getPoint).filter(Boolean);
    if (!valid.length) return [0, 0];
    const total = valid.reduce((acc, point) => [acc[0] + point[0], acc[1] + point[1]], [0, 0]);
    return [total[0] / valid.length, total[1] / valid.length];
  };
  const distance = (a, b) => {
    const pa = points[a];
    const pb = points[b];
    if (!pa || !pb) return 0;
    return Math.hypot(pa[0] - pb[0], pa[1] - pb[1]);
  };

  const faceWidth = Math.max(distance(234, 454), distance(127, 356), 0.001);
  const mouthOpen = clamp01(distance(13, 14) / faceWidth * 5.8);
  const smile = clamp01(distance(61, 291) / faceWidth * 1.65 - 0.72);
  const leftEyeOpen = clamp01(distance(159, 145) / faceWidth * 8.0);
  const rightEyeOpen = clamp01(distance(386, 374) / faceWidth * 8.0);
  const browLift = clamp01((distance(105, 159) + distance(334, 386)) / faceWidth * 2.2 - 0.35);
  const nose = getPoint(1) || midpoint(4, 5);
  const leftCheek = getPoint(234) || getPoint(127);
  const rightCheek = getPoint(454) || getPoint(356);
  const eyeCenter = midpoint(33, 133, 159, 145, 362, 263, 386, 374);
  const mouthCenter = midpoint(13, 14, 61, 291);
  const midFaceX = leftCheek && rightCheek ? (leftCheek[0] + rightCheek[0]) * 0.5 : 0.5;
  const halfFaceWidth = leftCheek && rightCheek ? Math.max(Math.abs(rightCheek[0] - leftCheek[0]) * 0.5, 0.001) : 0.5;
  const verticalSpan = Math.max(Math.abs(mouthCenter[1] - eyeCenter[1]), 0.001);
  const headYaw = clampSigned(((nose?.[0] ?? midFaceX) - midFaceX) / halfFaceWidth * 1.25);
  const headPitch = clampSigned((((nose?.[1] ?? eyeCenter[1]) - eyeCenter[1]) / verticalSpan - 0.52) * 1.85);

  return {
    mouthOpen: Number(mouthOpen.toFixed(3)),
    smile: Number(smile.toFixed(3)),
    eyeOpen: Number(((leftEyeOpen + rightEyeOpen) * 0.5).toFixed(3)),
    browLift: Number(browLift.toFixed(3)),
    headYaw: Number(headYaw.toFixed(3)),
    headPitch: Number(headPitch.toFixed(3)),
    keypointCount: points.length,
  };
}

function getMl5FeatureMap() {
  const map = new Array(468).fill(0);
  for (const i of [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173, 263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]) map[i] = 1;
  for (const i of [1, 2, 4, 5, 6, 19, 94, 97, 98, 99, 100, 101, 102, 164, 168, 195, 197]) map[i] = 2;
  for (const i of [0, 11, 12, 13, 14, 15, 16, 17, 37, 38, 39, 40, 41, 61, 62, 72, 73, 74, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 91, 95, 146, 178, 179, 180, 181, 183, 184, 185, 186, 267, 269, 270, 271, 272, 273, 291, 292, 302, 303, 304, 306, 307, 308, 310, 311, 312, 314, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325]) map[i] = 3;
  for (const i of [46, 52, 53, 55, 63, 65, 66, 67, 70, 105, 107, 336, 296, 300, 334, 293, 295, 282, 283, 285]) map[i] = 5;
  for (const i of [58, 127, 132, 136, 148, 149, 150, 152, 172, 176, 234, 288, 297, 361, 365, 377, 378, 379, 397, 400, 432, 434, 454]) map[i] = 4;
  return map;
}

function buildMl5PointCloud() {
  const kps = latestFaceMesh?.keypoints;
  if (!Array.isArray(kps) || kps.length < 100 || !dom.video.videoWidth || !dom.video.videoHeight) return null;

  const w = dom.video.videoWidth;
  const h = dom.video.videoHeight;
  const lc = kps[234] || kps[127];
  const rc = kps[454] || kps[356];
  const chin = kps[152];
  const brow = kps[10];
  if (!lc || !rc || !chin) return null;

  const cx = (lc.x + rc.x) * 0.5 / w;
  const cy = (lc.y + rc.y) * 0.5 / h;
  const fw = Math.max(Math.abs(rc.x - lc.x) / w, 0.1);
  const browNY = brow ? brow.y / h : cy - fw * 0.7;
  const scale = Math.max(fw, Math.abs(chin.y / h - browNY) * 0.9, 0.1);

  const fm = getMl5FeatureMap();
  const fz = [0, 0.06, 0.18, 0.12, -0.04, 0.04, -0.12];

  return kps.map((kp, i) => {
    if (!kp || !Number.isFinite(kp.x) || !Number.isFinite(kp.y)) return null;
    const nx = (kp.x / w - cx) / scale;
    const ny = -(kp.y / h - cy) / scale;
    const r = Math.sqrt(nx * nx + ny * ny * 0.82);
    const f = fm[i] || 0;
    return { x: nx, y: ny, z: Math.max(0, 0.38 - r * 0.68) + (fz[f] || 0), feature: f };
  }).filter(Boolean);
}

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

function clampSigned(value) {
  return Math.min(1, Math.max(-1, value));
}

function buildLiveFeatureMap(faceLandmarks) {
  if (!Array.isArray(faceLandmarks) || faceLandmarks.length < 20) {
    return [];
  }

  const featureLoops = [
    { feature: "mouth", z: 30, closed: true, indices: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291] },
    { feature: "mouth", z: 24, closed: true, indices: [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308] },
    { feature: "leftEye", z: 16, closed: true, indices: [33, 160, 158, 133, 153, 144] },
    { feature: "rightEye", z: 16, closed: true, indices: [362, 385, 387, 263, 373, 380] },
    { feature: "leftBrow", z: 8, closed: false, indices: [70, 63, 105, 66, 107] },
    { feature: "rightBrow", z: 8, closed: false, indices: [336, 296, 334, 293, 300] },
    { feature: "nose", z: 38, closed: false, indices: [168, 6, 197, 195, 5, 4, 1] },
    {
      feature: "jaw",
      z: -8,
      closed: false,
      indices: [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454],
    },
  ];

  const livePoints = [];

  for (const loop of featureLoops) {
    const points = loop.indices
      .map((index) => faceLandmarks[index] || null)
      .filter(Boolean);

    if (points.length < 2) {
      continue;
    }

    const segments = loop.closed ? points.length : points.length - 1;
    for (let i = 0; i < segments; i += 1) {
      const current = points[i];
      const next = points[(i + 1) % points.length];
      const steps = 4;

      for (let step = 0; step < steps; step += 1) {
        const t = step / steps;
        livePoints.push({
          feature: loop.feature,
          x: lerp(current[0], next[0], t),
          y: lerp(current[1], next[1], t),
          z: loop.z,
        });
      }
    }
  }

  return livePoints;
}

function buildHairShellPayload(context, width, height, faceLandmarks) {
  if (!context || !Array.isArray(faceLandmarks) || faceLandmarks.length < 20 || !width || !height) {
    return [];
  }

  const imageData = context.getImageData(0, 0, width, height).data;
  const getPoint = (index) => faceLandmarks[index] || null;
  const averagePoint = (indices) => {
    const valid = indices.map(getPoint).filter(Boolean);
    if (!valid.length) return null;
    return valid.reduce((acc, point) => [acc[0] + point[0], acc[1] + point[1]], [0, 0]).map((value) => value / valid.length);
  };
  const samplePixel = (x, y) => {
    const px = Math.max(0, Math.min(width - 1, Math.round(x * (width - 1))));
    const py = Math.max(0, Math.min(height - 1, Math.round(y * (height - 1))));
    const index = (py * width + px) * 4;
    return [
      imageData[index],
      imageData[index + 1],
      imageData[index + 2],
      imageData[index + 3],
    ];
  };

  const leftCheek = getPoint(234) || getPoint(127);
  const rightCheek = getPoint(454) || getPoint(356);
  const chin = getPoint(152);
  const browCenter = averagePoint([70, 105, 107, 336, 334, 300]);
  const eyeCenter = averagePoint([33, 133, 362, 263, 159, 145, 386, 374]);
  if (!leftCheek || !rightCheek || !chin || !browCenter || !eyeCenter) {
    return [];
  }

  const faceWidth = Math.max(Math.abs(rightCheek[0] - leftCheek[0]), 0.18);
  const faceHeight = Math.max(Math.abs(chin[1] - browCenter[1]) * 1.32, faceWidth * 1.04);
  const centerX = (leftCheek[0] + rightCheek[0]) * 0.5;
  const centerY = browCenter[1] + faceHeight * 0.16;
  const outerRx = faceWidth * 0.9;
  const outerRy = faceHeight * 0.9;
  const innerRx = faceWidth * 0.56;
  const innerRy = faceHeight * 0.72;
  const topY = browCenter[1] - faceHeight * 0.72;
  const bottomY = chin[1] + faceHeight * 0.34;
  const leftX = centerX - outerRx * 1.06;
  const rightX = centerX + outerRx * 1.06;

  const skinSampleIndices = [1, 4, 5, 6, 197, 195, 234, 454, 33, 263, 2, 98, 327];
  const skinSamples = skinSampleIndices
    .map(getPoint)
    .filter(Boolean)
    .map(([x, y]) => samplePixel(x, y))
    .filter((pixel) => pixel[3] > 0);
  if (!skinSamples.length) {
    return buildFallbackHairShellPayload(faceLandmarks);
  }

  const skinTone = skinSamples.reduce(
    (acc, pixel) => [acc[0] + pixel[0], acc[1] + pixel[1], acc[2] + pixel[2]],
    [0, 0, 0],
  ).map((value) => value / skinSamples.length);

  const points = [];
  const cols = 62;
  const rows = 76;
  const maxCount = Math.min(Math.round(Number(dom.pointCount.value) * 0.16), 2200);

  for (let row = 0; row < rows; row += 1) {
    const y = lerp(topY, bottomY, row / (rows - 1));
    for (let col = 0; col < cols; col += 1) {
      const x = lerp(leftX, rightX, col / (cols - 1));
      const dx = (x - centerX) / outerRx;
      const dy = (y - centerY) / outerRy;
      const outer = dx * dx + dy * dy;
      if (outer > 1.06) {
        continue;
      }

      const innerDx = (x - centerX) / innerRx;
      const innerDy = (y - (browCenter[1] + chin[1]) * 0.5) / innerRy;
      const inner = innerDx * innerDx + innerDy * innerDy;
      const upperBias = clamp01((browCenter[1] - y + faceHeight * 0.4) / (faceHeight * 0.84));
      const sideBias = clamp01((Math.abs(x - centerX) - faceWidth * 0.24) / (faceWidth * 0.55));
      const belowJaw = y > chin[1] + faceHeight * 0.04;
      const shellMask = inner > 0.94 || upperBias > 0.22 || sideBias > 0.18 || belowJaw;
      if (!shellMask) {
        continue;
      }

      const [r, g, b, a] = samplePixel(x, y);
      if (a < 10) {
        continue;
      }

      const luma = (0.2126 * r) + (0.7152 * g) + (0.0722 * b);
      const skinDistance = Math.hypot(r - skinTone[0], g - skinTone[1], b - skinTone[2]);
      let likelihood = clamp01((skinDistance - 14) / 92) * 0.72 + clamp01((168 - luma) / 124) * 0.28;

      if (upperBias > 0.5) {
        likelihood += upperBias * 0.12;
      }
      if (belowJaw && sideBias < 0.18) {
        likelihood *= 0.42;
      }
      if (likelihood < 0.22) {
        continue;
      }

      const crownCurve = Math.sqrt(Math.max(0, 1 - Math.min(outer, 1)));
      const sideSign = x < centerX ? -1 : 1;
      const depth = clampSigned(((crownCurve - 0.46) * 1.18) + sideBias * sideSign * 0.18);

      points.push({
        x: clamp01(x),
        y: clamp01(y),
        z: depth,
        strength: clamp01(likelihood),
      });
    }
  }

  if (!points.length) {
    return buildFallbackHairShellPayload(faceLandmarks);
  }

  points.sort((a, b) => b.strength - a.strength);
  const basePoints = buildFallbackHairShellPayload(faceLandmarks, 0.72);
  if (points.length <= maxCount) {
    return mergeHairShellPayload(points, basePoints);
  }

  const sampled = [];
  const stride = points.length / maxCount;
  for (let i = 0; i < maxCount; i += 1) {
    sampled.push(points[Math.floor(i * stride)]);
  }
  return mergeHairShellPayload(sampled, basePoints);
}

function buildFallbackHairShellPayload(faceLandmarks, strengthScale = 1) {
  if (!Array.isArray(faceLandmarks) || faceLandmarks.length < 20) {
    return [];
  }

  const getPoint = (index) => faceLandmarks[index] || null;
  const averagePoint = (indices) => {
    const valid = indices.map(getPoint).filter(Boolean);
    if (!valid.length) return null;
    return valid.reduce((acc, point) => [acc[0] + point[0], acc[1] + point[1]], [0, 0]).map((value) => value / valid.length);
  };

  const leftCheek = getPoint(234) || getPoint(127);
  const rightCheek = getPoint(454) || getPoint(356);
  const chin = getPoint(152);
  const browCenter = averagePoint([70, 105, 107, 336, 334, 300]);
  if (!leftCheek || !rightCheek || !chin || !browCenter) {
    return [];
  }

  const faceWidth = Math.max(Math.abs(rightCheek[0] - leftCheek[0]), 0.18);
  const faceHeight = Math.max(Math.abs(chin[1] - browCenter[1]) * 1.34, faceWidth * 1.08);
  const centerX = (leftCheek[0] + rightCheek[0]) * 0.5;
  const crownY = browCenter[1] - faceHeight * 0.76;
  const jawY = chin[1] + faceHeight * 0.3;
  const outerRx = faceWidth * 0.98;
  const outerRy = faceHeight * 0.94;
  const points = [];
  const arcs = 34;
  const layers = 24;

  for (let layer = 0; layer < layers; layer += 1) {
    const layerT = layer / Math.max(1, layers - 1);
    const scaleX = lerp(0.62, 1.16, layerT);
    const scaleY = lerp(0.54, 1.08, layerT);
    const yOffset = lerp(-0.08, 0.22, layerT) * outerRy;
    const zOffset = lerp(0.5, -0.16, layerT);
    for (let i = 0; i < arcs; i += 1) {
      const theta = lerp(Math.PI * 1.04, Math.PI * -0.04, i / Math.max(1, arcs - 1));
      const x = centerX + Math.cos(theta) * outerRx * scaleX;
      const y = crownY + yOffset + Math.sin(theta) * outerRy * scaleY;
      if (y > jawY) {
        continue;
      }

      const sideBias = Math.abs(Math.cos(theta));
      const crownBias = clamp01((browCenter[1] - y + faceHeight * 0.2) / (faceHeight * 0.92));
      const strength = clamp01((0.48 + sideBias * 0.28 + crownBias * 0.34) * strengthScale);
      points.push({
        x: clamp01(x),
        y: clamp01(y),
        z: clampSigned(zOffset + Math.cos(theta * 0.65) * 0.1),
        strength,
      });
    }
  }

  return points;
}

function mergeHairShellPayload(primary, secondary) {
  if (!secondary.length) {
    return primary;
  }
  if (!primary.length) {
    return secondary;
  }
  return [...primary, ...secondary];
}

async function loadDemoCloudFallback() {
  setLoading(true, "Loading procedural demo cloud for particle-system testing...");

  try {
    const response = await fetch(`${apiBase()}/api/demo-points?count=${Number(dom.pointCount.value)}`, {
      cache: "no-store",
      headers: {
        Accept: "application/json",
      },
    });
    const payload = await readJsonResponse(response, "Could not load demo cloud");
    if (!response.ok) {
      throw new ApiError(extractErrorMessage(payload, "Could not load demo cloud"), payload);
    }

    loadPointCloudIntoSketch(payload);
    window.hologramSketch?.setHairShell([]);
    const liveHint = dom.video.srcObject
      ? " Demo cloud loaded. Your live faceMesh features and expressions now drive the hologram."
      : " Demo cloud loaded. Start the camera to drive it with your live expressions.";
    setStatus(liveHint);
  } catch (error) {
    setStatus(`Could not load demo cloud: ${error.message}`, "error");
  } finally {
    setLoading(false);
  }
}

function loadPointCloudIntoSketch(payload, hairPayload = null) {
  const points = payload.points || [];
  const meta = payload.meta || {};
  window.hologramSketch?.loadPointCloud(points, meta);
  if (hairPayload !== null) {
    window.hologramSketch?.setHairShell(hairPayload);
  }

  const source = meta.mode === "live-camera"
    ? "Live DECA"
    : meta.mode === "live-facemesh"
      ? "Live faceMesh"
    : meta.source === "deca"
      ? "DECA"
      : "Demo";
  const expression = meta.expression?.mouthOpen !== undefined
    ? ` | mouth ${Math.round(meta.expression.mouthOpen * 100)}%`
    : "";
  dom.cloudMeta.textContent = `${source} cloud loaded${expression}`;
}

async function refreshHealth() {
  try {
    const response = await fetch(`${apiBase()}/api/health`, {
      cache: "no-store",
      headers: {
        Accept: "application/json",
      },
    });
    const payload = await readJsonResponse(response, "Health check failed");
    if (payload?.deca?.configured) {
      setStatus("Backend online. DECA appears configured for inference.");
    }
  } catch {
    // The page can still be viewed as static HTML; API calls will show errors.
  }
}

function updateSliderReadouts() {
  dom.pointCountValue.textContent = dom.pointCount.value;
  dom.noiseStrengthValue.textContent = dom.noiseStrength.value;
  dom.rotationStrengthValue.textContent = dom.rotationStrength.value;
  dom.particleSizeValue.textContent = dom.particleSize.value;
  dom.expansionValue.textContent = dom.expansion.value;
  dom.movementValue.textContent = dom.movement.value;
  dom.brightnessValue.textContent = dom.brightness.value;
  dom.hueValue.textContent = dom.hue.value;

  window.hologramSketch?.setNoiseStrength(Number(dom.noiseStrength.value) / 100);
  window.hologramSketch?.setRotationStrength(Number(dom.rotationStrength.value) / 100);
  window.hologramSketch?.setParticleSize(Number(dom.particleSize.value) / 100);
  window.hologramSketch?.setExpansion(Number(dom.expansion.value) / 100);
  window.hologramSketch?.setMovement(Number(dom.movement.value) / 100);
  window.hologramSketch?.setBrightness(Number(dom.brightness.value) / 100);
  window.hologramSketch?.setHue(Number(dom.hue.value) / 100);
}

function setLoading(isLoading, message = "") {
  dom.generate.disabled = isLoading || !lastCaptureDataUrl;
  dom.startCamera.disabled = isLoading;
  dom.captureFace.disabled = isLoading || !dom.video.srcObject;

  if (message) {
    setStatus(message, isLoading ? "loading" : "");
  } else if (!isLoading) {
    dom.status.classList.remove("loading");
  }
}


function setStatus(message, mode = "") {
  dom.status.textContent = message;
  dom.status.classList.toggle("error", mode === "error");
  dom.status.classList.toggle("loading", mode === "loading");
}

function apiBase() {
  if (window.DECA_API_BASE) {
    return window.DECA_API_BASE.replace(/\/$/, "");
  }
  // When served by the backend, use the current origin
  // When opened as a file://, default to local development server
  return window.location.protocol === "file:" ? "http://127.0.0.1:8000" : window.location.origin;
}

async function readJsonResponse(response, fallbackMessage) {
  const text = await response.text();
  const trimmed = text.trim();

  if (!trimmed) {
    throw new ApiError(`${fallbackMessage}: the server returned an empty response.`, {
      status: response.status,
      contentType: response.headers.get("content-type") || "",
    });
  }

  try {
    return JSON.parse(trimmed);
  } catch {
    const preview = trimmed.replace(/\s+/g, " ").slice(0, 120);
    const looksLikeHtml = /^<!doctype html|^<html/i.test(preview);
    const explanation = looksLikeHtml
      ? "the server returned an HTML page instead of JSON"
      : "the server returned invalid JSON";

    throw new ApiError(`${fallbackMessage}: ${explanation}.`, {
      status: response.status,
      contentType: response.headers.get("content-type") || "",
      preview,
    });
  }
}

function extractErrorMessage(payload, fallbackMessage) {
  const detail = payload?.detail;
  if (typeof detail === "string" && detail) {
    return detail;
  }
  if (detail && typeof detail.message === "string" && detail.message) {
    return detail.message;
  }
  return fallbackMessage;
}

class ApiError extends Error {
  constructor(message, payload) {
    super(message);
    this.name = "ApiError";
    this.payload = payload;
  }
}
