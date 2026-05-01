/* p5.js WEBGL particle hologram.

   The point cloud becomes a set of particle targets. Each particle has a
   position, target, velocity, acceleration, noise offset, brightness, and size,
   following a Nature-of-Code-style force/update/render loop.
*/

const EMITTER_Y = 250;
const FACE_SHIFT_Y = -46;
const HOLOGRAM_GROUP_Y = 14;

let canvasReady = false;
let stageElement = null;
let controlDeckElement = null;
let pendingCloud = null;
let particles = [];
let meta = {};
let emergence = 0;
let expressionDrivenMode = false;
let liveFeatureOverlayMode = false;
let mouseYaw = 0;
let mousePitch = 0;
let smoothedYaw = 0;
let smoothedPitch = 0;
let starfield = [];
let pointerInStage = false;
let stagePointerX = 0;
let stagePointerY = 0;
let liveExpressionTarget = neutralExpression();
let liveExpressionCurrent = neutralExpression();
let liveFeatureMapTarget = [];
let liveFeatureMapCurrent = [];
let liveFeatureGroupsCurrent = emptyLiveFeatureGroups();
let hairShellTarget = [];
let hairShellParticles = [];
let emotionIntensity = 0;

const hologramConfig = {
  noiseStrength: 0.44,
  rotationStrength: 0.58,
  autoRotate: 0.18,
  scanlineStrength: 0,
  particleSize: 0.46,
  expansion: 1.18,
  movement: 0.36,
  brightness: 0.96,
  hue: 0.52,
};

window.hologramSketch = {
  loadPointCloud(points, cloudMeta = {}) {
    pendingCloud = { points, meta: cloudMeta };
    if (canvasReady) {
      createParticlesFromCloud(points, cloudMeta);
    }
  },
  streamPointCloud(points, cloudMeta = {}) {
    pendingCloud = { points, meta: cloudMeta, stream: true };
    if (canvasReady) {
      streamParticlesFromCloud(points, cloudMeta);
    }
  },
  setNoiseStrength(value) {
    hologramConfig.noiseStrength = value;
  },
  setRotationStrength(value) {
    hologramConfig.rotationStrength = value;
  },
  setParticleSize(value) {
    hologramConfig.particleSize = value;
  },
  setExpansion(value) {
    hologramConfig.expansion = value;
  },
  setMovement(value) {
    hologramConfig.movement = value;
  },
  setBrightness(value) {
    hologramConfig.brightness = value;
  },
  setHue(value) {
    hologramConfig.hue = value;
  },
  isLoaded() {
    return particles.length > 0;
  },
  setLiveMode(enabled) {
    expressionDrivenMode = enabled;
    liveFeatureOverlayMode = enabled;
    if (!enabled) {
      liveExpressionTarget = neutralExpression();
      liveExpressionCurrent = neutralExpression();
      liveFeatureMapTarget = [];
      liveFeatureMapCurrent = [];
      liveFeatureGroupsCurrent = emptyLiveFeatureGroups();
    }
  },
  setLiveExpression(metrics) {
    liveExpressionTarget = normalizeLiveExpression(metrics);
  },
  setLiveFeatureMap(points) {
    liveFeatureMapTarget = normalizeLiveFeatureMap(points);
  },
  setHairShell(points) {
    hairShellTarget = normalizeHairShell(points);
    if (canvasReady) {
      createHairShellParticles(hairShellTarget);
    }
  },
};

function setup() {
  stageElement = document.getElementById("hologram-canvas");
  controlDeckElement = document.querySelector(".control-deck");
  const cnv = createCanvas(stageElement.clientWidth, stageElement.clientHeight, WEBGL);
  cnv.parent(stageElement);
  cnv.elt.style.touchAction = "none";
  bindStagePointerTracking();
  pixelDensity(Math.min(2, window.devicePixelRatio || 1));
  colorMode(RGB, 255, 255, 255, 255);
  noiseDetail(4, 0.48);

  starfield = Array.from({ length: 140 }, () => ({
    x: random(-width * 0.55, width * 0.55),
    y: random(-height * 0.5, height * 0.45),
    z: random(-480, 120),
    pulse: random(TWO_PI),
  }));

  canvasReady = true;
  if (pendingCloud) {
    if (pendingCloud.stream) {
      streamParticlesFromCloud(pendingCloud.points, pendingCloud.meta);
    } else {
      createParticlesFromCloud(pendingCloud.points, pendingCloud.meta);
    }
  }
  if (hairShellTarget.length) {
    createHairShellParticles(hairShellTarget);
  }
}

function draw() {
  const t = millis() * 0.001;
  background(1, 6, 14);
  blendMode(ADD);

  drawStarfield(t);
  drawProjectorBase(t);

  updateMouseRotation();
  emergence = min(1, emergence + 0.011 + hologramConfig.movement * 0.008);
  updateLiveExpressionState();
  updateLiveFeatureState();

  // Keep the face mostly front-facing so reconstructed eyes, nose, mouth, and
  // jaw remain readable. The mouse can still rotate it more dramatically.
  const autoYaw = sin(t * 0.45) * (0.06 + hologramConfig.movement * 0.14) * (pointerInStage ? 0.24 : 1);
  const hover = sin(t * 1.2) * (3 + hologramConfig.movement * 10);
  const centerX = stageFocusX();
  const expressionYaw = expressionDrivenMode ? liveExpressionCurrent.headYaw * 0.34 : 0;
  const expressionPitch = expressionDrivenMode ? liveExpressionCurrent.headPitch * 0.22 : 0;

  push();
  translate(centerX, hover + HOLOGRAM_GROUP_Y, 0);
  rotateX(expressionPitch + smoothedPitch * (0.22 + hologramConfig.rotationStrength * 0.82) + sin(t * 0.7) * 0.025 * hologramConfig.movement);
  rotateY(autoYaw + expressionYaw + smoothedYaw * (0.3 + hologramConfig.rotationStrength * 1.08));
  rotateZ(sin(t * 0.33) * 0.018 * hologramConfig.movement);

  updateAndRenderParticles(t);
  pop();

  blendMode(BLEND);
}

function windowResized() {
  stageElement = document.getElementById("hologram-canvas");
  resizeCanvas(stageElement.clientWidth, stageElement.clientHeight);
}

function createParticlesFromCloud(points, cloudMeta = {}) {
  meta = cloudMeta;
  emergence = 0;
  const targets = fitPointCloud(points);
  const bounds = targetBounds(targets);
  expressionDrivenMode = Boolean(meta.liveDriven || meta.source === "procedural-demo");
  liveFeatureOverlayMode = Boolean(meta.liveFeatureOverlay ?? (meta.source === "procedural-demo"));
  if (!expressionDrivenMode) {
    liveExpressionTarget = neutralExpression();
    liveExpressionCurrent = neutralExpression();
    liveFeatureMapTarget = [];
    liveFeatureMapCurrent = [];
    liveFeatureGroupsCurrent = emptyLiveFeatureGroups();
  }
  particles = targets.map((target, index) => new HologramParticle(
    target.position,
    index,
    target.feature,
    buildExpressionRegions(target.position, bounds, target.feature),
  ));
  syncHairShellWithTargets(targets, bounds);
}

function streamParticlesFromCloud(points, cloudMeta = {}) {
  meta = cloudMeta;
  const targets = fitPointCloud(points);
  if (!targets.length) {
    return;
  }

  const bounds = targetBounds(targets);
  expressionDrivenMode = Boolean(meta.liveDriven || meta.source === "procedural-demo");
  liveFeatureOverlayMode = Boolean(meta.liveFeatureOverlay ?? (meta.source === "procedural-demo"));

  if (particles.length !== targets.length) {
    particles = targets.map((target, index) => new HologramParticle(
      target.position,
      index,
      target.feature,
      buildExpressionRegions(target.position, bounds, target.feature),
    ));
    emergence = 1;
    return;
  }

  for (let i = 0; i < particles.length; i += 1) {
    const particle = particles[i];
    const target = targets[i];
    particle.target = target.position.copy();
    particle.feature = target.feature;
    particle.regions = buildExpressionRegions(target.position, bounds, target.feature);
  }

  if (meta.source !== "live-ml5") {
    syncHairShellWithTargets(targets, bounds);
  }
  emergence = 1;
}

function fitPointCloud(points) {
  if (!Array.isArray(points) || points.length === 0) {
    return [];
  }

  const parsed = points
    .map((point) => {
      if (Array.isArray(point)) {
        return {
          xyz: [point[0], point[1], point[2]],
          feature: Number.isFinite(point[3]) ? point[3] : 0,
        };
      }
      return {
        xyz: [point.x, point.y, point.z],
        feature: Number.isFinite(point.feature) ? point.feature : 0,
      };
    })
    .filter((point) => point.xyz.every(Number.isFinite));

  if (!parsed.length) {
    return [];
  }

  // Backend points are normalized, but this extra fit keeps hand-edited JSON or
  // OBJ-derived data centered if you experiment later.
  const mins = [Infinity, Infinity, Infinity];
  const maxs = [-Infinity, -Infinity, -Infinity];
  for (const point of parsed) {
    for (let i = 0; i < 3; i += 1) {
      mins[i] = min(mins[i], point.xyz[i]);
      maxs[i] = max(maxs[i], point.xyz[i]);
    }
  }

  const center = [
    (mins[0] + maxs[0]) * 0.5,
    (mins[1] + maxs[1]) * 0.5,
    (mins[2] + maxs[2]) * 0.5,
  ];
  const extent = max(maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2]) || 1;

  return parsed.map((point) => {
    const nx = (point.xyz[0] - center[0]) / extent;
    const ny = (point.xyz[1] - center[1]) / extent;
    const nz = (point.xyz[2] - center[2]) / extent;
    return {
      position: createVector(
        nx * 315,
        -ny * 365 + FACE_SHIFT_Y,
        nz * 245,
      ),
      feature: point.feature,
    };
  });
}

function updateAndRenderParticles(t) {
  for (const particle of particles) {
    particle.update(t, emergence, hologramConfig.noiseStrength);
  }
  for (const particle of hairShellParticles) {
    particle.update(t, emergence, hologramConfig.noiseStrength * 0.8);
  }

  for (const particle of particles) {
    particle.render(t);
  }
  for (const particle of hairShellParticles) {
    particle.render(t);
  }

  drawLiveFeatureOverlay(t);
}

function targetBounds(targets) {
  const bounds = {
    minX: Infinity,
    maxX: -Infinity,
    minY: Infinity,
    maxY: -Infinity,
    minZ: Infinity,
    maxZ: -Infinity,
  };

  for (const target of targets) {
    bounds.minX = min(bounds.minX, target.position.x);
    bounds.maxX = max(bounds.maxX, target.position.x);
    bounds.minY = min(bounds.minY, target.position.y);
    bounds.maxY = max(bounds.maxY, target.position.y);
    bounds.minZ = min(bounds.minZ, target.position.z);
    bounds.maxZ = max(bounds.maxZ, target.position.z);
  }

  return bounds;
}

function updateMouseRotation() {
  const targetYaw = pointerInStage ? map(stagePointerX, 0, width, -1.12, 1.12) : 0;
  const targetPitch = pointerInStage ? map(stagePointerY, 0, height, -0.72, 0.72) : 0;
  const pointerLerp = pointerInStage ? 0.24 : 0.1;
  const settleLerp = pointerInStage ? 0.28 : 0.14;

  mouseYaw = lerp(mouseYaw, targetYaw, pointerLerp);
  mousePitch = lerp(mousePitch, targetPitch, pointerLerp);
  smoothedYaw = lerp(smoothedYaw, mouseYaw, settleLerp);
  smoothedPitch = lerp(smoothedPitch, mousePitch, settleLerp);
}

function bindStagePointerTracking() {
  if (!stageElement) return;

  const updatePointer = (event) => {
    const rect = stageElement.getBoundingClientRect();
    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;
    const inside = localX >= 0 && localX <= rect.width && localY >= 0 && localY <= rect.height;
    pointerInStage = inside;
    if (inside) {
      stagePointerX = constrain(localX, 0, rect.width);
      stagePointerY = constrain(localY, 0, rect.height);
    }
  };

  stageElement.addEventListener("pointermove", updatePointer);
  window.addEventListener("pointermove", updatePointer, { passive: true });

  stageElement.addEventListener("pointerenter", () => {
    pointerInStage = true;
  });

  stageElement.addEventListener("pointerleave", () => {
    pointerInStage = false;
  });
}

class HologramParticle {
  constructor(target, index, feature = 0, regions = buildExpressionRegions(target, targetBounds([{ position: target }]), feature)) {
    this.index = index;
    this.feature = feature;
    this.target = target.copy();
    this.regions = regions;
    this.origin = randomEmitterPosition();
    this.position = this.origin.copy().add(random(-10, 10), random(-4, 18), random(-10, 10));
    this.velocity = p5.Vector.random3D().mult(random(0.2, 1.6));
    this.acceleration = createVector(0, 0, 0);
    this.seed = random(1000);
    this.delay = random(0, 0.42);
    this.size = random(0.45, 1.15) + featureVisual(feature).sizeBoost;
    this.hairStrength = 0;
    this.alpha = 0;
  }

  update(t, globalEmergence, noiseStrength) {
    const localProgress = smoothstep(constrain((globalEmergence - this.delay) / (1 - this.delay), 0, 1));
    const unstableTarget = this.animatedTarget(t, noiseStrength, localProgress);
    const currentTarget = p5.Vector.lerp(this.origin, unstableTarget, localProgress);

    const arrive = p5.Vector.sub(currentTarget, this.position).mult(0.055 + localProgress * 0.025);
    this.applyForce(arrive);

    const visual = featureVisual(this.feature);
    const driftScale = (0.2 + localProgress * 0.8) * noiseStrength * visual.driftScale * hologramConfig.movement;
    const drift = createVector(
      noise(this.seed, t * 0.55) - 0.5,
      noise(this.seed + 31.7, t * 0.5) - 0.5,
      noise(this.seed + 73.2, t * 0.45) - 0.5,
    ).mult(driftScale * 7.5);
    this.applyForce(drift);

    this.velocity.add(this.acceleration);
    this.velocity.mult(0.84 + (1 - hologramConfig.movement) * 0.08);
    this.position.add(this.velocity);
    this.acceleration.mult(0);

    const flicker = noise(this.seed + 11.0, t * 3.2);
    const highlight = 0.88 + noise(this.seed + 21.0, t * 1.4) * 0.16;
    const strengthBoost = this.feature === 6 ? 0.55 + this.hairStrength * 0.85 : 1;
    this.alpha = 185 * hologramConfig.brightness * visual.alphaBoost * localProgress * (0.48 + flicker * 0.7) * highlight * strengthBoost;
  }

  animatedTarget(t, noiseStrength, progress) {
    const pulse = sin(t * 1.6 + this.seed) * (0.8 + hologramConfig.movement * 3.4);
    const visual = featureVisual(this.feature);
    const expandedTarget = createVector(
      this.target.x * hologramConfig.expansion,
      (this.target.y - FACE_SHIFT_Y) * hologramConfig.expansion + FACE_SHIFT_Y,
      this.target.z * hologramConfig.expansion,
    );
    const expressionScale = meta.source === "live-ml5" ? 0.12 : 1.0;
    const expressionOffset = expressionDrivenMode
      ? this.expressionOffset(liveExpressionCurrent).mult(expressionScale)
      : createVector(0, 0, 0);
    const turbulence = createVector(
      noise(this.seed + 1.2, t * 0.8) - 0.5,
      noise(this.seed + 5.6, t * 0.75) - 0.5,
      noise(this.seed + 9.8, t * 0.7) - 0.5,
    ).mult(noiseStrength * 11.0 * progress * visual.driftScale * hologramConfig.movement);

    return expandedTarget.add(expressionOffset).add(turbulence).add(0, pulse, 0);
  }

  expressionOffset(expression) {
    const mouthDelta = expression.mouthOpen;
    const smileDelta = expression.smile;
    const eyeDelta = expression.eyeOpen - 0.42;
    const browDelta = expression.browLift;
    const jawDrop = mouthDelta * 0.7;

    const offset = createVector(0, 0, 0);

    if (this.regions.mouth > 0) {
      offset.y += mouthDelta * this.regions.mouth * this.regions.mouthSplit * 22;
      offset.z += mouthDelta * this.regions.mouth * 12;
      offset.x += smileDelta * this.regions.smile * this.regions.side * 16;
      offset.y -= smileDelta * this.regions.smile * 11;
    }

    const eyeWeight = this.regions.leftEye + this.regions.rightEye;
    if (eyeWeight > 0) {
      offset.y += eyeDelta * eyeWeight * this.regions.eyeSplit * 18;
      offset.z += eyeDelta * eyeWeight * 4;
    }

    if (this.regions.brows > 0) {
      offset.y -= browDelta * this.regions.brows * 18;
    }

    if (this.regions.jaw > 0) {
      offset.y += jawDrop * this.regions.jaw * 20;
      offset.z += jawDrop * this.regions.jaw * 5;
    }

    return offset.add(this.structuralFeatureOffset());
  }

  structuralFeatureOffset() {
    if (!expressionDrivenMode) {
      return createVector(0, 0, 0);
    }

    const offset = createVector(0, 0, 0);
    offset.add(this.pullTowardFeature("mouth", this.regions.mouth * 0.78));
    offset.add(this.pullTowardFeature("nose", this.regions.nose * 0.62));
    offset.add(this.pullTowardFeature("jaw", this.regions.jaw * 0.46));

    if (this.regions.leftEye > 0) {
      offset.add(this.pullTowardFeature("leftEye", this.regions.leftEye * 0.88));
    }
    if (this.regions.rightEye > 0) {
      offset.add(this.pullTowardFeature("rightEye", this.regions.rightEye * 0.88));
    }
    if (this.regions.brows > 0) {
      offset.add(this.pullTowardFeature(this.regions.side < 0 ? "leftBrow" : "rightBrow", this.regions.brows * 0.72));
    }

    return offset;
  }

  pullTowardFeature(featureName, strength) {
    if (strength <= 0.001) {
      return createVector(0, 0, 0);
    }

    const featurePoints = liveFeatureGroupsCurrent[featureName];
    if (!featurePoints || !featurePoints.length) {
      return createVector(0, 0, 0);
    }

    let bestPoint = null;
    let bestScore = Infinity;

    for (const point of featurePoints) {
      const dx = point.x - this.target.x;
      const dy = point.y - this.target.y;
      const dz = point.z - this.target.z;
      const score = dx * dx + dy * dy * 1.1 + dz * dz * 0.2;
      if (score < bestScore) {
        bestScore = score;
        bestPoint = point;
      }
    }

    if (!bestPoint) {
      return createVector(0, 0, 0);
    }

    return p5.Vector.sub(bestPoint, this.target).mult(strength);
  }

  applyForce(force) {
    this.acceleration.add(force);
  }

  render(t) {
    if (this.alpha <= 1) {
      return;
    }

    const depthBoost = map(this.position.z, -230, 230, 0.55, 1.1, true);
    const flicker = 0.72 + 0.28 * noise(this.seed + 200, t * 8.0);
    const alpha = constrain(this.alpha * depthBoost * flicker, 0, 230);

    if (expressionDrivenMode && emotionIntensity > 0.05 && this.regions) {
      const regionWeight = max(
        this.regions.mouth,
        this.regions.leftEye + this.regions.rightEye,
        this.regions.brows,
      );
      if (regionWeight > 0.04) {
        stroke(18, 44, 172, alpha * emotionIntensity * regionWeight * 0.52);
        strokeWeight(this.size * 2.8 * (1 + hologramConfig.particleSize * 0.8));
        point(this.position.x, this.position.y, this.position.z);
      }
    }

    if (this.index % 3 === 0) {
      stroke(29, 164, 255, alpha * 0.2);
      strokeWeight(this.size * (1.4 + hologramConfig.particleSize * 1.2));
      point(this.position.x, this.position.y, this.position.z);
    }

    const visual = featureVisual(this.feature);
    const whiteCore = this.index % 9 === 0 || this.feature > 0 ? 70 : 0;
    const hairCore = this.feature === 6 ? 70 + this.hairStrength * 118 : 0;
    const shifted = shiftedColor(visual.color);
    stroke(
      shifted[0] + whiteCore + hairCore,
      shifted[1] + whiteCore * 0.2 + hairCore * 0.28,
      shifted[2] + hairCore * 0.14,
      alpha,
    );
    strokeWeight(this.size * (this.feature === 6 ? 1.04 : 0.45) * (1 + hologramConfig.particleSize * 1.2));
    point(this.position.x, this.position.y, this.position.z);
  }
}

function featureVisual(feature) {
  switch (Math.round(feature)) {
    case 1:
      return { color: [190, 255, 255], sizeBoost: 0.36, alphaBoost: 1.55, driftScale: 0.36 };
    case 2:
      return { color: [120, 245, 255], sizeBoost: 0.22, alphaBoost: 1.35, driftScale: 0.42 };
    case 3:
      return { color: [235, 255, 255], sizeBoost: 0.42, alphaBoost: 1.75, driftScale: 0.32 };
    case 4:
      return { color: [92, 220, 255], sizeBoost: 0.12, alphaBoost: 1.18, driftScale: 0.5 };
    case 5:
      return { color: [160, 250, 255], sizeBoost: 0.18, alphaBoost: 1.28, driftScale: 0.42 };
    case 6:
      return { color: [160, 238, 255], sizeBoost: 0.7, alphaBoost: 1.88, driftScale: 1.22 };
    default:
      return { color: [85, 230, 255], sizeBoost: 0, alphaBoost: 1, driftScale: 1 };
  }
}

function neutralExpression() {
  return {
    mouthOpen: 0,
    smile: 0,
    eyeOpen: 0.42,
    browLift: 0,
    headYaw: 0,
    headPitch: 0,
  };
}

function normalizeLiveExpression(metrics) {
  if (!metrics) {
    return neutralExpression();
  }

  return {
    mouthOpen: constrain(metrics.mouthOpen ?? 0, 0, 1),
    smile: constrain(metrics.smile ?? 0, 0, 1),
    eyeOpen: constrain(metrics.eyeOpen ?? 0.42, 0, 1),
    browLift: constrain(metrics.browLift ?? 0, 0, 1),
    headYaw: constrain(metrics.headYaw ?? 0, -1, 1),
    headPitch: constrain(metrics.headPitch ?? 0, -1, 1),
  };
}

function updateLiveExpressionState() {
  const target = expressionDrivenMode ? liveExpressionTarget : neutralExpression();
  const easing = expressionDrivenMode ? 0.26 : 0.1;

  liveExpressionCurrent.mouthOpen = lerp(liveExpressionCurrent.mouthOpen, target.mouthOpen, easing);
  liveExpressionCurrent.smile = lerp(liveExpressionCurrent.smile, target.smile, easing);
  liveExpressionCurrent.eyeOpen = lerp(liveExpressionCurrent.eyeOpen, target.eyeOpen, easing);
  liveExpressionCurrent.browLift = lerp(liveExpressionCurrent.browLift, target.browLift, easing);
  liveExpressionCurrent.headYaw = lerp(liveExpressionCurrent.headYaw, target.headYaw, easing * 0.95);
  liveExpressionCurrent.headPitch = lerp(liveExpressionCurrent.headPitch, target.headPitch, easing * 0.95);

  emotionIntensity = expressionDrivenMode
    ? constrain(
        liveExpressionCurrent.mouthOpen * 0.5 +
        liveExpressionCurrent.smile * 0.25 +
        liveExpressionCurrent.browLift * 0.15 +
        abs(liveExpressionCurrent.headYaw) * 0.1,
        0, 1,
      )
    : 0;
}

function normalizeLiveFeatureMap(points) {
  if (!Array.isArray(points)) {
    return [];
  }

  return points
    .filter((point) => Number.isFinite(point?.x) && Number.isFinite(point?.y))
    .map((point) => ({
      feature: point.feature || "feature",
      x: point.x,
      y: point.y,
      z: Number.isFinite(point.z) ? point.z : 0,
    }));
}

function normalizeHairShell(points) {
  if (!Array.isArray(points)) {
    return [];
  }

  return points
    .filter((point) => Number.isFinite(point?.x) && Number.isFinite(point?.y))
    .map((point) => ({
      x: constrain(point.x, 0, 1),
      y: constrain(point.y, 0, 1),
      z: constrain(point.z ?? 0, -1, 1),
      strength: constrain(point.strength ?? 0.5, 0, 1),
    }));
}

function createHairShellParticles(points) {
  if (!points.length) {
    hairShellParticles = [];
    return;
  }

  hairShellParticles = points.map((point, index) => {
    const targetPosition = point.position?.copy ? point.position.copy() : mapHairShellPoint(point);
    const particle = new HologramParticle(targetPosition, 50000 + index, 6, zeroRegions());
    particle.size += 0.36 + point.strength * 0.52;
    particle.delay = random(0, 0.18);
    particle.hairStrength = point.strength;
    return particle;
  });
}

function syncHairShellWithTargets(targets, bounds) {
  const fallback = createFallbackHairShellFromTargets(targets, bounds);
  const combined = hairShellTarget.length
    ? [...hairShellTarget, ...fallback]
    : fallback;
  createHairShellParticles(combined);
}

function createFallbackHairShellFromTargets(targets, bounds) {
  if (!targets.length) {
    return [];
  }

  const centerX = (bounds.minX + bounds.maxX) * 0.5;
  const centerY = (bounds.minY + bounds.maxY) * 0.5;
  const width = max(1, bounds.maxX - bounds.minX);
  const height = max(1, bounds.maxY - bounds.minY);
  const points = [];
  const candidateTargets = targets.filter((target) => {
    const u = (target.position.x - bounds.minX) / width;
    const v = (target.position.y - bounds.minY) / height;
    const side = Math.abs(u - 0.5);
    return v < 0.72 && (v < 0.34 || side > 0.22);
  });

  for (const target of candidateTargets) {
    const u = (target.position.x - bounds.minX) / width;
    const v = (target.position.y - bounds.minY) / height;
    const side = (u - 0.5) * 2;
    const sideAbs = Math.abs(side);
    const crown = clamp01((0.42 - v) / 0.42);
    const upper = clamp01((0.68 - v) / 0.68);
    const outer = clamp01((sideAbs - 0.16) / 0.34);
    const influence = max(crown, outer * 0.82 + upper * 0.28);
    if (influence <= 0.06) {
      continue;
    }

    const outwardX = side * (18 + outer * 68 + crown * 24);
    const liftY = -(24 + crown * 92 + outer * 28);
    const depthZ = 28 + crown * 74 + outer * 34;

    points.push({
      position: createVector(
        target.position.x + outwardX,
        target.position.y + liftY,
        target.position.z + depthZ,
      ),
      strength: constrain(0.54 + influence * 0.54, 0, 1),
    });

    if (crown > 0.18) {
      points.push({
        position: createVector(
          target.position.x + outwardX * 0.52,
          target.position.y + liftY - 16 - crown * 18,
          target.position.z + depthZ + 12 + crown * 18,
        ),
        strength: constrain(0.48 + crown * 0.48, 0, 1),
      });
    }
  }

  return points;
}

function updateLiveFeatureState() {
  const target = expressionDrivenMode ? liveFeatureMapTarget : [];

  if (!target.length) {
    liveFeatureMapCurrent = [];
    liveFeatureGroupsCurrent = emptyLiveFeatureGroups();
    return;
  }

  if (liveFeatureMapCurrent.length !== target.length) {
    liveFeatureMapCurrent = target.map((point) => ({ ...point }));
    liveFeatureGroupsCurrent = groupLiveFeaturePoints(liveFeatureMapCurrent);
    return;
  }

  for (let i = 0; i < target.length; i += 1) {
    const current = liveFeatureMapCurrent[i];
    const next = target[i];
    current.feature = next.feature;
    current.x = lerp(current.x, next.x, 0.38);
    current.y = lerp(current.y, next.y, 0.38);
    current.z = lerp(current.z, next.z, 0.38);
  }

  liveFeatureGroupsCurrent = groupLiveFeaturePoints(liveFeatureMapCurrent);
}

function drawLiveFeatureOverlay(t) {
  if (!liveFeatureOverlayMode || !expressionDrivenMode || !liveFeatureMapCurrent.length) {
    return;
  }

  for (let i = 0; i < liveFeatureMapCurrent.length; i += 1) {
    const point = liveFeatureMapCurrent[i];
    const rendered = mapLiveFeaturePoint(point);
    const visual = liveFeatureVisual(point.feature);
    const flicker = 0.82 + 0.18 * noise(i * 0.13, t * 3.1);
    const alpha = visual.alpha * hologramConfig.brightness * emergence * flicker;

    stroke(visual.glow[0], visual.glow[1], visual.glow[2], alpha * 0.34);
    strokeWeight(visual.size * (1.2 + hologramConfig.particleSize * 1.35));
    point(rendered.x, rendered.y, rendered.z);

    stroke(visual.core[0], visual.core[1], visual.core[2], alpha);
    strokeWeight(visual.size * (0.72 + hologramConfig.particleSize * 1.05));
    point(rendered.x, rendered.y, rendered.z);
  }
}

function mapHairShellPoint(point) {
  const x = (point.x - 0.5) * 560;
  const y = (0.5 - point.y) * 640 + FACE_SHIFT_Y - 10;
  const crownLift = clamp01(0.5 - point.y) * 42;
  return createVector(
    x,
    y,
    point.z * 165 + crownLift,
  );
}

function mapLiveFeaturePoint(point) {
  const x = (point.x - 0.5) * 450;
  const y = (0.5 - point.y) * 540 + FACE_SHIFT_Y + 18;
  const depthBias = point.feature === "nose"
    ? 26
    : point.feature === "mouth"
      ? 16
      : point.feature === "leftEye" || point.feature === "rightEye"
        ? 12
        : 4;

  return createVector(
    x,
    y,
    point.z + depthBias,
  );
}

function liveFeatureVisual(feature) {
  switch (feature) {
    case "mouth":
      return { core: [245, 252, 255], glow: shiftedColor([160, 235, 255]), size: 1.7, alpha: 220 };
    case "leftEye":
    case "rightEye":
      return { core: [228, 250, 255], glow: shiftedColor([120, 220, 255]), size: 1.5, alpha: 205 };
    case "leftBrow":
    case "rightBrow":
      return { core: [220, 246, 255], glow: shiftedColor([135, 210, 255]), size: 1.25, alpha: 168 };
    case "nose":
      return { core: [230, 250, 255], glow: shiftedColor([145, 228, 255]), size: 1.35, alpha: 188 };
    case "jaw":
      return { core: [214, 238, 255], glow: shiftedColor([120, 200, 245]), size: 1.05, alpha: 112 };
    default:
      return { core: [220, 245, 255], glow: shiftedColor([130, 220, 255]), size: 1.1, alpha: 144 };
  }
}

function buildExpressionRegions(position, bounds, feature) {
  const rangeX = max(1, bounds.maxX - bounds.minX);
  const rangeY = max(1, bounds.maxY - bounds.minY);
  const u = constrain((position.x - bounds.minX) / rangeX, 0, 1);
  const v = constrain((position.y - bounds.minY) / rangeY, 0, 1);
  const centered = 1 - min(1, abs(u - 0.5) * 2);
  const side = u < 0.5 ? -1 : 1;

  const mouth = softBand(v, 0.56, 0.8, 0.1) * softBand(centered, 0.08, 1, 0.18);
  const smile = mouth * Math.pow(min(1, abs(u - 0.5) * 2), 0.9);
  const leftEye = softBand(v, 0.24, 0.42, 0.08) * softBand(u, 0.08, 0.42, 0.08);
  const rightEye = softBand(v, 0.24, 0.42, 0.08) * softBand(u, 0.58, 0.92, 0.08);
  const brows = softBand(v, 0.14, 0.29, 0.07) * softBand(centered, 0.1, 1, 0.18);
  const nose = softBand(v, 0.28, 0.66, 0.08) * softBand(centered, 0.34, 1, 0.18);
  const jaw = softBand(v, 0.74, 1.0, 0.08) * softBand(centered, 0.04, 1, 0.12);

  const featureBoost = {
    mouth: feature === 3 ? 0.65 : 0,
    eyes: feature === 1 ? 0.75 : 0,
    brows: feature === 5 ? 0.6 : 0,
    jaw: feature === 4 ? 0.45 : 0,
  };

  return {
    mouth: constrain(mouth + featureBoost.mouth, 0, 1),
    smile: constrain(smile + featureBoost.mouth * 0.35, 0, 1),
    mouthSplit: constrain((v - 0.675) / 0.12, -1, 1),
    leftEye: constrain(leftEye + featureBoost.eyes, 0, 1),
    rightEye: constrain(rightEye + featureBoost.eyes, 0, 1),
    eyeSplit: constrain((v - 0.335) / 0.08, -1, 1),
    brows: constrain(brows + featureBoost.brows, 0, 1),
    nose: constrain(nose + (feature === 2 ? 0.68 : 0), 0, 1),
    jaw: constrain(jaw + featureBoost.jaw, 0, 1),
    side,
  };
}

function emptyLiveFeatureGroups() {
  return {
    mouth: [],
    leftEye: [],
    rightEye: [],
    leftBrow: [],
    rightBrow: [],
    nose: [],
    jaw: [],
  };
}

function zeroRegions() {
  return {
    mouth: 0,
    smile: 0,
    mouthSplit: 0,
    leftEye: 0,
    rightEye: 0,
    eyeSplit: 0,
    brows: 0,
    nose: 0,
    jaw: 0,
    side: 0,
  };
}

function groupLiveFeaturePoints(points) {
  const groups = emptyLiveFeatureGroups();
  for (const point of points) {
    const mapped = mapLiveFeaturePoint(point);
    if (groups[point.feature]) {
      groups[point.feature].push(mapped);
    }
  }
  return groups;
}

function softBand(value, start, end, feather = 0.08) {
  if (value <= start - feather || value >= end + feather) {
    return 0;
  }
  if (value < start) {
    return (value - (start - feather)) / feather;
  }
  if (value > end) {
    return 1 - (value - end) / feather;
  }
  return 1;
}

function shiftedColor(base) {
  const [h, s, v] = rgbToHsv(base[0], base[1], base[2]);
  const hueShift = (hologramConfig.hue - 0.5) * 1.35;
  const shiftedHue = (h + hueShift + 1) % 1;
  const shiftedSaturation = constrain(s * 0.92 + abs(hologramConfig.hue - 0.5) * 0.16, 0.18, 1);
  const shiftedValue = constrain(v * 0.98 + 0.02, 0, 1);
  return hsvToRgb(shiftedHue, shiftedSaturation, shiftedValue);
}

function rgbToHsv(r, g, b) {
  const nr = r / 255;
  const ng = g / 255;
  const nb = b / 255;
  const maxValue = max(nr, ng, nb);
  const minValue = min(nr, ng, nb);
  const delta = maxValue - minValue;

  let hue = 0;
  if (delta > 0) {
    if (maxValue === nr) {
      hue = ((ng - nb) / delta) % 6;
    } else if (maxValue === ng) {
      hue = (nb - nr) / delta + 2;
    } else {
      hue = (nr - ng) / delta + 4;
    }
    hue /= 6;
    if (hue < 0) {
      hue += 1;
    }
  }

  const saturation = maxValue === 0 ? 0 : delta / maxValue;
  return [hue, saturation, maxValue];
}

function hsvToRgb(h, s, v) {
  const sector = floor(h * 6);
  const fraction = h * 6 - sector;
  const p = v * (1 - s);
  const q = v * (1 - fraction * s);
  const t = v * (1 - (1 - fraction) * s);

  switch (sector % 6) {
    case 0:
      return [v * 255, t * 255, p * 255];
    case 1:
      return [q * 255, v * 255, p * 255];
    case 2:
      return [p * 255, v * 255, t * 255];
    case 3:
      return [p * 255, q * 255, v * 255];
    case 4:
      return [t * 255, p * 255, v * 255];
    default:
      return [v * 255, p * 255, q * 255];
  }
}

function randomEmitterPosition() {
  const angle = random(TWO_PI);
  const radius = random(10, 124);
  return createVector(cos(angle) * radius, projectorY() + random(-8, 18), sin(angle) * radius);
}

function projectorY() {
  return min(height * 0.34, EMITTER_Y);
}

function stageFocusX() {
  if (!stageElement || !controlDeckElement) {
    return 0;
  }

  const stageRect = stageElement.getBoundingClientRect();
  const deckRect = controlDeckElement.getBoundingClientRect();
  const overlapsY = deckRect.bottom > stageRect.top && deckRect.top < stageRect.bottom;
  if (!overlapsY) {
    return 0;
  }

  const coveredWidth = max(0, min(stageRect.right, deckRect.right) - stageRect.left);
  return coveredWidth * 0.5;
}

function drawProjectorBase(t) {
  const y = height * 0.79;
  const x = width * 0.5 + stageFocusX();
  const color = shiftedColor([130, 238, 255]);
  const glow = shiftedColor([215, 250, 255]);
  const ringPulse = 1 + sin(t * 1.18) * 0.012;
  const outerW = min(width * 0.48, 720);
  const outerH = outerW * 0.24;
  const midW = outerW * 0.78;
  const midH = outerH * 0.76;
  const innerW = outerW * 0.54;
  const innerH = outerH * 0.58;
  const coreW = outerW * 0.22;
  const coreH = outerH * 0.22;

  push();
  resetMatrix();
  translate(-width * 0.5, -height * 0.5);

  noStroke();
  fill(color[0], color[1], color[2], 9 * hologramConfig.brightness);
  ellipse(x, y + 4, outerW * 1.1, outerH * 1.35);
  fill(color[0], color[1], color[2], 16 * hologramConfig.brightness);
  ellipse(x, y, outerW * 0.92, outerH * 1.05);
  fill(color[0], color[1], color[2], 18 * hologramConfig.brightness);
  ellipse(x, y, midW * 0.95, midH * 0.98);
  fill(glow[0], glow[1], glow[2], 24 * hologramConfig.brightness);
  ellipse(x, y, innerW * 0.88, innerH * 0.82);
  fill(220, 252, 255, 48 * hologramConfig.brightness);
  ellipse(x, y, coreW * 1.2, coreH * 1.15);

  noFill();
  stroke(color[0], color[1], color[2], 16 * hologramConfig.brightness);
  strokeWeight(18);
  ellipse(x, y, outerW, outerH);
  stroke(color[0], color[1], color[2], 58 * hologramConfig.brightness);
  strokeWeight(2.6);
  ellipse(x, y, outerW * ringPulse, outerH * ringPulse);
  stroke(glow[0], glow[1], glow[2], 92 * hologramConfig.brightness);
  strokeWeight(1.8);
  ellipse(x, y, midW, midH);
  ellipse(x, y, innerW * ringPulse, innerH * ringPulse);
  ellipse(x, y, innerW * 0.68, innerH * 0.68);
  ellipse(x, y, coreW, coreH);

  stroke(220, 252, 255, 74 * hologramConfig.brightness);
  strokeWeight(1.2);
  arc(x, y, outerW * 0.98, outerH * 0.98, PI + 0.2, TWO_PI - 0.2);
  arc(x, y, midW * 0.98, midH * 0.98, PI + 0.18, TWO_PI - 0.18);
  arc(x, y, innerW * 0.98, innerH * 0.98, PI + 0.14, TWO_PI - 0.14);

  const tickAngles = [-2.62, -2.18, -1.74, -1.3, -0.86, -0.42, 0.42, 0.86, 1.3, 1.74, 2.18, 2.62];
  for (const angle of tickAngles) {
    const innerA = (midW * 0.5) * cos(angle);
    const innerB = (midH * 0.5) * sin(angle);
    const outerA = (outerW * 0.5) * cos(angle);
    const outerB = (outerH * 0.5) * sin(angle);
    stroke(glow[0], glow[1], glow[2], 42 * hologramConfig.brightness);
    strokeWeight(1.3);
    line(x + innerA, y + innerB, x + outerA, y + outerB);
  }

  const beaconOffsets = [-outerW * 0.34, -outerW * 0.18, outerW * 0.18, outerW * 0.34];
  for (const offset of beaconOffsets) {
    stroke(glow[0], glow[1], glow[2], 38 * hologramConfig.brightness);
    strokeWeight(1.05);
    line(x + offset, y - outerH * 0.16, x + offset, y - outerH * 1.18);
    stroke(220, 252, 255, 88 * hologramConfig.brightness);
    strokeWeight(2.8);
    point(x + offset, y - outerH * 1.21);
  }
  pop();

  if (!particles.length) {
    return;
  }

  push();
  const pulse = 14 + sin(t * 1.65) * 2;
  stroke(color[0], color[1], color[2], 16 * hologramConfig.brightness);
  strokeWeight(3.2);
  line(stageFocusX(), projectorY() - 8, stageFocusX(), projectorY() - 50 - pulse);
  pop();
}

function drawStarfield(t) {
  strokeWeight(1);
  for (const star of starfield) {
    const alpha = 20 + 28 * noise(star.pulse, t * 0.24);
    stroke(85, 220, 255, alpha);
    point(star.x, star.y, star.z);
  }
}

function smoothstep(x) {
  return x * x * (3 - 2 * x);
}
