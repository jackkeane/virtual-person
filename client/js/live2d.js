/**
 * Live2D avatar renderer with placeholder fallback.
 *
 * Uses pixi-live2d-display (loaded via CDN in index.html) for real Live2D rendering.
 * Falls back to a canvas-drawn placeholder face when SDK or model is unavailable.
 */

let runtime = null;

const NEUTRAL_MOUTH = { open: 0.05, form: 0.5 };
const DEFAULT_EXPRESSION = {
  eyes: 0.7,
  brows: 0.5,
  mouth_open: NEUTRAL_MOUTH.open,
  mouth_form: NEUTRAL_MOUTH.form,
  head: 0,
};

const DEFAULT_BLEND_CONFIG = {
  smoothing: {
    expressionLambda: 7.0,
    mouthLambda: 20.0,
    speechActivityLambda: 14.0,
    bodyLambda: 8.0,
  },
  speech: {
    neutralOpen: NEUTRAL_MOUTH.open,
    neutralForm: NEUTRAL_MOUTH.form,
  },
  microMotion: {
    enabled: true,
    freqHz: 0.22,
    mouthOpenAmp: 0.012,
    headAmp: 0.015,
  },
  bodyMotion: {
    idleAmpX: 2.4,
    idleAmpY: 0.8,
    speakAmpX: 2.4,
    speakAmpY: 0.8,
    breathBase: 0.35,
    breathAmp: 0.25,
    maxX: 10,
    maxY: 4,
  },
};

// --------------- Utility ---------------

function clamp01(v, fallback = 0) {
  return Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : fallback;
}

function clamp(v, min, max, fallback = min) {
  if (!Number.isFinite(v)) return fallback;
  return Math.max(min, Math.min(max, v));
}

function smoothExp(current, target, lambda, dtSec) {
  if (!Number.isFinite(current) || !Number.isFinite(target)) return target;
  if (!Number.isFinite(lambda) || lambda <= 0 || dtSec <= 0) return target;
  const alpha = 1 - Math.exp(-lambda * dtSec);
  return current + (target - current) * alpha;
}

function lerp(a, b, t) { return a + (b - a) * t; }

function mergeConfig(base, patch = {}) {
  return {
    ...base,
    ...patch,
    smoothing: { ...base.smoothing, ...(patch.smoothing || {}) },
    speech: { ...base.speech, ...(patch.speech || {}) },
    microMotion: { ...base.microMotion, ...(patch.microMotion || {}) },
    bodyMotion: { ...base.bodyMotion, ...(patch.bodyMotion || {}) },
  };
}

// --------------- Blend Controller ---------------

function createBlendController(configPatch = {}, onApply) {
  const cfg = mergeConfig(DEFAULT_BLEND_CONFIG, configPatch);
  const state = {
    lastTickMs: 0,
    microPhase: 0,
    bodyScale: 1.0,
    expression: {
      current: { ...DEFAULT_EXPRESSION },
      target: { ...DEFAULT_EXPRESSION },
    },
    speech: {
      current: { mouth_open: cfg.speech.neutralOpen, mouth_form: cfg.speech.neutralForm },
      target: { mouth_open: cfg.speech.neutralOpen, mouth_form: cfg.speech.neutralForm },
      activeCurrent: 0,
      activeTarget: 0,
    },
    body: {
      angleX: 0,
      angleY: 0,
      breath: cfg.bodyMotion.breathBase,
      phase: 0,
    },
  };

  return {
    setExpressionTarget(params = {}) {
      const next = { ...state.expression.target };
      if (params.eyes !== undefined) next.eyes = clamp01(params.eyes, next.eyes);
      if (params.brows !== undefined) next.brows = clamp01(params.brows, next.brows);
      if (params.head !== undefined) next.head = clamp(params.head, -1, 1, next.head);
      if (params.mouth !== undefined) next.mouth_form = clamp01(params.mouth, next.mouth_form);
      if (params.mouth_form !== undefined) next.mouth_form = clamp01(params.mouth_form, next.mouth_form);
      if (params.mouth_open !== undefined) next.mouth_open = clamp01(params.mouth_open, next.mouth_open);
      state.expression.target = next;
    },

    setSpeechTarget(mouthOpen, mouthForm = 0.5) {
      state.speech.target.mouth_open = clamp01(mouthOpen, state.speech.target.mouth_open);
      state.speech.target.mouth_form = clamp01(mouthForm, state.speech.target.mouth_form);
    },

    setSpeaking(active) {
      state.speech.activeTarget = active ? 1 : 0;
      if (!active) {
        state.speech.target.mouth_open = cfg.speech.neutralOpen;
        state.speech.target.mouth_form = cfg.speech.neutralForm;
      }
    },

    setBodyMotionScale(scale = 1.0) {
      state.bodyScale = clamp(Number(scale), 0.4, 2.0, 1.0);
    },

    tick(nowMs = performance.now()) {
      if (!state.lastTickMs) state.lastTickMs = nowMs;
      const dtSec = Math.min(0.05, Math.max(0.001, (nowMs - state.lastTickMs) / 1000));
      state.lastTickMs = nowMs;

      const { expressionLambda, mouthLambda, speechActivityLambda, bodyLambda } = cfg.smoothing;

      const ec = state.expression.current;
      const et = state.expression.target;
      ec.eyes = smoothExp(ec.eyes, et.eyes, expressionLambda, dtSec);
      ec.brows = smoothExp(ec.brows, et.brows, expressionLambda, dtSec);
      ec.head = smoothExp(ec.head, et.head, expressionLambda, dtSec);
      ec.mouth_open = smoothExp(ec.mouth_open, et.mouth_open, expressionLambda, dtSec);
      ec.mouth_form = smoothExp(ec.mouth_form, et.mouth_form, expressionLambda, dtSec);

      const sp = state.speech;
      sp.activeCurrent = smoothExp(sp.activeCurrent, sp.activeTarget, speechActivityLambda, dtSec);
      sp.current.mouth_open = smoothExp(sp.current.mouth_open, sp.target.mouth_open, mouthLambda, dtSec);
      sp.current.mouth_form = smoothExp(sp.current.mouth_form, sp.target.mouth_form, mouthLambda, dtSec);

      let microOpen = 0;
      let microHead = 0;
      if (cfg.microMotion.enabled) {
        state.microPhase += dtSec * cfg.microMotion.freqHz * Math.PI * 2;
        microOpen = Math.sin(state.microPhase) * cfg.microMotion.mouthOpenAmp;
        microHead = Math.sin(state.microPhase * 0.8 + 0.4) * cfg.microMotion.headAmp;
      }

      const sw = clamp01(sp.activeCurrent, 0);
      const blendedOpen = clamp01(lerp(ec.mouth_open, sp.current.mouth_open, sw) + microOpen, ec.mouth_open);
      const blendedForm = clamp01(lerp(ec.mouth_form, sp.current.mouth_form, sw), ec.mouth_form);

      const bm = cfg.bodyMotion;
      const mouthEnergy = clamp01((blendedOpen - cfg.speech.neutralOpen) / 0.55, 0);
      const speakEnergy = clamp01(sw * 0.65 + mouthEnergy * 0.35, 0);
      state.body.phase += dtSec * (cfg.microMotion.freqHz * 0.7 + sw * 0.45) * Math.PI * 2;

      const ampX = lerp(bm.idleAmpX, bm.speakAmpX, speakEnergy) * state.bodyScale;
      const ampY = lerp(bm.idleAmpY, bm.speakAmpY, speakEnergy) * state.bodyScale;
      const targetBodyX = clamp(Math.sin(state.body.phase + 0.1) * ampX, -bm.maxX, bm.maxX, 0);
      const targetBodyY = clamp(Math.sin(state.body.phase * 0.85 + 1.0) * ampY, -bm.maxY, bm.maxY, 0);
      const targetBreath = clamp(bm.breathBase + Math.sin(state.body.phase * 0.5 + 0.7) * bm.breathAmp * (0.45 + 0.55 * speakEnergy), 0, 1, bm.breathBase);

      state.body.angleX = smoothExp(state.body.angleX, targetBodyX, bodyLambda, dtSec);
      state.body.angleY = smoothExp(state.body.angleY, targetBodyY, bodyLambda, dtSec);
      state.body.breath = smoothExp(state.body.breath, targetBreath, bodyLambda * 0.75, dtSec);

      onApply?.({
        eyes: clamp01(ec.eyes, DEFAULT_EXPRESSION.eyes),
        brows: clamp01(ec.brows, DEFAULT_EXPRESSION.brows),
        head: clamp(ec.head + microHead, -1, 1, DEFAULT_EXPRESSION.head),
        mouth_open: blendedOpen,
        mouth_form: blendedForm,
        speech_weight: sw,
        body_angle_x: clamp(state.body.angleX, -bm.maxX, bm.maxX, 0),
        body_angle_y: clamp(state.body.angleY, -bm.maxY, bm.maxY, 0),
        breath: clamp01(state.body.breath, bm.breathBase),
      }, nowMs);
    },
  };
}

// --------------- Placeholder Renderer ---------------

function createPlaceholderRenderer(canvasEl, options = {}) {
  const ctx = canvasEl.getContext('2d');
  const state = {
    mouthOpen: NEUTRAL_MOUTH.open,
    mouthForm: NEUTRAL_MOUTH.form,
    expression: { eyes: 0.7, brows: 0.5, mouth: 0.5, head: 0 },
  };

  const controller = createBlendController(options.blendConfig, (out) => {
    state.mouthOpen = out.mouth_open;
    state.mouthForm = out.mouth_form;
    state.expression.eyes = out.eyes;
    state.expression.brows = out.brows;
    state.expression.head = out.head;
    state.expression.mouth = out.mouth_form;
  });

  return {
    kind: 'placeholder',
    setExpression(params = {}) { controller.setExpressionTarget(params); },
    setMouth(mouthOpen, mouthForm) { controller.setSpeechTarget(mouthOpen, mouthForm); },
    setSpeaking(active) { controller.setSpeaking(active); },
    setBodyMotionScale(scale = 1.0) { controller.setBodyMotionScale(scale); },
    setSpeakingMotion(_opts = {}) { /* placeholder no-op */ },
    tick(t = performance.now()) {
      controller.tick(t);
      const w = canvasEl.width;
      const h = canvasEl.height;
      ctx.clearRect(0, 0, w, h);
      const cx = w / 2 + state.expression.head * 12 + Math.sin(t / 1000) * 4;
      const cy = h / 2;

      // Head
      ctx.fillStyle = '#fcd34d';
      ctx.beginPath();
      ctx.arc(cx, cy, 85, 0, Math.PI * 2);
      ctx.fill();

      // Eyes
      ctx.fillStyle = '#111827';
      const eyeOpen = 6 + state.expression.eyes * 10;
      ctx.fillRect(cx - 35, cy - 20, 10, eyeOpen);
      ctx.fillRect(cx + 25, cy - 20, 10, eyeOpen);

      // Mouth
      ctx.strokeStyle = '#1f2937';
      ctx.lineWidth = 4;
      ctx.beginPath();
      const mouthCurve = 0.35 + (state.mouthForm - 0.5) * 0.3;
      const start = Math.max(0.05, mouthCurve - 0.25) * Math.PI;
      const end = Math.min(0.95, mouthCurve + 0.25) * Math.PI;
      const mw = 20 + state.mouthOpen * 35;
      ctx.arc(cx, cy + 30, mw, start, end);
      ctx.stroke();
    },
  };
}

// --------------- Live2D Renderer (pixi-live2d-display) ---------------

async function createLive2DRenderer(canvasEl, options = {}) {
  const PIXI = globalThis.PIXI;
  // pixi-live2d-display registers Live2DModel on PIXI.live2d
  const Live2DModel = PIXI?.live2d?.Live2DModel;

  if (!PIXI) throw new Error('PixiJS not loaded');
  if (!Live2DModel) throw new Error('pixi-live2d-display not loaded (PIXI.live2d.Live2DModel missing)');

  // Check Cubism Core
  if (!globalThis.Live2DCubismCore) {
    throw new Error('Live2D Cubism Core not loaded (live2dcubismcore.min.js missing)');
  }

  const modelPath = options.modelUrl || '/client/assets/models/Hiyori/Hiyori.model3.json';

  // Create PIXI app on the canvas (v6 API)
  const pixiApp = new PIXI.Application({
    view: canvasEl,
    width: canvasEl.width,
    height: canvasEl.height,
    transparent: true,
    antialias: true,
    autoStart: true,
  });

  console.log('[Ani] Loading Live2D model from:', modelPath);

  // Load the model
  const model = await Live2DModel.from(modelPath, { autoInteract: false });

  console.log('[Ani] Model loaded, size:', model.width, 'x', model.height);

  // Scale and position the model to fit the canvas
  const scale = Math.min(canvasEl.width / model.width, canvasEl.height / model.height) * 0.7;
  model.scale.set(scale);
  model.anchor.set(0.5, 0.5);
  model.x = canvasEl.width / 2;
  model.y = canvasEl.height / 2;
  pixiApp.stage.addChild(model);

  // We need to inject our parameter values AFTER the internal model's
  // motion/physics/pose pipeline runs but BEFORE coreModel.update() renders.
  // Monkey-patch coreModel.update to achieve this.
  const coreModel = model.internalModel?.coreModel;
  if (!coreModel) throw new Error('Cannot access coreModel');

  // Build a param name → index map from the low-level model for reliable access.
  // coreModel.setParameterValueById expects CubismIdHandle, not strings — so we
  // write directly to the parameter values array by index instead.
  const paramMap = {};
  try {
    // pixi-live2d-display Cubism4: coreModel._model is the low-level CubismMoc model
    const lowModel = coreModel._model;
    const ids = lowModel?.parameters?.ids;
    if (ids) {
      for (let i = 0; i < ids.length; i++) {
        paramMap[ids[i]] = i;
      }
      console.log(`[Ani] Mapped ${Object.keys(paramMap).length} Live2D parameters`);
      console.log('[Ani] Mouth params:', 'ParamMouthOpenY' in paramMap, 'ParamMouthForm' in paramMap);
    }
  } catch (e) {
    console.warn('[Ani] Failed to build param index map:', e);
  }

  // Direct parameter write by index — guaranteed to work
  function setParam(id, value) {
    const idx = paramMap[id];
    if (idx === undefined) return;
    try {
      const vals = coreModel._model.parameters.values;
      if (vals) vals[idx] = value;
    } catch (_) {}
  }

  function getParam(id) {
    const idx = paramMap[id];
    if (idx === undefined) return undefined;
    try {
      const vals = coreModel._model.parameters.values;
      return vals ? vals[idx] : undefined;
    } catch (_) {
      return undefined;
    }
  }

  // Store the latest blend output to apply each frame
  let latestOut = null;
  let latestNowMs = 0;

  // Speaking motion trigger (separate from lipsync/body params)
  const speakingMotion = {
    enabled: false,
    group: 'TapBody',
    index: 0,
    cooldownMs: 2500,
    lastPlayedMs: 0,
    isSpeaking: false,
    armLockGraceMs: 2200,
    armLockUntilMs: 0,
  };

  function triggerSpeakingMotion(force = false) {
    if (!speakingMotion.enabled) return;
    const now = performance.now();
    if (!force && now - speakingMotion.lastPlayedMs < speakingMotion.cooldownMs) return;
    try {
      if (typeof model.motion === 'function') {
        model.motion(speakingMotion.group, speakingMotion.index);
        speakingMotion.lastPlayedMs = now;
      }
    } catch (_) {
      // Best-effort only; keep avatar stable even if motion API differs.
    }
  }

  // Cursor-aware gaze override (falls back to random wander when cursor is far).
  const gaze = {
    hasPointer: false,
    pointerX: canvasEl.width / 2,
    pointerY: canvasEl.height / 2,
    currentX: 0,
    currentY: 0,
  };

  canvasEl.addEventListener('pointermove', (evt) => {
    const rect = canvasEl.getBoundingClientRect();
    gaze.pointerX = evt.clientX - rect.left;
    gaze.pointerY = evt.clientY - rect.top;
    gaze.hasPointer = true;
  });

  const armParamIds = [
    'ParamArmLA', 'ParamArmRA', 'ParamArmLB', 'ParamArmRB',
    'ParamHandL', 'ParamHandR', 'ParamHandLB', 'ParamHandRB', 'ParamShoulder',
  ];
  const armNeutral = {};
  for (const id of armParamIds) {
    const v = getParam(id);
    if (Number.isFinite(v)) armNeutral[id] = v;
  }

  function setArmsNeutral() {
    for (const id of armParamIds) {
      const v = armNeutral[id];
      if (Number.isFinite(v)) setParam(id, v);
    }
  }

  const originalUpdate = coreModel.update.bind(coreModel);
  coreModel.update = function () {
    let shouldLockArms = false;
    if (latestOut) {
      const out = latestOut;
      const nowMs = latestNowMs;

      // Mouth — ParamMouthOpenY range [0,1], ParamMouthForm range [-1,1]
      setParam('ParamMouthOpenY', out.mouth_open);
      setParam('ParamMouthForm', out.mouth_form * 2 - 1);

      // Eyes [0,1]
      setParam('ParamEyeLOpen', out.eyes);
      setParam('ParamEyeROpen', out.eyes);

      // Brows [-1,1]
      const browVal = out.brows * 2 - 1;
      setParam('ParamBrowLY', browVal);
      setParam('ParamBrowRY', browVal);

      // Head angle [-30,30] degrees
      setParam('ParamAngleX', out.head * 30);

      // Body motion synced with speaking (if params exist on model)
      setParam('ParamBodyAngleX', out.body_angle_x ?? 0);
      setParam('ParamBodyAngleY', out.body_angle_y ?? 0);
      setParam('ParamBreath', out.breath ?? 0.35);

      // Eye gaze: track cursor when near avatar center; otherwise keep idle wander.
      const centerX = canvasEl.width / 2;
      const centerY = canvasEl.height / 2;
      const dx = gaze.pointerX - centerX;
      const dy = gaze.pointerY - centerY;
      const dist = Math.hypot(dx, dy);
      const trackCursor = gaze.hasPointer && dist <= 200;

      const targetEyeX = trackCursor
        ? clamp(dx / 200, -1, 1, 0)
        : Math.sin(nowMs / 4500 + 1.3) * 0.4;
      const targetEyeY = trackCursor
        ? clamp(dy / 200, -1, 1, 0)
        : Math.sin(nowMs / 5200 + 0.7) * 0.3;

      const dtSec = 1 / 60;
      gaze.currentX = smoothExp(gaze.currentX, targetEyeX, 4, dtSec);
      gaze.currentY = smoothExp(gaze.currentY, targetEyeY, 4, dtSec);

      setParam('ParamEyeBallX', gaze.currentX);
      setParam('ParamEyeBallY', gaze.currentY);

      // If speaking-motion preset is disabled, lock arms while speaking and keep a
      // longer grace window to avoid sentence-boundary pops between audio chunks.
      const nowMsPerf = performance.now();
      shouldLockArms = !speakingMotion.enabled && (
        speakingMotion.isSpeaking || nowMsPerf < speakingMotion.armLockUntilMs
      );
      if (shouldLockArms) setArmsNeutral();
    }
    originalUpdate();
    if (shouldLockArms) {
      // Enforce after internal motion/physics updates to prevent sporadic arm raises.
      setArmsNeutral();
    }
  };

  // Build blend controller — stores output for the patched update() to apply
  const controller = createBlendController(options.blendConfig, (out, nowMs) => {
    latestOut = out;
    latestNowMs = nowMs;
  });

  console.log('[Ani] Live2D renderer ready');

  return {
    kind: 'live2d',
    model,
    pixiApp,
    setExpression(params = {}) { controller.setExpressionTarget(params); },
    setMouth(mouthOpen, mouthForm = 0.5) { controller.setSpeechTarget(mouthOpen, mouthForm); },
    setSpeaking(active) {
      const next = !!active;
      controller.setSpeaking(next);
      if (next && !speakingMotion.isSpeaking) {
        // Rising edge: start a gesture when speech begins.
        triggerSpeakingMotion(true);
      }
      if (!next && speakingMotion.isSpeaking) {
        // Keep arm lock briefly after speaking ends to prevent end-of-utterance pop.
        speakingMotion.armLockUntilMs = performance.now() + speakingMotion.armLockGraceMs;
      }
      speakingMotion.isSpeaking = next;
    },
    setBodyMotionScale(scale = 1.0) { controller.setBodyMotionScale(scale); },
    setSpeakingMotion({ enabled, group, index, cooldownMs } = {}) {
      if (typeof enabled === 'boolean') speakingMotion.enabled = enabled;
      if (typeof group === 'string' && group) speakingMotion.group = group;
      if (Number.isInteger(index) && index >= 0) speakingMotion.index = index;
      if (Number.isFinite(cooldownMs) && cooldownMs > 200) speakingMotion.cooldownMs = cooldownMs;
    },
    tick(t = performance.now()) {
      controller.tick(t);
      if (speakingMotion.isSpeaking) {
        // Long utterances: occasionally refresh gesture.
        triggerSpeakingMotion(false);
      }
    },
  };
}

// --------------- Public API ---------------

export async function initLive2D(canvasEl, options = {}) {
  if (!canvasEl) throw new Error('initLive2D requires a canvas element');

  // Try Live2D first, fall back to placeholder
  if (options.live2dEnabled !== false) {
    try {
      runtime = await createLive2DRenderer(canvasEl, options);
      console.log('[Ani] Live2D renderer loaded:', runtime.kind);
      return { backend: runtime.kind };
    } catch (err) {
      console.warn('[Ani] Live2D init failed, using placeholder:', err.message);
    }
  }

  runtime = createPlaceholderRenderer(canvasEl, options);
  console.log('[Ani] Using placeholder renderer');
  return { backend: runtime.kind };
}

export function setExpression(params = {}) {
  runtime?.setExpression(params);
}

export function setMouth(mouthOpen, mouthForm = 0.5) {
  runtime?.setMouth(mouthOpen, mouthForm);
}

export function setSpeaking(active) {
  runtime?.setSpeaking(!!active);
}

export function setBodyMotionScale(scale = 1.0) {
  runtime?.setBodyMotionScale?.(scale);
}

export function setSpeakingMotion(opts = {}) {
  runtime?.setSpeakingMotion?.(opts);
}

export function tick() {
  runtime?.tick();
}
