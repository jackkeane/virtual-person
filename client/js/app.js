import { WSClient } from './ws.js';
import { addMessage, appendAssistantPartial, finalizeAssistantMessage, resetAssistantDraft, showThinking, removeThinking } from './chat.js';
import {
  playAudioBase64,
  startMicRecording,
  stopMicRecording,
  startAlwaysOnMic,
  stopAlwaysOnMic,
  isRecording,
  isAlwaysOnActive,
} from './audio.js';
import { initLive2D, setExpression, setMouth, setSpeaking, setBodyMotionScale, setSpeakingMotion, tick } from './live2d.js';
import { UI_CONFIG } from './ui-config.js';

const canvas = document.getElementById('avatarCanvas');
const stateEl = document.getElementById('state-indicator');

const speaking = {
  stateFlag: false,
  audioFlag: false,
};

let latestStateText = 'idle';
let latestStatusText = '';

function renderStateIndicator() {
  stateEl.textContent = latestStatusText || latestStateText || 'idle';
}

let lastTranscriptErrorAt = 0;
let autoMicGuardUntil = 0;
let autoTurnInFlight = false;
let autoTurnInFlightSince = 0;

const visemePlayer = {
  timeline: [],
  startedAtMs: 0,
  cursor: 0,
  fallbackPulseSeed: Math.random() * 1000,
};

const AVATAR_BLEND_TUNING = {
  // Optional runtime override for manual tuning from browser console:
  // window.__ANI_AVATAR_TUNING__ = { smoothing: { mouthLambda: 16 } }
  smoothing: {
    expressionLambda: 7.0,
    mouthLambda: 20.0,
    speechActivityLambda: 14.0,
  },
  microMotion: {
    enabled: true,
    freqHz: 0.22,
    mouthOpenAmp: 0.012,
    headAmp: 0.015,
  },
};

async function loadAvatarConfig() {
  try {
    const r = await fetch('/avatar/config');
    if (!r.ok) throw new Error('avatar config unavailable');
    const data = await r.json();
    return {
      live2dEnabled: data.live2d_enabled,
      modelPath: data.avatar_model_path,
      blendConfig: { ...AVATAR_BLEND_TUNING, ...(globalThis.__ANI_AVATAR_TUNING__ || {}) },
    };
  } catch (_err) {
    return {
      live2dEnabled: false,
      modelPath: '',
      blendConfig: { ...AVATAR_BLEND_TUNING, ...(globalThis.__ANI_AVATAR_TUNING__ || {}) },
    };
  }
}

function syncSpeakingLayer() {
  setSpeaking(isSpeakingNow());
}

function startVisemeTimeline(timeline = []) {
  visemePlayer.timeline = Array.isArray(timeline)
    ? [...timeline].sort((a, b) => (a.time_ms || 0) - (b.time_ms || 0))
    : [];
  visemePlayer.startedAtMs = performance.now();
  visemePlayer.cursor = 0;
  if (visemePlayer.timeline.length) {
    setSpeaking(true);
  }
}

function isSpeakingNow() {
  return speaking.stateFlag || speaking.audioFlag;
}

function updateVisemePlayback(nowMs) {
  if (visemePlayer.timeline.length) {
    const elapsed = nowMs - visemePlayer.startedAtMs;
    while (visemePlayer.cursor < visemePlayer.timeline.length) {
      const frame = visemePlayer.timeline[visemePlayer.cursor];
      if ((frame.time_ms || 0) > elapsed) break;
      setMouth(frame.mouth_open ?? 0.05, frame.mouth_form ?? 0.5);
      visemePlayer.cursor += 1;
    }

    if (visemePlayer.cursor >= visemePlayer.timeline.length && !isSpeakingNow()) {
      visemePlayer.timeline = [];
      setMouth(0.05, 0.5);
      syncSpeakingLayer();
    }
  }

  // Keep visible speaking motion when audio continues after timeline frames are exhausted.
  if (isSpeakingNow() && visemePlayer.cursor >= visemePlayer.timeline.length) {
    const pulse = 0.11 + Math.abs(Math.sin((nowMs + visemePlayer.fallbackPulseSeed) / 85)) * 0.16;
    setMouth(pulse, 0.5);
  }
}

const audioQueue = [];
let audioPlaying = false;

let _audioChunkIndex = 0;

function playNextAudioChunk() {
  if (audioPlaying || audioQueue.length === 0) return;
  const item = audioQueue.shift();
  const idx = ++_audioChunkIndex;
  console.log(`[Ani][audio] playing chunk #${idx}, queue remaining: ${audioQueue.length}, b64len: ${item.audioPayload?.length}`);

  const { audio, cleanup } = playAudioBase64(item.audioPayload, item.mimeHint);
  if (!audio) {
    console.warn(`[Ani][audio] chunk #${idx} failed to create Audio element, skipping`);
    // Try next chunk
    setTimeout(playNextAudioChunk, 10);
    return;
  }

  audioPlaying = true;
  speaking.audioFlag = true;
  syncSpeakingLayer();

  if (item.timeline?.length) startVisemeTimeline(item.timeline);

  const finish = () => {
    console.log(`[Ani][audio] chunk #${idx} finished`);
    cleanup?.();
    audioPlaying = false;
    if (audioQueue.length > 0) {
      playNextAudioChunk();
      return;
    }
    speaking.audioFlag = false;
    // Guard auto-mic for a short window to avoid replay/echo self-trigger.
    autoMicGuardUntil = Date.now() + 1200;
    if (!isSpeakingNow() && visemePlayer.cursor >= visemePlayer.timeline.length) {
      setMouth(0.05, 0.5);
    }
    syncSpeakingLayer();
  };

  audio.onended = finish;
  audio.onerror = finish;
}

function bindAudioPlayback(audioPayload, mimeHint, timeline) {
  audioQueue.push({ audioPayload, mimeHint, timeline });
  playNextAudioChunk();
}

const avatarConfig = await loadAvatarConfig();
await initLive2D(canvas, avatarConfig);
syncSpeakingLayer();

const LS_BODY_MOTION_SCALE = 'ani.bodyMotionScale';
const LS_SPEAKING_MOTION_PRESET = 'ani.speakingMotionPreset';
const LS_EMOTION_MOTION_ENABLED = 'ani.emotionMotionEnabled';
const LS_MOTION_TUNING_PRESET = 'ani.motionTuningPreset';
const LS_ACTIVE_TAB = 'ani.activeTab';
const LS_STT_LANGUAGE = 'ani.sttLanguage';
let sttLanguage = localStorage.getItem(LS_STT_LANGUAGE) || 'auto';

let speakingMotionPreset = 'none';
let emotionMotionEnabled = true;

function applySpeakingMotionPresetValue(v) {
  speakingMotionPreset = v;
  if (v === 'none') {
    setSpeakingMotion({ enabled: false });
    return;
  }
  if (v === 'tapbody') {
    setSpeakingMotion({ enabled: true, group: 'TapBody', index: 0, cooldownMs: 2500 });
    return;
  }
  if (v === 'idle-1') {
    setSpeakingMotion({ enabled: true, group: 'Idle', index: 0, cooldownMs: 3000 });
    return;
  }
  if (v === 'idle-2') {
    setSpeakingMotion({ enabled: true, group: 'Idle', index: 1, cooldownMs: 3000 });
    return;
  }
  if (v === 'idle-3') {
    setSpeakingMotion({ enabled: true, group: 'Idle', index: 2, cooldownMs: 3000 });
    return;
  }
}

function applyEmotionMotion(category = '') {
  if (!emotionMotionEnabled) return;
  if (!speakingMotionPresetEl) return;

  const c = String(category || '').toLowerCase();
  let mapped = speakingMotionPreset;
  // Per-emotion mapping examples:
  // happy -> Idle m02, serious/sad -> Idle m03
  if (c.includes('happy') || c.includes('joy') || c.includes('excited')) mapped = 'idle-2';
  else if (c.includes('serious') || c.includes('sad') || c.includes('angry') || c.includes('fear')) mapped = 'idle-3';

  speakingMotionPresetEl.value = mapped;
  applySpeakingMotionPresetValue(mapped);
}

const bodyMotionScaleEl = document.getElementById('bodyMotionScale');
const bodyMotionScaleValueEl = document.getElementById('bodyMotionScaleValue');
if (bodyMotionScaleEl) {
  const savedScale = Number(localStorage.getItem(LS_BODY_MOTION_SCALE) || bodyMotionScaleEl.value || 1);
  bodyMotionScaleEl.value = Number.isFinite(savedScale) ? String(savedScale) : '1.0';

  const applyBodyScale = () => {
    const scale = Number(bodyMotionScaleEl.value || 1);
    setBodyMotionScale(scale);
    if (bodyMotionScaleValueEl) bodyMotionScaleValueEl.textContent = `${scale.toFixed(1)}x`;
    localStorage.setItem(LS_BODY_MOTION_SCALE, String(scale));
  };
  bodyMotionScaleEl.addEventListener('input', applyBodyScale);
  applyBodyScale();
}

const speakingMotionPresetEl = document.getElementById('speakingMotionPreset');
if (speakingMotionPresetEl) {
  const savedPreset = localStorage.getItem(LS_SPEAKING_MOTION_PRESET);
  if (savedPreset) speakingMotionPresetEl.value = savedPreset;
  const savedEmotionMotion = localStorage.getItem(LS_EMOTION_MOTION_ENABLED);
  if (savedEmotionMotion != null) emotionMotionEnabled = savedEmotionMotion !== 'false';

  const applySpeakingMotionPreset = () => {
    const v = speakingMotionPresetEl.value;
    applySpeakingMotionPresetValue(v);
    localStorage.setItem(LS_SPEAKING_MOTION_PRESET, v);
  };
  speakingMotionPresetEl.addEventListener('change', applySpeakingMotionPreset);
  applySpeakingMotionPreset();
}

const emotionMotionToggleEl = document.getElementById('emotionMotionToggle');
if (emotionMotionToggleEl) {
  emotionMotionToggleEl.checked = !!emotionMotionEnabled;
  emotionMotionToggleEl.addEventListener('change', () => {
    emotionMotionEnabled = !!emotionMotionToggleEl.checked;
    localStorage.setItem(LS_EMOTION_MOTION_ENABLED, String(emotionMotionEnabled));
  });
}

const motionTuningPresetEl = document.getElementById('motionTuningPreset');
if (motionTuningPresetEl && bodyMotionScaleEl) {
  const savedTuning = localStorage.getItem(LS_MOTION_TUNING_PRESET);
  if (savedTuning) motionTuningPresetEl.value = savedTuning;

  const applyMotionTuningPreset = () => {
    const v = motionTuningPresetEl.value;
    localStorage.setItem(LS_MOTION_TUNING_PRESET, v);
    if (v === 'subtle') bodyMotionScaleEl.value = '0.8';
    else if (v === 'expressive') bodyMotionScaleEl.value = '1.3';
    else bodyMotionScaleEl.value = '1.0';
    bodyMotionScaleEl.dispatchEvent(new Event('input'));
  };

  motionTuningPresetEl.addEventListener('change', applyMotionTuningPreset);
  applyMotionTuningPreset();
}

function stripPersonaPrefix(text) {
  // Defensively strip any leftover [Name|tone] prefix from responses
  return (text || '').replace(/^\[[\w]+\|[^\]]*\]\s*/, '');
}

console.log('[Ani] app.js loaded v3-day3');

const ws = new WSClient((evt) => {
  if (evt.type === 'chat_partial' && evt.role === 'assistant') {
    appendAssistantPartial(stripPersonaPrefix(evt.text || ''));
  }
  if (evt.type === 'chat' && evt.role === 'assistant') {
    finalizeAssistantMessage(stripPersonaPrefix(evt.text || ''));
    if (evt.final) {
      autoTurnInFlight = false;
      autoTurnInFlightSince = 0;
      autoMicGuardUntil = Date.now() + 1200;
    }
  } else if (evt.type === 'chat') {
    addMessage(evt.role, stripPersonaPrefix(evt.text));
  }
  if (evt.type === 'state') {
    latestStateText = evt.state || latestStateText;
    if (!latestStatusText) renderStateIndicator();
    if (evt.state === 'thinking') showThinking();
    else removeThinking();
    speaking.stateFlag = evt.state === 'speaking';
    if (!isSpeakingNow() && visemePlayer.cursor >= visemePlayer.timeline.length) {
      setMouth(0.05, 0.5);
    }
    syncSpeakingLayer();
  }
  if (evt.type === 'status') {
    latestStatusText = (evt.text || '').trim();
    renderStateIndicator();
  }
  if (evt.type === 'transcript') {
    if (evt.text) {
      resetAssistantDraft();
      addMessage('user', evt.text);
    }
    else if (evt.error) {
      autoTurnInFlight = false;
      autoTurnInFlightSince = 0;
      autoMicGuardUntil = Date.now() + 800;
      // In auto mode, noisy misses are expected occasionally; don't spam chat.
      if (micMode !== 'auto') {
        const now = Date.now();
        if (now - lastTranscriptErrorAt > 3000) {
          addMessage('system', evt.error);
          lastTranscriptErrorAt = now;
        }
      }
    }
  }
  if (evt.type === 'expression') {
    setExpression(evt.params || {});
    applyEmotionMotion(evt.category || '');
  }
  if (evt.type === 'audio' && evt.chunk) bindAudioPlayback(evt.chunk, evt.mime, evt.timeline || []);
  if (evt.type === 'viseme' && evt.timeline?.length) startVisemeTimeline(evt.timeline);
}, (status) => {
  if (status === 'connected') {
    latestStateText = 'idle';
    latestStatusText = '';
    renderStateIndicator();
    autoTurnInFlight = false;
    autoTurnInFlightSince = 0;
  }
  else if (status === 'disconnected') {
    latestStatusText = '🔴 disconnected';
    renderStateIndicator();
    autoTurnInFlight = false;
    autoTurnInFlightSince = 0;
  }
  else if (status === 'reconnecting') {
    latestStatusText = '🟡 reconnecting...';
    renderStateIndicator();
    autoTurnInFlight = false;
    autoTurnInFlightSince = 0;
  }
  else if (status === 'error') {
    latestStatusText = '🔴 error';
    renderStateIndicator();
    autoTurnInFlight = false;
    autoTurnInFlightSince = 0;
  }
});
ws.connect();

const input = document.getElementById('chatInput');

const tabBtnChat = document.getElementById('tabBtnChat');
const tabBtnAvatar = document.getElementById('tabBtnAvatar');
const tabChat = document.getElementById('tabChat');
const tabAvatar = document.getElementById('tabAvatar');

function setTab(which = 'chat') {
  const isChat = which === 'chat';
  tabBtnChat?.classList.toggle('active', isChat);
  tabBtnAvatar?.classList.toggle('active', !isChat);
  tabChat?.classList.toggle('active', isChat);
  tabAvatar?.classList.toggle('active', !isChat);
}

tabBtnChat?.addEventListener('click', () => {
  setTab('chat');
  localStorage.setItem(LS_ACTIVE_TAB, 'chat');
});
tabBtnAvatar?.addEventListener('click', () => {
  setTab('avatar');
  localStorage.setItem(LS_ACTIVE_TAB, 'avatar');
});
setTab(localStorage.getItem(LS_ACTIVE_TAB) || 'chat');

const sttLanguageSelectEl = document.getElementById('sttLanguageSelect');
if (sttLanguageSelectEl) {
  if (!['auto', 'zh-CN', 'en'].includes(sttLanguage)) sttLanguage = 'auto';
  sttLanguageSelectEl.value = sttLanguage;
  sttLanguageSelectEl.addEventListener('change', () => {
    sttLanguage = sttLanguageSelectEl.value || 'auto';
    localStorage.setItem(LS_STT_LANGUAGE, sttLanguage);
  });
}

const eraseMemoryBtn = document.getElementById('eraseMemoryBtn');
eraseMemoryBtn?.addEventListener('click', async () => {
  const ok = window.confirm('Erase ALL memories? This cannot be undone.');
  if (!ok) return;
  try {
    const res = await fetch('/memory/erase?confirm=true', { method: 'DELETE' });
    const data = await res.json();
    if (data.ok) addMessage('system', `Memory erased (${data.deleted ?? 0} items).`);
    else addMessage('system', data.error || 'Failed to erase memory');
  } catch (_err) {
    addMessage('system', 'Failed to erase memory');
  }
});

const QUICK_CHIPS = UI_CONFIG.quickChips || [];

const quickChipsEl = document.getElementById('quickChips');
if (quickChipsEl) {
  quickChipsEl.innerHTML = '';
  QUICK_CHIPS.forEach(({ label, text }) => {
    const btn = document.createElement('button');
    btn.className = 'chip';
    btn.type = 'button';
    btn.textContent = label;
    btn.dataset.text = text;
    btn.addEventListener('click', () => {
      input.value = text;
      sendChatMessage();
    });
    quickChipsEl.appendChild(btn);
  });
}

function sendChatMessage() {
  const text = input.value.trim();
  if (!text) return;
  resetAssistantDraft();
  addMessage('user', text);
  ws.send({ type: 'chat', text, user_id: 'web_user' });
  input.value = '';
}

document.getElementById('sendBtn').onclick = sendChatMessage;

// Enter key sends; Shift+Enter inserts newline
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage();
  }
});

const micBtn = document.getElementById('micBtn');
let micModeBtn = document.getElementById('micModeBtn');
let micMode = 'manual'; // 'manual' | 'auto'

// Backward-compatible: if HTML cache/old template doesn't include the mode button,
// create it dynamically so app logic doesn't crash.
if (!micModeBtn && micBtn?.parentElement) {
  micModeBtn = document.createElement('button');
  micModeBtn.id = 'micModeBtn';
  micModeBtn.textContent = 'Mode: Manual';
  micBtn.parentElement.insertBefore(micModeBtn, micBtn);
}

function renderMicUi() {
  if (micModeBtn) {
    micModeBtn.textContent = micMode === 'manual' ? 'Mode: Manual' : 'Mode: Auto (2s)';
  }
  if (!micBtn) return;

  if (micMode === 'manual') {
    micBtn.textContent = isRecording() ? '⏹️' : '🎤';
    micBtn.style.background = isRecording() ? '#dc2626' : '';
  } else {
    micBtn.textContent = isAlwaysOnActive() ? '🟢 Auto ON' : '🎙️ Auto OFF';
    micBtn.style.background = isAlwaysOnActive() ? '#16a34a' : '';
  }
}

if (micModeBtn) micModeBtn.onclick = async (e) => {
  e.preventDefault();
  // Stop current capture before switching mode
  if (isAlwaysOnActive()) stopAlwaysOnMic();
  else if (isRecording()) await stopMicRecording();
  autoTurnInFlight = false;
  autoTurnInFlightSince = 0;

  micMode = micMode === 'manual' ? 'auto' : 'manual';
  renderMicUi();
};

if (micBtn) micBtn.onclick = async (e) => {
  e.preventDefault();
  if (micMode === 'manual') {
    if (isRecording()) {
      micBtn.textContent = '🎤';
      micBtn.style.background = '';
      const { base64, mimeType } = await stopMicRecording();
      if (base64) {
        ws.send({ type: 'audio', chunk: base64, mime: mimeType, user_id: 'web_user', stt_language: sttLanguage });
      }
      return;
    }

    try {
      await startMicRecording();
      micBtn.textContent = '⏹️';
      micBtn.style.background = '#dc2626';
      ws.send({ type: 'mic_start' });
    } catch (err) {
      console.error('[Ani] Mic access denied:', err);
      addMessage('system', 'Microphone access denied');
    }
    return;
  }

  // Auto mode: one click toggles always-on listening.
  if (isAlwaysOnActive()) {
    stopAlwaysOnMic();
    autoTurnInFlight = false;
    renderMicUi();
    return;
  }

  try {
    await startAlwaysOnMic({
      silenceMs: 99999,
      minSpeechMs: 0,
      maxSegmentMs: 4000,
      rmsThreshold: 0,
      minBlobBytes: 500,
      isSpeaking: isSpeakingNow,
      onSegment: ({ base64, mimeType, speechMs = 0, segmentMs = 0 }) => {
        if (!base64) return;
        const speechRatio = segmentMs > 0 ? speechMs / segmentMs : 0;

        const now = Date.now();

        // During avatar playback or guard window, skip (echo contamination).
        if (isSpeakingNow()) return;
        if (now < autoMicGuardUntil) return;

        // In-flight watchdog: release lock if backend didn't reply for too long.
        if (autoTurnInFlight && now - autoTurnInFlightSince > 12000) {
          autoTurnInFlight = false;
          autoTurnInFlightSince = 0;
        }
        if (autoTurnInFlight) return;

        autoTurnInFlight = true;
        autoTurnInFlightSince = Date.now();
        console.log('[Ani][auto-mic] segment sent', { bytes: base64.length, mimeType, speechMs, segmentMs, speechRatio });
        ws.send({ type: 'audio', chunk: base64, mime: mimeType, user_id: 'web_user', stt_language: sttLanguage });
      },
    });
    ws.send({ type: 'mic_start' });
    autoMicGuardUntil = Date.now() + 500;
    renderMicUi();
  } catch (err) {
    console.error('[Ani] Auto mic access denied:', err);
    addMessage('system', 'Microphone access denied');
  }
};

renderMicUi();

function frame(now) {
  updateVisemePlayback(now);
  tick();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
