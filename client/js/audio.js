let _recorder = null;
let _stream = null;
let _chunks = [];
let _onRecordingDone = null;

// Auto-listen mode state
let _autoRaf = null;
let _audioCtx = null;
let _analyser = null;
let _source = null;
let _autoConfig = null;
let _autoSegmentCallback = null;
let _autoSpeaking = false;
let _autoSilenceMs = 0;
let _autoSpeechMs = 0;
let _autoSegmentMs = 0;
let _autoLastTick = 0;
let _autoCooldownUntil = 0;
let _autoDebugLastLog = 0;
let _autoNoiseFloor = 0;
let _autoNoiseCalibrated = false;
let _autoNoiseSamples = [];
let _autoHeaderPrefix = null; // EBML+track prefix bytes (before first Cluster element)

/**
 * Start recording from the microphone.
 * Returns a promise that resolves when recording starts.
 */
export async function startMicRecording() {
  if (_recorder && _recorder.state === 'recording') return;

  _chunks = [];
  _stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate: 16000,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: false,  // Prevent browser from reducing mic volume
    },
  });

  const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    ? 'audio/webm;codecs=opus'
    : MediaRecorder.isTypeSupported('audio/webm')
      ? 'audio/webm'
      : '';

  _recorder = new MediaRecorder(_stream, mimeType ? { mimeType } : {});

  _recorder.ondataavailable = (e) => {
    if (e.data.size > 0) _chunks.push(e.data);
  };

  _recorder.onstop = async () => {
    const blob = new Blob(_chunks, { type: _recorder.mimeType });
    const buffer = await blob.arrayBuffer();
    const base64 = arrayBufferToBase64(buffer);
    _onRecordingDone?.(base64, _recorder.mimeType);
    _onRecordingDone = null;

    releaseMic();
  };

  _recorder.start(100);
}

/**
 * Stop recording and return the audio as base64.
 */
export function stopMicRecording() {
  return new Promise((resolve) => {
    if (!_recorder || _recorder.state !== 'recording') {
      resolve({ base64: '', mimeType: '' });
      return;
    }
    _onRecordingDone = (base64, mimeType) => resolve({ base64, mimeType });
    _recorder.stop();
  });
}

/**
 * Start always-on listening mode. Sends a segment callback when silence threshold is reached.
 */
export async function startAlwaysOnMic({
  onSegment,
  isSpeaking = () => false,
  silenceMs = 2000,
  minSpeechMs = 500,
  maxSegmentMs = 20000,
  cooldownMs = 300,
  rmsThreshold = 0.022,
  minBlobBytes = 6000,
} = {}) {
  if (_recorder && _recorder.state === 'recording') return;

  _autoConfig = { silenceMs, minSpeechMs, maxSegmentMs, cooldownMs, rmsThreshold, minBlobBytes };
  _autoSegmentCallback = onSegment;
  _chunks = [];

  _stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate: 16000,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: false,
    },
  });

  const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    ? 'audio/webm;codecs=opus'
    : MediaRecorder.isTypeSupported('audio/webm')
      ? 'audio/webm'
      : '';

  _recorder = new MediaRecorder(_stream, mimeType ? { mimeType } : {});
  _autoHeaderPrefix = null;
  _recorder.ondataavailable = (e) => {
    if (e.data.size > 0) {
      _chunks.push(e.data);
    }
  };
  _recorder.onstop = () => {
    stopAlwaysOnMic();
  };
  _recorder.start(100);

  _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  _source = _audioCtx.createMediaStreamSource(_stream);
  _analyser = _audioCtx.createAnalyser();
  _analyser.fftSize = 1024;
  _source.connect(_analyser);

  _autoSpeaking = false;
  _autoSilenceMs = 0;
  _autoSpeechMs = 0;
  _autoSegmentMs = 0;
  _autoLastTick = performance.now();
  _autoCooldownUntil = 0;
  _autoHeaderPrefix = null;
  _autoNoiseFloor = 0;
  _autoNoiseCalibrated = false;
  _autoNoiseSamples = [];
  _autoDebugLastLog = 0;

  const tick = async () => {
    if (!_analyser || !_recorder || _recorder.state !== 'recording') return;
    // Resume AudioContext if it got suspended (e.g., after audio playback)
    if (_audioCtx && _audioCtx.state === 'suspended') {
      _audioCtx.resume().catch(() => {});
    }

    const now = performance.now();
    const dt = Math.max(0, now - _autoLastTick);
    _autoLastTick = now;

    const rms = currentRms(_analyser);
    // Skip client-side speech detection — always treat as speech.
    // Server-side Silero VAD handles real speech/noise filtering.
    const isSpeech = true;

    _autoSegmentMs += dt;
    if (isSpeech) {
      _autoSpeaking = true;
      _autoSilenceMs = 0;
      _autoSpeechMs += dt;
    } else if (_autoSpeaking) {
      _autoSilenceMs += dt;
    }

    const shouldCommitBySilence =
      _autoSpeaking &&
      _autoSpeechMs >= _autoConfig.minSpeechMs &&
      _autoSilenceMs >= _autoConfig.silenceMs;

    const shouldCommitByMaxLen =
      _autoSpeaking &&
      _autoSpeechMs >= _autoConfig.minSpeechMs &&
      _autoSegmentMs >= _autoConfig.maxSegmentMs;

    if ((shouldCommitBySilence || shouldCommitByMaxLen) && now >= _autoCooldownUntil) {
      _autoCooldownUntil = now + _autoConfig.cooldownMs;
      await flushAutoSegment();
      _autoSpeaking = false;
      _autoSilenceMs = 0;
      _autoSpeechMs = 0;
      _autoSegmentMs = 0;
    }

    _autoRaf = requestAnimationFrame(() => {
      tick().catch((err) => console.error('[Ani] auto mic tick failed', err));
    });
  };

  _autoRaf = requestAnimationFrame(() => {
    tick().catch((err) => console.error('[Ani] auto mic start failed', err));
  });
}

export function stopAlwaysOnMic() {
  if (_autoRaf) {
    cancelAnimationFrame(_autoRaf);
    _autoRaf = null;
  }
  _autoSegmentCallback = null;
  _autoConfig = null;

  if (_source) {
    try { _source.disconnect(); } catch (_) {}
    _source = null;
  }
  _analyser = null;

  if (_audioCtx) {
    try { _audioCtx.close(); } catch (_) {}
    _audioCtx = null;
  }

  if (_recorder && _recorder.state === 'recording') {
    _recorder.stop();
  }

  releaseMic();
  _chunks = [];
}

export function isRecording() {
  return _recorder?.state === 'recording';
}

export function isAlwaysOnActive() {
  return !!_autoConfig && isRecording();
}

function currentRms(analyser) {
  const data = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(data);
  let sumSq = 0;
  for (let i = 0; i < data.length; i++) {
    const v = (data[i] - 128) / 128;
    sumSq += v * v;
  }
  return Math.sqrt(sumSq / data.length);
}

async function flushAutoSegment() {
  if (!_chunks.length || !_recorder) return;

  let blob = new Blob(_chunks, { type: _recorder.mimeType });
  let buffer = await blob.arrayBuffer();

  if (!_autoHeaderPrefix) {
    // Build a stable WebM prefix (EBML + track metadata) from the first segment,
    // up to (but excluding) the first Cluster element (0x1F43B675).
    _autoHeaderPrefix = extractWebmHeaderPrefix(new Uint8Array(buffer));
  } else {
    // Subsequent segments from MediaRecorder may start at Cluster and miss EBML.
    // Prepend the saved prefix so ffmpeg can decode them.
    blob = new Blob([_autoHeaderPrefix, ..._chunks], { type: _recorder.mimeType });
    buffer = await blob.arrayBuffer();
  }

  _chunks = [];

  if (!buffer.byteLength || buffer.byteLength < (_autoConfig?.minBlobBytes || 0)) return;
  const base64 = arrayBufferToBase64(buffer);
  _autoSegmentCallback?.({
    base64,
    mimeType: _recorder.mimeType,
    speechMs: _autoSpeechMs,
    segmentMs: _autoSegmentMs,
  });
}

function releaseMic() {
  _stream?.getTracks().forEach((t) => t.stop());
  _stream = null;
  _recorder = null;
}

function extractWebmHeaderPrefix(bytes) {
  // Cluster element ID in WebM/Matroska: 0x1F43B675
  const CLUSTER = [0x1f, 0x43, 0xb6, 0x75];
  for (let i = 0; i <= bytes.length - 4; i++) {
    if (
      bytes[i] === CLUSTER[0] &&
      bytes[i + 1] === CLUSTER[1] &&
      bytes[i + 2] === CLUSTER[2] &&
      bytes[i + 3] === CLUSTER[3]
    ) {
      // Keep only metadata prefix; exclude first cluster payload.
      return bytes.slice(0, i);
    }
  }
  // Fallback: if Cluster not found, keep entire first segment as prefix.
  return bytes;
}

function detectAudioMime(base64Str) {
  try {
    const raw = atob(base64Str.slice(0, 16));
    const bytes = new Uint8Array([...raw].map((c) => c.charCodeAt(0)));
    if (bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46) return 'audio/wav';
    if (bytes[0] === 0x49 && bytes[1] === 0x44 && bytes[2] === 0x33) return 'audio/mpeg';
    if (bytes[0] === 0xff && (bytes[1] & 0xe0) === 0xe0) return 'audio/mpeg';
    if (bytes[0] === 0x4f && bytes[1] === 0x67 && bytes[2] === 0x67 && bytes[3] === 0x53) return 'audio/ogg';
    if (bytes[0] === 0x1a && bytes[1] === 0x45 && bytes[2] === 0xdf && bytes[3] === 0xa3) return 'audio/webm';
  } catch (_) {
    /* fall through */
  }
  return 'application/octet-stream';
}

export function playAudioBase64(audioBase64, mimeHint) {
  const mime = mimeHint || detectAudioMime(audioBase64);

  // Use Blob URL instead of huge data URI to avoid browser playback instability
  // on long WAV chunks.
  const binary = atob(audioBase64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

  const blob = new Blob([bytes], { type: mime });
  const objectUrl = URL.createObjectURL(blob);
  const audio = new Audio(objectUrl);

  const cleanup = () => {
    try { URL.revokeObjectURL(objectUrl); } catch (_) {}
  };

  const playPromise = audio.play().catch(() => {});
  return { audio, playPromise, cleanup };
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}
