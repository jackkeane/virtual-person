#!/usr/bin/env python
"""Standalone CosyVoice TTS worker.

Runs in the `cosyvoice` conda env. Communicates via stdin/stdout JSON protocol.
Usage:
  echo '{"text":"你好","lang":"zh"}' | ~/anaconda3/envs/cosyvoice/bin/python cosyvoice_worker.py

Input  (JSON per line): {"text": "...", "lang": "zh|en", "prompt_text": "...", "prompt_wav": "..."}
Output (JSON per line): {"ok": true, "audio_b64": "...", "sample_rate": 22050, "duration_ms": 1234, "latency_ms": 567}
       or               {"ok": false, "error": "..."}
"""
import json
import sys
import os
import time
import io
import base64
import wave

# Setup paths
sys.path.insert(0, "/home/zz79jk/CosyVoice")
sys.path.insert(0, "/home/zz79jk/CosyVoice/third_party/Matcha-TTS")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_model = None
_sample_rate = 22050

DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
DEFAULT_PROMPT_WAV = "/home/zz79jk/CosyVoice/asset/zero_shot_prompt.wav"
MODEL_DIR = "/home/zz79jk/CosyVoice/pretrained_models/CosyVoice2-0.5B"


def _load_model():
    global _model, _sample_rate
    if _model is not None:
        return
    from cosyvoice.cli.cosyvoice import AutoModel
    _model = AutoModel(model_dir=MODEL_DIR)
    _sample_rate = _model.sample_rate


def _tensor_to_wav_bytes(tensor, sr):
    """Convert a 1-D or (1, N) torch tensor to WAV bytes with anti-crackle smoothing."""
    import torch
    if tensor.dim() > 1:
        tensor = tensor.squeeze(0)

    # Remove DC offset and avoid hard clipping distortion.
    tensor = tensor - tensor.mean()
    peak = torch.max(torch.abs(tensor))
    if peak > 0:
        tensor = tensor / peak * 0.92

    # Tiny fade-in/out to avoid edge clicks.
    n = tensor.shape[-1]
    fade = min(int(sr * 0.008), max(1, n // 20))  # up to 8ms
    if fade > 1 and n > fade * 2:
        ramp = torch.linspace(0.0, 1.0, fade, device=tensor.device, dtype=tensor.dtype)
        tensor[:fade] = tensor[:fade] * ramp
        tensor[-fade:] = tensor[-fade:] * torch.flip(ramp, dims=[0])

    tensor = tensor.clamp(-1.0, 1.0)
    pcm = (tensor * 32767).to(torch.int16).cpu().numpy().tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return buf.getvalue()


def synthesize(req: dict) -> dict:
    t0 = time.time()
    try:
        _load_model()
    except Exception as e:
        return {"ok": False, "error": f"model_load_failed: {e}"}

    text = req.get("text", "").strip()
    if not text:
        return {"ok": False, "error": "empty_text"}

    lang = req.get("lang", "zh")
    prompt_text = req.get("prompt_text", DEFAULT_PROMPT_TEXT)
    prompt_wav = req.get("prompt_wav", DEFAULT_PROMPT_WAV)

    try:
        if lang == "en":
            # Cross-lingual for English
            gen = _model.inference_cross_lingual(
                f"<|en|>{text}",
                prompt_wav,
                stream=False,
            )
        else:
            # Zero-shot for Chinese (and default)
            gen = _model.inference_zero_shot(
                text,
                prompt_text,
                prompt_wav,
                stream=False,
            )

        import torch
        parts = []
        for chunk in gen:
            piece = chunk.get("tts_speech")
            if piece is None:
                continue
            if piece.dim() > 1:
                piece = piece.squeeze(0)
            parts.append(piece)

        if not parts:
            return {"ok": False, "error": "no_audio_generated"}

        audio_tensor = torch.cat(parts, dim=-1)

        wav_bytes = _tensor_to_wav_bytes(audio_tensor, _sample_rate)
        duration_ms = int(audio_tensor.shape[-1] / _sample_rate * 1000)
        latency_ms = int((time.time() - t0) * 1000)

        return {
            "ok": True,
            "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
            "sample_rate": _sample_rate,
            "duration_ms": duration_ms,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def main():
    """Read JSON lines from stdin, respond on stdout."""
    # Redirect stderr to devnull so tqdm / logging don't corrupt stdout JSON
    import os as _os
    _real_stderr_fd = _os.dup(2)
    _devnull = _os.open(_os.devnull, _os.O_WRONLY)
    _os.dup2(_devnull, 2)
    _os.close(_devnull)
    # Also suppress tqdm via env
    _os.environ["TQDM_DISABLE"] = "1"

    # If called with --warmup, just load model and exit
    if "--warmup" in sys.argv:
        try:
            _load_model()
            print(json.dumps({"ok": True, "action": "warmup", "sample_rate": _sample_rate}), flush=True)
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}), flush=True)
        return

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"ok": False, "error": f"json_parse: {e}"}), flush=True)
            continue
        result = synthesize(req)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
