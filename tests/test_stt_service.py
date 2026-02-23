from __future__ import annotations

from types import SimpleNamespace

from app.voice.stt_service import FasterWhisperSTTService


class _DummyModel:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def transcribe(self, audio_path, language=None, beam_size=5, vad_filter=True):
        self.calls.append(language)
        return self.responses.pop(0)


def _resp(text: str, lang: str = "zh", prob: float = 0.95):
    segments = [SimpleNamespace(text=text)] if text else []
    info = SimpleNamespace(language=lang, language_probability=prob)
    return segments, info


def test_language_hint_pass_through(monkeypatch):
    monkeypatch.setattr("app.voice.stt_service.config.stt_language_hint", "zh")
    svc = FasterWhisperSTTService()
    model = _DummyModel([_resp("你好", "zh", 0.99)])
    svc._model = model
    svc._load_attempted = True

    audio = svc._build_silent_wav()
    out = svc.transcribe(audio)

    assert out == "你好"
    assert model.calls[0] == "zh"


def test_fallback_to_auto_when_first_pass_empty(monkeypatch):
    monkeypatch.setattr("app.voice.stt_service.config.stt_language_hint", "zh")
    svc = FasterWhisperSTTService()
    model = _DummyModel([
        _resp("", "zh", 0.2),
        _resp("今天天气不错", "zh", 0.9),
    ])
    svc._model = model
    svc._load_attempted = True

    out = svc.transcribe(svc._build_silent_wav())

    assert out == "今天天气不错"
    assert model.calls == ["zh", None]


def test_warmup_does_not_crash_when_model_unavailable(monkeypatch):
    monkeypatch.setattr("app.voice.stt_service.config.stt_warmup_enabled", True)

    class _ImmediateThread:
        def __init__(self, target=None, name=None, daemon=None):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    def _boom(self):
        raise RuntimeError("no model")

    monkeypatch.setattr("app.voice.stt_service.threading.Thread", _ImmediateThread)
    monkeypatch.setattr(FasterWhisperSTTService, "_ensure_model", _boom)

    svc = FasterWhisperSTTService()
    assert svc._warmup_started is True
