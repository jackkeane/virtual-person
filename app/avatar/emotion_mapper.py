from __future__ import annotations

from dataclasses import dataclass


AFFECT_PRESETS: dict[str, dict] = {
    "neutral": {"eyes": 0.7, "brows": 0.5, "mouth": 0.5, "head": 0.0},
    "happy": {"eyes": 0.9, "brows": 0.7, "mouth": 0.8, "head": 0.1},
    "sad": {"eyes": 0.4, "brows": 0.3, "mouth": 0.2, "head": -0.1},
    "surprised": {"eyes": 1.0, "brows": 1.0, "mouth": 0.6, "head": 0.0},
    "thinking": {"eyes": 0.6, "brows": 0.5, "mouth": 0.4, "head": 0.3},
    "confused": {"eyes": 0.7, "brows": 0.8, "mouth": 0.45, "head": -0.2},
}

RULES: dict[str, list[str]] = {
    "happy": ["happy", "glad", "great", "wonderful", "awesome", "😊", "thanks", "yay"],
    "sad": ["sorry", "unfortunately", "sad", "miss", "regret", "😢"],
    "surprised": ["wow", "unexpected", "really?", "omg", "!?"],
    "thinking": ["hmm", "let me think", "consider", "...", "perhaps"],
    "confused": ["not sure", "unclear", "what do you mean", "confused", "maybe"],
}


@dataclass
class EmotionResult:
    category: str
    intensity: float
    expression: dict
    transition_ms: int


class EmotionMapper:
    def analyze(self, text: str) -> EmotionResult:
        raw = (text or "").lower()
        if not raw.strip():
            return EmotionResult("neutral", 0.0, AFFECT_PRESETS["neutral"], 300)

        scores = {"neutral": 0.15}
        for category, keywords in RULES.items():
            score = 0.0
            for kw in keywords:
                if kw in raw:
                    score += 0.25
            scores[category] = min(score, 1.0)

        # punctuation signal
        exclamations = raw.count("!")
        questions = raw.count("?")
        dots = raw.count("...")
        scores["happy"] = min(1.0, scores.get("happy", 0.0) + exclamations * 0.08)
        scores["surprised"] = min(1.0, scores.get("surprised", 0.0) + (exclamations + questions) * 0.06)
        scores["thinking"] = min(1.0, scores.get("thinking", 0.0) + dots * 0.12)

        category = max(scores, key=scores.get)
        intensity = max(0.0, min(scores[category], 1.0))
        if intensity < 0.2:
            category = "neutral"
            intensity = 0.1

        target = self._apply_intensity(AFFECT_PRESETS[category], intensity)
        transition_ms = int(300 + 200 * intensity)
        return EmotionResult(category, intensity, target, transition_ms)

    @staticmethod
    def lerp_expression(current: dict, target: dict, progress: float) -> dict:
        p = max(0.0, min(progress, 1.0))
        out = {}
        keys = set(current.keys()) | set(target.keys())
        for key in keys:
            c = float(current.get(key, 0.0))
            t = float(target.get(key, 0.0))
            out[key] = c + (t - c) * p
        return out

    @staticmethod
    def _apply_intensity(preset: dict, intensity: float) -> dict:
        base = AFFECT_PRESETS["neutral"]
        out = {}
        for k, v in preset.items():
            b = base.get(k, 0.0)
            out[k] = max(0.0, min(1.0, b + (v - b) * intensity))
        return out
