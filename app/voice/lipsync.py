from __future__ import annotations

PHONEME_TO_VISEME = {
    "AA": "open_back", "AE": "wide", "AH": "mid", "AO": "round", "AW": "round",
    "AY": "wide", "B": "closed", "CH": "rest", "D": "rest", "EH": "mid",
    "ER": "mid", "EY": "wide", "F": "bite", "G": "rest", "HH": "open",
    "IH": "mid", "IY": "wide", "JH": "rest", "K": "rest", "L": "tip",
    "M": "closed", "N": "closed", "NG": "closed", "OW": "round", "OY": "round",
    "P": "closed", "R": "mid", "S": "teeth", "SH": "teeth", "T": "rest",
    "TH": "tip", "UH": "round", "UW": "round", "V": "bite", "W": "round",
    "Y": "wide", "Z": "teeth", "SIL": "sil",
}

VISEME_PARAMS = {
    "sil": {"mouth_open": 0.05, "mouth_form": 0.5},
    "closed": {"mouth_open": 0.02, "mouth_form": 0.45},
    "open": {"mouth_open": 0.75, "mouth_form": 0.55},
    "open_back": {"mouth_open": 0.85, "mouth_form": 0.4},
    "wide": {"mouth_open": 0.55, "mouth_form": 0.85},
    "mid": {"mouth_open": 0.45, "mouth_form": 0.6},
    "round": {"mouth_open": 0.55, "mouth_form": 0.25},
    "bite": {"mouth_open": 0.3, "mouth_form": 0.35},
    "teeth": {"mouth_open": 0.25, "mouth_form": 0.65},
    "tip": {"mouth_open": 0.35, "mouth_form": 0.5},
    "rest": {"mouth_open": 0.15, "mouth_form": 0.5},
}


def phoneme_to_viseme(phoneme: str) -> str:
    token = (phoneme or "SIL").strip().upper()
    return PHONEME_TO_VISEME.get(token, PHONEME_TO_VISEME.get(token[:2], "rest"))


def build_viseme_timeline(phoneme_timestamps: list[dict], smoothing: float = 0.35) -> list[dict]:
    if not phoneme_timestamps:
        return [{"time_ms": 0, "viseme": "sil", **VISEME_PARAMS["sil"]}]

    timeline: list[dict] = []
    prev = VISEME_PARAMS["sil"]

    for item in phoneme_timestamps:
        viseme = phoneme_to_viseme(item.get("phoneme", "SIL"))
        cur = VISEME_PARAMS[viseme]
        open_v = prev["mouth_open"] + (cur["mouth_open"] - prev["mouth_open"]) * (1 - smoothing)
        form_v = prev["mouth_form"] + (cur["mouth_form"] - prev["mouth_form"]) * (1 - smoothing)
        timeline.append(
            {
                "time_ms": int(item.get("start_ms", 0)),
                "viseme": viseme,
                "mouth_open": round(open_v, 4),
                "mouth_form": round(form_v, 4),
            }
        )
        prev = {"mouth_open": open_v, "mouth_form": form_v}

    return timeline
