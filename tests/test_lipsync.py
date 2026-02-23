from app.voice.lipsync import build_viseme_timeline, phoneme_to_viseme


def test_viseme_mapping_for_known_phonemes():
    assert phoneme_to_viseme("AA") == "open_back"
    assert phoneme_to_viseme("M") == "closed"
    assert phoneme_to_viseme("ZZ") in {"teeth", "rest"}


def test_timeline_output_for_test_phonemes():
    phonemes = [
        {"phoneme": "M", "start_ms": 0, "end_ms": 100},
        {"phoneme": "AA", "start_ms": 110, "end_ms": 220},
        {"phoneme": "SIL", "start_ms": 230, "end_ms": 320},
    ]
    timeline = build_viseme_timeline(phonemes)
    assert len(timeline) == 3
    assert all("mouth_open" in p and "mouth_form" in p for p in timeline)


def test_timeline_time_order_and_range():
    phonemes = [
        {"phoneme": "M", "start_ms": 0, "end_ms": 80},
        {"phoneme": "IY", "start_ms": 80, "end_ms": 170},
        {"phoneme": "UW", "start_ms": 170, "end_ms": 260},
    ]
    timeline = build_viseme_timeline(phonemes)

    times = [item["time_ms"] for item in timeline]
    assert times == sorted(times)

    for frame in timeline:
        assert 0 <= frame["mouth_open"] <= 1
        assert 0 <= frame["mouth_form"] <= 1


def test_empty_timeline_has_neutral_sil_frame():
    timeline = build_viseme_timeline([])
    assert timeline == [{"time_ms": 0, "viseme": "sil", "mouth_open": 0.05, "mouth_form": 0.5}]
