from app.avatar.emotion_mapper import EmotionMapper


def test_happy_sad_neutral_detection_and_intensity():
    mapper = EmotionMapper()

    happy = mapper.analyze("I'm so happy for you!!! Wonderful news!")
    assert happy.category == "happy"
    assert happy.intensity > 0.7

    sad = mapper.analyze("I'm sorry, unfortunately this is sad.")
    assert sad.category == "sad"
    assert sad.intensity >= 0.5

    neutral = mapper.analyze("The report has been uploaded.")
    assert neutral.category == "neutral"
    assert 0.0 <= neutral.intensity <= 1.0


def test_emotion_lerp_blending():
    mapper = EmotionMapper()
    current = {"eyes": 0.2, "mouth": 0.2}
    target = {"eyes": 1.0, "mouth": 0.8}
    mid = mapper.lerp_expression(current, target, 0.5)
    assert 0.55 <= mid["eyes"] <= 0.65
    assert 0.45 <= mid["mouth"] <= 0.55
