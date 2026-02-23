from app.avatar.state_machine import ConversationStateMachine


def test_state_transitions_include_speaking_to_listening_and_any_to_idle():
    sm = ConversationStateMachine()
    assert sm.state == "idle"

    sm.transition("listening")
    sm.transition("thinking")
    sm.transition("speaking")
    sm.transition("listening")  # newly allowed
    sm.transition("idle")

    # any state -> idle should be allowed
    sm.transition("thinking")
    sm.transition("idle")
    assert sm.state == "idle"


def test_illegal_transition_is_non_fatal_and_keeps_current_state():
    sm = ConversationStateMachine()
    sm.transition("listening")

    snap = sm.transition("speaking")  # still illegal from listening
    assert snap.state == "listening"
    assert sm.state == "listening"


def test_expression_params_in_bounds():
    sm = ConversationStateMachine()
    for state in ["idle", "listening", "thinking", "speaking"]:
        expr = sm.expression_for(state)
        for v in expr.values():
            assert 0.0 <= float(v) <= 1.0
