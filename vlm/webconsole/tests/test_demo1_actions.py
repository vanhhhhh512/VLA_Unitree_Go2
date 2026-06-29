from vlm.webconsole.demo1.actions import match_action, ACTIONS


def test_match_stand_up_vi_en():
    assert match_action("đứng dậy")["api_id"] == 1004
    assert match_action("stand up")["api_id"] == 1004


def test_match_hello():
    assert match_action("say hello")["api_id"] == 1016
    assert match_action("xin chào")["api_id"] == 1016


def test_match_dance_specific_over_generic():
    assert match_action("dance 2")["api_id"] == 1023   # cụ thể thắng "dance"
    assert match_action("nhảy")["api_id"] == 1022


def test_match_sit():
    assert match_action("ngồi")["api_id"] == 1009


def test_question_is_not_action():
    assert match_action("is the dog sitting?") is None


def test_non_action_returns_none():
    assert match_action("is the bottle on the microwave?") is None
    assert match_action("describe the scene") is None


def test_actions_table_has_required_fields():
    for a in ACTIONS:
        assert {"api_id", "vi", "desc", "keys"} <= set(a)
        assert isinstance(a["keys"], list) and a["keys"]
