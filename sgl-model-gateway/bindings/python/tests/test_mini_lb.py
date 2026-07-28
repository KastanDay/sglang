from sglang_router.mini_lb import _ensure_shared_request_id


def test_shared_request_id_uses_bootstrap_room():
    request = {"bootstrap_room": 7}

    _ensure_shared_request_id(request)

    assert request["rid"] == "pd-7"


def test_shared_batch_request_ids_use_bootstrap_rooms():
    request = {"bootstrap_room": [7, 8]}

    _ensure_shared_request_id(request)

    assert request["rid"] == ["pd-7", "pd-8"]


def test_explicit_request_id_is_preserved():
    request = {"bootstrap_room": 7, "rid": "caller-rid"}

    _ensure_shared_request_id(request)

    assert request["rid"] == "caller-rid"
