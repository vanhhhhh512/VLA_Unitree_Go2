import os
from vlm.webconsole.demo1.rooms import (
    Room, MAP_BOUNDS, random_rooms, save_rooms, load_rooms,
)


def test_random_rooms_keys_and_bounds():
    rooms = random_rooms()
    assert set(rooms) == {"kitchen", "living_room", "bedroom"}
    xmin, xmax, ymin, ymax = MAP_BOUNDS
    for r in rooms.values():
        assert isinstance(r, Room)
        assert xmin <= r.x <= xmax
        assert ymin <= r.y <= ymax
    assert "microwave" in rooms["kitchen"].landmarks


def test_save_and_load_roundtrip(tmp_path):
    rooms = {"kitchen": Room("kitchen", 1.0, 2.0, 0.5, ["microwave", "fridge"])}
    p = os.path.join(tmp_path, "rooms.yaml")
    save_rooms(rooms, p)
    loaded = load_rooms(p)
    assert loaded["kitchen"].x == 1.0
    assert loaded["kitchen"].landmarks == ["microwave", "fridge"]
    assert loaded["kitchen"].name == "kitchen"
    assert loaded["kitchen"].face is None


def test_face_roundtrip(tmp_path):
    rooms = {"kitchen": Room("kitchen", -0.6, -0.26, 0.0, ["microwave"],
                             face=[0.1, -0.26])}
    p = os.path.join(tmp_path, "rooms.yaml")
    save_rooms(rooms, p)
    loaded = load_rooms(p)
    assert loaded["kitchen"].face == [0.1, -0.26]
