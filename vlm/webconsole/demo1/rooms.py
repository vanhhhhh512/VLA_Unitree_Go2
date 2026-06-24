"""Định nghĩa phòng + toạ độ trên map."""
import random
from dataclasses import dataclass, asdict

import yaml

# (xmin, xmax, ymin, ymax) theo mét, suy từ cty.yaml
MAP_BOUNDS = (-2.118, 6.732, -5.803, 3.847)

_LANDMARKS = {
    "kitchen": ["microwave", "fridge", "stove", "sink"],
    "living_room": ["sofa", "tv", "coffee table"],
    "bedroom": ["bed", "wardrobe", "lamp"],
}


@dataclass
class Room:
    name: str
    x: float
    y: float
    yaw: float
    landmarks: list
    face: list = None  # [fx, fy] điểm robot quay mặt vào; None -> dùng yaw


def random_rooms():
    xmin, xmax, ymin, ymax = MAP_BOUNDS
    rooms = {}
    for name, lm in _LANDMARKS.items():
        rooms[name] = Room(
            name=name,
            x=round(random.uniform(xmin + 0.5, xmax - 0.5), 2),
            y=round(random.uniform(ymin + 0.5, ymax - 0.5), 2),
            yaw=round(random.uniform(-3.14, 3.14), 2),
            landmarks=list(lm),
        )
    return rooms


def save_rooms(rooms, path):
    data = {}
    for name, r in rooms.items():
        d = asdict(r)
        d.pop("name")
        if d.get("face") is None:
            d.pop("face", None)
        data[name] = d
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_rooms(path):
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    rooms = {}
    for name, d in data.items():
        face = d.get("face")
        if face is not None:
            face = [float(face[0]), float(face[1])]
        rooms[name] = Room(
            name=name,
            x=float(d["x"]), y=float(d["y"]), yaw=float(d.get("yaw", 0.0)),
            landmarks=list(d.get("landmarks", [])),
            face=face,
        )
    return rooms
