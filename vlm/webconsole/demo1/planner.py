"""Planner: câu lệnh -> phòng đích + vật cần quan sát (Qwen text)."""
import re
import json
from dataclasses import dataclass

import numpy as np

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_REQUIRED = ("room", "target_object", "observation_question", "reasoning")


class PlanError(Exception):
    pass


@dataclass
class Plan:
    room: str
    target_object: str
    observation_question: str
    reasoning: str


def parse_plan(text, rooms):
    m = _JSON_RE.search(text or "")
    if not m:
        raise PlanError("No JSON found in planner output.")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise PlanError(f"Invalid planner JSON: {e}")
    for k in _REQUIRED:
        if k not in data:
            raise PlanError(f"Missing key '{k}' in plan.")
    if data["room"] not in rooms:
        raise PlanError(
            f"Room '{data['room']}' is not available. Valid rooms: {list(rooms)}"
        )
    return Plan(
        room=data["room"],
        target_object=str(data["target_object"]),
        observation_question=str(data["observation_question"]),
        reasoning=str(data["reasoning"]),
    )


def build_prompt(command, rooms):
    lines = [f"- {n}: {', '.join(r.landmarks)}" for n, r in rooms.items()]
    rooms_block = "\n".join(lines)
    return (
        "You are a task planner for a mobile robot. Here are the rooms and their "
        "landmarks:\n"
        f"{rooms_block}\n\n"
        f"User command: \"{command}\"\n\n"
        "Choose ONE room the robot should go to in order to answer the command, the "
        "object to look at, and the observation question. Reply ONLY with JSON in this "
        "exact format (in English):\n"
        '{"room": "<room name>", "target_object": "<object>", '
        '"observation_question": "<question>", "reasoning": "<short explanation>"}'
    )


class Planner:
    def __init__(self, engine):
        self.engine = engine

    def plan(self, command, rooms):
        prompt = build_prompt(command, rooms)
        blank = np.zeros((8, 8, 3), dtype=np.uint8)
        text = "".join(self.engine.stream_infer(blank, prompt))
        return parse_plan(text, rooms)
