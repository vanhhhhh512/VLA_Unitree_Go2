"""Orchestrator demo1: plan -> nav -> perceive -> answer."""
from .planner import PlanError

_STOP = {"type": "error", "message": "⏹ Đã dừng theo yêu cầu."}


class Agent:
    def __init__(self, planner, navigator, frame_source, perception, rooms):
        self.planner = planner
        self.navigator = navigator
        self.frame_source = frame_source
        self.perception = perception
        self.rooms = rooms

    def run(self, command, cancel=None):
        def cancelled():
            return cancel is not None and cancel.is_set()

        # 1) PLAN
        yield {"type": "step", "id": "plan", "status": "running",
               "title": "Planning the task..."}
        try:
            plan = self.planner.plan(command, self.rooms)
        except PlanError as e:
            yield {"type": "error", "message": f"Planner: {e}"}
            return
        if cancelled():
            yield _STOP
            return
        yield {"type": "token", "step": "plan", "text": plan.reasoning}
        yield {"type": "step", "id": "plan", "status": "done",
               "data": {"room": plan.room, "target_object": plan.target_object}}

        room = self.rooms[plan.room]

        # 2) NAV
        yield {"type": "step", "id": "nav", "status": "running",
               "title": f"Navigating to {room.name}",
               "data": {"room": room.name, "x": room.x, "y": room.y}}
        success = False
        for ev in self.navigator.go_to(room):
            if cancelled():
                self._cancel_nav()
                yield {"type": "step", "id": "nav", "status": "error"}
                yield _STOP
                return
            if ev["kind"] == "feedback":
                yield {"type": "nav", "distance_remaining": ev["distance_remaining"]}
            elif ev["kind"] == "error":
                yield {"type": "step", "id": "nav", "status": "error"}
                yield {"type": "error", "message": f"Navigation: {ev['message']}"}
                return
            elif ev["kind"] == "done":
                success = ev.get("success", False)
        if not success:
            yield {"type": "step", "id": "nav", "status": "error"}
            yield {"type": "error", "message": "Navigation failed."}
            return
        yield {"type": "step", "id": "nav", "status": "done"}

        if cancelled():
            yield _STOP
            return

        # 3) PERCEIVE
        yield {"type": "step", "id": "perceive", "status": "running",
               "title": f"Observing the {plan.target_object}"}
        frame = self.frame_source.get_latest_frame()
        if frame is None:
            yield {"type": "step", "id": "perceive", "status": "error"}
            yield {"type": "error", "message": "No camera image available."}
            return
        parts = []
        for chunk in self.perception.observe(
            frame, plan.target_object, plan.observation_question
        ):
            if cancelled():
                yield {"type": "step", "id": "perceive", "status": "error"}
                yield _STOP
                return
            parts.append(chunk)
            yield {"type": "token", "step": "perceive", "text": chunk}
        result = self.perception.finalize(
            "".join(parts), frame, plan.target_object,
            question=plan.observation_question)
        if result.annotated_jpeg_b64:
            yield {"type": "image", "data": result.annotated_jpeg_b64}
        yield {"type": "step", "id": "perceive", "status": "done"}

        # 4) ANSWER — câu trả lời ngôn ngữ tự nhiên của VLM cho đúng câu hỏi.
        answer = result.answer or f"({result.state})"
        yield {"type": "answer", "text": answer, "state": result.state}

    def _cancel_nav(self):
        """Hủy goal Nav2 đang chạy (nếu navigator hỗ trợ)."""
        cancel = getattr(self.navigator, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass
