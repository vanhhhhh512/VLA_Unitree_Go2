"""FastAPI server cho demo1 (DI + mock mode)."""
import os
import asyncio
import threading
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse

from ..vlm_engine import encode_frame_jpeg
from .motion import parse_motion
from .actions import ACTIONS, match_action

WEBUI = Path(__file__).parent.parent / "webui" / "demo1.html"
ASSETS = Path(__file__).parent.parent / "webui" / "assets"


class MockFrameSource:
    is_mock = True
    is_connected = False

    def __init__(self):
        f = np.full((480, 640, 3), 245, dtype=np.uint8)
        cv2.putText(f, "DEMO_MOCK - no robot", (60, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2)
        self._f = f

    def get_latest_frame(self):
        return self._f


class MockNavigator:
    def go_to(self, room):
        for d in (2.0, 1.0, 0.3):
            yield {"kind": "feedback", "distance_remaining": d}
        yield {"kind": "done", "success": True}

    def cancel(self):
        pass


class MockMotion:
    def run(self, cmd, cancel=None):
        import math
        if cmd.kind == "move":
            title = f"Moving {'forward' if cmd.value >= 0 else 'backward'} {abs(cmd.value):.2f} m"
        else:
            title = f"Turning {'left' if cmd.value >= 0 else 'right'} {math.degrees(abs(cmd.value)):.0f}°"
        yield {"type": "step", "id": "motion", "status": "running", "title": title}
        for d in (0.6, 0.3, 0.1):
            yield {"type": "nav", "distance_remaining": d}
        yield {"type": "step", "id": "motion", "status": "done"}
        yield {"type": "answer", "text": f"Done — {title.lower()}.", "state": "UNKNOWN"}

    def estop(self):
        pass


class MockAction:
    def run(self, act, cancel=None):
        yield {"type": "step", "id": "action", "status": "running",
               "title": f"Action: {act['vi']}"}
        yield {"type": "step", "id": "action", "status": "done"}
        yield {"type": "answer", "text": f"Đã gửi lệnh: {act['vi']}.", "state": "UNKNOWN"}


def create_agent_app(agent, frame_source, motion=None, action=None):
    app = FastAPI()
    app.state.busy = False

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return WEBUI.read_text(encoding="utf-8")

    @app.get("/assets/{name}")
    async def asset(name: str):
        p = ASSETS / name
        if not p.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(str(p))

    @app.get("/actions")
    async def actions():
        return JSONResponse(ACTIONS)

    @app.get("/status")
    async def status():
        return JSONResponse({
            "connected": bool(getattr(frame_source, "is_connected", False)),
            "mock": bool(getattr(frame_source, "is_mock", False)),
        })

    @app.get("/video_feed")
    async def video_feed():
        async def gen():
            while True:
                frame = frame_source.get_latest_frame()
                if frame is not None:
                    jpg = encode_frame_jpeg(frame)
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + jpg + b"\r\n")
                await asyncio.sleep(0.05)
        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        # Vòng nhận lệnh KHÔNG bị chặn khi đang chạy job -> bắt được "stop".
        job = {"task": None, "cancel": None}
        try:
            while True:
                data = await websocket.receive_json()
                msg_action = (data or {}).get("action")
                if msg_action == "stop":
                    if job["cancel"] is not None:
                        job["cancel"].set()          # báo agent dừng
                    continue
                if msg_action == "estop":
                    # Dừng khẩn cấp: hủy job + phanh robot ngay (publish vận tốc 0).
                    if job["cancel"] is not None:
                        job["cancel"].set()
                    est = getattr(motion, "estop", None)
                    if callable(est):
                        try:
                            est()
                        except Exception:
                            pass
                    await websocket.send_json(
                        {"type": "error", "message": "🛑 EMERGENCY STOP — robot halted."})
                    continue
                command = (data or {}).get("command", "").strip()
                if not command:
                    continue
                if job["task"] is not None and not job["task"].done():
                    await websocket.send_json(
                        {"type": "error", "message": "Đang xử lý lệnh khác."})
                    continue
                cancel = threading.Event()
                job["cancel"] = cancel
                # Định tuyến: chuyển động (move/turn) -> motion; hành động (đứng/ngồi/
                # chào/nhảy...) -> action; còn lại -> agentic (Qwen + Nav2 + YOLO).
                mc = parse_motion(command)
                ac = match_action(command)
                if mc is not None and motion is not None:
                    producer = lambda: motion.run(mc, cancel)        # noqa: E731
                elif ac is not None and action is not None:
                    producer = lambda: action.run(ac, cancel)        # noqa: E731
                else:
                    producer = lambda: agent.run(command, cancel=cancel)  # noqa: E731
                job["task"] = asyncio.create_task(_run_job(websocket, producer))
        except WebSocketDisconnect:
            if job["cancel"] is not None:
                job["cancel"].set()
            return

    return app


async def _run_job(websocket, producer):
    """Chạy producer() (generator event) trong executor, đẩy ra WS; có thể cancel."""
    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def produce():
        try:
            for ev in producer():
                asyncio.run_coroutine_threadsafe(q.put(ev), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                q.put({"type": "error", "message": str(e)}), loop)
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

    loop.run_in_executor(None, produce)
    while True:
        ev = await q.get()
        if ev is None:
            break
        try:
            await websocket.send_json(ev)
        except Exception:
            break


def main():
    import uvicorn
    from .rooms import load_rooms, random_rooms, save_rooms
    from .planner import Planner
    from .perception import Perception
    from .agent import Agent
    from ..vlm_engine import VLMEngine

    cfg = Path(__file__).parent.parent / "config" / "rooms.yaml"
    if cfg.is_file():
        rooms = load_rooms(str(cfg))
    else:
        rooms = random_rooms()
        cfg.parent.mkdir(parents=True, exist_ok=True)
        save_rooms(rooms, str(cfg))

    mock = os.getenv("DEMO_MOCK") == "1"
    engine = VLMEngine()

    motion = None
    action = None
    if mock:
        frame_source = MockFrameSource()
        navigator = MockNavigator()
        motion = MockMotion()
        action = MockAction()
        print("[demo1] DEMO_MOCK=1 -> không cần ROS/robot.")
        if os.getenv("VLM_SKIP_MODEL") != "1":
            engine.load()
    else:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from .navigator import Nav2Navigator
        from .ros_frame_source import RosFrameSource
        import threading
        rclpy.init()
        node = rclpy.create_node("strikerobot_demo1")
        frame_source = RosFrameSource(node)
        # Node camera dùng executor RIÊNG, không phải global executor —
        # nếu dùng global thì BasicNavigator (cũng dùng global) sẽ báo
        # "Executor is already spinning".
        cam_exec = SingleThreadedExecutor()
        cam_exec.add_node(node)
        threading.Thread(target=cam_exec.spin, daemon=True).start()
        navigator = Nav2Navigator()
        try:
            from .motion import MotionController
            motion = MotionController(node)
            print("[demo1] Motion controller (cmd_vel_joy) sẵn sàng.")
        except Exception as e:
            print(f"[demo1] Motion controller lỗi ({e}); chế độ lệnh tay tắt.")
        try:
            from .actions import ActionController
            action = ActionController(node)
            print("[demo1] Action controller (webrtc_req) sẵn sàng.")
        except Exception as e:
            print(f"[demo1] Action controller lỗi ({e}); chế độ hành động tắt.")
        engine.load()

    detector = None
    if os.getenv("USE_YOLO", "1") == "1":
        try:
            from .yolo_detector import YoloDetector
            detector = YoloDetector()
            print(f"[demo1] YOLO detector sẵn sàng ({len(detector.names)} lớp COCO).")
        except Exception as e:
            print(f"[demo1] YOLO không dùng được ({e}); fallback Qwen grounding.")

    planner = Planner(engine)
    perception = Perception(engine, detector=detector)
    agent = Agent(planner, navigator, frame_source, perception, rooms)

    app = create_agent_app(agent, frame_source, motion=motion, action=action)
    try:
        # timeout_graceful_shutdown: đừng chờ vô tận luồng MJPEG /video_feed khi Ctrl+C.
        uvicorn.run(app, host="0.0.0.0", port=8001, timeout_graceful_shutdown=3)
    finally:
        if not mock:
            # Tắt ROS sạch để tránh "terminate called" lúc thoát.
            try:
                cam_exec.shutdown()
            except Exception:
                pass
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
