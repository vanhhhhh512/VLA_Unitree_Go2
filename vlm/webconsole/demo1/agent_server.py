"""FastAPI server cho demo1 (DI + mock mode)."""
import os
import asyncio
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse

from ..vlm_engine import encode_frame_jpeg

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


def create_agent_app(agent, frame_source):
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
        try:
            while True:
                data = await websocket.receive_json()
                command = (data or {}).get("command", "").strip()
                if not command:
                    continue
                if app.state.busy:
                    await websocket.send_json(
                        {"type": "error", "message": "Đang xử lý lệnh khác."})
                    continue
                app.state.busy = True
                try:
                    loop = asyncio.get_event_loop()
                    q: asyncio.Queue = asyncio.Queue()

                    def produce():
                        try:
                            for ev in agent.run(command):
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
                        await websocket.send_json(ev)
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
                finally:
                    app.state.busy = False
        except WebSocketDisconnect:
            return

    return app


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

    if mock:
        frame_source = MockFrameSource()
        navigator = MockNavigator()
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

    app = create_agent_app(agent, frame_source)
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
