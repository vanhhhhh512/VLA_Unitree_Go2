"""FastAPI server cho demo1 — VLM nav (VLA loop) thuần, KHÔNG Nav2.

Định tuyến lệnh từ GUI:
  - "move/turn ..." (parse_motion)  -> MotionController        (lệnh tay)
  - "ngồi/chào/nhảy..." (match_action) -> ActionController     (sport API)
  - còn lại (ngôn ngữ tự nhiên)     -> NavLoopAgent (VLA loop) -> VLM suy luận
    ra hành động từng bước, tự chấp hành qua MotionController, lặp tới khi xong.
"""
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

    def _publish(self, vx, wz):
        pass

    def _stop(self):
        pass

    def front_distance(self):
        return None


class MockAction:
    def run(self, act, cancel=None):
        yield {"type": "step", "id": "action", "status": "running",
               "title": f"Action: {act['vi']}"}
        yield {"type": "step", "id": "action", "status": "done"}
        yield {"type": "answer", "text": f"Đã gửi lệnh: {act['vi']}.", "state": "UNKNOWN"}


def create_agent_app(frame_source, motion=None, action=None, navloop=None,
                     annotator=None):
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

    @app.get("/debug")
    async def debug():
        """Số liệu tức thời cho dashboard GUI (poll ~5Hz). Tính stateless từ annotator+lidar."""
        box = annotator.target_box() if annotator is not None else None
        h = annotator.frame_height() if annotator is not None else None
        w = annotator.frame_width() if annotator is not None else None
        gap = int(h - box[3]) if (box is not None and h) else None
        offset = None
        if box is not None and w:
            cx = (box[0] + box[2]) / 2.0
            offset = round(((cx - w / 2.0) / (w / 2.0)) * 45.0, 1)   # FOV~90 -> nửa 45°
        obstacle = None
        fd = getattr(motion, "front_distance", None)
        if callable(fd):
            try:
                d = fd()
                obstacle = round(d, 2) if d is not None else None
            except Exception:
                obstacle = None
        return JSONResponse({
            "target_detected": box is not None,
            "label": getattr(annotator, "label", None) if box is not None else None,
            "yolo_gap_px": gap,
            "stop_px": getattr(navloop, "stop_bottom_px", None),
            "center_offset_deg": offset,
            "center_tol_deg": getattr(navloop, "center_tol_deg", None),
            "obstacle_m": obstacle,
            "control": getattr(navloop, "control", None),
        })

    @app.get("/video_feed")
    async def video_feed():
        async def gen():
            last_id = None
            last_chunk = None
            while True:
                frame = frame_source.get_latest_frame()
                # CHỈ encode lại khi FRAME MỚI (camera ~2fps -> khỏi encode trùng 30 lần/s,
                # đỡ tốn CPU web -> nhường CPU cho driver decode video -> cam đỡ lag).
                if frame is not None and id(frame) != last_id:
                    last_id = id(frame)
                    out = annotator.render(frame) if annotator is not None else frame
                    jpg = encode_frame_jpeg(out)
                    last_chunk = (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                  + jpg + b"\r\n")
                if last_chunk is not None:
                    yield last_chunk
                await asyncio.sleep(0.033)
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
                        job["cancel"].set()          # báo job dừng
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
                # Khoanh vùng YOLO theo vật thể nhắc trong lệnh, lên camera trực tiếp.
                if annotator is not None:
                    annotator.set_target_from_text(command)
                if job["task"] is not None and not job["task"].done():
                    await websocket.send_json(
                        {"type": "error", "message": "Đang xử lý lệnh khác."})
                    continue
                cancel = threading.Event()
                job["cancel"] = cancel
                # Định tuyến: lệnh tay move/turn -> motion; sport action -> action;
                # còn lại (ngôn ngữ tự nhiên) -> VLA loop (VLM suy luận, KHÔNG Nav2).
                mc = parse_motion(command)
                ac = match_action(command)
                if mc is not None and motion is not None:
                    producer = lambda: motion.run(mc, cancel)        # noqa: E731
                elif ac is not None and action is not None:
                    producer = lambda: action.run(ac, cancel)        # noqa: E731
                elif navloop is not None:
                    producer = lambda: navloop.run(command, cancel=cancel)  # noqa: E731
                else:
                    await websocket.send_json(
                        {"type": "error",
                         "message": "VLM nav chưa sẵn sàng (cần VLM + camera + motion)."})
                    continue
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
    from ..vlm_engine import VLMEngine

    mock = os.getenv("DEMO_MOCK") == "1"
    engine = VLMEngine()
    # Não ngoài (NaVILA server / API) -> KHÔNG nạp Qwen local (đỡ ~7GB VRAM, tránh OOM khi
    # NaVILA server cùng GPU). Chỉ nạp Qwen cho brain 'local'.
    _ext_brain = os.getenv("VLA_BRAIN", "local").lower() in ("navila", "api")

    motion = None
    action = None
    if mock:
        frame_source = MockFrameSource()
        motion = MockMotion()
        action = MockAction()
        print("[demo1] DEMO_MOCK=1 -> không cần ROS/robot.")
        if os.getenv("VLM_SKIP_MODEL") != "1" and not _ext_brain:
            engine.load()
    else:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from .ros_frame_source import RosFrameSource
        import threading
        rclpy.init()
        node = rclpy.create_node("strikerobot_demo1")
        frame_source = RosFrameSource(node)
        # Spin node trong thread riêng để nhận camera + /odom + /scan cho MotionController.
        cam_exec = SingleThreadedExecutor()
        cam_exec.add_node(node)
        threading.Thread(target=cam_exec.spin, daemon=True).start()
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
        if not _ext_brain:
            engine.load()
        else:
            print(f"[demo1] Não ngoài ({os.getenv('VLA_BRAIN')}) -> bỏ nạp Qwen local (tiết kiệm VRAM).")

    detector = None
    if os.getenv("USE_YOLO", "1") == "1":
        try:
            from .yolo_detector import YoloDetector
            detector = YoloDetector()
            print(f"[demo1] YOLO detector sẵn sàng ({len(detector.names)} lớp COCO).")
        except Exception as e:
            print(f"[demo1] YOLO không dùng được ({e}); bỏ gợi ý vật cản.")

    # YOLO khoanh vùng vật thể mục tiêu (theo prompt) lên camera trực tiếp.
    annotator = None
    if detector is not None:
        try:
            from .annotator import LiveAnnotator
            annotator = LiveAnnotator(frame_source, detector)
            annotator.start()
            print("[demo1] Live YOLO annotator sẵn sàng (khoanh vùng theo prompt).")
        except Exception as e:
            print(f"[demo1] Annotator không bật được ({e}).")

    # VLM nav (VLA loop): mọi lệnh ngôn ngữ tự nhiên do VLM suy luận, KHÔNG qua Nav2.
    navloop = None
    if motion is not None and (engine.loaded or _ext_brain):
        try:
            from .navloop import NavLoopAgent, make_brain
            brain = make_brain(engine if engine.loaded else None)
            navloop = NavLoopAgent(brain, frame_source, motion, detector=detector,
                                   annotator=annotator)
            print(f"[demo1] VLM nav (VLA loop) sẵn sàng — brain="
                  f"{os.getenv('VLA_BRAIN', 'local')}.")
        except Exception as e:
            print(f"[demo1] VLM nav không bật được ({e}).")

    app = create_agent_app(frame_source, motion=motion, action=action,
                           navloop=navloop, annotator=annotator)
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
