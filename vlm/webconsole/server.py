"""FastAPI server cho VLM Web Console."""
import os
import base64
import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

from .vlm_engine import parse_boxes, draw_boxes, encode_frame_jpeg

WEBUI = Path(__file__).parent / "webui" / "index.html"


def create_app(source, engine):
    app = FastAPI()
    app.state.busy = False

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return WEBUI.read_text(encoding="utf-8")

    @app.get("/status")
    async def status():
        return JSONResponse({
            "connected": bool(source.is_connected),
            "mock": bool(source.is_mock),
        })

    @app.get("/video_feed")
    async def video_feed():
        async def gen():
            while True:
                frame = source.get_latest_frame()
                if frame is not None:
                    jpg = encode_frame_jpeg(frame)
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + jpg + b"\r\n")
                await asyncio.sleep(0.05)  # ~20 fps
        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                prompt = (data or {}).get("prompt", "").strip()
                if not prompt:
                    continue
                if app.state.busy:
                    await websocket.send_json({"type": "busy"})
                    continue
                app.state.busy = True
                try:
                    await _handle_prompt(websocket, source, engine, prompt)
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
                finally:
                    app.state.busy = False
        except WebSocketDisconnect:
            return

    return app


async def _handle_prompt(websocket, source, engine, prompt):
    frame = source.get_latest_frame()
    if frame is None:
        await websocket.send_json(
            {"type": "error", "message": "Chưa có ảnh từ robot."}
        )
        return

    snapshot = frame.copy()
    loop = asyncio.get_event_loop()
    full = []

    # Bơm generator (blocking) qua executor, đẩy token ra queue của event loop.
    def produce(q):
        try:
            for chunk in engine.stream_infer(snapshot, prompt):
                asyncio.run_coroutine_threadsafe(q.put(("token", chunk)), loop)
            asyncio.run_coroutine_threadsafe(q.put(("end", None)), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(q.put(("err", str(e))), loop)

    q: asyncio.Queue = asyncio.Queue()
    loop.run_in_executor(None, produce, q)

    while True:
        kind, payload = await q.get()
        if kind == "token":
            full.append(payload)
            await websocket.send_json({"type": "token", "text": payload})
        elif kind == "err":
            await websocket.send_json({"type": "error", "message": payload})
            return
        else:  # end
            break

    text = "".join(full)
    h, w = snapshot.shape[:2]
    boxes = parse_boxes(text, w, h)
    if boxes:
        drawn = draw_boxes(snapshot, boxes, prompt)
        b64 = base64.b64encode(encode_frame_jpeg(drawn)).decode("ascii")
        await websocket.send_json({"type": "image", "data": b64})
    await websocket.send_json({"type": "done"})


def main():
    import uvicorn
    from .frame_source import FrameSource
    from .vlm_engine import VLMEngine

    source = FrameSource(os.getenv("ROBOT_IP"))
    engine = VLMEngine()
    if os.getenv("VLM_SKIP_MODEL") == "1":
        print("[server] VLM_SKIP_MODEL=1 -> bỏ qua load model (chỉ xem GUI). "
              "Gửi lệnh sẽ báo lỗi cho tới khi bỏ cờ này.")
    else:
        print("[server] Loading VLM model... (lần đầu sẽ tải ~vài GB)")
        engine.load()

    app = create_app(source, engine)

    @app.on_event("startup")
    async def _startup():
        await source.start()

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
