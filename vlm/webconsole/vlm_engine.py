"""VLM helpers + Qwen2.5-VL engine."""
import re
import threading

import cv2
import numpy as np
from PIL import Image as PILImage

BOX_PATTERN = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]")


def parse_boxes(text, width, height):
    """Parse box '[ymin,xmin,ymax,xmax]' (0-1000) -> [(x1,y1,x2,y2)] theo pixel."""
    boxes = []
    for ymin, xmin, ymax, xmax in BOX_PATTERN.findall(text):
        x1 = int(int(xmin) * width / 1000.0)
        y1 = int(int(ymin) * height / 1000.0)
        x2 = int(int(xmax) * width / 1000.0)
        y2 = int(int(ymax) * height / 1000.0)
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        boxes.append((x1, y1, x2, y2))
    return boxes


def draw_boxes(frame_bgr, boxes, label):
    """Vẽ box + label lên bản copy của frame, trả frame mới."""
    out = frame_bgr.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out, label, (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
        )
    return out


def encode_frame_jpeg(frame_bgr, quality=80):
    """Encode BGR frame thành JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def build_messages(pil_image, prompt):
    """Tạo messages cho Qwen chat template."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


class VLMEngine:
    """Wrapper Qwen2.5-VL với streaming token."""

    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", device=None):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

    @property
    def loaded(self):
        return self.model is not None and self.processor is not None

    def load(self):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("[VLMEngine] WARNING: chạy trên CPU, inference sẽ rất chậm.")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        print(f"[VLMEngine] Model loaded on {self.device}.")

    def stream_infer(self, frame_bgr, prompt):
        if not self.loaded:
            raise RuntimeError("VLMEngine chưa load(). Gọi load() trước.")

        from transformers import TextIteratorStreamer
        from qwen_vl_utils import process_vision_info

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)
        messages = build_messages(pil_image, prompt)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        # Kích thước ảnh model THỰC SỰ nhìn (sau smart_resize) — toạ độ bbox của
        # Qwen2.5-VL nằm trong không gian pixel này; perception dùng để scale lại.
        self.last_image_size = image_inputs[0].size if image_inputs else None
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs = dict(**inputs, max_new_tokens=256, streamer=streamer)
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        thread.join()
