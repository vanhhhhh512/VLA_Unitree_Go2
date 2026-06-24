#!/usr/bin/env python3
import asyncio
import cv2
import torch
import numpy as np
import re
import os
import sys
import threading
from PIL import Image as PILImage

# Ensure go2_robot_sdk can be imported
sys.path.insert(0, "/home/dsc-labs/ros2_ws/src/go2_robot_sdk")

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False

from go2_robot_sdk.infrastructure.webrtc.go2_connection import Go2Connection

class VLMProcessor:
    def __init__(self, prompt="phone"):
        self.prompt = prompt
        
        if not HAS_QWEN:
            print("Missing transformers or qwen_vl_utils. Please pip install them.")
            sys.exit(1)
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Qwen2.5-VL-3B-Instruct model on {self.device}...")
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        print("Model loaded successfully.")
        
        self.latest_frame = None
        self.display_frame = None
        self.is_processing = False
        
        # Start display thread
        self.display_thread = threading.Thread(target=self._display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()
        
    def process_frame(self, frame):
        # frame is a BGR numpy array
        if self.is_processing:
            return # Skip if already processing
            
        self.is_processing = True
        
        def run_inference():
            try:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = PILImage.fromarray(rgb_image)
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": f"Detect {self.prompt}."},
                        ],
                    }
                ]
                
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.device)
                
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
                matches = re.findall(pattern, output_text)
                
                h, w, _ = frame.shape
                drawn_frame = frame.copy()
                
                for match in matches:
                    ymin, xmin, ymax, xmax = [int(x) for x in match]
                    
                    x1 = int(xmin * w / 1000.0)
                    y1 = int(ymin * h / 1000.0)
                    x2 = int(xmax * w / 1000.0)
                    y2 = int(ymax * h / 1000.0)
                    
                    cv2.rectangle(drawn_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(drawn_frame, self.prompt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                self.display_frame = drawn_frame
                
            except Exception as e:
                print(f"Error processing image: {e}")
            finally:
                self.is_processing = False
                
        # Run inference in a background thread
        threading.Thread(target=run_inference, daemon=True).start()

    def _display_loop(self):
        cv2.namedWindow("VLM Detection", cv2.WINDOW_NORMAL)
        while True:
            # Show the frame with detections if available, otherwise just show the latest raw frame
            frame_to_show = self.display_frame if self.display_frame is not None else self.latest_frame
            
            if frame_to_show is not None:
                cv2.imshow("VLM Detection", frame_to_show)
                
            if cv2.waitKey(30) & 0xFF == ord('q'):
                print("Exiting...")
                os._exit(0)

async def main():
    robot_ip = os.getenv("ROBOT_IP")
    if not robot_ip:
        print("Error: Please set the ROBOT_IP environment variable.")
        print("Example: export ROBOT_IP='192.168.123.161'")
        return
        
    prompt = os.getenv("VLM_PROMPT", "phone")
    
    print(f"Initializing VLM Processor for prompt: '{prompt}'...")
    vlm = VLMProcessor(prompt=prompt)
    
    async def on_video_frame(track, robot_id):
        print("WebRTC Video stream connected!")
        while True:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                vlm.latest_frame = img
                
                # Offload VLM processing
                vlm.process_frame(img)
            except Exception as e:
                print(f"Video frame error or stream closed: {e}")
                break
                
    def on_validated(robot_num):
        print(f"Robot {robot_num} validated.")
        # Turn off traffic saving to ensure video stream is stable
        asyncio.create_task(conn.disableTrafficSaving(True))
        
        # Subscribe to all RTC topics to ensure video and data start flowing
        import json
        from go2_robot_sdk.domain.constants import RTC_TOPIC
        try:
            for topic in RTC_TOPIC.values():
                conn.data_channel.send(json.dumps({"type": "subscribe", "topic": topic}))
            print("Subscribed to all WebRTC topics.")
        except Exception as e:
            print(f"Failed to subscribe to topics: {e}")

    print(f"Connecting to Unitree Go2 at {robot_ip}...")
    conn = Go2Connection(
        robot_ip=robot_ip,
        robot_num=0,
        token="",
        on_validated=on_validated,
        on_video_frame=on_video_frame,
        decode_lidar=False
    )
    
    await conn.connect()
    
    try:
        # Keep the event loop running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
