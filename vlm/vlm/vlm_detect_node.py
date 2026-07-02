#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import re

# Try importing Qwen modules
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False

from PIL import Image as PILImage

class VLMDetectNode(Node):
    def __init__(self):
        super().__init__('vlm_detect_node')
        
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('prompt', 'phone')
        
        self.image_topic = self.get_parameter('image_topic').value
        self.prompt = self.get_parameter('prompt').value
        
        self.bridge = CvBridge()
        
        if not HAS_QWEN:
            self.get_logger().error("Missing transformers or qwen_vl_utils. Please run: pip install transformers qwen-vl-utils torch torchvision")
            return
            
        self.get_logger().info("Loading Qwen2.5-VL-3B-Instruct model...")
        
        # Load model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        self.get_logger().info(f"Model loaded on {self.device}. Subscribing to {self.image_topic}")
        
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )
        
        self.publisher = self.create_publisher(Image, '/vlm/debug_image', 10)
        
        # Process one frame at a time
        self.is_processing = False

    def image_callback(self, msg):
        if self.is_processing:
            return
        self.is_processing = True
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            current_prompt = self.get_parameter('prompt').value
            
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
            
            # Request bounding box using "<|box_2d|>"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": f"Detect {current_prompt}."},
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
            
            self.get_logger().info(f"Model output: {output_text}")
            
            # Find coordinates
            pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
            matches = re.findall(pattern, output_text)
            
            h, w, _ = cv_image.shape
            
            for match in matches:
                ymin, xmin, ymax, xmax = [int(x) for x in match]
                
                x1 = int(xmin * w / 1000.0)
                y1 = int(ymin * h / 1000.0)
                x2 = int(xmax * w / 1000.0)
                y2 = int(ymax * h / 1000.0)
                
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_image, current_prompt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            self.publisher.publish(out_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
        finally:
            self.is_processing = False

def main(args=None):
    rclpy.init(args=args)
    node = VLMDetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
