"""BƯỚC 1 — Kiểm tra khả thi NaVILA trên 5070 (KHÔNG robot, KHÔNG lidar, KHÔNG ROS).

Chỉ trả lời 2 câu hỏi cửa ải:
  1) Model 8B có NẠP nổi trên 16GB VRAM không? (thử fp16, và 4-bit nếu tràn)
  2) 1 bước suy luận (8 frame + câu lệnh) MẤT BAO LÂU + TỐN BAO NHIÊU VRAM?

Đặt file này TRONG repo NaVILA đã clone (để `from llava...` import được), chạy bằng conda env 'navila'.

    conda run -n navila python navila_probe.py --model-path ~/navila-ckpt
    conda run -n navila python navila_probe.py --model-path ~/navila-ckpt --load-4bit
    # ảnh thật: bỏ 8 ảnh .jpg vào 1 thư mục rồi:
    conda run -n navila python navila_probe.py --model-path ~/navila-ckpt --frames-dir ~/frames8

Nếu prompt/conv-mode chưa đúng chuẩn NaVILA thì OUTPUT có thể lạ — KHÔNG sao, mục tiêu
Bước 1 là ĐO VRAM + TỐC ĐỘ (cửa ải khả thi), tinh chỉnh prompt để Bước 2.
"""
import os
import time
import glob
import argparse

import torch
from PIL import Image


def log(msg):
    print(f"[probe] {msg}", flush=True)


def vram(tag):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1e9
        p = torch.cuda.max_memory_allocated() / 1e9
        log(f"VRAM {tag}: đang dùng {a:.2f} GB | đỉnh {p:.2f} GB")


def load_frames(frames_dir, n):
    if frames_dir:
        paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")) +
                       glob.glob(os.path.join(frames_dir, "*.png")))[:n]
        if paths:
            log(f"Dùng {len(paths)} ảnh thật từ {frames_dir}")
            imgs = [Image.open(p).convert("RGB") for p in paths]
            while len(imgs) < n:            # thiếu thì lặp ảnh cuối cho đủ n
                imgs.append(imgs[-1])
            return imgs
    log(f"Không có --frames-dir -> tạo {n} frame xám giả (chỉ để đo tài nguyên).")
    return [Image.new("RGB", (384, 384), (127, 127, 127)) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="thư mục checkpoint hoặc HF id")
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--frames-dir", default=None, help="thư mục chứa 8 ảnh (không có -> ảnh giả)")
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--conv-mode", default=os.getenv("NAVILA_CONV", "llama_3"))
    ap.add_argument("--instruction", default="Move to the water bottle then stop.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--load-4bit", action="store_true", help="lượng tử 4-bit cho vừa 16GB")
    ap.add_argument("--load-8bit", action="store_true")
    args = ap.parse_args()

    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
    from llava.conversation import conv_templates
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    model_name = get_model_name_from_path(args.model_path)
    log(f"Nạp model '{model_name}' (4bit={args.load_4bit}, 8bit={args.load_8bit})…")
    t0 = time.time()
    # Ký hiệu builder VILA có thể khác bản -> thử kèm 4/8-bit, tràn thì fallback.
    kw = {"torch_dtype": torch.float16}
    if args.load_4bit:
        kw["load_4bit"] = True
    if args.load_8bit:
        kw["load_8bit"] = True
    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, model_name, args.model_base, **kw)
    except TypeError as e:
        log(f"builder không nhận kwarg lượng tử ({e}); nạp mặc định fp16.")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, model_name, args.model_base)
    log(f"NẠP XONG sau {time.time()-t0:.1f}s. context_len={context_len}")
    vram("sau khi nạp model")

    images = load_frames(args.frames_dir, args.num_frames)
    images_tensor = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16)
    log(f"images_tensor shape = {tuple(images_tensor.shape)}")

    interleaved = (DEFAULT_IMAGE_TOKEN + "\n") * (args.num_frames - 1)
    qs = (
        "Imagine you are a robot programmed for navigation tasks. You have been given a video "
        f'of historical observations {interleaved}, and current observation {DEFAULT_IMAGE_TOKEN}\n. '
        f'Your assigned task is: "{args.instruction}" '
        "Analyze this series of images to decide your next action, which could be turning left or right by a specific "
        "degree, moving forward a certain distance, or stop if the task is completed."
    )
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    log("Chạy suy luận thử (warm-up)…")
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    with torch.inference_mode():
        _ = model.generate(input_ids, images=images_tensor, do_sample=False,
                           max_new_tokens=args.max_new_tokens, use_cache=True,
                           pad_token_id=tokenizer.eos_token_id)

    log("Đo latency 3 lần…")
    lat = []
    for i in range(3):
        t = time.time()
        with torch.inference_mode():
            out = model.generate(input_ids, images=images_tensor, do_sample=False,
                                 max_new_tokens=args.max_new_tokens, use_cache=True,
                                 pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        lat.append(time.time() - t)
    text = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

    vram("đỉnh khi suy luận")
    log(f"Latency mỗi bước: {[round(x,2) for x in lat]} s (trung bình {sum(lat)/len(lat):.2f}s)")
    log(f"OUTPUT model:\n---\n{text}\n---")
    log("XONG BƯỚC 1. Xem: (a) VRAM đỉnh có < 16GB không, (b) latency mỗi bước.")


if __name__ == "__main__":
    main()
