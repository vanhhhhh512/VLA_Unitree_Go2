#!/usr/bin/env bash
# ============================================================================
# Tái lập MÔI TRƯỜNG NaVILA chạy được trên GPU Blackwell (RTX 50xx, sm_120).
# Đóng băng toàn bộ "port" đã làm tay ở Bước 1:
#   - conda env 'navila' Python 3.10 (accept ToS)
#   - torch 2.7 cu128 (thay torch==2.3.0 ghim -> 2.3 không có kernel sm_120)
#   - VILA (-e .) + deps pinned; transformers_replace patch
#   - flash_attn STUB bằng SDPA (Blackwell không có flash-attn)
#   - deepspeed + cuda-nvcc + CUDA_HOME (qua import check)
#   - bitsandbytes 0.49 (4-bit Blackwell) + vá integration fp16_statistics
#   - modeling_llama: decoder eager + rotary API mới
#   - tải checkpoint a8cheng/navila-llama3-8b-8f
# Idempotent: chạy lại nhiều lần không hỏng.
#
#   bash setup_navila_blackwell.sh
# ============================================================================
set -e

NAVILA_DIR="${NAVILA_DIR:-$HOME/NaVILA}"
CKPT_DIR="${CKPT_DIR:-$HOME/navila-ckpt}"
ENV="${ENV:-navila}"
TORCH_VER="${TORCH_VER:-2.7.0}"
TV_VER="${TV_VER:-0.22.0}"
CUDA_IDX="${CUDA_IDX:-cu128}"       # cu128 cho Blackwell
CUDA_NVCC="${CUDA_NVCC:-12.8}"

source ~/miniconda3/etc/profile.d/conda.sh
PY() { conda run -n "$ENV" python "$@"; }
PIP() { conda run -n "$ENV" pip "$@"; }
say() { echo -e "\n\033[1;36m== $* ==\033[0m"; }

# ---------------------------------------------------------------------------
say "0. conda ToS + tạo env $ENV (python 3.10)"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
if ! conda env list | grep -q "/envs/$ENV$"; then
    conda create -n "$ENV" python=3.10 -y
else
    echo "env $ENV đã có."
fi

# ---------------------------------------------------------------------------
say "1. clone NaVILA + nới pin torch"
if [ ! -d "$NAVILA_DIR" ]; then
    git clone https://github.com/AnjieCheng/NaVILA.git "$NAVILA_DIR"
fi
cd "$NAVILA_DIR"
# torch==2.3.0 -> torch (không ghim) để giữ 2.7 Blackwell
sed -i 's/"torch==2.3.0", "torchvision==0.18.0",/"torch", "torchvision",/' pyproject.toml || true

# ---------------------------------------------------------------------------
say "2. torch $TORCH_VER $CUDA_IDX (Blackwell)"
PIP install --upgrade pip >/dev/null
PY - <<'EOF' 2>/dev/null && echo "torch Blackwell đã OK, bỏ qua" || NEED_TORCH=1
import torch, sys
assert torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 12
print("ok")
EOF
if [ "${NEED_TORCH:-0}" = "1" ]; then
    PIP install "torch==$TORCH_VER" "torchvision==$TV_VER" --index-url "https://download.pytorch.org/whl/$CUDA_IDX"
fi

# ---------------------------------------------------------------------------
say "3. cài VILA (-e .) + deps"
PIP install -e . 2>&1 | tail -2
PIP install deepspeed==0.14.4 2>&1 | tail -1
PIP install -U "bitsandbytes>=0.45.0" 2>&1 | tail -1
conda install -n "$ENV" -c nvidia "cuda-nvcc=$CUDA_NVCC" -y 2>&1 | tail -1

SITE=$(PY -c 'import site; print(site.getsitepackages()[0])')
echo "site-packages: $SITE"

# ---------------------------------------------------------------------------
say "4. patch transformers_replace của VILA"
cp -r ./llava/train/transformers_replace/* "$SITE/transformers/" 2>/dev/null || true

# ---------------------------------------------------------------------------
say "5. tạo STUB flash_attn (SDPA)"
STUB="$SITE/flash_attn"
mkdir -p "$STUB"
cat > "$STUB/__init__.py" <<'PYEOF'
import torch, torch.nn.functional as F
def _sdpa(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False):
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return o.transpose(1, 2).contiguous()
def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    return _sdpa(q, k, v, dropout_p, softmax_scale, causal)
def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    return _sdpa(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], dropout_p, softmax_scale, causal)
def _unavail(*a, **k):
    raise RuntimeError("flash_attn varlen not supported in SDPA stub")
flash_attn_varlen_func = flash_attn_varlen_qkvpacked_func = flash_attn_unpadded_qkvpacked_func = _unavail
__version__ = "0.0.0-stub"
PYEOF
cat > "$STUB/flash_attn_interface.py" <<'PYEOF'
from . import flash_attn_func, flash_attn_qkvpacked_func, _unavail
flash_attn_varlen_func = flash_attn_varlen_qkvpacked_func = flash_attn_unpadded_qkvpacked_func = _unavail
PYEOF
cat > "$STUB/bert_padding.py" <<'PYEOF'
from . import _unavail
pad_input = unpad_input = index_first_axis = _unavail
PYEOF

# ---------------------------------------------------------------------------
say "6. patch code (idempotent) — intern import, modeling_llama eager+rotary, bnb"
NAVILA_DIR="$NAVILA_DIR" SITE="$SITE" PY - <<'PYEOF'
import os
nav = os.environ["NAVILA_DIR"]; site = os.environ["SITE"]

def patch(path, old, new, tag):
    s = open(path).read()
    if new in s:
        print(f"  [skip] {tag} (đã patch)"); return
    assert old in s, f"KHÔNG thấy mẫu cho {tag} ở {path}"
    open(path, "w").write(s.replace(old, new)); print(f"  [ok]  {tag}")

# 6a. intern flash_attention import an toàn
p = f"{nav}/llava/model/multimodal_encoder/intern/flash_attention.py"
patch(p,
'''try:  # v1
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
except:  # v2
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func

from flash_attn.bert_padding import pad_input, unpad_input''',
'''try:
    try:
        from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
    except Exception:
        from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
except Exception:
    flash_attn_unpadded_qkvpacked_func = None
    pad_input = unpad_input = None''',
"intern flash_attention import")

# 6b. modeling_llama: decoder self_attn -> eager
ml = f"{site}/transformers/models/llama/modeling_llama.py"
patch(ml,
'''        self.self_attn = (
            # LlamaAttention(config=config)
            LlamaFlashAttention2(config=config)
        )''',
'''        self.self_attn = (
            LlamaAttention(config=config)  # PATCH Blackwell: eager thay flash
            # LlamaFlashAttention2(config=config)
        )''',
"llama decoder -> eager")

# 6c. modeling_llama: eager rotary API mới
patch(ml,
'''        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)''',
'''        cos, sin = self.rotary_emb(value_states, position_ids)  # PATCH: rotary API mới
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)''',
"llama eager rotary")

# 6d. bnb integration: nhận fp16_statistics
bp = f"{site}/transformers/integrations/bitsandbytes.py"
patch(bp,
"def set_module_quantized_tensor_to_device(module, tensor_name, device, value=None, quantized_stats=None):",
"def set_module_quantized_tensor_to_device(module, tensor_name, device, value=None, quantized_stats=None, fp16_statistics=None):\n    if quantized_stats is None and fp16_statistics is not None:\n        quantized_stats = fp16_statistics",
"bnb fp16_statistics alias")
print("patch xong.")
PYEOF

# ---------------------------------------------------------------------------
say "7. tải checkpoint + ép attn eager trong config"
if [ ! -f "$CKPT_DIR/config.json" ]; then
    PATH="$HOME/.local/bin:$PATH" hf download a8cheng/navila-llama3-8b-8f --local-dir "$CKPT_DIR"
else
    echo "checkpoint đã có ở $CKPT_DIR"
fi
CKPT_DIR="$CKPT_DIR" PY - <<'PYEOF'
import os, json
for name in ("llm/config.json", "config.json"):
    p = os.path.join(os.environ["CKPT_DIR"], name)
    if os.path.exists(p):
        d = json.load(open(p)); d["_attn_implementation"] = "eager"; d["attn_implementation"] = "eager"
        json.dump(d, open(p, "w"), indent=2); print("  eager ->", name)
PYEOF

# ---------------------------------------------------------------------------
say "8. VERIFY: torch Blackwell + import llava"
PY -c "import torch; print('torch', torch.__version__, 'sm', torch.cuda.get_device_capability(0))"
CUDA_HOME="$HOME/miniconda3/envs/$ENV" PY -c "import llava; print('llava import OK')" 2>&1 | tail -1

echo -e "\n\033[1;32m== XONG. Chạy thử: ==\033[0m"
echo "cd $NAVILA_DIR && CUDA_HOME=\$HOME/miniconda3/envs/$ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\"
echo "  conda run -n $ENV --no-capture-output python navila_probe.py --model-path $CKPT_DIR --load-4bit --frames-dir ~/frames8"
