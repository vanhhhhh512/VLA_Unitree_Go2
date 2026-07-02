import os
import sys

# Đảm bảo `vlm.webconsole` import được khi chạy pytest từ bất kỳ đâu.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
