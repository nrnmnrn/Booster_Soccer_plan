"""MJX 環境模組

提供 GPU 加速的 MuJoCo 訓練環境，包含：
- soccer_env.py: MJX 足球環境
- preprocessor_jax.py: JAX 版本的 87 維 observation preprocessor
"""

from pathlib import Path

# 資源目錄路徑
ASSETS_DIR = Path(__file__).parent / "assets"
