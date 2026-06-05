"""Photomanager 配置"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# 自定义存储路径（优先使用环境变量，否则使用默认路径）
_env_data_path = os.environ.get("PHOTOMANAGER_DATA_PATH")
if _env_data_path:
    DATA_DIR = Path(_env_data_path)
else:
    DATA_DIR = BASE_DIR / "data"

IMAGES_DIR = DATA_DIR / "images"
METADATA_PATH = DATA_DIR / "metadata.json"

# CLIP 模型
CLIP_MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# BLIP 模型
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# 分类阈值
CLASSIFICATION_THRESHOLD_HIGH = 0.30
CLASSIFICATION_THRESHOLD_LOW = 0.20

# 预设大类
DEFAULT_CATEGORIES = [
    "动物", "建筑", "风景", "人物",
    "食物", "交通", "植物", "物品",
    "艺术", "科技",
]

# 搜索
TOP_K_RESULTS = 20
SIMILARITY_SCALE = 100.0

# 服务器
HOST = "127.0.0.1"
PORT = 8000

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
