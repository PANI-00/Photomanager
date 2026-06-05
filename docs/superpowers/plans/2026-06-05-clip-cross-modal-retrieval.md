# CLIP 跨模态检索系统 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建基于 CLIP + BLIP 的 Web 跨模态检索系统，支持图片导入/管理/分类/自然语言搜索

**Architecture:** FastAPI 后端提供 REST API，原生前端单页应用，CLIP 负责特征提取与检索，BLIP 负责图像描述生成，本地文件持久化

**Tech Stack:** Python 3.14, FastAPI, CLIP ViT-B/32, BLIP-base, vanilla HTML/CSS/JS

---

## 文件结构

```
Zero-Shot-Classification-and-Retrieval-System-Based-on-CLIP/
├── config.py               # 配置文件（阈值、预设类别、路径）
├── models/
│   ├── __init__.py
│   ├── clip_service.py     # CLIP 模型封装
│   └── caption_service.py  # BLIP 模型封装
├── services/
│   ├── __init__.py
│   ├── classifier.py       # 分类引擎
│   └── search.py           # 检索服务
├── main.py                 # FastAPI 应用入口 + 路由
├── data/
│   ├── images/             # 图片存储目录（gitignore）
│   └── metadata.json       # 元数据持久化
├── static/
│   ├── index.html          # 前端页面
│   ├── style.css           # 样式
│   └── app.js              # 交互逻辑
├── requirements.txt        # Python 依赖
└── README.md               # 项目说明
```

---

### Task 1: 配置文件 config.py

**Files:**
- Create: `config.py`

- [ ] **Step 1: 编写 config.py**

```python
"""系统配置"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
METADATA_PATH = DATA_DIR / "metadata.json"

# CLIP 模型
CLIP_MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# BLIP 模型
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# 分类阈值
CLASSIFICATION_THRESHOLD_HIGH = 0.30   # 明确归入
CLASSIFICATION_THRESHOLD_LOW = 0.20    # 边缘匹配，低于此值创建新类别

# 预设大类（中/英文皆可，CLIP 会自动理解语义）
DEFAULT_CATEGORIES = [
    "动物", "建筑", "风景", "人物",
    "食物", "交通", "植物", "物品",
    "艺术", "科技",
]

# 搜索
TOP_K_RESULTS = 20
SIMILARITY_SCALE = 100.0  # 相似度放大系数

# 服务器
HOST = "127.0.0.1"
PORT = 8000

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
```

- [ ] **Step 2: 检查文件内容正确**

---

### Task 2: 依赖文件 requirements.txt

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: 编写 requirements.txt**

```txt
fastapi>=0.115.0
uvicorn>=0.34.0
Pillow>=11.0.0
requests>=2.32.0
git+https://github.com/openai/CLIP.git
torch>=2.5.0
transformers>=4.47.0
python-multipart>=0.0.19
aiofiles>=24.1.0
```

- [ ] **Step 2: 确认依赖清单完整**

---

### Task 3: CLIP 服务 models/clip_service.py

**Files:**
- Create: `models/__init__.py`（空文件）
- Create: `models/clip_service.py`

- [ ] **Step 1: 编写 models/__init__.py**

```python
```
（空文件）

- [ ] **Step 2: 编写 models/clip_service.py**

```python
"""CLIP 模型封装：图像/文本特征提取"""

import clip
import torch
from PIL import Image
from config import CLIP_MODEL_NAME, DEVICE


class CLIPService:
    """单例模式封装 CLIP 模型，提供特征提取接口。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._preprocess = None
        return cls._instance

    @property
    def model(self):
        if self._model is None:
            print("正在加载 CLIP 模型...")
            self._model, self._preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
            self._model.eval()
            print("CLIP 模型加载完成。")
        return self._model

    @property
    def preprocess(self):
        if self._preprocess is None:
            _ = self.model  # 触发加载
        return self._preprocess

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """提取单张图片的 L2 归一化特征向量 (1, 512)"""
        image_input = self.preprocess(image).unsqueeze(0).to(DEVICE)
        features = self.model.encode_image(image_input)
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu()

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """批量提取图片特征 (N, 512)"""
        image_inputs = torch.stack([self.preprocess(img) for img in images]).to(DEVICE)
        features = self.model.encode_image(image_inputs)
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu()

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """批量编码文本 (N, 512)"""
        text_tokens = clip.tokenize(texts).to(DEVICE)
        features = self.model.encode_text(text_tokens)
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu()

    @torch.no_grad()
    def similarity(self, image_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """计算图像与文本的余弦相似度矩阵"""
        return 100.0 * image_feat @ text_feat.T
```

- [ ] **Step 3: 确认文件创建完毕**

---

### Task 4: BLIP 描述服务 models/caption_service.py

**Files:**
- Create: `models/caption_service.py`

- [ ] **Step 1: 编写 models/caption_service.py**

```python
"""BLIP 图像描述生成封装"""

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from config import BLIP_MODEL_NAME, DEVICE


class CaptionService:
    """使用 BLIP 为图片生成自然语言描述。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._processor = None
            cls._instance._model = None
        return cls._instance

    @property
    def processor(self):
        if self._processor is None:
            print("正在加载 BLIP 模型...")
            self._processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
            self._model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(DEVICE)
            self._model.eval()
            print("BLIP 模型加载完成。")
        return self._processor

    @property
    def model(self):
        if self._model is None:
            _ = self.processor
        return self._model

    @torch.no_grad()
    def generate_caption(self, image: Image.Image) -> str:
        """为单张图片生成英文描述"""
        inputs = self.processor(image, return_tensors="pt").to(DEVICE)
        out = self.model.generate(**inputs, max_new_tokens=50)
        return self.processor.decode(out[0], skip_special_tokens=True)
```

- [ ] **Step 2: 确认文件创建完毕**

---

### Task 5: 分类引擎 services/classifier.py

**Files:**
- Create: `services/__init__.py`（空文件）
- Create: `services/classifier.py`

- [ ] **Step 1: 编写 services/__init__.py**

```python
```
（空文件）

- [ ] **Step 2: 编写 services/classifier.py**

```python
"""分类引擎：混合模式——预设大类 + 自动创建新类别"""

import torch
from config import (
    DEFAULT_CATEGORIES,
    CLASSIFICATION_THRESHOLD_HIGH,
    CLASSIFICATION_THRESHOLD_LOW,
)


def classify_image(
    image_feature: torch.Tensor,
    clip_service,
    existing_categories: list[str] | None = None,
) -> tuple[str, float, str]:
    """
    对图片特征进行分类。

    返回: (大类名称, 相似度分数, 匹配类型)
    匹配类型: "exact" / "边缘" / "新类别"
    """
    # 合并预设大类 + 已有自动创建的类别
    all_categories = list(DEFAULT_CATEGORIES)
    if existing_categories:
        for cat in existing_categories:
            if cat not in all_categories:
                all_categories.append(cat)

    # 计算图片与各类别名称的 CLIP 相似度
    text_features = clip_service.encode_text(all_categories)
    scores = image_feat @ text_features.T  # (1, N)

    max_score, max_idx = scores[0].max(dim=-1)
    max_score = max_score.item()

    category = all_categories[max_idx]

    if max_score >= CLASSIFICATION_THRESHOLD_HIGH:
        match_type = "exact"
    elif max_score >= CLASSIFICATION_THRESHOLD_LOW:
        match_type = "边缘"
    else:
        category = f"{category}_自动类别"
        match_type = "新类别"

    return category, max_score, match_type
```

- [ ] **Step 3: 确认文件创建完毕**

---

### Task 6: 检索服务 services/search.py

**Files:**
- Create: `services/search.py`

- [ ] **Step 1: 编写 services/search.py**

```python
"""检索服务：自然语言搜索匹配图片"""

from config import TOP_K_RESULTS


def search_images(
    query: str,
    clip_service,
    metadata_list: list[dict],
) -> list[dict]:
    """
    自然语言检索图片。

    参数:
        query: 用户输入的搜索文本
        clip_service: CLIP 服务实例
        metadata_list: 图片元数据列表

    返回:
        [{id, filename, caption, category, similarity, source, url}, ...]
    """
    if not metadata_list:
        return []

    # 编码查询文本
    query_feat = clip_service.encode_text([query])  # (1, 512)

    # 堆叠所有图片特征
    all_features = []
    for item in metadata_list:
        all_features.append(item["feature"])
    all_features = __import__("torch").cat(all_features, dim=0)  # (N, 512)

    # 余弦相似度
    similarity_scores = 100.0 * query_feat @ all_features.T  # (1, N)

    # 排序取 Top-K
    values, indices = similarity_scores[0].topk(k=min(TOP_K_RESULTS, len(metadata_list)))

    results = []
    for val, idx in zip(values, indices):
        item = metadata_list[idx]
        results.append({
            "id": item["id"],
            "filename": item["filename"],
            "caption": item.get("caption", ""),
            "category": item.get("category", "未分类"),
            "similarity": round(val.item(), 2),
            "source": item.get("source", "unknown"),
            "url": f"/api/images/file/{item['id']}",
        })

    return results
```

- [ ] **Step 2: 确认文件创建完毕**

---

### Task 7: FastAPI 主入口 main.py

**Files:**
- Create: `main.py`

- [ ] **Step 1: 编写 main.py**

```python
"""CLIP 跨模态检索系统 — FastAPI 应用入口"""

import json
import uuid
from io import BytesIO
from pathlib import Path

import aiofiles
import requests
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from config import IMAGES_DIR, METADATA_PATH
from models.clip_service import CLIPService
from models.caption_service import CaptionService
from services.classifier import classify_image
from services.search import search_images

app = FastAPI(title="CLIP 跨模态检索系统")

clip_service = CLIPService()
caption_service = CaptionService()


# ==================== 数据持久化 ====================

def _load_metadata() -> list[dict]:
    """从磁盘加载元数据"""
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 反序列化时恢复 feature 为 tensor
            for item in data:
                if "feature" in item and isinstance(item["feature"], list):
                    item["feature"] = torch.tensor(item["feature"]).unsqueeze(0)
            return data
    return []


def _save_metadata(data: list[dict]) -> None:
    """保存元数据到磁盘（序列化 tensor 为列表）"""
    serializable = []
    for item in data:
        item_copy = dict(item)
        if "feature" in item_copy and isinstance(item_copy["feature"], torch.Tensor):
            item_copy["feature"] = item_copy["feature"].squeeze(0).tolist()
        serializable.append(item_copy)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


# 启动时加载元数据
metadata_store = _load_metadata()


# ==================== 图片处理 ====================

def _process_and_index_image(image: Image.Image, filename: str, source: str) -> dict:
    """处理单张图片：提取特征 → 生成描述 → 分类 → 返回元数据"""
    # 保存图片
    image_id = str(uuid.uuid4())
    ext = Path(filename).suffix if Path(filename).suffix else ".jpg"
    image_path = IMAGES_DIR / f"{image_id}{ext}"
    image.save(image_path, format="JPEG" if ext.lower() in (".jpg", ".jpeg") else "PNG")

    # CLIP 提取特征
    image_feat = clip_service.encode_image(image)

    # BLIP 生成描述
    caption = caption_service.generate_caption(image)

    # 分类
    existing_categories = list({
        item.get("category", "") for item in metadata_store
        if item.get("category", "") not in ["animal", "building", "landscape",
                                              "person", "food", "transportation",
                                              "plant", "object", "art", "tech"]
    })
    category, score, match_type = classify_image(
        image_feat, clip_service, existing_categories
    )

    metadata = {
        "id": image_id,
        "filename": filename,
        "source": source,
        "caption": caption,
        "category": category,
        "match_type": match_type,
        "similarity_score": round(score, 3),
        "feature": image_feat,
        "created_at": __import__("datetime").datetime.now().isoformat(),
    }
    return metadata


# ==================== API 路由 ====================

@app.get("/api/stats")
async def get_stats():
    """系统统计信息"""
    categories = {}
    for item in metadata_store:
        cat = item.get("category", "未分类")
        categories[cat] = categories.get(cat, 0) + 1

    return {
        "total_images": len(metadata_store),
        "total_categories": len(categories),
        "categories": categories,
    }


@app.get("/api/categories")
async def get_categories():
    """获取所有类别列表"""
    categories_set = set()
    for item in metadata_store:
        categories_set.add(item.get("category", "未分类"))
    return {"categories": sorted(categories_set)}


@app.get("/api/images")
async def get_images(category: str | None = None):
    """获取图片列表，可按分类筛选"""
    results = []
    for item in metadata_store:
        if category and item.get("category") != category:
            continue
        results.append({
            "id": item["id"],
            "filename": item["filename"],
            "caption": item.get("caption", ""),
            "category": item.get("category", "未分类"),
            "score": item.get("similarity_score", 0),
            "source": item.get("source", "unknown"),
            "created_at": item.get("created_at", ""),
            "url": f"/api/images/file/{item['id']}",
        })
    return {"images": results, "total": len(results)}


@app.get("/api/images/file/{image_id}")
async def get_image_file(image_id: str):
    """返回图片文件"""
    for item in metadata_store:
        if item["id"] == image_id:
            # 查找对应的文件
            for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
                path = IMAGES_DIR / f"{image_id}{ext}"
                if path.exists():
                    return FileResponse(str(path))
            raise HTTPException(status_code=404, detail="图片文件未找到")
    raise HTTPException(status_code=404, detail="图片 ID 不存在")


@app.post("/api/images/upload")
async def upload_images(files: list[UploadFile] = File(...)):
    """上传单张/多张图片"""
    results = []
    errors = []

    for file in files:
        try:
            contents = await file.read()
            image = Image.open(BytesIO(contents)).convert("RGB")
            metadata = _process_and_index_image(image, file.filename, "upload")
            metadata_store.append(metadata)
            results.append({
                "id": metadata["id"],
                "filename": file.filename,
                "caption": metadata["caption"],
                "category": metadata["category"],
            })
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})

    _save_metadata(metadata_store)
    return {"imported": len(results), "errors": errors, "results": results}


class URLInput(BaseModel):
    urls: list[str]


@app.post("/api/images/from-url")
async def import_from_url(data: URLInput):
    """从 URL 导入图片"""
    results = []
    errors = []

    for url in data.urls:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            image = Image.open(BytesIO(resp.content)).convert("RGB")
            metadata = _process_and_index_image(image, url.rsplit("/", 1)[-1], "url")
            metadata["source_url"] = url
            metadata_store.append(metadata)
            results.append({
                "id": metadata["id"],
                "url": url,
                "caption": metadata["caption"],
                "category": metadata["category"],
            })
        except Exception as e:
            errors.append({"url": url, "error": str(e)})

    _save_metadata(metadata_store)
    return {"imported": len(results), "errors": errors, "results": results}


@app.delete("/api/images/{image_id}")
async def delete_image(image_id: str):
    """删除图片"""
    for i, item in enumerate(metadata_store):
        if item["id"] == image_id:
            # 删除文件
            for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
                path = IMAGES_DIR / f"{image_id}{ext}"
                if path.exists():
                    path.unlink()
            metadata_store.pop(i)
            _save_metadata(metadata_store)
            return {"deleted": image_id}

    raise HTTPException(status_code=404, detail="图片 ID 不存在")


class SearchQuery(BaseModel):
    query: str


@app.post("/api/search")
async def search(data: SearchQuery):
    """自然语言检索图片"""
    if not data.query.strip():
        return {"results": [], "query": data.query}

    if not metadata_store:
        return {"results": [], "query": data.query}

    results = search_images(data.query, clip_service, metadata_store)
    return {"results": results, "query": data.query, "total": len(results)}


# ==================== 静态文件服务 ====================

app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT
    print(f"启动服务: http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
```

- [ ] **Step 2: 确认文件创建完毕**

---

### Task 8: 前端 HTML static/index.html

**Files:**
- Create: `static/index.html`

- [ ] **Step 1: 编写 static/index.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CLIP 跨模态检索系统</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>

<header>
    <div class="header-left">
        <h1>🔍 CLIP 跨模态检索系统</h1>
    </div>
    <div class="header-right">
        <span id="stats-display">加载中...</span>
    </div>
</header>

<main>
    <!-- 工具栏 -->
    <div class="toolbar">
        <button onclick="document.getElementById('file-input').click()">📤 上传图片</button>
        <input type="file" id="file-input" multiple accept="image/*" style="display:none">
        <button onclick="showUrlImport()">🔗 从 URL 导入</button>
        <button onclick="refreshAll()">🔄 刷新</button>
    </div>

    <!-- URL 导入弹窗 -->
    <div id="url-modal" class="modal" style="display:none">
        <div class="modal-content">
            <span class="close" onclick="closeUrlImport()">&times;</span>
            <h3>从 URL 导入图片</h3>
            <textarea id="url-input" rows="4" placeholder="每行一个图片链接&#10;https://example.com/image1.jpg&#10;https://example.com/image2.png"></textarea>
            <button onclick="importUrls()">导入</button>
        </div>
    </div>

    <!-- Loading -->
    <div id="loading" class="loading" style="display:none">
        <div class="spinner"></div>
        <span id="loading-text">处理中...</span>
    </div>

    <!-- 主内容区 -->
    <div class="content">
        <!-- 左侧分类 -->
        <aside class="sidebar">
            <h3>📁 分类</h3>
            <ul id="category-list">
                <li class="category-item active" data-category="" onclick="filterByCategory(this, '')">
                    📋 全部
                </li>
            </ul>
        </aside>

        <!-- 右侧图片展示 -->
        <section class="gallery-section">
            <!-- 搜索栏 -->
            <div class="search-bar">
                <input type="text" id="search-input" placeholder="输入自然语言搜索图片..." onkeydown="if(event.key==='Enter') searchImages()">
                <button onclick="searchImages()">🔎 搜索</button>
                <button id="clear-search" class="btn-clear" style="display:none" onclick="clearSearch()">✕ 清除</button>
            </div>

            <div id="gallery" class="gallery">
                <div class="empty-state">暂无图片，请先导入。</div>
            </div>
        </section>
    </div>
</main>

<script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: 确认文件创建完毕**

---

### Task 9: 前端样式 static/style.css

**Files:**
- Create: `static/style.css`

- [ ] **Step 1: 编写 static/style.css**

```css
/* ==================== 全局 ==================== */
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
    background: #f0f2f5;
    color: #1a1a2e;
    min-height: 100vh;
}

/* ==================== 头部 ==================== */
header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white;
    padding: 16px 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
header h1 { font-size: 20px; font-weight: 600; }
#stats-display { font-size: 14px; opacity: 0.9; }

/* ==================== 工具栏 ==================== */
.toolbar {
    padding: 16px 24px;
    background: white;
    border-bottom: 1px solid #e0e0e0;
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}
.toolbar button {
    padding: 8px 18px;
    background: #0f3460;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    transition: background 0.2s;
}
.toolbar button:hover { background: #1a5276; }

/* ==================== 主体 ==================== */
.content {
    display: flex;
    height: calc(100vh - 140px);
}

/* ==================== 侧栏 ==================== */
.sidebar {
    width: 200px;
    background: white;
    border-right: 1px solid #e0e0e0;
    padding: 16px;
    overflow-y: auto;
    flex-shrink: 0;
}
.sidebar h3 { font-size: 14px; margin-bottom: 12px; color: #555; }
.category-item {
    padding: 8px 12px;
    cursor: pointer;
    border-radius: 6px;
    font-size: 14px;
    margin-bottom: 4px;
    list-style: none;
    transition: background 0.2s;
}
.category-item:hover { background: #e3f0ff; }
.category-item.active { background: #0f3460; color: white; }
.category-count {
    float: right;
    background: rgba(0,0,0,0.1);
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 12px;
}
.category-item.active .category-count { background: rgba(255,255,255,0.2); }

/* ==================== 画廊 ==================== */
.gallery-section {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 16px 24px;
    overflow: hidden;
}

/* 搜索栏 */
.search-bar {
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
}
.search-bar input {
    flex: 1;
    padding: 10px 16px;
    border: 2px solid #ddd;
    border-radius: 8px;
    font-size: 15px;
    outline: none;
    transition: border-color 0.2s;
}
.search-bar input:focus { border-color: #0f3460; }
.search-bar button {
    padding: 10px 24px;
    background: #0f3460;
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
}
.search-bar button:hover { background: #1a5276; }
.btn-clear { background: #e74c3c !important; }
.btn-clear:hover { background: #c0392b !important; }

/* 图片网格 */
.gallery {
    flex: 1;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 16px;
    overflow-y: auto;
    padding: 4px;
}

.image-card {
    background: white;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: transform 0.2s, box-shadow 0.2s;
    position: relative;
}
.image-card:hover { transform: translateY(-3px); box-shadow: 0 4px 16px rgba(0,0,0,0.12); }

.image-card img {
    width: 100%;
    height: 160px;
    object-fit: cover;
    display: block;
}

.image-card .card-body {
    padding: 10px 12px;
}
.image-card .card-category {
    font-size: 11px;
    color: #0f3460;
    background: #e3f0ff;
    padding: 2px 8px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 4px;
}
.image-card .card-caption {
    font-size: 13px;
    color: #333;
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.image-card .card-score {
    font-size: 12px;
    color: #888;
    margin-top: 4px;
}
.image-card .delete-btn {
    position: absolute;
    top: 6px;
    right: 6px;
    background: rgba(0,0,0,0.5);
    color: white;
    border: none;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    cursor: pointer;
    font-size: 14px;
    line-height: 24px;
    text-align: center;
    opacity: 0;
    transition: opacity 0.2s;
}
.image-card:hover .delete-btn { opacity: 1; }
.image-card .delete-btn:hover { background: rgba(231, 76, 60, 0.8); }

/* 搜索结果高亮 */
.image-card.search-match { border: 2px solid #0f3460; }

/* 空状态 */
.empty-state {
    grid-column: 1 / -1;
    text-align: center;
    padding: 60px 20px;
    color: #999;
    font-size: 16px;
}

/* ==================== 加载动画 ==================== */
.loading {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.5);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    color: white;
}
.spinner {
    width: 40px;
    height: 40px;
    border: 4px solid rgba(255,255,255,0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-bottom: 12px;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ==================== 弹窗 ==================== */
.modal {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 999;
}
.modal-content {
    background: white;
    padding: 24px;
    border-radius: 12px;
    width: 90%;
    max-width: 500px;
}
.modal-content h3 { margin-bottom: 12px; }
.modal-content textarea {
    width: 100%;
    padding: 10px;
    border: 2px solid #ddd;
    border-radius: 8px;
    font-size: 14px;
    margin-bottom: 12px;
    resize: vertical;
}
.modal-content button {
    padding: 8px 20px;
    background: #0f3460;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
}
.close {
    float: right;
    font-size: 24px;
    cursor: pointer;
    color: #999;
}
.close:hover { color: #333; }
```

- [ ] **Step 2: 确认文件创建完毕**

---

### Task 10: 前端交互逻辑 static/app.js

**Files:**
- Create: `static/app.js`

- [ ] **Step 1: 编写 static/app.js**

```javascript
// ==================== 状态管理 ====================
let currentFilter = '';
let isSearchMode = false;

// ==================== 初始化 ====================
document.addEventListener('DOMContentLoaded', () => {
    refreshAll();
    setupFileUpload();
});

// ==================== API 工具 ====================
async function apiGet(path) {
    const resp = await fetch(path);
    if (!resp.ok) throw new Error(`GET ${path} failed: ${resp.status}`);
    return resp.json();
}

async function apiPost(path, body) {
    const resp = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(`POST ${path} failed: ${resp.status}`);
    return resp.json();
}

async function apiDelete(path) {
    const resp = await fetch(path, { method: 'DELETE' });
    if (!resp.ok) throw new Error(`DELETE ${path} failed: ${resp.status}`);
    return resp.json();
}

// ==================== 文件上传 ====================
function setupFileUpload() {
    const input = document.getElementById('file-input');
    input.addEventListener('change', async () => {
        if (input.files.length === 0) return;
        showLoading('上传中...');
        const formData = new FormData();
        for (const file of input.files) {
            formData.append('files', file);
        }
        try {
            const resp = await fetch('/api/images/upload', { method: 'POST', body: formData });
            const result = await resp.json();
            alert(`导入完成：${result.imported} 张成功，${result.errors.length} 张失败`);
            await refreshAll();
        } catch (e) {
            alert('上传失败：' + e.message);
        } finally {
            hideLoading();
            input.value = '';
        }
    });
}

// ==================== URL 导入 ====================
function showUrlImport() {
    document.getElementById('url-modal').style.display = 'flex';
}
function closeUrlImport() {
    document.getElementById('url-modal').style.display = 'none';
}

async function importUrls() {
    const text = document.getElementById('url-input').value.trim();
    if (!text) return;
    const urls = text.split('\n').map(s => s.trim()).filter(Boolean);
    closeUrlImport();
    showLoading('从 URL 导入中...');
    try {
        const result = await apiPost('/api/images/from-url', { urls });
        alert(`导入完成：${result.imported} 张成功，${result.errors.length} 张失败`);
        document.getElementById('url-input').value = '';
        await refreshAll();
    } catch (e) {
        alert('导入失败：' + e.message);
    } finally {
        hideLoading();
    }
}

// ==================== 刷新 ====================
async function refreshAll() {
    await Promise.all([loadStats(), loadCategories(), loadImages()]);
}

// ==================== 统计 ====================
async function loadStats() {
    try {
        const stats = await apiGet('/api/stats');
        document.getElementById('stats-display').textContent =
            `📷 ${stats.total_images} 张图片 | 📁 ${stats.total_categories} 个分类`;
    } catch (e) {
        document.getElementById('stats-display').textContent = '加载失败';
    }
}

// ==================== 分类列表 ====================
async function loadCategories() {
    try {
        const data = await apiGet('/api/categories');
        const stats = await apiGet('/api/stats');
        const list = document.getElementById('category-list');
        // 保留 "全部" 项
        list.innerHTML = '<li class="category-item active" data-category="" onclick="filterByCategory(this, \'\')">📋 全部</li>';

        for (const cat of data.categories) {
            const li = document.createElement('li');
            li.className = 'category-item';
            li.dataset.category = cat;
            li.textContent = `📁 ${cat}`;
            li.addEventListener('click', () => filterByCategory(li, cat));

            // 显示数量
            const count = stats.categories[cat] || 0;
            const span = document.createElement('span');
            span.className = 'category-count';
            span.textContent = count;
            li.appendChild(span);

            list.appendChild(li);
        }
    } catch (e) {
        console.error('加载分类失败:', e);
    }
}

// ==================== 分类筛选 ====================
function filterByCategory(el, category) {
    document.querySelectorAll('.category-item').forEach(item => item.classList.remove('active'));
    el.classList.add('active');
    currentFilter = category;
    isSearchMode = false;
    document.getElementById('search-input').value = '';
    document.getElementById('clear-search').style.display = 'none';
    loadImages();
}

// ==================== 加载图片 ====================
async function loadImages() {
    const gallery = document.getElementById('gallery');
    gallery.innerHTML = '<div class="empty-state">⏳ 加载中...</div>';

    try {
        const params = currentFilter ? `?category=${encodeURIComponent(currentFilter)}` : '';
        const data = await apiGet(`/api/images${params}`);

        if (data.images.length === 0) {
            gallery.innerHTML = '<div class="empty-state">暂无图片，请先导入。</div>';
            return;
        }

        gallery.innerHTML = '';
        for (const img of data.images) {
            const card = createImageCard(img);
            gallery.appendChild(card);
        }
    } catch (e) {
        gallery.innerHTML = `<div class="empty-state">❌ 加载失败：${e.message}</div>`;
    }
}

// ==================== 搜索 ====================
async function searchImages() {
    const query = document.getElementById('search-input').value.trim();
    if (!query) return;

    isSearchMode = true;
    document.getElementById('clear-search').style.display = 'inline-block';
    showLoading('搜索中...');

    try {
        const data = await apiPost('/api/search', { query });

        const gallery = document.getElementById('gallery');
        gallery.innerHTML = '';

        if (data.results.length === 0) {
            gallery.innerHTML = '<div class="empty-state">未找到匹配图片，请尝试其他关键词。</div>';
            return;
        }

        for (const img of data.results) {
            const card = createImageCard({
                id: img.id,
                filename: img.filename,
                caption: img.caption,
                category: img.category,
                score: img.similarity,
                url: img.url,
                source: img.source,
                created_at: '',
            }, true);
            gallery.appendChild(card);
        }
    } catch (e) {
        alert('搜索失败：' + e.message);
    } finally {
        hideLoading();
    }
}

function clearSearch() {
    document.getElementById('search-input').value = '';
    document.getElementById('clear-search').style.display = 'none';
    isSearchMode = false;
    // 重置分类选中状态
    document.querySelectorAll('.category-item').forEach(item => item.classList.remove('active'));
    document.querySelector('.category-item[data-category=""]')?.classList.add('active');
    currentFilter = '';
    loadImages();
}

// ==================== 图片卡片 ====================
function createImageCard(img, isSearch = false) {
    const card = document.createElement('div');
    card.className = 'image-card' + (isSearch ? ' search-match' : '');

    card.innerHTML = `
        <img src="${img.url}" alt="${img.filename}" loading="lazy"
             onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 200 160%22><rect fill=%22%23ddd%22 width=%22200%22 height=%22160%22/><text x=%2250%%22 y=%2250%%22 text-anchor=%22middle%22 fill=%22%23999%22 font-size=%2214%22>加载失败</text></svg>'">
        <button class="delete-btn" onclick="deleteImage('${img.id}')">✕</button>
        <div class="card-body">
            <div class="card-category">${img.category}</div>
            <div class="card-caption">${img.caption || '无描述'}</div>
            ${isSearch ? `<div class="card-score">匹配度: ${img.score}%</div>` : ''}
        </div>
    `;
    return card;
}

// ==================== 删除图片 ====================
async function deleteImage(id) {
    if (!confirm('确定删除这张图片吗？')) return;
    try {
        await apiDelete(`/api/images/${id}`);
        await refreshAll();
    } catch (e) {
        alert('删除失败：' + e.message);
    }
}

// ==================== Loading ====================
function showLoading(text) {
    document.getElementById('loading-text').textContent = text || '处理中...';
    document.getElementById('loading').style.display = 'flex';
}
function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}
```

- [ ] **Step 2: 确认文件创建完毕**

---

### Task 11: README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 编写 README.md**

```markdown
# CLIP 零样本图像与语义跨模态检索系统

基于 OpenAI CLIP + BLIP 的跨模态检索 Web 应用。

## 功能

- 📤 **导入图片** — 支持本地上传和 URL 导入
- 🏷️ **自动分类** — CLIP 零样本分类，混合模式（预设 + 自动创建）
- 📝 **生成描述** — BLIP 自动为每张图片生成自然语言描述
- 🔍 **语义搜索** — 输入自然语言，精准匹配图片
- 📁 **分类管理** — 按大类浏览和筛选图片

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py

# 打开浏览器访问
# http://127.0.0.1:8000
```

首次启动会自动下载 CLIP 和 BLIP 模型。图片导入后特征和描述会被缓存，搜索毫秒级响应。

## 技术栈

- **后端**: FastAPI + Uvicorn
- **视觉模型**: CLIP ViT-B/32（特征提取）、BLIP-base（图像描述）
- **前端**: 原生 HTML + CSS + JavaScript
- **存储**: 本地文件系统

## 项目结构

```
├── main.py              # FastAPI 入口 + API 路由
├── config.py            # 配置（阈值/模型名/预设类别）
├── models/
│   ├── clip_service.py  # CLIP 特征提取
│   └── caption_service.py  # BLIP 图像描述
├── services/
│   ├── classifier.py    # 分类引擎
│   └── search.py        # 检索服务
├── static/
│   ├── index.html       # 前端页面
│   ├── style.css        # 样式
│   └── app.js           # 交互逻辑
├── data/
│   ├── images/          # 图片存储
│   └── metadata.json    # 元数据
├── requirements.txt
└── README.md
```

## 配置

编辑 `config.py` 可调整：
- 预设分类类别
- 分类匹配阈值
- 检索结果数量
- 服务器地址和端口
```

- [ ] **Step 2: 确认文件创建完毕**

---

### Task 12: .gitignore 更新

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: 创建/更新 .gitignore**

```gitignore
# Python
__pycache__/
*.pyc
.venv/

# 项目数据
data/images/*
!data/images/.gitkeep
data/metadata.json

# IDE
.idea/
```

- [ ] **Step 2: 创建 data/images/.gitkeep**

确保 `data/images/` 目录被 git 跟踪。

---

### Task 13: 端到端验证

- [ ] **Step 1: 安装依赖并启动服务**

```bash
cd /d/academy/git_hub/Zero-Shot-Classification-and-Retrieval-System-Based-on-CLIP
.venv/Scripts/activate
pip install -r requirements.txt
python main.py
```

- [ ] **Step 2: 在浏览器打开 `http://127.0.0.1:8000`**

验证：
- 页面正常渲染，统计显示 0 张图片
- 上传几张测试图片
- 查看图片是否自动分类和生成描述
- 输入自然语言搜索，验证结果匹配
- 删除图片验证删除功能
- 刷新页面验证持久化

- [ ] **Step 3: 测试通过后提交并推送**

```bash
git add -A
git commit -m "feat: 实现CLIP跨模态检索系统Web版
- FastAPI后端提供REST API
- CLIP特征提取+BLIP图像描述
- 图片分类引擎（预设+自动创建）
- 原生前端单页应用
- 本地文件持久化"
git push
```
