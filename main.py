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
