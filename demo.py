"""
CLIP 零样本分类与跨模态检索系统
基于 OpenAI CLIP (ViT-B/32)，无需额外训练即可分类和检索。
"""
import os
import sys
from pathlib import Path
from typing import Optional

import clip
import torch
from PIL import Image
import requests


# ==================== 配置 ====================
CACHE_DIR = Path("image_cache")
CACHE_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 默认图像数据集（可替换为本地路径或自己的 URL 列表）
DEFAULT_IMAGE_SOURCES: list[str] = [
    'https://globalimg.sucai999.com/preimg/AF76E3/700/AF76E3/25/250355ed900c6c9a0e1887ef9502fe04.jpg?x-oss-process=image/resize,w_320/format,webp',
    'https://globalimg.sucai999.com/preimg/6D28AC/700/6D28AC/201/5b4a759c5788ca45d4395211521bf2c.jpg?x-oss-process=image/resize,w_320/format,webp',
    'https://globalimg.sucai999.com/preimg/DBC456/700/DBC456/103/2c1f0905d93e64af2799d80c67df5.jpg?x-oss-process=image/resize,w_320/format,webp',
    'https://globalimg.sucai999.com/preimg/8AA05E/700/8AA05E/103/95e47c5820ab2b738f0e9e2ddc6ca8.jpg?x-oss-process=image/resize,w_320/format,webp',
    'https://globalimg.sucai999.com/preimg/E625C8/700/E625C8/148/546aeb60b2446bf0c4aa247454d32d50.jpg?x-oss-process=image/resize,w_320/format,webp',
    'https://globalimg.sucai999.com/preimg/E625C8/700/E625C8/111/f436fc2891516b652564367cd2d859a7.jpg?x-oss-process=image/resize,w_320/format,webp',
    'https://globalimg.sucai999.com/preimg/123D82/700/123D82/198/d849caf1496cf78e70e08a2a9265e762.jpg?x-oss-process=image/resize,w_320/format,webp',
]

TOP_K = 5               # 检索结果数量
TIMEOUT_SEC = 15        # 下载超时（秒）


# ==================== 模型管理 ====================

class CLIPModel:
    """封装 CLIP 模型的加载与推理。"""

    def __init__(self, model_name: str = "ViT-B/32"):
        print(f"正在加载 CLIP 模型 ({model_name}) 到 {DEVICE} ...")
        self.model, self.preprocess = clip.load(model_name, device=DEVICE)
        print("模型加载完成。\n")

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """批量编码图像，返回 L2 归一化后的特征向量 (N, D)。"""
        image_input = torch.stack([self.preprocess(img) for img in images]).to(DEVICE)
        features = self.model.encode_image(image_input)
        features /= features.norm(dim=-1, keepdim=True)
        return features

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """编码文本列表，返回 L2 归一化后的特征向量 (N, D)。"""
        text_inputs = clip.tokenize(texts).to(DEVICE)
        features = self.model.encode_text(text_inputs)
        features /= features.norm(dim=-1, keepdim=True)
        return features


# ==================== 图像加载 ====================

def _load_image_from_url(url: str) -> Optional[Image.Image]:
    """从 URL 下载图像，失败时返回 None。"""
    try:
        resp = requests.get(url, timeout=TIMEOUT_SEC)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"  ⚠ 下载失败: {url}\n    {e}")
        return None


def _load_image_from_local(path: str) -> Optional[Image.Image]:
    """从本地路径加载图像，失败时返回 None。"""
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"  ⚠ 读取失败: {path}\n    {e}")
        return None


def load_images(sources: list[str]) -> list[Image.Image]:
    """
    从 URL 或本地路径批量加载图像。
    自动跳过加载失败的项并提示。
    """
    images: list[Image.Image] = []
    for src in sources:
        img = _load_image_from_local(src) if os.path.isfile(src) else _load_image_from_url(src)
        if img is not None:
            images.append(img)
    return images


# ==================== 搜索逻辑 ====================

def _show_topk(
    scores: torch.Tensor,
    labels: list[str],
    top_k: int = TOP_K,
    unit: str = "%",
) -> None:
    """打印 Top-K 得分结果。"""
    k = min(top_k, len(labels))
    values, indices = scores.topk(k=k, dim=-1)
    for val, idx in zip(values, indices):
        print(f"  {labels[idx]}: {val.item():.2f}{unit}")


# ==================== 功能函数 ====================

def zero_shot_classification(model_handler: CLIPModel) -> None:
    """
    零样本分类：用户输入一张图像，模型预测它属于预定义类别中的哪一类。
    """
    raw = input("输入图片链接或本地路径:\n").strip()
    if not raw:
        print("输入为空，返回主菜单。")
        return

    img = _load_image_from_local(raw) if os.path.isfile(raw) else _load_image_from_url(raw)
    if img is None:
        return

    # 自定义类别（可在此修改或改为从输入读取）
    text_labels = [
        "a photo of a cat", "a photo of a dog", "a photo of a bird",
        "a photo of a car", "a photo of a house", "a photo of a flower",
        "a photo of gundam", "a photo of a landscape",
    ]

    with torch.no_grad():
        image_feat = model_handler.preprocess(img).unsqueeze(0).to(DEVICE)
        image_feat = model_handler.model.encode_image(image_feat)
        image_feat /= image_feat.norm(dim=-1, keepdim=True)

        text_feat = model_handler.encode_text(text_labels)

        # 相似度（余弦相似度 × 100）
        similarity = 100.0 * image_feat @ text_feat.T

    print("\n📷 图像分类预测结果：")
    _show_topk(similarity[0], text_labels)


def cross_modal_retrieval(model_handler: CLIPModel, image_sources: list[str], images: list[Image.Image]) -> None:
    """
    跨模态检索：用户输入一段文本，从图像库中找出最匹配的图像。
    """
    if not images:
        print("⚠ 图像库为空，无法检索。请检查图片 URL 或本地路径。")
        return

    query = input("输入检索文本 (English):\n").strip()
    if not query:
        print("输入为空，返回主菜单。")
        return

    with torch.no_grad():
        query_feat = model_handler.encode_text([query])
        image_feat = model_handler.encode_images(images)

        # 余弦相似度
        similarity = 100.0 * query_feat @ image_feat.T

    print(f"\n🔍 Query '{query}' 匹配得分 (Top-{TOP_K})：")
    values, indices = similarity[0].topk(k=min(TOP_K, len(images)), dim=-1)
    for val, idx in zip(values, indices):
        print(f"  {image_sources[idx]}  —  {val.item():.2f}%")


# ==================== 主流程 ====================

def main_menu() -> str:
    """显示主菜单并返回用户选项。"""
    print("\n" + "=" * 45)
    print("  CLIP 零样本分类 & 跨模态检索系统")
    print("=" * 45)
    print("  1. 📷 图片分类（零样本）")
    print("  2. 🔍 文本检索图像（跨模态）")
    print("  q. ❌ 退出")
    return input("请选择操作 (1 / 2 / q): ").strip()


def main() -> None:
    # 加载模型（只加载一次）
    try:
        clip_model = CLIPModel()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)

    # 加载图像库
    print(f"正在加载 {len(DEFAULT_IMAGE_SOURCES)} 张图像...")
    images = load_images(DEFAULT_IMAGE_SOURCES)
    print(f"成功加载 {len(images)} / {len(DEFAULT_IMAGE_SOURCES)} 张图像\n")

    while True:
        choice = main_menu()
        if choice == "q":
            print("再见！")
            break
        elif choice == "1":
            zero_shot_classification(clip_model)
        elif choice == "2":
            cross_modal_retrieval(clip_model, DEFAULT_IMAGE_SOURCES, images)
        else:
            print("无效选项，请输入 1、2 或 q。")


if __name__ == "__main__":
    from io import BytesIO   # 仅在主模块导入，避免循环依赖
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断，退出。")
        sys.exit(0)
