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
