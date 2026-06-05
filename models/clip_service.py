"""CLIP 模型封装：图像/文本特征提取
所有 heavy import (torch / clip) 延迟到实际使用时才做，加速模块导入。
"""

from PIL import Image


class CLIPService:
    """单例模式封装 CLIP 模型，提供特征提取接口。"""

    _instance = None
    _MODEL_NAME = "ViT-B/32"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._preprocess = None
            cls._instance._device = None
        return cls._instance

    @property
    def _DEVICE(self):
        if self._device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def model(self):
        if self._model is None:
            import clip
            import torch
            print("正在加载 CLIP 模型...")
            self._model, self._preprocess = clip.load(self._MODEL_NAME, device=self._DEVICE)
            self._model.eval()
            print("CLIP 模型加载完成。")
        return self._model

    @property
    def preprocess(self):
        if self._preprocess is None:
            _ = self.model  # 触发加载
        return self._preprocess

    def encode_image(self, image: Image.Image):
        """提取单张图片的 L2 归一化特征向量 (1, 512)"""
        import torch
        image_input = self.preprocess(image).unsqueeze(0).to(self._DEVICE)
        with torch.no_grad():
            features = self.model.encode_image(image_input)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu()

    def encode_images(self, images: list[Image.Image]):
        """批量提取图片特征 (N, 512)"""
        import torch
        image_inputs = torch.stack([self.preprocess(img) for img in images]).to(self._DEVICE)
        with torch.no_grad():
            features = self.model.encode_image(image_inputs)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu()

    def encode_text(self, texts: list[str]):
        """批量编码文本 (N, 512)"""
        import clip
        import torch
        text_tokens = clip.tokenize(texts).to(self._DEVICE)
        with torch.no_grad():
            features = self.model.encode_text(text_tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu()

    def similarity(self, image_feat, text_feat):
        """计算图像与文本的余弦相似度矩阵"""
        return 100.0 * image_feat @ text_feat.T
