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
