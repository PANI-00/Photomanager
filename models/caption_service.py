"""轻量级图片描述生成：vit-gpt2（约 250MB）
优先读取本地缓存，若失败则通过 hf-mirror.com 下载（国内友好）。
如果两者都失败，标记为不可用，应用降级运行。
"""

import os

from PIL import Image

# 国内镜像（HuggingFace 在中国被墙）— 全局生效
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_MIRROR = "https://hf-mirror.com"


class CaptionService:
    """vit-gpt2 封装，单例模式，延迟加载"""

    _instance = None
    _MODEL_ID = "nlpconnect/vit-gpt2-image-captioning"
    available = False  # 标记模型是否可用

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._processor = None
            cls._instance._tokenizer = None
        return cls._instance

    def _try_load(self):
        """尝试加载模型，先从本地缓存，失败则走镜像"""
        from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

        # 先试本地缓存
        try:
            self._model = VisionEncoderDecoderModel.from_pretrained(
                self._MODEL_ID, local_files_only=True
            )
            self._processor = ViTImageProcessor.from_pretrained(
                self._MODEL_ID, local_files_only=True
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._MODEL_ID, local_files_only=True
            )
            self.available = True
            print("Caption 模型从本地缓存加载成功")
            return
        except Exception:
            print("Caption 模型本地缓存未找到，尝试从镜像下载…")

        # 走国内镜像
        old_hf = os.environ.get("HF_ENDPOINT")
        os.environ["HF_ENDPOINT"] = _MIRROR
        try:
            self._model = VisionEncoderDecoderModel.from_pretrained(self._MODEL_ID)
            self._processor = ViTImageProcessor.from_pretrained(self._MODEL_ID)
            self._tokenizer = AutoTokenizer.from_pretrained(self._MODEL_ID)
            self.available = True
            print("Caption 模型从镜像下载成功")
        except Exception as e:
            print(f"Caption 模型加载失败（镜像也不可用）: {e}")
            self.available = False
        finally:
            if old_hf:
                os.environ["HF_ENDPOINT"] = old_hf
            elif "HF_ENDPOINT" in os.environ:
                del os.environ["HF_ENDPOINT"]

    @property
    def model(self):
        if self._model is None:
            self._try_load()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self._try_load()
        return self._processor

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._try_load()
        return self._tokenizer

    def generate(self, image: Image.Image, max_length: int = 30) -> str:
        """生成单张图片的描述文字"""
        import torch
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=3,
                no_repeat_ngram_size=2,
            )
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()

    def generate_batch(self, images: list[Image.Image], max_length: int = 30) -> list[str]:
        """批量生成描述（分批处理，避免 OOM）"""
        import torch
        results = []
        batch_size = 4
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs.pixel_values,
                    max_length=max_length,
                    num_beams=3,
                    no_repeat_ngram_size=2,
                )
            for ids in output_ids:
                caption = self.tokenizer.decode(ids, skip_special_tokens=True)
                results.append(caption.strip())
        return results
