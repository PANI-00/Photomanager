"""中 → 英翻译服务（transformers，单例延迟加载）
模型：Helsinki-NLP/opus-mt-zh-en（约 300MB）
首次加载自动从 hf-mirror.com 下载，后续从本地缓存读取。
"""

import os
import threading

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class Translator:
    """中文 → 英文翻译，单例，后台延迟加载"""

    _instance = None
    _lock = threading.Lock()
    available = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._tokenizer = None
            cls._instance._ready = False
        return cls._instance

    def _try_load(self):
        """尝试加载翻译模型（可被后台线程调用）"""
        with self._lock:
            if self._ready and self.available:
                return  # 已就绪，跳过
            from transformers import MarianTokenizer, MarianMTModel

            model_id = "Helsinki-NLP/opus-mt-zh-en"
            try:
                # 先试本地缓存
                self._tokenizer = MarianTokenizer.from_pretrained(
                    model_id, local_files_only=True
                )
                self._model = MarianMTModel.from_pretrained(
                    model_id, local_files_only=True
                )
                self._ready = True
                self.available = True
                print("翻译模型就绪（本地缓存）")
                return
            except Exception:
                # 本地没有 → 尝试下载
                pass

            print("翻译模型本地缓存未找到，从镜像下载… (~300MB)")
            try:
                self._tokenizer = MarianTokenizer.from_pretrained(model_id)
                self._model = MarianMTModel.from_pretrained(model_id)
                self._ready = True
                self.available = True
                print("翻译模型就绪（已下载）")
            except Exception as e:
                print(f"翻译模型下载失败: {e}")
                # 不设置 self._ready = True，允许下次重试
                self.available = False

    @property
    def ready(self) -> bool:
        return self._ready and self._model is not None

    def translate(self, text: str) -> str:
        """将中文文本翻译为英文，失败时返回原文

        注意：只处理含中文的文本，纯英文直接跳过。
        """
        if not text or not self._model or not self._tokenizer:
            return text
        # 纯英文不翻译（避免模型画蛇添足）
        import re
        if not re.search(r"[一-鿿]", text):
            return text
        import torch
        try:
            inputs = self._tokenizer(text, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs, max_length=128, num_beams=4
                )
            result = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 去掉尾部标点（句号等会影响 CLIP 编码）
            result = result.strip().rstrip(".,!?;:")
            return result
        except Exception as e:
            print(f"翻译失败: {e}")
            return text
