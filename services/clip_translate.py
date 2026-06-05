"""CLIP 多语言文本编码器：中 → 英跨语言查询扩展
利用 CLIP 共享向量空间，找到中文查询最近的英文词/词组，
无需额外翻译模型。
"""

import re
import torch

# 内置常用英文概念（CLIP 空间中的锚点）
_BUILTIN_VOCAB = [
    # 动物
    "monkey", "cat", "dog", "bird", "fish", "horse", "cow", "sheep",
    "pig", "chicken", "duck", "bear", "deer", "fox", "wolf", "rabbit",
    "tiger", "lion", "elephant", "panda", "snake", "turtle", "butterfly",
    "penguin", "dolphin", "whale", "squirrel", "frog",
    # 人物
    "person", "people", "man", "woman", "child", "baby", "boy", "girl",
    "crowd", "family", "friend", "face", "hand",
    # 自然
    "forest", "tree", "flower", "mountain", "beach", "ocean", "river",
    "lake", "waterfall", "sky", "cloud", "sunset", "sunrise", "rainbow",
    "snow", "rain", "field", "grass", "garden", "park",
    # 建筑 & 城市
    "building", "house", "city", "street", "road", "bridge", "church",
    "temple", "castle", "tower", "wall", "door", "window", "store",
    # 交通
    "car", "bus", "truck", "train", "airplane", "boat", "ship", "bicycle",
    "motorcycle", "parking", "traffic", "driving",
    # 物品
    "phone", "camera", "computer", "laptop", "book", "chair", "table",
    "bottle", "cup", "bag", "backpack", "umbrella", "clock", "key",
    "lamp", "toy", "ball", "flag", "statue", "sign",
    # 颜色
    "red", "blue", "green", "yellow", "white", "black", "purple", "orange",
    "pink", "brown", "gray", "golden", "colorful",
    # 场景
    "indoor", "outdoor", "night", "day", "morning", "evening", "kitchen",
    "bedroom", "living room", "office", "restaurant", "market",
    "pool", "parking lot", "playground", "stadium",
    # 动作
    "running", "walking", "sitting", "standing", "jumping", "eating",
    "drinking", "cooking", "sleeping", "reading", "smiling", "dancing",
    "flying", "driving", "riding", "playing",
    # 形容词
    "old", "new", "big", "small", "tall", "short", "long", "beautiful",
    "cute", "happy", "sad", "clean", "dirty", "bright", "dark",
    # 其他常见
    "water", "fire", "light", "shadow", "wood", "metal", "glass",
    "food", "fruit", "vegetable", "cake", "bread", "wall",
]


def _clean_word(word: str) -> str:
    """去掉标点，保留字母"""
    return re.sub(r"[^a-zA-Z]", "", word).strip().lower()


def _extract_caption_words(captions: list[str]) -> list[str]:
    """从 caption 列表中提取有意义的英文单词"""
    words = set()
    for cap in captions:
        if not cap:
            continue
        for w in cap.split():
            cleaned = _clean_word(w)
            if cleaned and len(cleaned) > 1:
                words.add(cleaned)
    return sorted(words)


class CLIPTranslator:
    """用 CLIP 向量空间做中文 → 英文的跨语言查询扩展

    原理：
        CLIP 文本编码器是双语的——"猴子"和"monkey"的向量在空间中接近。
        我们预计算一批英文锚点的向量，对中文查询找最近的英文词。
    """

    def __init__(self, clip_service, caption_words: list[str] | None = None):
        self.clip = clip_service
        self._vocab: list[str] = []
        self._embeddings: torch.Tensor | None = None
        self._ready = False

        # 合并内置词表 + caption 词表
        vocab_set = set(w.lower() for w in _BUILTIN_VOCAB)
        if caption_words:
            for w in caption_words:
                vocab_set.add(w.lower())
        # 去重排序以保持一致性
        self._vocab = sorted(vocab_set)

    def build(self) -> int:
        """预计算所有英文锚点的 CLIP 向量"""
        if not self._vocab:
            return 0

        embeddings_list = []
        # 分批编码，避免一次太多
        batch_size = 64
        for i in range(0, len(self._vocab), batch_size):
            batch = self._vocab[i : i + batch_size]
            emb = self.clip.encode_text(batch)  # (N, 512)
            embeddings_list.append(emb)

        self._embeddings = torch.cat(embeddings_list, dim=0)  # (V, 512)
        self._ready = True
        return len(self._vocab)

    def expand_query(self, query: str, top_k: int = 3) -> str:
        """将中文查询扩展为英文关键词

        对于英文查询直接返回原文；
        对于中文/混合查询，找出最近的英文词 + 保留原英文词。
        """
        from services.chinese_dict import has_chinese

        if not has_chinese(query):
            return query  # 纯英文，不需要扩展

        if not self._ready:
            return query  # 词表未建好，保底

        # 提取查询中的英文词（保留）
        english_words = set(
            _clean_word(w) for w in query.split() if _clean_word(w)
        )
        english_words.discard("")

        # 中文部分 → CLIP 找最近英文词
        query_emb = self.clip.encode_text([query])  # (1, 512)
        scores = 100.0 * query_emb @ self._embeddings.T  # (1, V)
        values, indices = scores[0].topk(k=min(top_k, len(self._vocab)))

        expanded = set(english_words)
        for idx in indices.tolist():
            word = self._vocab[idx]
            if word not in expanded:
                expanded.add(word)

        result = " ".join(sorted(expanded))
        return result if result else query
