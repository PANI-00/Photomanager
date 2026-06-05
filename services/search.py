"""检索服务：CLIP 图片语义搜索 + Caption 关键词命中加分 + 中文翻译"""

import re
import torch
from config import TOP_K_RESULTS
from services.chinese_dict import has_chinese, chinese_to_english
from services.translator import Translator


def _query_keywords(text: str) -> set[str]:
    """提取英文查询中的有效关键词（去停用词）"""
    STOP = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "with",
        "and", "or", "is", "are", "was", "were", "be", "been", "being",
        "it", "its", "this", "that", "these", "those", "has", "have",
        "had", "do", "does", "did", "will", "would", "can", "could",
        "may", "might", "shall", "should", "up", "down", "out", "off",
        "over", "under", "by", "from", "as", "but", "not", "no", "nor",
        "so", "if", "then", "than", "too", "very", "just", "about",
        "also", "more", "some", "any", "each", "every", "all", "both",
        "few", "most", "other", "into", "through", "during", "before",
        "after", "above", "below", "between", "around", "near",
        "back", "front", "still", "yet", "already", "even", "much",
        "many", "only", "own", "same", "one", "two", "three",
    }
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return words - STOP


def search_images(
    query: str,
    clip_service,
    metadata_list: list[dict],
) -> list[dict]:
    """
    图片搜索：CLIP 图片语义为主 + Caption 关键词命中加分。

    中文查询自动翻译为英文（翻译模型优先 → 词典兜底）。

    参数:
        query: 用户输入的搜索文本（中文/英文/混合均可）
        clip_service: CLIP 服务实例
        metadata_list: 图片元数据列表

    返回:
        [{id, filename, caption, category, similarity, source, url}, ...]
    """
    if not metadata_list:
        return []

    # ── 0. 中文 → 英文翻译 ──
    search_query = query
    if has_chinese(query):
        # 词典翻译（稳定可靠）
        dict_result = chinese_to_english(query) if not None else None
        dict_ok = dict_result is not None and dict_result != query

        # 模型翻译（更准确，但可能丢语义）
        translator = Translator()
        model_ok = translator.ready
        if model_ok:
            model_result = translator.translate(query)
            model_ok = bool(model_result) and model_result != query

        if model_ok and dict_ok:
            # 两者合并：模型翻译 + 词典关键词，扩大覆盖
            search_query = model_result
            extra = dict_result
            # 加词典词作为补充关键词（让 CLIP 同时理解两个语义）
            for w in extra.split():
                if w.lower() not in search_query.lower():
                    search_query += " " + w
        elif model_ok:
            search_query = model_result
        elif dict_ok:
            search_query = dict_result

    # ── 1. CLIP 图片语义匹配（主力） ──
    query_feat = clip_service.encode_text([search_query])  # (1, 512)
    all_features = torch.cat([item["feature"] for item in metadata_list], dim=0)  # (N, 512)
    scores = 100.0 * query_feat @ all_features.T  # (1, N)

    # ── 2. Caption 关键词命中加分 ──
    query_kw = _query_keywords(search_query)
    if query_kw:
        for i, item in enumerate(metadata_list):
            cap = (item.get("caption") or "").lower()
            hits = sum(1 for w in query_kw if w in cap)
            if hits:
                scores[0, i] += hits * 5.0  # 每个关键词 +5 分

    # ── 3. 排序取 Top-K ──
    values, indices = scores[0].topk(k=min(TOP_K_RESULTS, len(metadata_list)))

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
