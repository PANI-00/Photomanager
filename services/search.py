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
