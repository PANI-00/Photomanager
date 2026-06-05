"""分类引擎：混合模式——预设大类 + 自动创建新类别"""

import torch
from config import (
    DEFAULT_CATEGORIES,
    CLASSIFICATION_THRESHOLD_HIGH,
    CLASSIFICATION_THRESHOLD_LOW,
)


def classify_image(
    image_feat: torch.Tensor,
    clip_service,
    existing_categories: list[str] | None = None,
) -> tuple[str, float, str]:
    """
    对图片特征进行分类。

    返回: (大类名称, 相似度分数, 匹配类型)
    匹配类型: "exact" / "边缘" / "新类别"
    """
    # 合并预设大类 + 已有自动创建的类别
    all_categories = list(DEFAULT_CATEGORIES)
    if existing_categories:
        for cat in existing_categories:
            if cat not in all_categories:
                all_categories.append(cat)

    # 计算图片与各类别名称的 CLIP 相似度
    text_features = clip_service.encode_text(all_categories)
    scores = image_feat @ text_features.T  # (1, N)

    max_score, max_idx = scores[0].max(dim=-1)
    max_score = max_score.item()

    category = all_categories[max_idx]

    if max_score >= CLASSIFICATION_THRESHOLD_HIGH:
        match_type = "exact"
    elif max_score >= CLASSIFICATION_THRESHOLD_LOW:
        match_type = "边缘"
    else:
        category = f"{category}_自动类别"
        match_type = "新类别"

    return category, max_score, match_type
