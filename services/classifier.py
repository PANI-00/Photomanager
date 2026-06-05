"""分类引擎：预设大类 + 用户自定义类别
所有图片归入最匹配的类别（预设 + 自定义），不自动创建新类别。
"""

import torch
from config import DEFAULT_CATEGORIES


def classify_image(
    image_feat: torch.Tensor,
    clip_service,
    extra_categories: list[str] | None = None,
) -> tuple[str, float]:
    """
    对图片特征进行分类。

    参数:
        extra_categories: 用户自定义的额外类别列表
    返回: (类别名称, 相似度分数)
    """
    all_categories = list(DEFAULT_CATEGORIES)
    if extra_categories:
        for cat in extra_categories:
            if cat and cat not in all_categories:
                all_categories.append(cat)

    text_features = clip_service.encode_text(all_categories)  # (N, 512)
    scores = image_feat @ text_features.T  # (1, N)

    max_score, max_idx = scores[0].max(dim=-1)
    max_score = max_score.item()

    return all_categories[max_idx], max_score
