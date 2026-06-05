# CLIP 零样本图像与语义跨模态检索系统 — 设计文档

## 概述

基于 OpenAI CLIP + BLIP 的跨模态检索 Web 应用。用户可导入图片（上传/URL），系统自动生成描述并归类，支持自然语言搜索匹配图片。

## 技术栈

| 层 | 技术 |
|----|------|
| 后端框架 | FastAPI (Python) |
| 前端 | 原生 HTML + CSS + JavaScript |
| 视觉模型 | CLIP ViT-B/32 (特征提取) + BLIP-base (图像描述) |
| 持久化 | 本地文件 (data/images/ + data/metadata.json) |
| 运行环境 | Python 3.14 + 虚拟环境 (.venv) |

## 架构

```
Browser (HTML/CSS/JS)  ←REST API→  FastAPI Backend → Model Layer (CLIP + BLIP)
                                                       ↓
                                              File Storage (images + metadata)
```

三层分离：前端展示 → API 逻辑 → 模型推理。图片导入时预计算特征和描述并缓存，搜索时仅做向量相似度计算（毫秒级）。

## API 设计

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | /api/images/upload | 上传单张/多张图片 |
| POST | /api/images/from-url | 从 URL 导入图片 |
| GET | /api/images | 获取图片列表（含分类信息） |
| DELETE | /api/images/{id} | 删除图片 |
| POST | /api/search | 自然语言检索图片 |
| GET | /api/categories | 获取所有大类列表 |
| GET | /api/stats | 系统统计信息 |

## 数据持久化

- 图片文件：`data/images/{uuid}.{ext}`
- 元数据：`data/metadata.json`（含特征向量 512-dim、描述、分类、标签）
- metadata.json 结构：`[{id, filename, source, category, labels, caption, feature, created_at}]`

## 分类引擎（混合模式）

1. 预设大类：动物、建筑、风景、人物、食物、交通、植物、物品
2. CLIP 计算图片与各类别名称的相似度
3. 最高分 ≥ 0.30 → 归入该类；0.20~0.30 → 边缘匹配；< 0.20 → 自动创建新类别
4. 类别名用最高分标签自动命名（如"机械"、"抽象艺术"）

## 搜索流程

1. CLIP 编码用户文本 → 文本特征向量
2. 与所有图片特征计算余弦相似度
3. 按匹配度排序 → 按大类分组 → 返回前端分类展示

## 前端功能

- 图片导入（上传 + URL，含进度反馈）
- 分类筛选（左侧树形分类列表）
- 图片网格浏览
- 自然语言检索（回车/点击搜索，结果分类展示 + 相似度显示）
- 图片删除

## 项目结构

```
├── main.py                 # FastAPI 入口 + 路由
├── config.py               # 配置（阈值、模型名、默认类别）
├── models/
│   ├── clip_service.py     # CLIP 特征提取
│   └── caption_service.py  # BLIP 图像描述
├── services/
│   ├── classifier.py       # 分类引擎
│   └── search.py           # 检索服务
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── data/
│   ├── images/
│   └── metadata.json
├── requirements.txt
└── README.md
```

## 错误处理

- 图片格式不支持 → 前后端双重校验，返回明确错误
- URL 下载失败 → 跳过该图片，不影响其他
- 模型加载失败 → 启动时检查并提示安装指引
- metadata.json 损坏 → 自动备份并重建
- 搜索无结果 → 提示更换关键词

## 测试方向

- 单元测试：特征提取、分类引擎、相似度计算
- API 测试：各接口请求/响应格式
- 边界测试：空库搜索、重复导入、超大图片

## 后续扩展可能

- 支持更多图片格式
- 批量导入进度增强
- 图片详情弹窗
- 导出检索结果
