# 📸 Photomanager

> 基于 CLIP 的桌面照片管理器 — 零样本分类 × 自然语言搜索 × AI 描述生成

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![GUI](https://img.shields.io/badge/GUI-CustomTkinter-teal)
![Model](https://img.shields.io/badge/Model-CLIP%20ViT--B%2F32-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ 功能

| 功能 | 说明 |
|------|------|
| 🔍 **AI 语义搜索** | 用自然语言搜照片，支持中文/英文 — "猴子"、"black women"、"红色汽车" |
| 🏷️ **零样本分类** | 自动分类到动物/建筑/风景/人物等类别，无需训练 |
| 📝 **AI 描述生成** | vit-gpt2 自动为每张图片生成英文描述（后台加载，不阻塞操作） |
| 🌐 **中文搜索翻译** | 内置 Helsinki-NLP 翻译模型，中文搜索自动转英文，准确匹配 CLIP 语义 |
| 🎨 **个性化设置** | 主题色/字体/字号/字体颜色/明暗模式/网格列数，全 GUI 可调 |
| 🖼️ **图片详情页** | 查看原图、描述、分类、相似度，支持删除索引 |
| ⚡ **丝滑动画** | 60fps ease-out 缓动页面切换动画 |
| 📁 **本地索引** | 不复制图片文件，只记录路径索引，节省空间 |

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Windows 10/11（Linux/Mac 需自行适配路径）

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/PANI-00/Photomanager.git
cd Photomanager

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行
python desktop_app.py
```

首次启动会自动下载模型（约 5~15 分钟，取决于网络）：
- **CLIP ViT-B/32** — 特征提取 + 语义搜索（~300MB）
- **vit-gpt2-image-captioning** — 图片描述生成（~460MB）
- **opus-mt-zh-en** — 中文搜索翻译（~312MB，可选）

国内用户自动使用 `hf-mirror.com` 镜像下载。

### 一键启动

双击 **`启动 Photomanager.vbs`**（Windows 隐藏控制台窗口）。

## 🖥️ 界面预览

```
┌──────────────────────────────────────────────────────┐
│  📸 PM                                              │
│  [📁 导入图片]                                      │
│  ──────────────────        ┌──────────────────────┐ │
│  🗂 分类                  │ 🔍 用文字搜索图片…     │ │
│  ┌──────────────────┐     └──────────────────────┘ │
│  │ 🏠 全部图片     │     ┌──┐ ┌──┐ ┌──┐ ┌──┐      │
│  │ 动物  (5)       │     │🖼️│ │🖼️│ │🖼️│ │🖼️│      │
│  │ 建筑  (3)       │     │  │ │  │ │  │ │  │      │
│  │ 风景  (8)       │     │a monkey…│a car…│      │
│  │ 人物  (6)       │     │ 动  │ 交  │ 风  │      │
│  │ …              │     └──┘ └──┘ └──┘ └──┘      │
│  └──────────────────┘                              │
│  📊 33张 · 8个分类                                  │
│  ⚙ 设置                     ⏳ 模型就绪 (12s)      │
└──────────────────────────────────────────────────────┘
```

## 🧠 技术架构

```
┌──────────────┐    ┌───────────────────────────┐
│  用户输入     │    │  CustomTkinter 桌面 GUI   │
│  (中文/英文)  │───→│  desktop_app.py           │
└──────────────┘    └───────────┬───────────────┘
                                │
                    ┌───────────┴────────────┐
                    │      搜索/分类请求      │
                    └───────────┬────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
          ▼                     ▼                     ▼
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │ CLIP 语义搜索 │    │ 中文→英文翻译 │    │ 零样本分类    │
  │ search.py    │    │ translator.py│    │ classifier.py│
  │              │    │  opus-mt-zh  │    │              │
  │ 图片特征 60%  │    │  -en 模型    │    │ CLIP 文本编码 │
  │ caption  40% │    │  词典兜底    │    │ + 余弦相似度  │
  └──────────────┘    └──────────────┘    └──────────────┘
```

### 关键组件

| 文件 | 职责 |
|------|------|
| `desktop_app.py` | 主窗口、UI 构建、页面导航、模型调度 |
| `models/clip_service.py` | CLIP 特征提取（单例、惰性加载） |
| `models/caption_service.py` | vit-gpt2 图片描述（单例、后台加载） |
| `services/search.py` | 混合搜索：CLIP 语义 + Caption 关键词 |
| `services/translator.py` | 中文→英文翻译（Helsinki-NLP 模型） |
| `services/chinese_dict.py` | 中文搜索词词典（翻译模型未就绪时兜底） |
| `services/classifier.py` | 零样本分类引擎 |
| `config.py` | 全局配置 |

## ⚙️ 配置

所有设置可通过 GUI 设置页调整：

- **主题模式**: Dark / Light / System
- **配色主题**: teal / purple / grey / blue / dark-blue / green
- **字体**: 字号 9~32px，可选字体族
- **字体颜色**: 12 种预设色 + 自定义
- **网格列数**: 3~6 列缩略图
- **语言**: 中文 / English

配置文件保存于 `data/settings.json`。

## 📁 项目结构

```
├── desktop_app.py            # 桌面主程序
├── config.py                 # 全局配置
├── entry.py                  # 启动入口
├── requirements.txt          # Python 依赖
├── .gitignore
├── README.md
├── models/
│   ├── clip_service.py       # CLIP 特征提取
│   └── caption_service.py    # vit-gpt2 图片描述
├── services/
│   ├── search.py             # 混合搜索
│   ├── classifier.py         # 零样本分类
│   ├── translator.py         # 中→英翻译
│   ├── chinese_dict.py       # 中文搜索词典
│   └── clip_translate.py     # CLIP 跨语言（实验性）
├── themes/                   # 配色主题 JSON
│   ├── teal.json
│   ├── purple.json
│   └── grey.json
├── data/                     # 用户数据（不提交）
│   ├── images/               # 导入的图片文件
│   ├── metadata.json         # 图片元数据 + 特征
│   ├── settings.json         # 用户设置
│   └── categories.json       # 自定义分类
└── 启动 Photomanager.vbs     # 一键启动脚本
```

## 📦 存储占用

| 项目 | 大小 |
|------|------|
| 项目代码 | ~60 MB |
| CLIP ViT-B/32 | ~300 MB（运行时加载） |
| vit-gpt2 caption | ~460 MB（运行时加载） |
| opus-mt-zh-en 翻译模型 | ~312 MB（可选） |
| PyTorch + transformers | ~650 MB（基础框架） |
| **合计** | **~1.8 GB**（不含翻译模型则为 ~1.5 GB） |

## 🔧 开发

### 添加自定义分类

在设置页的「分类管理」中添加，或直接编辑 `config.py` 中的 `DEFAULT_CATEGORIES`。

### 导入图片

支持 JPG / PNG / GIF / WebP 格式，图片文件保留在原位，只记录路径索引。
