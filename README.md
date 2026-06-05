# CLIP 零样本图像与语义跨模态检索系统

基于 OpenAI CLIP + BLIP 的跨模态检索 Web 应用。

## 功能

- 📤 **导入图片** — 支持本地上传和 URL 导入
- 🏷️ **自动分类** — CLIP 零样本分类，混合模式（预设 + 自动创建）
- 📝 **生成描述** — BLIP 自动为每张图片生成自然语言描述
- 🔍 **语义搜索** — 输入自然语言，精准匹配图片
- 📁 **分类管理** — 按大类浏览和筛选图片

## 快速开始

### 方式一：一键启动（推荐）

双击项目根目录的 **`CLIP检索系统.exe`**，自动完成所有操作：

```
双击 .exe
  → 检查 Python 依赖（torch, CLIP, transformers...）
  → 缺失则自动 pip install（首次 5~15 分钟）
  → 启动 FastAPI 服务
  → 打开浏览器
  → 控制台窗口保持运行（关闭即停止服务）
```

> ⚠️ 要求：系统需已安装 Python 3.8+，且 `pip` 可用。

### 方式二：命令行启动

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py

# 打开浏览器访问
# http://127.0.0.1:8000
```

首次启动会自动下载 CLIP 和 BLIP 模型（约 2GB）。图片导入后特征和描述会被缓存，搜索毫秒级响应。

## 技术栈

- **后端**: FastAPI + Uvicorn
- **视觉模型**: CLIP ViT-B/32（特征提取）、BLIP-base（图像描述）
- **前端**: 原生 HTML + CSS + JavaScript
- **存储**: 本地文件系统

## 项目结构

```
├── main.py              # FastAPI 入口 + API 路由
├── config.py            # 配置（阈值/模型名/预设类别）
├── models/
│   ├── clip_service.py  # CLIP 特征提取
│   └── caption_service.py  # BLIP 图像描述
├── services/
│   ├── classifier.py    # 分类引擎
│   └── search.py        # 检索服务
├── static/
│   ├── index.html       # 前端页面
│   ├── style.css        # 样式
│   └── app.js           # 交互逻辑
├── data/
│   ├── images/          # 图片存储
│   └── metadata.json    # 元数据
├── requirements.txt
├── CLIP检索系统.exe      # 一键启动器（双击运行）
└── README.md
```

## 配置

编辑 `config.py` 可调整：
- 预设分类类别
- 分类匹配阈值
- 检索结果数量
- 服务器地址和端口
