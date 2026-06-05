#!/usr/bin/env python
"""
Photomanager Desktop — 轻量级本地照片管理器
基于 CLIP 零样本分类 + 自然语言语义搜索
使用 CustomTkinter 构建原生桌面 UI，无需浏览器
"""
import datetime
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import customtkinter as ctk
from PIL import Image as PILImage
from tkinter import filedialog, messagebox

# 项目根目录
_PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import IMAGES_DIR, DATA_DIR, METADATA_PATH, CATEGORIES_PATH, SETTINGS_PATH, DEFAULT_CATEGORIES
from models.clip_service import CLIPService
from models.caption_service import CaptionService
from services.classifier import classify_image
from services.search import search_images
from services.translator import Translator

# ============================================================
# 全局外观设置
# ============================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme(str(_PROJECT_ROOT / "themes" / "teal.json"))
# 注意：用户保存在 settings.json 中的配色主题将在 __init__ 中延迟加载
THUMB_SIZE = 180
PAD = 12


# ============================================================
# 加载遮罩
# ============================================================
class LoadingOverlay(ctk.CTkToplevel):
    """模型加载期间的模态遮罩"""

    def __init__(self, parent):
        super().__init__(parent)
        self.overrideredirect(True)
        self.configure(fg_color="#2b2b2b")
        self.attributes("-alpha", 0.95)
        self.geometry("420x170")

        # 居中
        self.update_idletasks()
        try:
            pw, ph = parent.winfo_width(), parent.winfo_height()
            px, py = parent.winfo_x(), parent.winfo_y()
            x = px + (pw - 420) // 2
            y = py + (ph - 170) // 2
            self.geometry(f"+{x}+{y}")
        except Exception:
            pass

        self.label = ctk.CTkLabel(
            self, text="正在加载 AI 模型…",
            font=ctk.CTkFont(size=16)
        )
        self.label.pack(pady=(35, 8))

        self.status = ctk.CTkLabel(
            self, text="首次加载需下载模型 (~500MB)\n请耐心等待…",
            font=ctk.CTkFont(size=12), text_color="#888888"
        )
        self.status.pack(pady=2)

        self.progress = ctk.CTkProgressBar(self, mode="indeterminate", width=320)
        self.progress.pack(pady=10)
        self.progress.start()

        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", lambda: None)

    def set_msg(self, text: str):
        self.label.configure(text=text)


# ============================================================
# 图片详情窗口
# ============================================================
class _DetailPage(ctk.CTkFrame):
    """图片详情页（内嵌于主窗口）"""

    def __init__(self, parent, app, item):
        super().__init__(parent)
        self.app = app
        self.item = item
        self._build_ui()

    def _build_ui(self):
        scroll = ctk.CTkScrollableFrame(self, corner_radius=0)
        scroll.pack(fill="both", expand=True)

        # 图片
        load_path = None
        orig = self.item.get("original_path")
        if orig and os.path.isfile(orig):
            load_path = orig
        else:
            image_id = self.item["id"]
            for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
                p = IMAGES_DIR / f"{image_id}{ext}"
                if p.exists():
                    load_path = str(p)
                    break

        if load_path:
            try:
                pil = PILImage.open(load_path)
                pw, ph = pil.size
                max_w = self.app.main_area.winfo_width() - 80 or 700
                scale = min(max_w / pw, 520 / ph, 1.0)
                ns = (int(pw * scale), int(ph * scale))
                pil = pil.resize(ns, PILImage.LANCZOS)
                img = ctk.CTkImage(pil, size=ns)
                lbl = ctk.CTkLabel(scroll, image=img, text="")
                lbl.image = img  # keep ref
                lbl.pack(pady=(20, 10))
            except Exception:
                ctk.CTkLabel(scroll, text="[图片加载失败]").pack(pady=30)
        else:
            ctk.CTkLabel(scroll, text="[文件不存在或已被移动]").pack(pady=30)

        # 信息卡片
        card = ctk.CTkFrame(scroll, corner_radius=10)
        card.pack(fill="x", padx=30, pady=5)
        info_data = [
            ("文件名", self.item.get("filename", "")),
            ("路径", self.item.get("original_path", self.item.get("source_url", ""))),
            ("描述", self.item.get("caption", "") or "无"),
            ("分类", self.item.get("category", "?")),
            ("匹配度", str(self.item.get("similarity_score", 0))),
            ("来源", self.item.get("source", "")),
            ("时间", self.item.get("created_at", "")),
        ]
        for i, (k, v) in enumerate(info_data):
            row_f = ctk.CTkFrame(card, fg_color="transparent")
            row_f.pack(fill="x", padx=15, pady=4)
            ctk.CTkLabel(row_f, text=k + ":", font=ctk.CTkFont(weight="bold", size=12),
                         width=60, anchor="w").pack(side="left")
            ctk.CTkLabel(row_f, text=str(v), wraplength=500, anchor="w",
                         font=ctk.CTkFont(size=12)).pack(side="left", fill="x", expand=True)

        # 操作按钮
        btn_row = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_row.pack(pady=(15, 25))
        ctk.CTkButton(
            btn_row, text="🗑 移除此索引",
            fg_color="#c0392b", hover_color="#e74c3c",
            command=self._delete
        ).pack(side="left", padx=5)

    def _delete(self):
        if not messagebox.askyesno("确认", "确定移除这张图片的索引？\n原始文件不会被删除。"):
            return
        self.app._delete_image_by_id(self.item["id"])
        self.app._go_back()


# ============================================================
# 分类管理器
# ============================================================


class _CategoryPage(ctk.CTkFrame):
    """分类管理页（内嵌）"""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build_ui()

    def _build_ui(self):
        scroll = ctk.CTkScrollableFrame(self, corner_radius=0)
        scroll.pack(fill="both", expand=True, padx=20, pady=10)

        ctk.CTkLabel(scroll, text="预设分类（不可删除）",
                     font=ctk.CTkFont(size=13, weight="bold"), anchor="w"
                     ).pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(scroll, text="动物  建筑  风景  人物  食物  交通  植物  物品  艺术  科技",
                     font=ctk.CTkFont(size=11), text_color="#888888", anchor="w"
                     ).pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(scroll, text="自定义分类",
                     font=ctk.CTkFont(size=13, weight="bold"), anchor="w"
                     ).pack(fill="x", pady=(0, 4))

        self.list_frame = ctk.CTkScrollableFrame(scroll, corner_radius=6, height=200)
        self.list_frame.pack(fill="x", pady=(0, 10))
        self._refresh_list()

        add_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        add_frame.pack(fill="x", pady=(4, 8))
        self.new_cat_var = ctk.StringVar()
        ctk.CTkEntry(add_frame, placeholder_text="输入新分类名称…",
                      textvariable=self.new_cat_var).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(add_frame, text="添加", width=60, command=self._add_category).pack(side="right")

        ctk.CTkButton(scroll, text="🔄 重新分类所有图片",
                      command=self._reclassify, height=35).pack(pady=(4, 6))

        ctk.CTkButton(scroll, text="← 返回设置", fg_color="transparent",
                      border_width=1, command=lambda: self.app._go_back()).pack(pady=(2, 10))

    def _refresh_list(self):
        for w in self.list_frame.winfo_children():
            w.destroy()
        cats = self.app.user_categories
        if not cats:
            ctk.CTkLabel(self.list_frame, text="暂无自定义分类",
                         font=ctk.CTkFont(size=11), text_color="#666666").pack(pady=20)
            return
        for cat in sorted(cats):
            row = ctk.CTkFrame(self.list_frame, fg_color="transparent")
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=cat, anchor="w").pack(side="left", fill="x", expand=True)
            ctk.CTkButton(row, text="✕", width=28, fg_color="#c0392b", hover_color="#e74c3c",
                          command=lambda c=cat: self._delete_category(c)).pack(side="right")

    def _add_category(self):
        name = self.new_cat_var.get().strip()
        if not name:
            return
        if name in DEFAULT_CATEGORIES or name in self.app.user_categories:
            messagebox.showinfo("提示", "该分类已存在")
            return
        self.app.user_categories.append(name)
        self.new_cat_var.set("")
        self.app._on_categories_saved(self.app.user_categories)
        self._refresh_list()

    def _delete_category(self, cat: str):
        if not messagebox.askyesno("确认", f"确定删除分类「{cat}」？已归入该分类的图片将被重新分类。"):
            return
        self.app.user_categories.remove(cat)
        self.app._on_categories_saved(self.app.user_categories)
        self.app._reclassify_all()
        self._refresh_list()

    def _reclassify(self):
        self.app._reclassify_all()
        messagebox.showinfo("完成", "已重新分类所有图片")

    def _refresh_ui(self):
        """供缓存复用：刷新分类列表"""
        self._refresh_list()

class _SettingsPage(ctk.CTkFrame):
    """设置页面，嵌入主内容区，非独立窗口"""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.settings = app.settings

        self._build_ui()

    def _build_ui(self):
        main = ctk.CTkScrollableFrame(self, corner_radius=0)
        main.pack(fill="both", expand=True, padx=30, pady=10)

        # ── Appearance ──
        ctk.CTkLabel(main, text="界面风格" if _current_lang == "zh" else "Appearance",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", pady=(0, 8))

        row1 = ctk.CTkFrame(main, fg_color="transparent")
        row1.pack(fill="x", pady=3)
        ctk.CTkLabel(row1, text="主题模式:" if _current_lang == "zh" else "Theme:",
                     width=120, anchor="w").pack(side="left")
        self.mode_var = ctk.StringVar(value=self.settings.get("appearance_mode", "dark"))
        ctk.CTkOptionMenu(row1, values=["dark", "light", "system"],
                          variable=self.mode_var,
                          command=self._on_mode_change).pack(side="right")

        row2 = ctk.CTkFrame(main, fg_color="transparent")
        row2.pack(fill="x", pady=3)
        ctk.CTkLabel(row2, text="配色主题:" if _current_lang == "zh" else "Color:",
                     width=120, anchor="w").pack(side="left")
        self.theme_var = ctk.StringVar(value=self.settings.get("theme_file", "teal.json"))
        ctk.CTkOptionMenu(row2,
                          values=["teal.json", "purple.json", "grey.json",
                                  "blue.json", "dark-blue.json", "green.json"],
                          variable=self.theme_var).pack(side="right")
        ctk.CTkLabel(main, text="（配色修改需重启应用）" if _current_lang == "zh" else "(restart required)",
                     font=ctk.CTkFont(size=10), text_color="#888888").pack(anchor="e")

        ctk.CTkFrame(main, height=1).pack(fill="x", pady=14)

        # ── Language ──
        ctk.CTkLabel(main, text="语言" if _current_lang == "zh" else "Language",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", pady=(0, 8))
        row3 = ctk.CTkFrame(main, fg_color="transparent")
        row3.pack(fill="x", pady=3)
        ctk.CTkLabel(row3, text="语言:" if _current_lang == "zh" else "Language:",
                     width=120, anchor="w").pack(side="left")
        self.lang_var = ctk.StringVar(value=self.settings.get("language", "zh"))
        ctk.CTkOptionMenu(row3, values=["中文", "English"],
                          variable=self.lang_var).pack(side="right")
        ctk.CTkLabel(main, text="（修改语言需重启应用）" if _current_lang == "zh" else "(restart required)",
                     font=ctk.CTkFont(size=10), text_color="#888888").pack(anchor="e")

        ctk.CTkFrame(main, height=1).pack(fill="x", pady=14)

        # ── Display ──
        ctk.CTkLabel(main, text="显示" if _current_lang == "zh" else "Display",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", pady=(0, 8))
        row4 = ctk.CTkFrame(main, fg_color="transparent")
        row4.pack(fill="x", pady=3)
        ctk.CTkLabel(row4, text="网格列数:" if _current_lang == "zh" else "Grid columns:",
                     width=120, anchor="w").pack(side="left")
        self.grid_var = ctk.StringVar(value=str(self.settings.get("grid_columns", 5)))
        ctk.CTkOptionMenu(row4, values=["3", "4", "5", "6"],
                          variable=self.grid_var).pack(side="right")

        # ── Font ──
        row5 = ctk.CTkFrame(main, fg_color="transparent")
        row5.pack(fill="x", pady=3)
        ctk.CTkLabel(row5, text="字体大小:" if _current_lang == "zh" else "Font size:",
                     width=120, anchor="w").pack(side="left")
        self.font_size_var = ctk.StringVar(value=str(self.settings.get("font_size", 13)))
        ctk.CTkOptionMenu(row5, values=["9", "10", "11", "12", "13", "14", "15", "16", "18", "20", "22", "24", "28", "32"],
                          variable=self.font_size_var).pack(side="right")

        row6 = ctk.CTkFrame(main, fg_color="transparent")
        row6.pack(fill="x", pady=3)
        ctk.CTkLabel(row6, text="字体:" if _current_lang == "zh" else "Font:",
                     width=120, anchor="w").pack(side="left")
        self.font_family_var = ctk.StringVar(value=self.settings.get("font_family", "Microsoft YaHei"))
        ctk.CTkOptionMenu(row6, values=["Microsoft YaHei", "Segoe UI", "Roboto", "Arial", "Consolas"],
                          variable=self.font_family_var).pack(side="right")
        ctk.CTkLabel(main, text="（字体修改需重启应用）" if _current_lang == "zh" else "(restart required)",
                     font=ctk.CTkFont(size=10), text_color="#888888").pack(anchor="e")

        # ── Font Color ──
        row7 = ctk.CTkFrame(main, fg_color="transparent")
        row7.pack(fill="x", pady=3)
        ctk.CTkLabel(row7, text="字体颜色:" if _current_lang == "zh" else "Text color:",
                     width=120, anchor="w").pack(side="left")
        self.font_color_var = ctk.StringVar(value=self.settings.get("font_color", ""))
        color_swatches = [
            ("", "默认", "#888888"),
            ("#FFFFFF", "", "#FFFFFF"),
            ("#E8E8E8", "", "#E8E8E8"),
            ("#AAAAAA", "", "#AAAAAA"),
            ("#555555", "", "#555555"),
            ("#1A1A1A", "", "#1A1A1A"),
            ("#4A90D9", "", "#4A90D9"),
            ("#E74C3C", "", "#E74C3C"),
            ("#2ECC71", "", "#2ECC71"),
            ("#F39C12", "", "#F39C12"),
            ("#9B59B6", "", "#9B59B6"),
            ("#1ABC9C", "", "#1ABC9C"),
        ]
        swatch_frame = ctk.CTkFrame(row7, fg_color="transparent")
        swatch_frame.pack(side="right")
        for val, label, bg in color_swatches:
            is_default = val == ""
            btn = ctk.CTkButton(
                swatch_frame, text=label or "", width=26, height=26,
                fg_color=bg if not is_default else "transparent",
                border_width=2 if is_default else 1,
                border_color="#666666",
                hover_color=bg if not is_default else "#444444",
                command=lambda v=val: self.font_color_var.set(v),
            )
            btn.pack(side="left", padx=2)

        ctk.CTkLabel(main, text="（修改字体颜色需重启应用）" if _current_lang == "zh" else "(restart required)",
                     font=ctk.CTkFont(size=10), text_color="#888888").pack(anchor="e")

        ctk.CTkFrame(main, height=1).pack(fill="x", pady=14)

        # ── Categories ──
        ctk.CTkLabel(main, text="分类管理" if _current_lang == "zh" else "Categories",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", pady=(0, 8))
        ctk.CTkButton(main, text="打开分类管理器" if _current_lang == "zh" else "Open Manager",
                      command=self._open_cat_mgr).pack(anchor="w", pady=3)

        ctk.CTkFrame(main, height=1).pack(fill="x", pady=14)

        # ── Save ──
        btn_row = ctk.CTkFrame(main, fg_color="transparent")
        btn_row.pack(fill="x", pady=(6, 0))
        ctk.CTkButton(btn_row, text="保存设置" if _current_lang == "zh" else "Save Settings",
                      command=self._save, height=38,
                      font=ctk.CTkFont(size=13, weight="bold")).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btn_row, text="返回主页" if _current_lang == "zh" else "Back",
                      command=self.app._go_back,
                      fg_color="transparent", border_width=1).pack(side="left")

    def _refresh_ui(self):
        """刷新设置值（当从其他地方修改了设置后调用）"""
        self.settings = self.app.settings
        self.mode_var.set(self.settings.get("appearance_mode", "dark"))
        self.theme_var.set(self.settings.get("theme_file", "teal.json"))
        self.lang_var.set("中文" if self.settings.get("language", "zh") == "zh" else "English")
        self.grid_var.set(str(self.settings.get("grid_columns", 5)))
        self.font_size_var.set(str(self.settings.get("font_size", 13)))
        self.font_family_var.set(self.settings.get("font_family", "Microsoft YaHei"))
        self.font_color_var.set(self.settings.get("font_color", ""))

    def _on_mode_change(self, mode: str):
        ctk.set_appearance_mode(mode)
        self.update()

    def _open_cat_mgr(self):
        self.app._navigate_to("categories")

    def _save(self):
        lang_map = {"中文": "zh", "English": "en"}
        old = dict(self.settings)  # 旧值快照
        self.settings["appearance_mode"] = self.mode_var.get()
        self.settings["theme_file"] = self.theme_var.get()
        new_lang = lang_map.get(self.lang_var.get(), "zh")
        self.settings["language"] = new_lang
        self.settings["grid_columns"] = int(self.grid_var.get())
        self.settings["font_size"] = int(self.font_size_var.get())
        self.settings["font_family"] = self.font_family_var.get()
        self.settings["font_color"] = self.font_color_var.get()

        # 检查需要重启的设置项是否变动
        restart_keys = {"theme_file", "language", "font_size", "font_family", "font_color"}
        need_restart = any(self.settings[k] != old.get(k) for k in restart_keys)

        self.app._save_settings(self.settings)
        global _current_lang
        _current_lang = new_lang

        if need_restart:
            msg = ("部分设置需重启生效，正在重启…" if _current_lang == "zh"
                   else "Some settings require a restart. Restarting…")
            self.app._go_back()
            messagebox.showinfo(
                "提示" if _current_lang == "zh" else "Info",
                msg,
            )
            self.app.after(200, self.app._restart_app)
        else:
            self.app._go_back()

class PhotoManager(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Photomanager")
        self.geometry("900x600")
        self.minsize(900, 600)

        # 核心服务
        self.clip = CLIPService()
        self.captioner = CaptionService()

        # 数据
        self.metadata = self._load_metadata()
        self.filtered = list(self.metadata)
        self.filter_mode: str = "all"   # "all" | "category:<name>" | "search"
        self.thumb_cache: dict[str, ctk.CTkImage] = {}
        self.user_categories = self._load_categories()
        self.settings = self._load_settings()
        # 从设置加载用户保存的配色主题（覆盖模块级默认值）
        _saved_theme = self.settings.get("theme_file", "teal.json")
        _theme_path = _PROJECT_ROOT / "themes" / _saved_theme
        if _theme_path.exists():
            ctk.set_default_color_theme(str(_theme_path))
        self._apply_font_color()

        global _current_lang
        _current_lang = self.settings.get("language", "zh")

        # 模型状态
        self.models_ready = False

        # 构建 UI
        self._build_ui()

        # 启动模型加载
        self.overlay = None
        self.after(10, self._init_models)


    def _fade_in(self):
        """启动时窗口淡入动画"""
        for i in range(1, 11):
            self.after(i * 25, lambda v=i/10: self.attributes("-alpha", v))

    # ────────────────────────────── 字体辅助 ──────────────────────────────

    def _apply_font_color(self):
        """将 font_color 设置应用到全局主题"""
        fc = self.settings.get("font_color", "")
        if fc and hasattr(ctk, 'ThemeManager'):
            for _key in ('CTkLabel','CTkButton','CTkEntry','CTkOptionMenu',
                         'CTkComboBox','CTkTextbox','CTkRadioButton',
                         'CTkCheckBox','CTkSwitch'):
                entry = ctk.ThemeManager.theme.get(_key)
                if entry and 'text_color' in entry:
                    entry['text_color'] = [fc, fc]

    def _font(self, size=13, weight="normal"):
        """根据当前设置创建字体对象"""
        family = self.settings.get("font_family", "Microsoft YaHei")
        sz = self.settings.get("font_size", 13)
        return ctk.CTkFont(family=family, size=sz if size == 13 else size, weight=weight)

    # ────────────────────────────── 模型加载 ──────────────────────────────

    def _init_models(self):
        self.overlay = LoadingOverlay(self)
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        t0 = time.time()
        self.after(0, lambda: self.overlay.set_msg("正在加载 CLIP 模型 (ViT-B/32)…"))
        try:
            _ = self.clip.model
            _ = self.clip.preprocess
        except Exception as e:
            self.after(0, lambda: self._log_error(f"CLIP 加载失败: {e}"))

        # CLIP 加载完毕 → 标记就绪，用户可操作
        elapsed = time.time() - t0
        self.models_ready = True
        self.after(0, lambda: self._on_models_loaded(elapsed))

        # Caption 模型后台加载（不阻塞主功能）
        self.after(0, lambda: self.overlay.set_msg("正在加载描述生成模型 (vit-gpt2)…"))
        threading.Thread(target=self._load_caption, daemon=True).start()

        # 翻译模型后台加载（中→英，约 300MB）
        threading.Thread(target=self._load_translator, daemon=True).start()

    def _load_translator(self):
        """后台线程：加载中文→英文翻译模型（用于中文搜索）"""
        try:
            t = Translator()
            t._try_load()
            if t.available:
                print("翻译模型就绪")
        except Exception as e:
            print(f"翻译模型加载失败: {e}")

    def _load_caption(self):
        """后台线程：加载 vit-gpt2 并生成描述"""
        try:
            self.after(0, lambda: self.overlay and self.overlay.set_msg("正在加载描述生成模型 (vit-gpt2)…"))
            self.captioner._try_load()
            if self.captioner.available:
                self.after(0, lambda: self.overlay and self.overlay.set_msg("正在生成图片描述…"))
                self._generate_captions_batch()
                # 为已有 caption 补算 CLIP 特征（跨语言搜索用）
                n = self._backfill_caption_features()
                if n:
                    print(f"已补算 {n} 条 caption 的 CLIP 特征")
                self.after(0, lambda: self.model_lbl.configure(text="✅ 描述模型就绪"))
        except Exception as e:
            self.after(0, lambda: self._log_error(f"Caption 后台加载失败: {e}"))

    def _generate_captions_batch(self):
        """为 metadata 中没有 caption 的图片批量生成描述"""
        from PIL import Image as PILImage
        import os

        to_process = [m for m in self.metadata if not m.get("caption")]
        if not to_process:
            return

        images = []
        indices = []
        for idx, m in enumerate(self.metadata):
            if not m.get("caption"):
                orig = m.get("original_path")
                if orig and os.path.isfile(orig):
                    try:
                        img = PILImage.open(orig).convert("RGB")
                        images.append(img)
                        indices.append(idx)
                    except Exception:
                        continue

        if not images:
            return

        captions = self.captioner.generate_batch(images, max_length=30)
        for idx, cap in zip(indices, captions):
            if cap:
                self.metadata[idx]["caption"] = cap

        # 为新 caption 计算 CLIP 文本特征（跨语言搜索用）
        for idx, cap in zip(indices, captions):
            if cap:
                try:
                    feat = self.clip.encode_text([cap])  # (1, 512)
                    self.metadata[idx]["caption_feature"] = feat
                except Exception:
                    pass

        self._save_metadata()

    def _backfill_caption_features(self):
        """为已有 caption 但缺 caption_feature 的图片补算 CLIP 特征"""
        count = 0
        for m in self.metadata:
            if m.get("caption") and "caption_feature" not in m:
                try:
                    m["caption_feature"] = self.clip.encode_text([m["caption"]])
                    count += 1
                except Exception:
                    pass
        if count:
            self._save_metadata()
        return count

    @staticmethod
    def _log_error(msg: str):
        import traceback as _tb
        from pathlib import Path as _P
        try:
            log = _P(__file__).with_name("entry.log")
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')} [desktop] {msg}\n")
        except Exception:
            pass

    def _on_models_loaded(self, elapsed: float):
        if self.overlay:
            self.overlay.destroy()
            self.overlay = None
        self.model_lbl.configure(text=f"✅ 模型就绪 ({elapsed:.0f}s)")
        self._refresh_thumbnails()

    # ────────────────────────────── UI 构建 ──────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ====== 侧边栏 ======
        sb = ctk.CTkFrame(self, width=220, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(
            sb, text="📸 PM",
            font=self._font(16, weight="bold")
        ).grid(row=0, column=0, pady=(15, 8))

        self.import_btn = ctk.CTkButton(
            sb, text="📁 导入图片", command=self._import_local
        )
        self.import_btn.grid(row=1, column=0, padx=PAD, pady=2, sticky="ew")

        ctk.CTkFrame(sb, height=2).grid(
            row=2, column=0, padx=PAD, pady=(6, 6), sticky="ew"
        )

        ctk.CTkLabel(
            sb, text="🗂 分类", font=self._font(13, weight="bold")
        ).grid(row=4, column=0, pady=(0, 2))

        self.cat_scroll = ctk.CTkScrollableFrame(
            sb, corner_radius=0, fg_color="transparent"
        )
        self.cat_scroll.grid(row=5, column=0, sticky="nsew")
        self.cat_scroll.grid_columnconfigure(0, weight=1)

        # "全部" 按钮
        self._cat_btns: dict[str, ctk.CTkButton] = {}
        self._cat_all = ctk.CTkButton(
            self.cat_scroll, text="🏠 全部图片",
            anchor="w", fg_color="transparent",
            command=lambda: self._set_filter("all")
        )
        self._cat_all.grid(row=0, column=0, sticky="ew", pady=1)
        self._cat_btns["all"] = self._cat_all

        # 底部状态
        self.stats_lbl = ctk.CTkLabel(
            sb, text="", font=self._font(11)
        )
        self.stats_lbl.grid(row=6, column=0, padx=PAD, pady=(0, 8), sticky="w")

        self.model_lbl = ctk.CTkLabel(
            sb, text="⏳ 模型加载中…", font=self._font(11), text_color="#f39c12"
        )
        self.model_lbl.grid(row=7, column=0, padx=PAD, pady=(0, 10), sticky="w")

        ctk.CTkButton(
            sb, text="⚙ 设置", fg_color="transparent", border_width=1,
            font=self._font(11), command=self._open_settings
        ).grid(row=9, column=0, padx=PAD, pady=(2, 10), sticky="ew")

        # ====== 主内容区（页面系统）======
        self.main_area = ctk.CTkFrame(self, corner_radius=0)
        self.main_area.grid(row=0, column=1, sticky="nsew")
        self.main_area.grid_columnconfigure(0, weight=1)
        self.main_area.grid_rowconfigure(1, weight=1)

        # 顶部导航栏（搜索 / 页面标题 + 返回按钮）
        self.nav_bar = ctk.CTkFrame(self.main_area, corner_radius=0, fg_color="transparent")
        self.nav_bar.grid(row=0, column=0, sticky="ew", padx=0, pady=(8, 0))
        self.nav_bar.grid_columnconfigure(0, weight=1)

        # 返回按钮（默认隐藏）
        self.back_btn = ctk.CTkButton(
            self.nav_bar, text="← 返回", width=60,
            fg_color="transparent", border_width=0,
            font=self._font(13),
            command=self._go_back
        )
        # 搜索控件
        self.search_var = ctk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._on_search_change())
        self.search_entry = ctk.CTkEntry(
            self.nav_bar, placeholder_text="🔍 用文字搜索图片…",
            textvariable=self.search_var, height=36
        )
        self.search_entry.bind("<Return>", lambda e: self._do_search())

        self.search_btn = ctk.CTkButton(
            self.nav_bar, text="搜索", width=70, command=self._do_search
        )

        self.clear_btn = ctk.CTkButton(
            self.nav_bar, text="✕", width=30,
            fg_color="transparent", border_width=1,
            command=self._clear_search
        )

        self._show_search_bar()  # 初始显示搜索栏

        # 页面容器（存放主页 / 设置页等）
        self.page_container = ctk.CTkFrame(self.main_area, corner_radius=0, fg_color="transparent")
        self.page_container.grid(row=1, column=0, sticky="nsew")
        self.page_container.grid_columnconfigure(0, weight=1)
        self.page_container.grid_rowconfigure(0, weight=1)

        # 主页 = 缩略图网格
        self.grid_frame = ctk.CTkScrollableFrame(self.page_container, corner_radius=0)
        self.grid_frame.grid(row=0, column=0, sticky="nsew")
        self.grid_frame.grid_columnconfigure(tuple(range(self.settings.get("grid_columns", 5))), weight=1)

        # 设置页（延迟创建）
        self.settings_page = None

        # 底部状态栏
        status = ctk.CTkFrame(self.main_area, height=24, corner_radius=0)
        status.grid(row=2, column=0, sticky="ew")
        self.status_msg = ctk.CTkLabel(
            status, text="就绪", font=self._font(11), anchor="w"
        )
        self.status_msg.grid(row=0, column=0, padx=PAD, sticky="w")

        # 初始化侧边栏
        self._refresh_categories()
        self._update_stats()
        # 初始显示提示
        self._refresh_thumbnails()

    # ────────────────────────────── 状态提示 ──────────────────────────────

    def _status(self, msg: str):
        self.status_msg.configure(text=msg)

    # ────────────────────────────── 页面导航 ──────────────────────────────

    def _show_search_bar(self):
        """显示搜索栏，隐藏返回按钮"""
        self.back_btn.grid_forget()
        self.search_entry.grid(row=0, column=0, padx=(PAD, 4), pady=(0, 8), sticky="ew")
        self.search_btn.grid(row=0, column=1, padx=(0, 4), pady=(0, 8))
        self.clear_btn.grid(row=0, column=2, padx=(0, PAD), pady=(0, 8))

    def _show_back_bar(self, title: str):
        """显示返回按钮 + 页面标题，隐藏搜索栏"""
        self.search_entry.grid_forget()
        self.search_btn.grid_forget()
        self.clear_btn.grid_forget()
        self.back_btn.configure(text=f"\u2190  {title}")
        self.back_btn.grid(row=0, column=0, padx=PAD, pady=(0, 8), sticky="w")


    # ────────────────────────────── 页面导航系统 ──────────────────────────────

    def _navigate_to(self, page: str, **kwargs):
        """切换到指定页面（带动画）"""
        if not hasattr(self, '_page_stack'):
            self._page_stack = ["main"]

        old = None
        for child in self.page_container.winfo_children():
            if child.winfo_ismapped():
                old = child
                break

        if page == "detail":
            item = kwargs.get("item")
            if not item:
                return
            new_page = _DetailPage(self.page_container, self, item)
        elif page == "settings":
            if not hasattr(self, '_settings_page_obj') or self._settings_page_obj is None:
                self._settings_page_obj = _SettingsPage(self.page_container, self)
            new_page = self._settings_page_obj
            new_page._refresh_ui()
        elif page == "categories":
            if not hasattr(self, '_cat_page_obj') or self._cat_page_obj is None:
                self._cat_page_obj = _CategoryPage(self.page_container, self)
            new_page = self._cat_page_obj
            # 分类页内容动态刷新（不重建）
            self._cat_page_obj._refresh_ui()
        else:
            return

        self._page_stack.append(page)

        # 让 tkinter 先完成新页面的布局计算
        self.update_idletasks()

        w = self.page_container.winfo_width() or 900
        new_page.place(x=w, y=0, relwidth=1, relheight=1)
        new_page.tkraise()  # new 覆盖在 old 之上

        if old:
            steps = 8
            for i in range(1, steps + 1):
                t = i / steps
                ease = 1 - (1 - t) * (1 - t)  # ease-out 缓出
                x = w * (1 - ease)
                self.after(i * 16, lambda x=x, n=new_page: (
                    n.place(x=x, y=0, relwidth=1, relheight=1),
                ))
            self.after(steps * 16 + 20, lambda o=old: self._finish_slide(o, new_page))
        else:
            new_page.place(x=0, y=0, relwidth=1, relheight=1)

        titles = {"detail": "图片详情", "settings": "设置", "categories": "分类管理"}
        self._show_back_bar(titles.get(page, ""))

    def _finish_slide(self, old_page, new_page):
        try:
            old_page.place_forget()
        except Exception:
            pass
        new_page.place(x=0, y=0, relwidth=1, relheight=1)

    def _go_back(self):
        if not hasattr(self, '_page_stack') or len(self._page_stack) <= 1:
            self._show_search_bar()
            self._show_page(self.grid_frame)
            self._refresh_thumbnails()
            self._page_stack = ["main"]
            return

        self._page_stack.pop()
        prev_page = self._page_stack[-1]

        if prev_page == "main":
            self._show_search_bar()
            self._show_page(self.grid_frame)
            self._refresh_thumbnails()
        elif prev_page == "settings" and hasattr(self, '_settings_page_obj') and self._settings_page_obj:
            self._settings_page_obj._refresh_ui()
            self._show_back_bar("设置")
            self._show_page(self._settings_page_obj)
        elif prev_page == "categories" and hasattr(self, '_cat_page_obj') and self._cat_page_obj:
            self._cat_page_obj._refresh_ui()
            self._show_back_bar("分类管理")
            self._show_page(self._cat_page_obj)
        else:
            # fallback: 回到主页
            self._show_search_bar()
            self._show_page(self.grid_frame)
            self._refresh_thumbnails()
            self._page_stack = ["main"]

    def _show_page(self, page):
        """显示指定页面，隐藏 page_container 中其他页面"""
        for child in self.page_container.winfo_children():
            if child is not page:
                try:
                    child.place_forget()
                except Exception:
                    pass
        # grid_frame 是 CTkScrollableFrame，必须用 grid() 保持内部 canvas 尺寸正确
        if page is self.grid_frame:
            page.grid(row=0, column=0, sticky="nsew")
        else:
            page.place(x=0, y=0, relwidth=1, relheight=1)
        self.update_idletasks()

    @staticmethod
    def _safe_destroy(widget):
        try:
            widget.destroy()
        except Exception:
            pass

    def _load_metadata(self) -> list[dict]:
        import torch
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                if "feature" in d and isinstance(d["feature"], list):
                    d["feature"] = torch.tensor(d["feature"]).unsqueeze(0)
                if "caption_feature" in d and isinstance(d["caption_feature"], list):
                    d["caption_feature"] = torch.tensor(d["caption_feature"]).unsqueeze(0)
            return data
        return []

    def _save_metadata(self):
        import torch
        ser = []
        for d in self.metadata:
            c = dict(d)
            if "feature" in c and isinstance(c["feature"], torch.Tensor):
                c["feature"] = c["feature"].squeeze(0).tolist()
            if "caption_feature" in c and isinstance(c["caption_feature"], torch.Tensor):
                c["caption_feature"] = c["caption_feature"].squeeze(0).tolist()
            ser.append(c)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(ser, f, ensure_ascii=False, indent=2)

    # ────────────────────────────── 用户自定义分类 ──────────────────────────────

    # ────────────────────────────── 设置 ──────────────────────────────

    def _load_settings(self) -> dict:
        if SETTINGS_PATH.exists():
            try:
                with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                    s = json.load(f)
                    return {**dict(appearance_mode="dark", color_theme="teal", language="zh", grid_columns=5, show_captions=True, theme_file="teal.json", font_size=13, font_family="Microsoft YaHei", font_color=""), **s}
            except Exception:
                pass
        return dict(dict(appearance_mode="dark", color_theme="teal", language="zh", grid_columns=5, show_captions=True, theme_file="teal.json", font_size=13, font_family="Microsoft YaHei", font_color=""))

    def _save_settings(self, s: dict):
        self.settings = s
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)

    def _open_settings(self):
        self._navigate_to("settings")

    def _load_categories(self) -> list[str]:
        if CATEGORIES_PATH.exists():
            try:
                with open(CATEGORIES_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_categories(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(CATEGORIES_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.user_categories, f, ensure_ascii=False, indent=2)

    # ────────────────────────────── 导入 ──────────────────────────────

    def _import_local(self):
        if not self.models_ready:
            messagebox.showinfo("提示", "AI 模型正在加载中，请稍候再导入图片。")
            return
        files = filedialog.askopenfilenames(
            title="选择要导入的图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.gif *.webp"), ("所有文件", "*.*")]
        )
        if files:
            self._process_batch(list(files), "local")


    def _process_batch(self, paths: list[str], source: str):
        n = len(paths)
        self.import_btn.configure(state="disabled", text=f"⏳ 导入中 (0/{n})…")
        self._status(f"正在处理 {n} 张图片…")

        def worker():
            imported, errors = 0, []
            for p in paths:
                try:
                    img = PILImage.open(p).convert("RGB")
                    meta = self._process_one(img, Path(p).name, source, original_path=p)
                    self.metadata.append(meta)
                    imported += 1
                    self.after(0, lambda i=imported: self.import_btn.configure(
                        text=f"⏳ 导入中 ({i}/{n})…"
                    ))
                except Exception as e:
                    errors.append({"file": p, "error": str(e)})
                    self._log_error(f"导入失败 {Path(p).name}: {e}")
                    import traceback as _tb
                    self._log_error(_tb.format_exc())
            self._save_metadata()
            self.after(0, lambda: self._on_import_done(imported, errors))

        threading.Thread(target=worker, daemon=True).start()

    def _process_one(self, pil_img: PILImage.Image, filename: str, source: str,
                     original_path: str | None = None) -> dict:
        """处理单张图片：提取特征 → 描述 → 分类
        不复制图片文件，只记录原始路径作为索引。
        """
        image_id = str(uuid.uuid4())

        feat = self.clip.encode_image(pil_img)

        cat, score = classify_image(feat, self.clip, self.user_categories)

        return {
            "id": image_id,
            "filename": filename,
            "source": source,
            "original_path": original_path or "",
            "category": cat,
            "similarity_score": round(score, 3),
            "feature": feat,
            "created_at": datetime.datetime.now().isoformat(),
        }

    def _on_import_done(self, imported: int, errors: list[dict]):
        self.import_btn.configure(state="normal", text="📁 导入图片")

        msg = f"✅ 导入完成：{imported} 张成功"
        if errors:
            msg += f"，{len(errors)} 张失败"
        messagebox.showinfo("导入结果", msg)

        self._refresh_categories()
        self._update_stats()
        if self.filter_mode == "all":
            self._refresh_thumbnails()
        else:
            self._set_filter(self.filter_mode)
        self._status(f"共 {len(self.metadata)} 张图片")

    # ────────────────────────────── 缩略图 ──────────────────────────────

    def _refresh_thumbnails(self):
        """重建缩略图网格"""
        for w in self.grid_frame.winfo_children():
            w.destroy()
        self.thumb_cache.clear()

        data = self.filtered if self.filter_mode != "all" else self.metadata

        if not data:
            ctk.CTkLabel(
                self.grid_frame,
                text="✨ 暂无图片\n点击左侧「导入图片」开始整理你的照片",
                font=self._font(14)
            ).grid(row=0, column=0, columnspan=self.settings.get("grid_columns", 5), pady=100)
            return

        for idx, item in enumerate(data):
            gc = self.settings.get("grid_columns", 5)
            r, c = idx // gc, idx % gc
            frame = ctk.CTkFrame(
                self.grid_frame, corner_radius=10,
                border_width=1, border_color=["gray78", "gray25"]
            )
            frame.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")
            frame.grid_columnconfigure(0, weight=1)
            # Hover border highlight
            frame.bind("<Enter>", lambda e, f=frame: f.configure(border_color=["#17979E", "#10676B"]))
            frame.bind("<Leave>", lambda e, f=frame: f.configure(border_color=["gray78", "gray25"]))

            # 缩略图
            img = self._get_thumb(item)
            if img:
                lbl = ctk.CTkLabel(frame, image=img, text="")
            else:
                lbl = ctk.CTkLabel(
                    frame, text="🖼️", font=self._font(32)
                )
            lbl.grid(row=0, column=0, padx=2, pady=(2, 0))

            # 描述
            cap = item.get("caption", "")
            cap_display = (cap[:25] + "…") if len(cap) > 25 else cap
            ctk.CTkLabel(
                frame, text=cap_display, font=self._font(9)
            ).grid(row=1, column=0, padx=2, pady=(1, 0))

            # 分类 + 分数
            cat = item.get("category", "?")
            score = item.get("similarity_score", 0)
            ctk.CTkLabel(
                frame, text=f"{cat} ({score})",
                font=self._font(8), text_color="#999999"
            ).grid(row=2, column=0, padx=2, pady=(0, 2))

            # 点击查看详情
            img_id = item["id"]
            for widget in (frame, lbl):
                widget.bind(
                    "<Button-1>",
                    lambda e, i=img_id: self._view_image(i)
                )

    def _get_thumb(self, item: dict) -> ctk.CTkImage | None:
        image_id = item["id"]
        if image_id in self.thumb_cache:
            return self.thumb_cache[image_id]

        # 优先原始路径（本地索引模式）
        orig = item.get("original_path")
        if orig and os.path.isfile(orig):
            try:
                pil = PILImage.open(orig)
                pil.thumbnail((THUMB_SIZE, THUMB_SIZE), PILImage.LANCZOS)
                ct = ctk.CTkImage(pil, size=pil.size)
                self.thumb_cache[image_id] = ct
                return ct
            except Exception:
                pass

        # 回退：从 data/images/ 加载（URL 导入的图片保存于此）
        for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            path = IMAGES_DIR / f"{image_id}{ext}"
            if path.exists():
                try:
                    pil = PILImage.open(path)
                    pil.thumbnail((THUMB_SIZE, THUMB_SIZE), PILImage.LANCZOS)
                    ct = ctk.CTkImage(pil, size=pil.size)
                    self.thumb_cache[image_id] = ct
                    return ct
                except Exception:
                    return None
        return None

    # ────────────────────────────── 搜索 & 筛选 ──────────────────────────────

    def _on_search_change(self):
        """搜索框内容变化时仅清空按钮状态；用户按 Enter 或点击搜索按钮才执行搜索"""
        pass

    def _clear_search(self):
        self.search_var.set("")
        self.filter_mode = "all"
        self.filtered = list(self.metadata)
        self._refresh_thumbnails()
        self._update_cat_style("all")
        self._status("")

    def _do_search(self):
        q = self.search_var.get().strip()
        if not q or not self.metadata:
            return
        self._status(f"🔍 正在搜索「{q}」…")

        def worker():
            results = search_images(q, self.clip, self.metadata)
            ids = {r["id"] for r in results}
            self.filtered = [m for m in self.metadata if m["id"] in ids]
            # 按搜索排序
            rank = {r["id"]: i for i, r in enumerate(results)}
            self.filtered.sort(key=lambda m: rank.get(m["id"], 999))
            self.after(0, lambda: self._search_done(q))

        threading.Thread(target=worker, daemon=True).start()

    def _search_done(self, q: str):
        self.filter_mode = "search"
        self._refresh_thumbnails()
        self._update_cat_style("search")
        self._status(f"🔍 「{q}」: 找到 {len(self.filtered)} 张")

    def _set_filter(self, cat: str):
        if cat == "all":
            self.filtered = list(self.metadata)
            self.filter_mode = "all"
        else:
            self.filtered = [m for m in self.metadata if m.get("category") == cat]
            self.filter_mode = cat
        self.search_var.set("")
        self._refresh_thumbnails()
        self._update_cat_style(cat)
        label = "全部" if cat == "all" else cat
        self._status(f"{label}: {len(self.filtered)} 张")

    def _update_cat_style(self, active: str):
        for name, btn in self._cat_btns.items():
            btn.configure(fg_color=["#1a7075", "#1a4a4d"] if name == active else "transparent")

    # ────────────────────────────── 查看图片 ──────────────────────────────

    def _view_image(self, image_id: str):
        item = next((m for m in self.metadata if m["id"] == image_id), None)
        if not item:
            return

        self._navigate_to("detail", item=item)

    def _delete_image_by_id(self, img_id: str):
        self.metadata = [m for m in self.metadata if m["id"] != img_id]
        self._save_metadata()
        self.thumb_cache.pop(img_id, None)
        self._refresh_categories()
        self._update_stats()
        self._refresh_thumbnails()
        self._status(f"已删除，共 {len(self.metadata)} 张")

    # ────────────────────────────── 侧边栏 ──────────────────────────────

    # ────────────────────────────── 分类管理 ──────────────────────────────

    def _open_category_manager(self):
        self._navigate_to("categories")

    def _on_categories_saved(self, categories: list[str]):
        self.user_categories = categories
        self._save_categories()

    def _reclassify_all(self):
        """重新分类所有已有图片"""
        if not self.metadata:
            return
        self._status("正在重新分类所有图片...")
        def worker():
            for m in self.metadata:
                try:
                    from PIL import Image as _PILImage
                    orig = m.get("original_path", "")
                    if orig and os.path.isfile(orig):
                        img = _PILImage.open(orig).convert("RGB")
                        feat = self.clip.encode_image(img)
                        cat, score = classify_image(feat, self.clip, self.user_categories)
                        m["category"] = cat
                        m["similarity_score"] = round(score, 3)
                except Exception:
                    pass
            self._save_metadata()
            self.after(0, lambda: self._on_reclassify_done())

        threading.Thread(target=worker, daemon=True).start()

    def _on_reclassify_done(self):
        self._refresh_categories()
        self._update_stats()
        if self.filter_mode == "all":
            self._refresh_thumbnails()
        else:
            self._set_filter(self.filter_mode)
        self._status(f"重新分类完成，共 {len(self.metadata)} 张图片")


    def _refresh_categories(self):
        # 移除旧的分类按钮
        for name, btn in list(self._cat_btns.items()):
            if name != "all":
                btn.destroy()
        self._cat_btns = {"all": self._cat_all}

        # 统计
        counts: dict[str, int] = {}
        for m in self.metadata:
            c = m.get("category", "未分类")
            counts[c] = counts.get(c, 0) + 1

        row = 1
        for cat in sorted(counts, key=lambda c: -counts[c]):
            btn = ctk.CTkButton(
                self.cat_scroll, text=f"{cat}  ({counts[cat]})",
                anchor="w", fg_color="transparent",
                command=lambda c=cat: self._set_filter(c)
            )
            btn.grid(row=row, column=0, sticky="ew", pady=1)
            self._cat_btns[cat] = btn
            row += 1

    def _update_stats(self):
        total = len(self.metadata)
        cats = len({m.get("category", "未分类") for m in self.metadata})
        self.stats_lbl.configure(text=f"📊 {total} 张 · {cats} 个分类")

    # ────────────────────────────── 清理 ──────────────────────────────

    def on_close(self):
        self._save_metadata()
        self._save_categories()
        self._save_settings(self.settings)
        self.destroy()

    def _restart_app(self):
        """保存数据后重启应用"""
        self._save_metadata()
        self._save_categories()
        self._save_settings(self.settings)
        self.destroy()
        subprocess.Popen([sys.executable, __file__])


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    # pythonw.exe 下 std 流为 None 时替换
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    if sys.stdin is None:
        sys.stdin = open(os.devnull, "r", encoding="utf-8")

    app = PhotoManager()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
