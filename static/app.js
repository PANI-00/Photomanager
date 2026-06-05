// ==================== 状态管理 ====================
let currentFilter = '';
let isSearchMode = false;

// ==================== 初始化 ====================
document.addEventListener('DOMContentLoaded', () => {
    refreshAll();
    setupFileUpload();
});

// ==================== API 工具 ====================
async function apiGet(path) {
    const resp = await fetch(path);
    if (!resp.ok) throw new Error(`GET ${path} failed: ${resp.status}`);
    return resp.json();
}

async function apiPost(path, body) {
    const resp = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(`POST ${path} failed: ${resp.status}`);
    return resp.json();
}

async function apiDelete(path) {
    const resp = await fetch(path, { method: 'DELETE' });
    if (!resp.ok) throw new Error(`DELETE ${path} failed: ${resp.status}`);
    return resp.json();
}

// ==================== 文件上传 ====================
function setupFileUpload() {
    const input = document.getElementById('file-input');
    input.addEventListener('change', async () => {
        if (input.files.length === 0) return;
        showLoading('上传中...');
        const formData = new FormData();
        for (const file of input.files) {
            formData.append('files', file);
        }
        try {
            const resp = await fetch('/api/images/upload', { method: 'POST', body: formData });
            const result = await resp.json();
            alert(`导入完成：${result.imported} 张成功，${result.errors.length} 张失败`);
            await refreshAll();
        } catch (e) {
            alert('上传失败：' + e.message);
        } finally {
            hideLoading();
            input.value = '';
        }
    });
}

// ==================== URL 导入 ====================
function showUrlImport() {
    document.getElementById('url-modal').style.display = 'flex';
}
function closeUrlImport() {
    document.getElementById('url-modal').style.display = 'none';
}

async function importUrls() {
    const text = document.getElementById('url-input').value.trim();
    if (!text) return;
    const urls = text.split('\n').map(s => s.trim()).filter(Boolean);
    closeUrlImport();
    showLoading('从 URL 导入中...');
    try {
        const result = await apiPost('/api/images/from-url', { urls });
        alert(`导入完成：${result.imported} 张成功，${result.errors.length} 张失败`);
        document.getElementById('url-input').value = '';
        await refreshAll();
    } catch (e) {
        alert('导入失败：' + e.message);
    } finally {
        hideLoading();
    }
}

// ==================== 刷新 ====================
async function refreshAll() {
    await Promise.all([loadStats(), loadCategories(), loadImages()]);
}

// ==================== 统计 ====================
async function loadStats() {
    try {
        const stats = await apiGet('/api/stats');
        document.getElementById('stats-display').textContent =
            `📷 ${stats.total_images} 张图片 | 📁 ${stats.total_categories} 个分类`;
    } catch (e) {
        document.getElementById('stats-display').textContent = '加载失败';
    }
}

// ==================== 分类列表 ====================
async function loadCategories() {
    try {
        const data = await apiGet('/api/categories');
        const stats = await apiGet('/api/stats');
        const list = document.getElementById('category-list');
        // 保留 "全部" 项
        list.innerHTML = '<li class="category-item active" data-category="" onclick="filterByCategory(this, \'\')">📋 全部</li>';

        for (const cat of data.categories) {
            const li = document.createElement('li');
            li.className = 'category-item';
            li.dataset.category = cat;
            li.textContent = `📁 ${cat}`;
            li.addEventListener('click', () => filterByCategory(li, cat));

            // 显示数量
            const count = stats.categories[cat] || 0;
            const span = document.createElement('span');
            span.className = 'category-count';
            span.textContent = count;
            li.appendChild(span);

            list.appendChild(li);
        }
    } catch (e) {
        console.error('加载分类失败:', e);
    }
}

// ==================== 分类筛选 ====================
function filterByCategory(el, category) {
    document.querySelectorAll('.category-item').forEach(item => item.classList.remove('active'));
    el.classList.add('active');
    currentFilter = category;
    isSearchMode = false;
    document.getElementById('search-input').value = '';
    document.getElementById('clear-search').style.display = 'none';
    loadImages();
}

// ==================== 加载图片 ====================
async function loadImages() {
    const gallery = document.getElementById('gallery');
    gallery.innerHTML = '<div class="empty-state">⏳ 加载中...</div>';

    try {
        const params = currentFilter ? `?category=${encodeURIComponent(currentFilter)}` : '';
        const data = await apiGet(`/api/images${params}`);

        if (data.images.length === 0) {
            gallery.innerHTML = '<div class="empty-state">暂无图片，请先导入。</div>';
            return;
        }

        gallery.innerHTML = '';
        for (const img of data.images) {
            const card = createImageCard(img);
            gallery.appendChild(card);
        }
    } catch (e) {
        gallery.innerHTML = `<div class="empty-state">❌ 加载失败：${e.message}</div>`;
    }
}

// ==================== 搜索 ====================
async function searchImages() {
    const query = document.getElementById('search-input').value.trim();
    if (!query) return;

    isSearchMode = true;
    document.getElementById('clear-search').style.display = 'inline-block';
    showLoading('搜索中...');

    try {
        const data = await apiPost('/api/search', { query });

        const gallery = document.getElementById('gallery');
        gallery.innerHTML = '';

        if (data.results.length === 0) {
            gallery.innerHTML = '<div class="empty-state">未找到匹配图片，请尝试其他关键词。</div>';
            return;
        }

        for (const img of data.results) {
            const card = createImageCard({
                id: img.id,
                filename: img.filename,
                caption: img.caption,
                category: img.category,
                score: img.similarity,
                url: img.url,
                source: img.source,
                created_at: '',
            }, true);
            gallery.appendChild(card);
        }
    } catch (e) {
        alert('搜索失败：' + e.message);
    } finally {
        hideLoading();
    }
}

function clearSearch() {
    document.getElementById('search-input').value = '';
    document.getElementById('clear-search').style.display = 'none';
    isSearchMode = false;
    // 重置分类选中状态
    document.querySelectorAll('.category-item').forEach(item => item.classList.remove('active'));
    document.querySelector('.category-item[data-category=""]')?.classList.add('active');
    currentFilter = '';
    loadImages();
}

// ==================== 图片卡片 ====================
function createImageCard(img, isSearch = false) {
    const card = document.createElement('div');
    card.className = 'image-card' + (isSearch ? ' search-match' : '');

    card.innerHTML = `
        <img src="${img.url}" alt="${img.filename}" loading="lazy"
             onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 200 160%22><rect fill=%22%23ddd%22 width=%22200%22 height=%22160%22/><text x=%2250%%22 y=%2250%%22 text-anchor=%22middle%22 fill=%22%23999%22 font-size=%2214%22>加载失败</text></svg>'">
        <button class="delete-btn" onclick="deleteImage('${img.id}')">✕</button>
        <div class="card-body">
            <div class="card-category">${img.category}</div>
            <div class="card-caption">${img.caption || '无描述'}</div>
            ${isSearch ? `<div class="card-score">匹配度: ${img.score}%</div>` : ''}
        </div>
    `;
    return card;
}

// ==================== 删除图片 ====================
async function deleteImage(id) {
    if (!confirm('确定删除这张图片吗？')) return;
    try {
        await apiDelete(`/api/images/${id}`);
        await refreshAll();
    } catch (e) {
        alert('删除失败：' + e.message);
    }
}

// ==================== Loading ====================
function showLoading(text) {
    document.getElementById('loading-text').textContent = text || '处理中...';
    document.getElementById('loading').style.display = 'flex';
}
function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}
