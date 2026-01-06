document.addEventListener('DOMContentLoaded', () => {
    const API_BASE = 'http://127.0.0.1:8288';
    // Tab Switching Logic
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('.view-section');
    const pageTitle = document.getElementById('pageTitle');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = item.dataset.tab;
            
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            
            sections.forEach(section => {
                section.classList.remove('active');
                if (section.id === `view-${targetId}`) {
                    section.classList.add('active');
                }
            });

            const titleMap = {
                'ask': '智能问答',
                'queue': '解析队列'
            };
            pageTitle.textContent = titleMap[targetId];

            if (targetId === 'queue') {
                currentPage = 1;
                fetchQueueData(queueSearch.value.trim(), currentFilter);
            }
        });
    });

    // Chat Interface Logic
    const chatContainer = document.querySelector('.chat-container');
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const messagesContainer = document.getElementById('messagesContainer');
    
    // Auto-resize textarea
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if(this.value === '') this.style.height = '24px';
    });

    function addMessage(text, role = 'user', meta = '') {
        if (chatContainer.classList.contains('is-initial')) {
            chatContainer.classList.remove('is-initial');
            setTimeout(() => {
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }, 50);
        }

        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;

        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerHTML = role === 'user'
            ? `<svg class="icon icon-sm" viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`
            : `<svg class="icon icon-sm" viewBox="0 0 24 24"><path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1v-1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2z"/></svg>`;

        const content = document.createElement('div');
        content.className = 'message-content';

        const textEl = document.createElement('div');
        textEl.className = role === 'bot' ? 'message-text assistant-markdown' : 'message-text';
        textEl.textContent = text || '';
        content.appendChild(textEl);
        
        if (role === 'bot') {
            const typing = document.createElement('div');
            typing.className = 'typing-indicator';
            typing.innerHTML = '<span></span><span></span><span></span>';
            content.appendChild(typing);

            const metaDiv = document.createElement('div');
            metaDiv.className = 'message-meta';
            metaDiv.textContent = meta || '';
            content.appendChild(metaDiv);
        }

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(content);
        messagesContainer.appendChild(msgDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        return textEl;
    }

    function escapeHtml(str) {
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function formatInlineMarkdown(text) {
        let html = escapeHtml(text);
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
        return html;
    }

    function convertPipeTables(md) {
        const lines = md.split(/\r?\n/);
        const out = [];
        const cellsOf = (l) => l.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(s => s.trim());
        const alignType = (s) => {
            if (/^:\-+:$/.test(s)) return 'center';
            if (/^:\-+$/.test(s)) return 'left';
            if (/^\-+:$/.test(s)) return 'right';
            return 'left';
        };
        const formatInlineLite = (t) => {
            let x = t;
            x = x.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
            x = x.replace(/\*(.+?)\*/g, '<em>$1</em>');
            return x;
        };
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            if (/^\s*\|/.test(line) && i + 1 < lines.length) {
                const alignLine = lines[i + 1];
                if (/^\s*\|(?:\s*:?-{1,}\s*\|)+\s*$/.test(alignLine)) {
                    const headers = cellsOf(line);
                    const aligns = cellsOf(alignLine).map(alignType);
                    let html = '<table><thead><tr>';
                    headers.forEach((h, idx) => {
                        html += `<th style="text-align:${aligns[idx] || 'left'}">${formatInlineLite(h)}</th>`;
                    });
                    html += '</tr></thead><tbody>';
                    i += 2;
                    while (i < lines.length && /^\s*\|/.test(lines[i])) {
                        const row = cellsOf(lines[i]);
                        html += '<tr>';
                        row.forEach((c, idx) => {
                            html += `<td style="text-align:${aligns[idx] || 'left'}">${formatInlineLite(c)}</td>`;
                        });
                        html += '</tr>';
                        i++;
                    }
                    html += '</tbody></table>';
                    out.push(html);
                    i--;
                    continue;
                }
            }
            out.push(line);
        }
        return out.join('\n');
    }

    function renderMarkdown(md) {
        md = convertPipeTables(md);
        if (typeof window !== 'undefined' && window.marked) {
            try {
                if (typeof window.marked.setOptions === 'function') {
                    window.marked.setOptions({
                        breaks: true,
                        gfm: true
                    });
                }
                let html;
                if (typeof window.marked.parse === 'function') {
                    html = window.marked.parse(md);
                } else if (typeof window.marked === 'function') {
                    html = window.marked(md);
                }
                if (html != null && typeof window !== 'undefined' && window.DOMPurify) {
                    return window.DOMPurify.sanitize(html);
                }
                if (html != null) return html;
            } catch (e) {
                console.error('Markdown parsing error:', e);
            }
        }

        const lines = md.split(/\r?\n/);
        const blocks = [];
        let inUl = false;
        let inOl = false;

        const closeLists = () => {
            if (inUl) {
                blocks.push('</ul>');
                inUl = false;
            }
            if (inOl) {
                blocks.push('</ol>');
                inOl = false;
            }
        };

        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) {
                closeLists();
                continue;
            }
            let match;
            if ((match = /^[-*]\s+(.+)$/.exec(trimmed))) {
                if (!inUl) {
                    closeLists();
                    blocks.push('<ul>');
                    inUl = true;
                }
                blocks.push('<li>' + formatInlineMarkdown(match[1]) + '</li>');
            } else if ((match = /^(\d+)\.\s+(.+)$/.exec(trimmed))) {
                if (!inOl) {
                    closeLists();
                    blocks.push('<ol>');
                    inOl = true;
                }
                blocks.push('<li>' + formatInlineMarkdown(match[2]) + '</li>');
            } else if (/^<\w+/.test(trimmed)) {
                closeLists();
                blocks.push(trimmed);
            } else {
                closeLists();
                blocks.push('<p>' + formatInlineMarkdown(trimmed) + '</p>');
            }
        }

        closeLists();
        const html = blocks.join('');
        if (typeof window !== 'undefined' && window.DOMPurify) {
            return window.DOMPurify.sanitize(html);
        }
        return html;
    }

    async function handleSend() {
        const text = chatInput.value.trim();
        if (!text) return;
        
        addMessage(text, 'user', '刚刚');
        chatInput.value = '';
        chatInput.style.height = '24px';
        
        const startTime = performance.now();
        const botTextEl = addMessage('', 'bot', '');
        const botContentEl = botTextEl.parentElement;
        const typingEl = botContentEl.querySelector('.typing-indicator');
        const metaEl = botContentEl.querySelector('.message-meta');
        let markdownBuffer = '';
        let hasReceivedToken = false;
        
        try {
            const res = await fetch(`${API_BASE}/assistant`, {
                method: 'POST',
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream'
                },
                body: JSON.stringify({ query: text })
            });

            if (!res.ok || !res.body) {
                throw new Error('网络错误');
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';
            let finished = false;
            const boundaryRegex = /\r?\n\r?\n/;

            while (!finished) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                let match;
                while ((match = boundaryRegex.exec(buffer)) !== null) {
                    const sseChunk = buffer.slice(0, match.index);
                    buffer = buffer.slice(match.index + match[0].length);

                    const lines = sseChunk.split(/\r?\n/);
                    for (const line of lines) {
                        if (line.startsWith('data:')) {
                            const raw = line.slice(5);
                            if (raw.trim() === '[DONE]') {
                                finished = true;
                                break;
                            } else {
                                if (!hasReceivedToken) {
                                    hasReceivedToken = true;
                                    if (typingEl) {
                                        typingEl.style.display = 'none';
                                    }
                                }
                                markdownBuffer += raw;
                                botTextEl.innerHTML = renderMarkdown(markdownBuffer);
                                messagesContainer.scrollTop = messagesContainer.scrollHeight;
                            }
                        }
                    }
                }
            }
        } catch (err) {
            botTextEl.textContent = '服务调用失败，请稍后重试。';
        } finally {
            const endTime = performance.now();
            const duration = ((endTime - startTime) / 1000).toFixed(1);
            if (metaEl) {
                metaEl.textContent = `耗时 ${duration} 秒`;
            }
            if (typingEl) {
                typingEl.style.display = 'none';
            }
        }
    }

    sendBtn.addEventListener('click', handleSend);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    // Queue Logic
    const queueBody = document.getElementById('queueBody');
    const queueSearch = document.getElementById('queueSearch');
    const filterBtns = document.querySelectorAll('.filter-btn');
    const uploadPdfBtn = document.getElementById('uploadPdfBtn');
    const uploadPdfInput = document.getElementById('uploadPdfInput');
    const uploadModal = document.getElementById('uploadModal');
    const modalConfirmBtn = document.getElementById('modalConfirmBtn');
    const prevPageBtn = document.getElementById('prevPageBtn');
    const nextPageBtn = document.getElementById('nextPageBtn');
    const pageInfo = document.getElementById('pageInfo');
    const pageSizeSelect = document.getElementById('pageSizeSelect');
    
    let queueData = [];
    let currentFilter = 'all';
    let currentPage = 1;
    let pageSize = 10;
    let totalCount = 0;
    let totalPages = 1;

    if (modalConfirmBtn && uploadModal) {
        modalConfirmBtn.addEventListener('click', () => {
            uploadModal.style.display = 'none';
        });
        uploadModal.addEventListener('click', (e) => {
            if (e.target === uploadModal) {
                uploadModal.style.display = 'none';
            }
        });
    }

    if (uploadPdfBtn && uploadPdfInput) {
        uploadPdfBtn.addEventListener('click', () => {
            uploadPdfInput.click();
        });
        uploadPdfInput.addEventListener('change', async () => {
            const file = uploadPdfInput.files && uploadPdfInput.files[0];
            uploadPdfInput.value = '';
            if (!file) return;
            const isPdf = (file.type === 'application/pdf') || /\.pdf$/i.test(file.name || '');
            const iconHtml = uploadPdfBtn.querySelector('svg') ? uploadPdfBtn.querySelector('svg').outerHTML : '';
            const originalHtml = uploadPdfBtn.innerHTML;
            if (!isPdf) {
                uploadPdfBtn.disabled = true;
                uploadPdfBtn.innerHTML = `${iconHtml} 仅支持 PDF`;
                setTimeout(() => {
                    uploadPdfBtn.innerHTML = originalHtml;
                    uploadPdfBtn.disabled = false;
                }, 1500);
                return;
            }
            try {
                const form = new FormData();
                form.append('file', file);
                uploadPdfBtn.disabled = true;
                uploadPdfBtn.innerHTML = `${iconHtml} 上传中...`;
                const res = await fetch(`${API_BASE}/upload_file`, {
                    method: 'POST',
                    mode: 'cors',
                    body: form
                });
                if (!res.ok) throw new Error('network error');
                await res.json();
                uploadPdfBtn.innerHTML = `${iconHtml} 已上传`;
                currentFilter = 'all';
                filterBtns.forEach(b => b.classList.remove('active'));
                const allBtn = Array.from(filterBtns).find(b => b.dataset.filter === 'all');
                if (allBtn) allBtn.classList.add('active');
                queueSearch.value = '';
                fetchQueueData('', currentFilter);
                setTimeout(() => {
                    fetchQueueData('', currentFilter);
                }, 1500);
                if (uploadModal) {
                    uploadModal.style.display = 'flex';
                }
                setTimeout(() => {
                    uploadPdfBtn.innerHTML = originalHtml;
                    uploadPdfBtn.disabled = false;
                }, 1200);
            } catch (e) {
                uploadPdfBtn.innerHTML = `${iconHtml} 上传失败`;
                setTimeout(() => {
                    uploadPdfBtn.innerHTML = originalHtml;
                    uploadPdfBtn.disabled = false;
                }, 1500);
            }
        });
    }
    function mapStatus(code) {
        if (code === 0) return 'pending';
        if (code === 1) return 'processing';
        if (code === 2) return 'success';
        if (code === 3) return 'error';
        return 'pending';
    }

    function mapType(code) {
        if (code === 0) return 'PDF';
        if (code === 1) return 'DOCX';
        if (code === 2) return 'TXT';
        return '未知';
    }

    function formatSize(size) {
        if (size == null) return '-';
        const num = Number(size);
        if (Number.isNaN(num)) return '-';
        return num.toFixed(1) + ' MB';
    }

    function buildStatusCode(filterKey) {
        if (filterKey === 'processing') return 1;
        if (filterKey === 'success') return 2;
        if (filterKey === 'failed') return 3;
        if (filterKey === 'pending') return 0;
        return null;
    }

    async function fetchQueueData(searchTerm = '', filterKey = 'all') {
        try {
            const payload = {};
            if (searchTerm) {
                payload.file_name = searchTerm;
            }
            const statusCode = buildStatusCode(filterKey);
            if (statusCode !== null) {
                payload.status = statusCode;
            }
            payload.page = currentPage;
            payload.page_size = pageSize;

            const res = await fetch(`${API_BASE}/document`, {
                method: 'POST',
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            if (!res.ok) {
                throw new Error('network error');
            }
            const result = await res.json();
            const dataObj = result && result.data ? result.data : {};
            const rawItems = Array.isArray(dataObj.items) ? dataObj.items : (Array.isArray(result.data) ? result.data : []);
            totalCount = Number(dataObj.total || 0);
            totalPages = Math.max(1, Math.ceil((totalCount || 0) / pageSize));
            const list = Array.isArray(rawItems) ? rawItems : [];
            queueData = list.map(item => ({
                id: item.id,
                name: item.file_name,
                status: mapStatus(item.status),
                type: mapType(item.type),
                size: formatSize(item.size),
                date: item.create_time || '-',
                minio_url: item.minio_url || ''
            }));
            renderQueue();
            renderPagination();
        } catch (e) {
            queueBody.innerHTML = `<tr><td colspan="6" class="empty-state">加载数据失败</td></tr>`;
        }
    }

    async function previewFile(fileId) {
        try {
            const res = await fetch(`${API_BASE}/file_preview`, {
                method: 'POST',
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/pdf'
                },
                body: JSON.stringify({ file_id: String(fileId) })
            });
            if (!res.ok) return;
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            window.open(url, '_blank', 'noopener,noreferrer');
            setTimeout(() => URL.revokeObjectURL(url), 60000);
        } catch (_) {}
    }

    async function downloadFile(fileId, fileName) {
        try {
            const res = await fetch(`${API_BASE}/file_download`, {
                method: 'POST',
                mode: 'cors',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_id: String(fileId) })
            });
            if (!res.ok) return;
            const blob = await res.blob();
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);
            link.href = url;
            link.download = fileName || 'download';
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
        } catch (_) {}
    }

    function renderQueue() {
        queueBody.innerHTML = '';

        const data = queueData;

        if (data.length === 0) {
            queueBody.innerHTML = `<tr><td colspan="6" class="empty-state">未找到符合条件的文件</td></tr>`;
            return;
        }

        data.forEach(item => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>
                    <div class="file-info">
                        <svg class="icon icon-sm file-icon" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                        ${item.name}
                    </div>
                </td>
                <td>${getStatusBadge(item.status)}</td>
                <td>${item.type}</td>
                <td>${item.size}</td>
                <td>${item.date}</td>
                <td>
                    <div class="actions-cell">
                        <button class="icon-btn ${item.minio_url ? '' : 'disabled'}" title="预览" data-action="preview" data-id="${item.id}">
                            <svg class="icon icon-sm" viewBox="0 0 24 24"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                        </button>
                        <button class="icon-btn ${item.minio_url ? '' : 'disabled'}" title="下载" data-action="download" data-id="${item.id}" data-name="${item.name}">
                            <svg class="icon icon-sm" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                        </button>
                    </div>
                </td>
            `;
            queueBody.appendChild(tr);

        });
    }

    function renderPagination() {
        if (pageInfo) {
            pageInfo.textContent = `第 ${currentPage}/${totalPages} 页，共 ${totalCount} 条`;
        }
        if (prevPageBtn) {
            if (currentPage <= 1) {
                prevPageBtn.classList.add('disabled');
            } else {
                prevPageBtn.classList.remove('disabled');
            }
        }
        if (nextPageBtn) {
            if (currentPage >= totalPages) {
                nextPageBtn.classList.add('disabled');
            } else {
                nextPageBtn.classList.remove('disabled');
            }
        }
        if (pageSizeSelect) {
            pageSizeSelect.value = String(pageSize);
        }
    }

    queueBody.addEventListener('click', (e) => {
        const btn = e.target.closest('button[data-action]');
        if (!btn || btn.classList.contains('disabled')) return;
        const id = btn.dataset.id;
        const name = btn.dataset.name;
        if (btn.dataset.action === 'preview') {
            previewFile(id);
        } else if (btn.dataset.action === 'download') {
            downloadFile(id, name);
        }
    });

    function getSimpleStatus(s) {
        if (s === 'success') return 'success';
        if (s === 'processing') return 'processing';
        if (s === 'error') return 'failed'; // map to filter key
        if (s === 'pending') return 'pending';
        return 'all';
    }

    function getStatusBadge(s) {
        const config = {
            success: { class: 'success', label: '已收录' },
            processing: { class: 'info', label: '解析中' },
            error: { class: 'error', label: '解析失败' },
            pending: { class: 'warning', label: '待处理' }
        };
        const c = config[s] || { class: 'neutral', label: '未知' };
        return `<span class="badge ${c.class}"><span class="badge-dot"></span>${c.label}</span>`;
    }

    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentFilter = btn.dataset.filter;
            currentPage = 1;
            fetchQueueData(queueSearch.value.trim(), currentFilter);
        });
    });

    queueSearch.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            currentPage = 1;
            fetchQueueData(queueSearch.value.trim(), currentFilter);
        }
    });
    if (prevPageBtn) {
        prevPageBtn.addEventListener('click', () => {
            if (currentPage > 1) {
                currentPage -= 1;
                fetchQueueData(queueSearch.value.trim(), currentFilter);
            }
        });
    }
    if (nextPageBtn) {
        nextPageBtn.addEventListener('click', () => {
            if (currentPage < totalPages) {
                currentPage += 1;
                fetchQueueData(queueSearch.value.trim(), currentFilter);
            }
        });
    }
    if (pageSizeSelect) {
        pageSizeSelect.addEventListener('change', () => {
            const v = parseInt(pageSizeSelect.value, 10);
            if (!Number.isNaN(v) && v > 0) {
                pageSize = v;
                currentPage = 1;
                fetchQueueData(queueSearch.value.trim(), currentFilter);
            }
        });
    }
    
    // Initial Render: do not load queue data until user enters queue view
});
