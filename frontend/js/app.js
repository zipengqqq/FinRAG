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
            
            // Update Nav State
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            
            // Update View State
            sections.forEach(section => {
                section.classList.remove('active');
                if (section.id === `view-${targetId}`) {
                    section.classList.add('active');
                }
            });

            // Update Title
            const titleMap = {
                'ask': '智能问答',
                'queue': '解析队列'
            };
            pageTitle.textContent = titleMap[targetId];
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
        // Transition from initial state if needed
        if (chatContainer.classList.contains('is-initial')) {
            chatContainer.classList.remove('is-initial');
            // Allow transition to happen before scrolling
            setTimeout(() => {
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }, 50);
        }

        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        // Simple SVG Icons for Avatars
        avatar.innerHTML = role === 'user' 
            ? `<svg class="icon icon-sm" viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`
            : `<svg class="icon icon-sm" viewBox="0 0 24 24"><path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1v-1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2z"/></svg>`;

        const content = document.createElement('div');
        content.className = 'message-content';
        content.textContent = text;
        
        // Add meta/refs if bot
        if (role === 'bot' && meta) {
            const metaDiv = document.createElement('div');
            metaDiv.className = 'message-meta';
            metaDiv.textContent = meta;
            content.appendChild(metaDiv);
        }

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(content);
        messagesContainer.appendChild(msgDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    async function handleSend() {
        const text = chatInput.value.trim();
        if (!text) return;
        
        addMessage(text, 'user', '刚刚');
        chatInput.value = '';
        chatInput.style.height = '24px';
        
        try {
            const res = await fetch(`${API_BASE}/assistant`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: text })
            });
            const data = await res.json();
            const reply = data && data.data ? data.data : '无响应';
            addMessage(reply, 'bot', 'AI 助手');
        } catch (err) {
            addMessage('服务调用失败，请稍后重试。', 'bot', '错误');
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
    
    // Mock Data
    const mockData = [
        { name: '2024_年度报告.pdf', status: 'success', type: 'PDF', size: '2.4 MB', date: '2025-10-24' },
        { name: 'Q3_市场分析报告.docx', status: 'processing', type: 'DOCX', size: '1.8 MB', date: '2025-10-23' },
        { name: '用户反馈数据_Raw.txt', status: 'error', type: 'TXT', size: '450 KB', date: '2025-10-22' },
        { name: '竞品分析_v2.pdf', status: 'pending', type: 'PDF', size: '5.1 MB', date: '2025-10-21' },
        { name: '内部备忘录_v2.docx', status: 'success', type: 'DOCX', size: '1.2 MB', date: '2025-10-20' },
    ];

    let currentFilter = 'all';

    function renderQueue() {
        const term = queueSearch.value.toLowerCase();
        queueBody.innerHTML = '';
        
        const filtered = mockData.filter(item => {
            if (currentFilter !== 'all' && getSimpleStatus(item.status) !== currentFilter) return false;
            return item.name.toLowerCase().includes(term);
        });

        if (filtered.length === 0) {
            queueBody.innerHTML = `<tr><td colspan="6" class="empty-state">未找到符合条件的文件</td></tr>`;
            return;
        }

        filtered.forEach(item => {
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
                        <button class="icon-btn" title="预览">
                            <svg class="icon icon-sm" viewBox="0 0 24 24"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                        </button>
                        <button class="icon-btn" title="下载">
                            <svg class="icon icon-sm" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                        </button>
                    </div>
                </td>
            `;
            queueBody.appendChild(tr);
        });
    }

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
            renderQueue();
        });
    });

    queueSearch.addEventListener('input', renderQueue);
    
    // Initial Render
    renderQueue();
});
