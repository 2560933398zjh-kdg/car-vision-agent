/* 识车笔记 · 主流程：会话管理、图片上传、对话收发。
   这里同时定义了全局的 API 封装与页面状态，供 records.js / logs.js 复用。 */

const AppState = {
  sessionId: null,
  imagePath: null,      // 本轮待发送的图片（发送后清空，下一轮靠会话上下文复用识别结果）
  imageUrl: null,
  busy: false,
};

/* ---------------------------------------------------------- 基础工具 */

async function api(path, options = {}) {
  const res = await fetch(path, options);
  let body = null;
  try {
    body = await res.json();
  } catch (err) {
    throw new Error(`服务返回了非 JSON 内容（HTTP ${res.status}）`);
  }
  if (!res.ok || body.ok === false) {
    throw new Error(body.error || `请求失败（HTTP ${res.status}）`);
  }
  return body;
}

function el(id) { return document.getElementById(id); }

function setAlert(message, kind = 'warn') {
  const bar = el('alert-bar');
  if (!message) {
    bar.classList.add('hidden');
    return;
  }
  bar.textContent = message;
  bar.classList.remove('hidden');
  bar.style.background = kind === 'error' ? 'var(--red-bg)' : 'var(--amber-bg)';
  bar.style.color = kind === 'error' ? 'var(--red)' : 'var(--amber)';
}

function formatTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/* ---------------------------------------------------------- 健康检查与会话 */

async function checkHealth() {
  const chip = el('health-chip');
  try {
    const body = await api('/api/health');
    if (body.ok) {
      chip.textContent = `就绪 · ${body.model}`;
      chip.className = 'chip ok';
      setAlert('');
    } else {
      chip.textContent = '配置不完整';
      chip.className = 'chip bad';
      setAlert(body.hint || '服务配置不完整', 'error');
    }
  } catch (err) {
    chip.textContent = '服务不可用';
    chip.className = 'chip bad';
    setAlert(`无法连接后端：${err.message}`, 'error');
  }
}

async function initSession() {
  try {
    const body = await api('/api/session', { method: 'POST' });
    AppState.sessionId = body.session_id;
    el('session-chip').textContent = `会话 ${body.session_id}`;
  } catch (err) {
    setAlert(`创建会话失败：${err.message}`, 'error');
  }
}

async function resetSession() {
  if (!AppState.sessionId) return;
  try {
    await api(`/api/session/${AppState.sessionId}/reset`, { method: 'POST' });
    el('messages').querySelectorAll('.bubble').forEach((n) => n.remove());
    el('chat-empty').classList.remove('hidden');
    if (window.renderToolLogs) renderToolLogs([]);
    setAlert('已清空会话上下文。');
    setTimeout(() => setAlert(''), 2500);
  } catch (err) {
    setAlert(`清空失败：${err.message}`, 'error');
  }
}

/* ---------------------------------------------------------- 图片上传 */

async function uploadFile(file) {
  if (!file) return;
  if (!AppState.sessionId) await initSession();

  const thumb = el('preview');
  thumb.style.backgroundImage = '';
  el('drop-title').textContent = '正在上传…';

  const form = new FormData();
  form.append('file', file);
  try {
    const body = await api('/api/upload', { method: 'POST', body: form });
    AppState.imagePath = body.image_path;
    AppState.imageUrl = body.url;
    thumb.style.backgroundImage = `url(${body.url})`;
    el('drop-title').textContent = '已附加图片，可以提问了';
    el('preview-name').textContent = `${body.filename} · ${body.size} · ${(body.bytes / 1024).toFixed(0)} KB`;
    setAlert('');
  } catch (err) {
    el('drop-title').textContent = '点击选择，或把图片拖到这里';
    el('preview-name').textContent = '支持 JPG / PNG / WEBP';
    setAlert(`上传失败：${err.message}`, 'error');
  }
}

function clearAttachedImage() {
  AppState.imagePath = null;
  AppState.imageUrl = null;
  el('preview').style.backgroundImage = '';
  el('drop-title').textContent = '点击选择，或把图片拖到这里';
  el('preview-name').textContent = '图片已发送，后续追问无需再次上传';
}

/* ---------------------------------------------------------- 对话渲染 */

function appendUserBubble(text, imageUrl) {
  el('chat-empty').classList.add('hidden');
  const wrap = document.createElement('div');
  wrap.className = 'bubble user';
  if (imageUrl) {
    const img = document.createElement('img');
    img.className = 'bubble-image';
    img.src = imageUrl;
    img.alt = '上传的车照';
    wrap.appendChild(img);
  }
  if (text) wrap.appendChild(document.createTextNode(text));
  el('messages').appendChild(wrap);
  scrollChat();
}

function appendBotBubble(text, toolCalls = [], isError = false) {
  const wrap = document.createElement('div');
  wrap.className = 'bubble bot' + (isError ? ' error' : '');
  wrap.appendChild(document.createTextNode(text));
  if (toolCalls.length) {
    const row = document.createElement('div');
    row.className = 'tag-row';
    toolCalls.forEach((call) => {
      const tag = document.createElement('span');
      tag.className = 'tag ' + (call.ok ? 'ok' : 'bad');
      tag.textContent = `${call.name} ${call.ok ? '成功' : '失败'}`;
      row.appendChild(tag);
    });
    wrap.appendChild(row);
  }
  el('messages').appendChild(wrap);
  scrollChat();
}

function appendTyping() {
  const wrap = document.createElement('div');
  wrap.className = 'typing';
  wrap.id = 'typing';
  wrap.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>'
    + '<span>智能体正在调用工具，通常需要 10~40 秒…</span>';
  el('messages').appendChild(wrap);
  scrollChat();
}

function removeTyping() {
  const node = el('typing');
  if (node) node.remove();
}

function scrollChat() {
  const box = el('messages');
  box.scrollTop = box.scrollHeight;
}

/* ---------------------------------------------------------- 发送 */

async function sendMessage(text) {
  if (AppState.busy) return;
  const message = (text || '').trim();
  const imagePath = AppState.imagePath;
  if (!message && !imagePath) {
    setAlert('请输入内容，或先上传一张图片');
    return;
  }
  if (!AppState.sessionId) await initSession();

  AppState.busy = true;
  el('send-btn').disabled = true;
  el('send-btn').textContent = '处理中';
  setAlert('');
  appendUserBubble(message, AppState.imageUrl);
  clearAttachedImage();
  appendTyping();

  try {
    const body = await api('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: AppState.sessionId, message, image_path: imagePath }),
    });
    removeTyping();
    appendBotBubble(body.answer, body.tool_calls || []);
    el('session-chip').textContent = `会话 ${AppState.sessionId} · 第 ${body.round} 轮`;
    if (window.renderToolLogs) renderToolLogs(body.tool_calls || [], true);
    if (window.renderRecords) renderRecords(body.records || []);
    if (window.renderStats && body.stats) renderStats(body.stats);
  } catch (err) {
    removeTyping();
    appendBotBubble(`请求失败：${err.message}`, [], true);
    setAlert(err.message, 'error');
  } finally {
    AppState.busy = false;
    el('send-btn').disabled = false;
    el('send-btn').textContent = '发送';
  }
}

/* ---------------------------------------------------------- 事件绑定 */

function bindEvents() {
  const zone = el('drop-zone');
  const input = el('file-input');

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });
  zone.setAttribute('tabindex', '0');
  el('pick-btn').addEventListener('click', (e) => { e.stopPropagation(); input.click(); });

  input.addEventListener('change', () => {
    if (input.files && input.files[0]) uploadFile(input.files[0]);
    input.value = '';
  });

  ['dragenter', 'dragover'].forEach((evt) => zone.addEventListener(evt, (e) => {
    e.preventDefault();
    zone.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach((evt) => zone.addEventListener(evt, (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
  }));
  zone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });

  // 整个页面都接受粘贴图片
  document.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
    if (item) uploadFile(item.getAsFile());
  });

  el('composer').addEventListener('submit', (e) => {
    e.preventDefault();
    const box = el('chat-input');
    const text = box.value;
    box.value = '';
    sendMessage(text);
  });

  el('chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      el('composer').requestSubmit();
    }
  });

  el('reset-btn').addEventListener('click', resetSession);

  el('quick-tasks').addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-task]');
    if (btn) sendMessage(btn.dataset.task);
  });
}

/* ---------------------------------------------------------- 启动 */

document.addEventListener('DOMContentLoaded', async () => {
  bindEvents();
  await checkHealth();
  await initSession();
  if (window.loadRecords) loadRecords();
});
