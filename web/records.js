/* 识车笔记面板：统计概览、记录列表、检索、编辑备注、删除、导出。
   依赖 app.js 里定义的 api() / el() / setAlert() / formatTime() / AppState。 */

let editingCarId = null;      // 正在编辑备注的记录
let cachedRecords = [];

/* ---------------------------------------------------------- 统计概览 */

function renderStats(stats) {
  const box = el('stats-cards');
  if (!stats) return;
  const cards = [
    ['记录总数', stats.total],
    ['带备注', stats.with_note],
    ['覆盖车身类型', stats.body_types],
  ];
  box.innerHTML = '';
  cards.forEach(([label, value]) => {
    const div = document.createElement('div');
    div.className = 'metric';
    const l = document.createElement('div');
    l.className = 'metric-label';
    l.textContent = label;
    const v = document.createElement('div');
    v.className = 'metric-value';
    v.textContent = String(value ?? 0);
    div.append(l, v);
    box.appendChild(div);
  });
}

/* ---------------------------------------------------------- 记录列表 */

function renderRecords(items) {
  cachedRecords = items || [];
  const list = el('records-list');
  list.innerHTML = '';

  if (!cachedRecords.length) {
    const li = document.createElement('li');
    li.className = 'muted small-text';
    li.textContent = '还没有记录。识别一款车型后告诉智能体“加入我的识车笔记”即可。';
    list.appendChild(li);
    return;
  }

  cachedRecords.forEach((rec) => {
    const li = document.createElement('li');
    li.className = 'record';

    const head = document.createElement('div');
    head.className = 'record-head';
    const name = document.createElement('span');
    name.className = 'record-name';
    name.textContent = rec.car_name;
    const id = document.createElement('span');
    id.className = 'record-id';
    id.textContent = rec.car_id;
    head.append(name, id);

    const sub = document.createElement('p');
    sub.className = 'record-sub';
    const conf = rec.confidence != null ? ` · 置信度 ${Number(rec.confidence).toFixed(2)}` : '';
    sub.textContent = `${rec.brand || '—'} · ${rec.body_type || '—'} · ${rec.sub_type || '—'}${conf}`;

    li.append(head, sub);

    if (editingCarId === rec.car_id) {
      const editor = document.createElement('div');
      editor.className = 'row-gap';
      const input = document.createElement('input');
      input.type = 'text';
      input.value = rec.note || '';
      input.placeholder = '输入备注，例如：小区门口看到的';
      input.id = 'note-editor';
      const save = document.createElement('button');
      save.className = 'btn small';
      save.type = 'button';
      save.textContent = '保存';
      save.addEventListener('click', () => saveNote(rec.car_id));
      const cancel = document.createElement('button');
      cancel.className = 'btn ghost small';
      cancel.type = 'button';
      cancel.textContent = '取消';
      cancel.addEventListener('click', () => { editingCarId = null; renderRecords(cachedRecords); });
      editor.append(input, save, cancel);
      li.appendChild(editor);
      setTimeout(() => input.focus(), 0);
    } else {
      const note = document.createElement('p');
      note.className = 'record-note';
      note.textContent = rec.note ? `备注：${rec.note}` : '暂无备注';
      li.appendChild(note);

      const actions = document.createElement('div');
      actions.className = 'record-actions';
      const edit = document.createElement('button');
      edit.className = 'btn ghost small';
      edit.type = 'button';
      edit.textContent = '编辑备注';
      edit.addEventListener('click', () => { editingCarId = rec.car_id; renderRecords(cachedRecords); });
      const del = document.createElement('button');
      del.className = 'btn danger small';
      del.type = 'button';
      del.textContent = '删除';
      del.addEventListener('click', () => deleteRecord(rec.car_id, rec.car_name));
      actions.append(edit, del);
      li.appendChild(actions);
    }

    const time = document.createElement('p');
    time.className = 'record-note';
    time.textContent = `更新于 ${formatTime(rec.updated_at)}`;
    li.appendChild(time);

    list.appendChild(li);
  });
}

/* ---------------------------------------------------------- 增删改查 */

async function loadRecords(keyword = '') {
  try {
    const query = keyword ? `?keyword=${encodeURIComponent(keyword)}` : '';
    const body = await api(`/api/records${query}`);
    renderRecords(body.items);
    renderStats(body.stats);
    if (keyword && !body.items.length) setAlert(`没有匹配「${keyword}」的记录。`);
    else setAlert('');
  } catch (err) {
    setAlert(`读取识车笔记失败：${err.message}`, 'error');
  }
}

async function saveNote(carId) {
  const input = el('note-editor');
  if (!input) return;
  try {
    await api(`/api/records/${carId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ note: input.value.trim() }),
    });
    editingCarId = null;
    await loadRecords(el('record-search').value.trim());
  } catch (err) {
    setAlert(`保存备注失败：${err.message}`, 'error');
  }
}

async function deleteRecord(carId, carName) {
  if (!window.confirm(`确定要从识车笔记中删除「${carName}」（${carId}）吗？该操作不可撤销。`)) return;
  try {
    const body = await api(`/api/records/${carId}`, { method: 'DELETE' });
    await loadRecords(el('record-search').value.trim());
    setAlert(`已删除 ${carName}，剩余 ${body.stats.total} 条记录。`);
    setTimeout(() => setAlert(''), 3000);
  } catch (err) {
    setAlert(`删除失败：${err.message}`, 'error');
  }
}

/* ---------------------------------------------------------- 事件绑定 */

document.addEventListener('DOMContentLoaded', () => {
  el('search-btn').addEventListener('click', () => loadRecords(el('record-search').value.trim()));
  el('record-search').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); loadRecords(el('record-search').value.trim()); }
  });
  el('export-csv').addEventListener('click', () => {
    window.location.href = '/api/records/export?format=csv';
  });
  el('export-json').addEventListener('click', () => {
    window.location.href = '/api/records/export?format=json';
  });
});

window.renderRecords = renderRecords;
window.renderStats = renderStats;
window.loadRecords = loadRecords;
