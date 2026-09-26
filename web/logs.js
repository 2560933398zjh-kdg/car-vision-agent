/* 工具调用记录面板：显示每次工具调用的名称、成功/失败、耗时，
   点击可展开查看参数与原始返回（任务书要求「提供工具调用记录」）。
   依赖 app.js 里定义的 el()。 */

const LOG_STORE = [];      // 累积本次会话的所有工具调用

function summariseArgs(args) {
  if (args === null || args === undefined) return '（参数解析失败）';
  const text = JSON.stringify(args);
  return text.length > 34 ? text.slice(0, 34) + '…' : text;
}

function renderToolLogs(toolCalls, append = false) {
  if (append) LOG_STORE.push(...toolCalls);
  else LOG_STORE.length = 0;

  el('log-count').textContent = `${LOG_STORE.length} 次`;
  const list = el('tool-logs');
  list.innerHTML = '';

  if (!LOG_STORE.length) {
    const li = document.createElement('li');
    li.className = 'muted small-text';
    li.textContent = '本会话还没有工具调用。';
    list.appendChild(li);
    return;
  }

  LOG_STORE.forEach((call, index) => {
    const li = document.createElement('li');

    const details = document.createElement('details');
    details.className = 'log';

    const summary = document.createElement('summary');
    const left = document.createElement('span');
    left.className = 'log-name';
    left.textContent = `${index + 1}. ${call.name}  ${summariseArgs(call.arguments)}`;
    const right = document.createElement('span');
    right.className = 'log-status ' + (call.ok ? 'ok' : 'bad');
    right.textContent = `${call.ok ? '成功' : '失败'} · ${Math.round(call.elapsed_ms)}ms`;
    summary.append(left, right);
    details.appendChild(summary);

    const body = document.createElement('div');
    body.className = 'log-body';

    const t1 = document.createElement('h4');
    t1.textContent = '调用参数';
    const p1 = document.createElement('pre');
    p1.textContent = JSON.stringify(call.arguments, null, 2);
    const t2 = document.createElement('h4');
    t2.textContent = '工具返回';
    const p2 = document.createElement('pre');
    p2.textContent = JSON.stringify(call.result, null, 2);

    body.append(t1, p1, t2, p2);
    details.appendChild(body);
    li.appendChild(details);
    list.appendChild(li);
  });
}

window.renderToolLogs = renderToolLogs;
