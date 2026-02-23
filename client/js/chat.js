let _thinkingEl = null;
let _activeAssistantEl = null;

function _hms() {
  const n = new Date();
  return [n.getHours(), n.getMinutes(), n.getSeconds()]
    .map((v) => String(v).padStart(2, '0'))
    .join(':');
}

export function addMessage(role, text) {
  removeThinking();
  const box = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = `message ${role}`;

  const label = role === 'user' ? 'You' : role === 'system' ? '⚙' : 'Ani';
  const tsSpan = `<span class="msg-time">${_hms()}</span>`;
  div.innerHTML = `<span class="msg-speaker">${label}:</span> <span class="msg-body">${_escHtml(text)}</span> ${tsSpan}`;

  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  if (role === 'assistant') _activeAssistantEl = div;
}

function _escHtml(s) {
  return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export function appendAssistantPartial(text) {
  if (!text) return;
  removeThinking();

  const box = document.getElementById('messages');
  if (!_activeAssistantEl) {
    _activeAssistantEl = document.createElement('div');
    _activeAssistantEl.className = 'message assistant';
    _activeAssistantEl.innerHTML = `<span class="msg-speaker">Ani:</span> <span class="msg-body"></span> <span class="msg-time">${_hms()}</span>`;
    box.appendChild(_activeAssistantEl);
  }

  const bodyEl = _activeAssistantEl.querySelector('.msg-body');
  if (bodyEl) bodyEl.textContent += text;
  box.scrollTop = box.scrollHeight;
}

export function finalizeAssistantMessage(text) {
  const finalText = text || '';
  if (_activeAssistantEl) {
    const bodyEl = _activeAssistantEl.querySelector('.msg-body');
    if (bodyEl) bodyEl.textContent = finalText;
    const tsEl = _activeAssistantEl.querySelector('.msg-time');
    if (tsEl) tsEl.textContent = _hms();
  } else {
    addMessage('assistant', finalText);
    return;
  }
  const box = document.getElementById('messages');
  box.scrollTop = box.scrollHeight;
  _activeAssistantEl = null;
}

export function resetAssistantDraft() {
  _activeAssistantEl = null;
}

export function showThinking() {
  if (_thinkingEl) return;
  const box = document.getElementById('messages');
  _thinkingEl = document.createElement('div');
  _thinkingEl.className = 'message thinking';
  _thinkingEl.textContent = 'Ani is thinking...';
  box.appendChild(_thinkingEl);
  box.scrollTop = box.scrollHeight;
}

export function removeThinking() {
  if (_thinkingEl) { _thinkingEl.remove(); _thinkingEl = null; }
}
