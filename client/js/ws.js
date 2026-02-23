export class WSClient {
  constructor(onEvent, onStatus) {
    this.onEvent = onEvent;
    this.onStatus = onStatus || (() => {});
    this.socket = null;
    this._reconnectMs = 1000;
    this._maxReconnectMs = 15000;
    this._reconnectTimer = null;
    this._intentionalClose = false;
  }

  connect() {
    this._intentionalClose = false;
    const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${protocol}://${location.host}/ws`;

    try {
      this.socket = new WebSocket(url);
    } catch (_) {
      this._scheduleReconnect();
      return;
    }

    this.socket.onopen = () => {
      this._reconnectMs = 1000;
      this.onStatus('connected');
    };

    this.socket.onmessage = (ev) => {
      try { this.onEvent(JSON.parse(ev.data)); } catch (_) {}
    };

    this.socket.onclose = () => {
      this.onStatus('disconnected');
      if (!this._intentionalClose) this._scheduleReconnect();
    };

    this.socket.onerror = () => {
      this.onStatus('error');
    };
  }

  _scheduleReconnect() {
    if (this._reconnectTimer) return;
    this.onStatus('reconnecting');
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this._reconnectMs = Math.min(this._reconnectMs * 1.5, this._maxReconnectMs);
      this.connect();
    }, this._reconnectMs);
  }

  send(data) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(data));
    }
  }

  close() {
    this._intentionalClose = true;
    if (this._reconnectTimer) { clearTimeout(this._reconnectTimer); this._reconnectTimer = null; }
    if (this.socket) this.socket.close();
  }
}
