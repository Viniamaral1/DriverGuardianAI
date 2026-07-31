const DG = {
  state: null,
  socket: null,
  listeners: [],

  emit(payload) {
    this.state = payload;
    document.querySelector("#top-state")?.replaceChildren(
      document.createTextNode(payload.metrics?.state || "READY")
    );
    document.querySelector("#side-system-state")?.replaceChildren(
      document.createTextNode(payload.metrics?.monitoring ? "Monitoring active" : "System ready")
    );
    this.listeners.forEach(fn => fn(payload));
  },

  subscribe(fn) {
    this.listeners.push(fn);
    if (this.state) fn(this.state);
  },

  connect() {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    this.socket = new WebSocket(`${protocol}://${location.host}/ws/live`);
    this.socket.onmessage = event => this.emit(JSON.parse(event.data));
    this.socket.onclose = () => window.setTimeout(() => this.connect(), 1800);
    this.socket.onerror = () => this.socket.close();
  },

  async request(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) throw new Error(await response.text());
    return response.json();
  },

  async post(url, body) {
    return this.request(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: body ? JSON.stringify(body) : undefined
    });
  },

  toast(message, type = "normal") {
    const toast = document.querySelector("#toast");
    if (!toast) return;
    toast.textContent = message;
    toast.dataset.type = type;
    toast.classList.add("visible");
    window.clearTimeout(this.toastTimer);
    this.toastTimer = window.setTimeout(() => toast.classList.remove("visible"), 2500);
  },

  applyTheme(theme, accent) {
    if (theme) {
      document.documentElement.dataset.theme = theme;
      localStorage.setItem("dg-theme", theme);
    }
    if (accent) {
      document.documentElement.dataset.accent = accent;
      localStorage.setItem("dg-accent", accent);
    }
  }
};

document.addEventListener("DOMContentLoaded", () => {
  const clock = document.querySelector("#clock");
  const calendar = document.querySelector("#calendar-date");

  const tick = () => {
    const now = new Date();
    if (clock) {
      clock.textContent = now.toLocaleTimeString("en-GB", {
        hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false
      });
    }
    if (calendar) {
      calendar.textContent = now.toLocaleDateString("en-GB", {
        weekday: "short", day: "2-digit", month: "short", year: "numeric"
      });
    }
  };

  tick();
  window.setInterval(tick, 1000);

  document.querySelector("#theme-toggle")?.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    DG.applyTheme(next);
    DG.toast(`${next[0].toUpperCase() + next.slice(1)} theme enabled`);
  });

  DG.connect();
});
