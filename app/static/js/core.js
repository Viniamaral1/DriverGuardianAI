const DG = {
  state: null,
  socket: null,
  listeners: [],
  emit(payload) {
    this.state = payload;
    document.querySelector("#top-state")?.replaceChildren(document.createTextNode(payload.metrics?.state || "READY"));
    document.querySelector("#side-system-state")?.replaceChildren(document.createTextNode(payload.metrics?.monitoring ? "Monitoring active" : "System ready"));
    this.listeners.forEach(fn => fn(payload));
  },
  subscribe(fn) { this.listeners.push(fn); if (this.state) fn(this.state); },
  connect() {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    this.socket = new WebSocket(`${protocol}://${location.host}/ws/live`);
    this.socket.onmessage = e => this.emit(JSON.parse(e.data));
    this.socket.onclose = () => setTimeout(() => this.connect(), 1800);
  },
  async post(url, body) {
    const response = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: body ? JSON.stringify(body) : undefined});
    if (!response.ok) throw new Error(await response.text());
    return response.json();
  },
  toast(message) {
    const toast = document.querySelector("#toast");
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add("visible");
    setTimeout(() => toast.classList.remove("visible"), 2300);
  },
  applyTheme(theme, accent) {
    if (theme) { document.documentElement.dataset.theme = theme; localStorage.setItem("dg-theme", theme); }
    if (accent) { document.documentElement.dataset.accent = accent; localStorage.setItem("dg-accent", accent); }
  }
};

document.addEventListener("DOMContentLoaded", () => {
  const clock = document.querySelector("#clock");
  const tick = () => { if (clock) clock.textContent = new Date().toLocaleTimeString([], {hour12:false}); };
  tick(); setInterval(tick, 1000);
  document.querySelector("#theme-toggle")?.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    DG.applyTheme(next);
  });
  DG.connect();
});
