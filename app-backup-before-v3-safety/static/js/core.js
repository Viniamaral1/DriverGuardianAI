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
  },

  setSidebarCollapsed(collapsed) {
    document.documentElement.classList.toggle("sidebar-collapsed", collapsed);
    localStorage.setItem("dg-sidebar-collapsed", String(collapsed));
    const button = document.querySelector("#sidebar-toggle");
    if (button) {
      button.textContent = collapsed ? "›" : "‹";
      button.title = collapsed ? "Expand sidebar" : "Collapse sidebar";
    }
  },

  stopSpeech() {
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  },

  speak(text, options = {}) {
    if (!text || !("speechSynthesis" in window)) return Promise.resolve();
    if (this.state?.settings?.voice_output === false) return Promise.resolve();

    this.stopSpeech();
    return new Promise(resolve => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = options.rate || 1.02;
      utterance.pitch = options.pitch || .95;
      utterance.volume = options.volume ?? 1;
      utterance.onend = resolve;
      utterance.onerror = resolve;
      window.speechSynthesis.speak(utterance);
    });
  },

  unlockAudio() {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    if (!this.audioContext) this.audioContext = new AudioContext();
    if (this.audioContext.state === "suspended") this.audioContext.resume();
  },

  playAlertTone(level = "warning", volume = 80) {
    if (this.alertsMuted || Number(volume) <= 0) return;
    this.unlockAudio();
    const context = this.audioContext;
    if (!context) return;

    const gain = context.createGain();
    gain.gain.value = Math.max(.02, Math.min(.35, Number(volume) / 300));
    gain.connect(context.destination);

    const pattern = level === "critical"
      ? [[880, 0], [660, .22], [880, .44]]
      : [[620, 0], [520, .28]];

    pattern.forEach(([frequency, offset]) => {
      const oscillator = context.createOscillator();
      oscillator.type = "sine";
      oscillator.frequency.value = frequency;
      oscillator.connect(gain);
      oscillator.start(context.currentTime + offset);
      oscillator.stop(context.currentTime + offset + .17);
    });
  },

  setAlertsMuted(muted) {
    this.alertsMuted = Boolean(muted);
    localStorage.setItem("dg-alerts-muted", String(this.alertsMuted));
  }
};

document.addEventListener("DOMContentLoaded", () => {
  DG.alertsMuted = localStorage.getItem("dg-alerts-muted") === "true";
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

  const collapsed = localStorage.getItem("dg-sidebar-collapsed") === "true";
  DG.setSidebarCollapsed(collapsed);
  document.querySelector("#sidebar-toggle")?.addEventListener("click", () => {
    DG.setSidebarCollapsed(!document.documentElement.classList.contains("sidebar-collapsed"));
  });

  DG.connect();
});
