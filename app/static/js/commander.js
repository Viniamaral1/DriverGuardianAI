const el = id => document.getElementById(id);
let lastSignature = "";

function renderConversation(items) {
  const signature = JSON.stringify(items);
  if (signature === lastSignature) return;
  lastSignature = signature;
  const box = el("conversation");
  box.innerHTML = items.map(m => `<article class="message ${m.role}"><div>${m.role === "assistant" ? "C" : "You"}</div><section><small>${m.role === "assistant" ? "COMMANDER" : "DRIVER"} · ${m.time}</small><p>${m.content}</p></section></article>`).join("");
  box.scrollTop = box.scrollHeight;
}

async function send(message) {
  const clean = message.trim();
  if (!clean) return;
  el("commander-input").value = "";
  const result = await DG.post("/api/commander/message", {message: clean});
  renderConversation(result.conversation);
  const voice = DG.state?.settings?.voice_output;
  if (voice && "speechSynthesis" in window) {
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance(result.response));
  }
}

DG.subscribe(({conversation, voice}) => {
  renderConversation(conversation);
  el("wake-status").textContent = voice.enabled ? "Wake word listening" : voice.available ? "Wake word ready" : "Wake word unavailable";
  el("wake-detail").textContent = voice.detail || voice.status;
});

el("commander-form").addEventListener("submit", e => { e.preventDefault(); send(el("commander-input").value); });
document.querySelectorAll("[data-command]").forEach(btn => btn.addEventListener("click", () => send(btn.dataset.command)));
el("clear-chat").addEventListener("click", async () => { const r = await DG.post("/api/commander/clear"); renderConversation(r.conversation); });
el("wake-start").addEventListener("click", async () => { const r = await DG.post("/api/commander/voice/start"); DG.toast(r.enabled ? "Wake word enabled" : r.detail); });
el("wake-stop").addEventListener("click", async () => { await DG.post("/api/commander/voice/stop"); DG.toast("Wake word disabled"); });

const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (Recognition) {
  const recognition = new Recognition();
  recognition.lang = "en-GB";
  recognition.onresult = e => send(e.results[0][0].transcript);
  recognition.onstart = () => el("browser-mic").classList.add("active");
  recognition.onend = () => el("browser-mic").classList.remove("active");
  el("browser-mic").addEventListener("click", () => recognition.start());
} else {
  el("browser-mic").disabled = true;
  el("browser-mic").title = "Browser speech recognition is unavailable";
}
