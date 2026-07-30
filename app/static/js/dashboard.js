"use strict";

const $ = (id) => document.getElementById(id);
const els = {
    currentDate: $("currentDate"), currentTime: $("currentTime"), systemBadge: $("systemBadge"), systemStatusText: $("systemStatusText"),
    cameraBadge: $("cameraBadge"), cameraStatusText: $("cameraStatusText"), cameraViewport: $("cameraViewport"), cameraMessage: $("cameraMessage"),
    faceDetectionText: $("faceDetectionText"), faceConfidence: $("faceConfidence"), sessionTime: $("sessionTime"), alertBadge: $("alertBadge"),
    fatigueProbability: $("fatigueProbability"), gaugeProgress: $("gaugeProgress"), driverStatus: $("driverStatus"), driverDescription: $("driverDescription"),
    earValue: $("earValue"), yawnValue: $("yawnValue"), blinkValue: $("blinkValue"), headTiltValue: $("headTiltValue"),
    commanderBadge: $("commanderBadge"), commanderStatusText: $("commanderStatusText"), commanderChat: $("commanderChat"), commanderForm: $("commanderForm"),
    commanderInput: $("commanderInput"), sendCommanderButton: $("sendCommanderButton"), microphoneButton: $("microphoneButton"), voiceToggleButton: $("voiceToggleButton"),
    clearChatButton: $("clearChatButton"), speechSupportMessage: $("speechSupportMessage"), eventLog: $("eventLog"), controlMessage: $("controlMessage"),
    startButton: $("startButton"), stopButton: $("stopButton"), toast: $("toast")
};

const GAUGE_LENGTH = 235.62;
let speechEnabled = true;
let recognition = null;
let lastConversationSignature = "";

function escapeHtml(value) { const div = document.createElement("div"); div.textContent = String(value); return div.innerHTML; }
function humanise(value) { return String(value || "unknown").replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase()); }
function formatTime(seconds) { const s = Math.max(0, parseInt(seconds,10)||0); return [Math.floor(s/3600), Math.floor((s%3600)/60), s%60].map(v=>String(v).padStart(2,"0")).join(":"); }
function updateClock(){ const now=new Date(); els.currentDate.textContent=new Intl.DateTimeFormat("en-GB",{weekday:"short",day:"2-digit",month:"short",year:"numeric"}).format(now); els.currentTime.textContent=now.toLocaleTimeString("en-GB",{hour12:false}); }

function renderState(state){
    const online=Boolean(state.monitoring);
    els.systemBadge.className=`status-badge ${online?"online":"standby"}`; els.systemStatusText.textContent=online?"Online":"Standby";
    els.cameraBadge.className=`mini-status ${online?"online":"offline"}`; els.cameraStatusText.textContent=humanise(state.camera_status);
    els.cameraViewport.classList.toggle("active",online); els.cameraMessage.textContent=online?"Vision system active":"Camera offline";
    els.faceDetectionText.textContent=online?"Tracking":"Inactive"; els.faceConfidence.textContent=`${Math.round((Number(state.face_confidence)||0)*100)}%`;
    els.sessionTime.textContent=formatTime(state.session_seconds); els.startButton.disabled=online; els.stopButton.disabled=!online;
    const p=Math.max(0,Math.min(1,Number(state.fatigue_probability)||0)); els.fatigueProbability.textContent=`${Math.round(p*100)}%`; els.gaugeProgress.style.strokeDashoffset=GAUGE_LENGTH*(1-p);
    els.gaugeProgress.style.stroke=p>=.75?"var(--critical)":p>=.45?"var(--warning)":"var(--success)";
    const level=state.alert_level||"standby"; els.alertBadge.className=`alert-badge ${level}`; els.alertBadge.textContent=humanise(level);
    els.driverStatus.textContent=humanise(state.driver_status); els.earValue.textContent=Number(state.ear||0).toFixed(3); els.yawnValue.textContent=Number(state.yawn_score||0).toFixed(3);
    els.blinkValue.textContent=parseInt(state.blink_rate||0,10); els.headTiltValue.textContent=Number(state.head_tilt||0).toFixed(1);
    els.driverDescription.textContent=!online?"Start monitoring to analyse the driver.":level==="critical"?"Immediate intervention may be required.":level==="warning"?"Possible drowsiness indicators are increasing.":"Driver behaviour remains within normal limits.";
    const cs=state.commander_status||"ready"; els.commanderStatusText.textContent=humanise(cs); els.commanderBadge.className=`mini-status ${cs==="ready"?"ready":""}`;
    els.controlMessage.textContent=online?"Driver monitoring session is active.":"System is ready.";
    renderLogs(state.logs); renderConversation(state.conversation);
}

function renderLogs(logs){ if(!Array.isArray(logs)||!logs.length){els.eventLog.innerHTML='<div class="event-empty">Waiting for system events.</div>';return;} els.eventLog.innerHTML=[...logs].reverse().map(x=>`<div class="event-item"><span class="event-time">${escapeHtml(x.time)}</span><span class="event-dot ${escapeHtml(x.level)}"></span><span class="event-message">${escapeHtml(x.message)}</span></div>`).join(""); }
function renderConversation(items){
    if(!Array.isArray(items)) return;
    const signature=JSON.stringify(items); if(signature===lastConversationSignature)return; lastConversationSignature=signature;
    els.commanderChat.innerHTML=items.map(x=>`<div class="chat-row ${x.role}"><div class="chat-bubble"><span class="chat-meta">${x.role==="user"?"You":"Commander"} · ${escapeHtml(x.time)}</span>${escapeHtml(x.message)}</div></div>`).join("");
    els.commanderChat.scrollTop=els.commanderChat.scrollHeight;
}
async function fetchStatus(){ try{const r=await fetch("/api/status",{cache:"no-store"}); if(!r.ok)throw new Error(r.status); renderState(await r.json());}catch(e){console.error(e);els.controlMessage.textContent="Unable to communicate with FastAPI.";} }
async function postCommand(url){ try{const r=await fetch(url,{method:"POST"}); const data=await r.json(); if(!r.ok)throw new Error(data.detail||r.status); showToast(data.message||"Command completed"); await fetchStatus();}catch(e){console.error(e);showToast("Command failed");} }
async function sendCommander(message){
    const text=String(message||"").trim(); if(!text)return;
    els.commanderInput.value=""; els.sendCommanderButton.disabled=true; els.commanderStatusText.textContent="Thinking";
    try{const r=await fetch("/api/commander/message",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:text})}); const data=await r.json(); if(!r.ok)throw new Error(data.detail||r.status); renderState(data.state); if(speechEnabled) speak(data.response);}
    catch(e){console.error(e);showToast("Commander could not process the command.");}
    finally{els.sendCommanderButton.disabled=false;els.commanderInput.focus();}
}
function speak(text){ if(!("speechSynthesis" in window))return; window.speechSynthesis.cancel(); const utterance=new SpeechSynthesisUtterance(text); utterance.rate=1; utterance.pitch=1; utterance.lang="en-GB"; window.speechSynthesis.speak(utterance); }
function setupRecognition(){
    const SpeechRecognition=window.SpeechRecognition||window.webkitSpeechRecognition;
    if(!SpeechRecognition){els.microphoneButton.disabled=true;els.speechSupportMessage.textContent="Browser speech recognition is unavailable here. Typed commands still work.";return;}
    recognition=new SpeechRecognition(); recognition.lang="en-GB"; recognition.interimResults=false; recognition.continuous=false;
    recognition.onstart=()=>{els.microphoneButton.classList.add("listening");els.speechSupportMessage.textContent="Listening…";};
    recognition.onend=()=>{els.microphoneButton.classList.remove("listening");els.speechSupportMessage.textContent="";};
    recognition.onerror=(e)=>{els.speechSupportMessage.textContent=`Microphone error: ${e.error}`;};
    recognition.onresult=(e)=>{const transcript=e.results[0][0].transcript;els.commanderInput.value=transcript;sendCommander(transcript);};
}
function showToast(message){els.toast.textContent=message;els.toast.classList.add("visible");setTimeout(()=>els.toast.classList.remove("visible"),2400);}
function registerEvents(){
    els.startButton.addEventListener("click",()=>postCommand("/api/monitoring/start")); els.stopButton.addEventListener("click",()=>postCommand("/api/monitoring/stop"));
    els.commanderForm.addEventListener("submit",e=>{e.preventDefault();sendCommander(els.commanderInput.value);});
    els.microphoneButton.addEventListener("click",()=>{if(recognition)recognition.start();});
    els.voiceToggleButton.addEventListener("click",()=>{speechEnabled=!speechEnabled;els.voiceToggleButton.classList.toggle("active",speechEnabled);els.voiceToggleButton.textContent=speechEnabled?"🔊":"🔇";if(!speechEnabled&&"speechSynthesis" in window)window.speechSynthesis.cancel();});
    els.clearChatButton.addEventListener("click",async()=>{await fetch("/api/commander/clear",{method:"POST"});lastConversationSignature="";await fetchStatus();});
    document.querySelectorAll("[data-command]").forEach(btn=>btn.addEventListener("click",()=>sendCommander(btn.dataset.command)));
}
function init(){updateClock();setInterval(updateClock,1000);registerEvents();setupRecognition();fetchStatus();setInterval(fetchStatus,1000);els.commanderInput.focus();}
document.addEventListener("DOMContentLoaded",init);
