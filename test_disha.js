const fs=require("fs"),assert=require("assert");
const logic=fs.readFileSync("src/disha.ts","utf8");
const app=fs.readFileSync("src/App.tsx","utf8");
const html=logic+app;
new Function(logic.replace(/^\/\/ @ts-nocheck\r?\nexport \{\};\r?\n/,""));
for(const id of ["canvas","eye","cam","ambientBtn","memoryBtn","caption","fatal","fatalTitle","fatalBody","input","mic"])
  assert(app.includes(`id="${id}"`));

const ids=new Set(JSON.parse(fs.readFileSync("hiyori_free/runtime/hiyori_free_t08.cdi3.json","utf8")).Parameters.map(p=>p.Id));
const animation=html.slice(html.indexOf("function write()"),html.indexOf("/* blinking"));
const used=[...animation.matchAll(/(?:set|add)\("([^"]+)"/g)].map(m=>m[1]);
assert.deepStrictEqual(used.filter(id=>!ids.has(id)),[]);
assert(html.includes("speakGemini"));
assert(html.includes('fetch("/api/live"'));
assert(html.includes("function playLiveAudio"));
assert(html.includes("function audioCtxRunning"));
assert(html.includes("AUDIO_RESUME_MS=750"));
assert(html.includes('if(risk.state==="no_evidence")'));
assert(html.includes("thinking(false); showCaption(text,persist)"));
assert(html.includes('fetch("/api/chat"'));
assert(html.includes('fetch("/api/tts"'));
assert(!html.includes("AQ."));
assert(!html.includes("ELEVENLABS"));
assert(!html.includes("ttsModel"));

const model=JSON.parse(fs.readFileSync("hiyori_free/runtime/hiyori_free_t08.model3.json","utf8"));
for(const [group,file] of [["ByeWave","disha_bye_wave.motion3.json"],["Laugh","disha_laugh.motion3.json"],["Comfort","disha_comfort.motion3.json"]]){
  assert.strictEqual(model.FileReferences.Motions[group][0].File,`motion/${file}`);
  const motion=JSON.parse(fs.readFileSync(`hiyori_free/runtime/motion/${file}`,"utf8"));
  assert.strictEqual(motion.Meta.Loop,false);
  assert.strictEqual(motion.Meta.CurveCount,motion.Curves.length);
  assert(motion.Curves.every(curve=>ids.has(curve.Id)));
  assert.strictEqual(motion.Meta.TotalSegmentCount,motion.Curves.reduce((n,curve)=>n+(curve.Segments.length-2)/3,0));
  assert.strictEqual(motion.Meta.TotalPointCount,motion.Curves.length+motion.Meta.TotalSegmentCount);
}
assert(html.includes('playGesture(gesture)'));
assert(html.includes('[gesture:wave]'));
assert(html.includes('[gesture:laugh]'));
assert(html.includes('const userFarewell='));
assert(html.includes('text="Bye yaar! "+text'));
assert(html.includes('locale:"hi-IN"'));
assert(html.includes("SpeechRecognitionPhrase"));
assert(html.includes("rec.maxAlternatives=3"));
console.log("PASS: Disha syntax, private APIs, Live2D parameters, wave, and laugh assets");

// ── memory: persisted across reloads, gated by explicit consent, erased on revoke ──
assert(html.includes('Policy.consent.session_memory=true'));
assert(html.includes("function loadMemory()"));
assert(html.includes("function saveMemory()"));
assert(html.includes('localStorage.removeItem(MEM_DATA)'));   // revoking consent deletes retained data, not just future writes
console.log("PASS: memory persistence wired to consent, deletes on revoke");

// ── barge-in: recognition survives her speech instead of being paused around it ──
assert(html.includes("function stopSpeaking()"));
assert(html.includes("BARGE_GRACE_MS"));
assert(html.includes("if(isDishaEcho(txt)) return"));
const echoSource=html.match(/function isDishaEcho\(text\)\{[\s\S]*?\n\}/)[0];
const isEcho=new Function("speech",`${echoSource};return isDishaEcho;`)({text:"Haan yaar, main sun rahi hoon"});
assert(isEcho("haan yaar main sun rahi hoon"));
assert(!isEcho("please stop now"));
assert(!/recPause\(\);\s*\/\/ she must not hear herself/.test(html));   // the old pause-on-speak line is gone
assert(html.includes("if(recWanted) setTimeout(recResume,150)"));        // resumes even mid-reply, not just !speech.on
console.log("PASS: barge-in replaces pause-around-speech with grace-window interrupt");

// ── STT fallback for browsers without SpeechRecognition ──
assert(html.includes('fetch("/api/stt"'));
assert(html.includes("function startPushToTalk()"));
assert(html.includes("function beginRecording()"));
assert(html.includes("MediaRecorder"));
console.log("PASS: push-to-talk STT fallback present");

// ── thinking animation: a distinct channel from listening/emotion, using
// eyeball params no other posture code touches ──
assert(html.includes("busyThinking"));
assert(html.includes('set("ParamEyeBallX"'));
assert(html.includes('set("ParamEyeBallY"'));
console.log("PASS: thinking animation wired");

// ── expanded gestures ──
for(const g of ["comfort","surprise","nod","shake"]) assert(html.includes(`[gesture:${g}]`));
assert(html.includes('gesture="comfort"'));   // auto-fallback on real risk turns
console.log("PASS: comfort/surprise/nod/shake gestures present");

// ── mood-varied TTS delivery ──
assert(html.includes('say(safe,risk.state!=="no_evidence",gesture,mood)'));
assert(html.includes("body:JSON.stringify({text,mood})"));
console.log("PASS: mood threaded from turn through to TTS request");

// ── low-latency voice: raw PCM is consumed and scheduled as it arrives ──
const server=fs.readFileSync("server.mjs","utf8");
assert(server.includes('CHAT_MODEL="gemini-3.5-flash-lite"'));
assert(server.includes('LIVE_MODEL="gemini-3.1-flash-live-preview"'));
assert(server.includes('voiceName:"Leda"'));
assert(server.includes("maxOutputTokens:512"));
assert(!server.includes("maxOutputTokens:160"));
assert(server.includes("try{session=await connect();}"));
assert(server.includes("setTimeout(()=>controller.abort(),8_000)"));
assert(server.includes('TTS_MODEL="gemini-3.1-flash-tts-preview"'));
assert(server.includes('TTS_FALLBACK_MODEL="gemini-2.5-flash-preview-tts"'));
assert(server.includes("[429,500,503].includes(upstream.status)"));
assert(server.includes("ttsPrimaryAfter=Date.now()+300_000"));
assert(server.includes("streamGenerateContent?alt=sse"));
assert(server.includes("try{response=await request();}catch{response=await request();}"));
assert(!html.includes("temperature:"));
assert(server.includes('voiceName:"Leda"'));
assert(server.includes("Synthesize natural speech."));
assert(server.includes("if(upstream.status===500) upstream=await request(model,stream)"));
assert(server.includes("AbortSignal.timeout(8_000)"));
assert(server.includes("AbortSignal.timeout(20_000)"));
assert(html.includes("AbortSignal.timeout(42_000)"));
assert(html.includes("AbortSignal.timeout(48_000)"));
assert(html.includes("r.body.getReader()"));
assert(html.includes("playSources.forEach"));
assert(!html.includes("speakBrowser"));
assert(!html.includes("SpeechSynthesisUtterance"));
assert(html.includes("maxOutputTokens:160"));
assert((html.match(/geminiReply\(msg,a,risk\)/g)||[]).length>=2);
assert((html.match(/audioCtxReady\(\);/g)||[]).length>=2);
assert(!html.includes("runTurn(null)"));
console.log("PASS: Gemini Live/Leda playback and TTS safety fallback present");

// ── ambient vision: opt-in, off by default, silent unless something changed ──
assert(html.includes("let ambientOn=false"));
assert(html.includes("function tryAmbientGlance()"));
assert(html.includes("NOTHING"));
console.log("PASS: ambient vision glance loop present and default-off");

// ── caption pin-to-read ──
assert(html.includes('capEl.addEventListener("click"'));
assert(html.includes('let capTimer=null,youLine="",pinned=false'));
console.log("PASS: caption pin-to-read present");
