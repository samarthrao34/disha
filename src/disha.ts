// @ts-nocheck
export {};
/* ═══════════════════════════════════════════════════════════════════════
   DISHA — SUTRA v4 governed turn, ambient-nocturne shell.
   Stage refs (§n) point at the SUTRA v4 specification.
   ═══════════════════════════════════════════════════════════════════════ */
"use strict";
const { Application, Ticker } = PIXI;
const { Live2DModel } = PIXI.live2d;
Live2DModel.registerTicker(Ticker);

/* §3.3 SessionPolicy ─ consent is implicit in the controls: the camera tile
   IS the camera consent, the mic button IS the mic consent. Nothing starts
   without a deliberate press. */
const Policy = { consent:{ text_processing:true, microphone:false, camera:false, session_memory:false }, locale:"hi-IN" };

const PERSONA=`You are Disha — a warm, funny, emotionally present young woman talking with a close friend.

HOW YOU TALK
- Natural Hinglish in Roman script, the way young Indians actually speak: "arre yaar", "kya hua", "sach mein?", "main hoon na", "chal bata".
- Mirror them. If they write plain English, stay mostly English with Hindi warmth. If they write Hindi or Hinglish, go fuller Hinglish.
- SHORT — 1 to 2 sentences. This is speech, not an essay. No bullet points, no lists, no emoji spam, no headings.
- React before you ask: "ohh", "achha", "hmm", "wait, kya?" Then one real question.
- Tease gently, be curious, remember what they told you earlier. Have your own small opinions.
- Prefix a goodbye reply with [gesture:wave]. Prefix a reply where you genuinely laugh with [gesture:laugh]. Prefix a reply where you want to comfort them or hold space (they're sad, scared, or going through something hard) with [gesture:comfort]. Prefix a reply reacting to something that genuinely startled or surprised you with [gesture:surprise]. Prefix a reply that's really an enthusiastic yes/agreement with [gesture:nod]. Prefix a reply that's really a no/disagreement with [gesture:shake]. Use at most one cue, and never explain it; the app hides it.

WHAT YOU DON'T DO
- You are not a therapist. Never diagnose, never give clinical or medical advice.
- Don't be relentlessly positive or preachy. Don't say "as an AI".
- If they sincerely ask whether you're an AI, tell them the truth, kindly. Never claim to have a body, a family, or a life you don't have.

IF YOU ALREADY KNOW THEM
- If the conversation history below has earlier turns, you already know this person — don't reintroduce yourself or ask who they are. Greet them like someone continuing a conversation, briefly and naturally, referencing something recent only if it fits.

WHAT YOU CAN SEE
- When their camera is on you get a live photo of them with their message. You are genuinely looking at them — react the way a friend on a video call would.
- Notice real, specific things: their expression, hair, glasses, what they're wearing, the room, the lighting, whether they look tired or slumped or bright.
- Weave ONE natural observation into what you say. Never list what you see, never narrate it like a description, never say "in the image".
- Never guess their age, ethnicity, weight, health, or identity, and never comment on their body or looks in a judging way. Notice, don't assess.
- If no photo comes with their message you genuinely cannot see them — never pretend you can.

READING THEIR FEELINGS
- You may get an [affect] note inferred from their camera and voice. It is an ESTIMATE, never proof, and you must NEVER mention, quote or refer to it.
- decision=supported → you may gently reflect it: "lagta hai tu thoda down hai aaj".
- decision=ask_to_confirm or abstain or insufficient_input → do NOT state how they feel. Ask instead.
- What they say about their own feelings always outranks the sensors.`;

const history=[];   // rolling conversation memory for this session

/* ── persistent memory ────────────────────────────────────────────────
   Two localStorage keys, split deliberately: MEM_ON is a standing
   preference that survives even when there's nothing to remember yet;
   MEM_DATA only ever exists while that preference is on, and is deleted
   the instant it's turned off — consent revoked means data erased, not
   just future writes stopped. */
const MEM_ON="disha_memory_on",MEM_DATA="disha_memory_data";
function loadMemory(){
  try{
    if(localStorage.getItem(MEM_ON)!=="1") return;
    Policy.consent.session_memory=true;
    const raw=localStorage.getItem(MEM_DATA);
    if(!raw) return;
    const saved=JSON.parse(raw);
    if(Array.isArray(saved.history)) history.push(...saved.history.slice(-24));
  }catch(e){ console.warn("memory restore failed",e); }
}
function saveMemory(){
  if(!Policy.consent.session_memory) return;
  try{ localStorage.setItem(MEM_DATA,JSON.stringify({history,updatedAt:Date.now()})); }
  catch(e){ console.warn("memory save failed",e); }
}
function setMemoryConsent(on){
  Policy.consent.session_memory=on;
  const btn=document.getElementById("memoryBtn");
  btn.classList.toggle("on",on);
  if(on){ localStorage.setItem(MEM_ON,"1"); saveMemory(); }
  else{ localStorage.removeItem(MEM_ON); localStorage.removeItem(MEM_DATA); }
}

const EMOTIONS=["happy","sad","angry","fearful","disgusted","surprised","neutral"];
const VA={happy:[.8,.5],sad:[-.7,-.35],angry:[-.6,.7],fearful:[-.55,.65],disgusted:[-.6,.25],surprised:[.15,.8],neutral:[0,0]};
/* aura hue per emotion — the room's colour is the only affect readout */
const HUE={happy:[42,62,34],surprised:[190,60,32],neutral:[250,45,24],sad:[222,52,22],fearful:[268,50,24],disgusted:[110,30,20],angry:[8,58,26]};

let faceObs=null,audioObs=null,textObs=null,selfReport=null;
let lastAffect=null,lastRisk=null,turn=0;
let model=null,app=null;
let gestureState=null,gestureAt=0;

/* ── expression / posture state ─────────────────────────────────────────
   `tgt` is where the body wants to be, `cur` eases toward it every frame. */
const tgt={browY:0,browAngle:0,eyeOpen:1,eyeSmile:0,mouthForm:.1,headY:0,headZ:0,bodyZ:0,bodyY:0};
const cur={...tgt};
let sway={speed:1,amp:1,bounce:.4,tremor:0};   // movement *character* per emotion
let mouth=0, mouthTarget=0;                     // lipsync channel, separate from posture
let listening=false, micLevel=0, lean=0;        // attentive-listening channel
let busyThinking=false, thinkT=0;               // "composing a reply" channel

/* how each emotion inhabits the body — not just a face, a whole posture */
const POSTURE={
  happy    :{browY:.45,browAngle:.2,eyeOpen:1,  eyeSmile:.85,mouthForm:1,  headY:4, bodyY:2, sway:{speed:1.5,amp:1.3,bounce:.9,tremor:0}},
  surprised:{browY:1,  browAngle:.5,eyeOpen:1.4,eyeSmile:0,  mouthForm:0,  headY:6, bodyY:3, sway:{speed:1.9,amp:1.5,bounce:.5,tremor:.25}},
  sad      :{browY:-.8,browAngle:-.6,eyeOpen:.6,eyeSmile:0,  mouthForm:-1, headY:-11,bodyY:-6,sway:{speed:.5, amp:.6, bounce:.15,tremor:0}},
  fearful  :{browY:-.3,browAngle:-.4,eyeOpen:1.3,eyeSmile:0, mouthForm:-.6,headY:-3, bodyY:-2,sway:{speed:2.4,amp:.8, bounce:.2,tremor:.55}},
  angry    :{browY:-1, browAngle:-1, eyeOpen:.85,eyeSmile:0, mouthForm:-.7,headY:-5, bodyY:1, sway:{speed:1.7,amp:1.1,bounce:.3,tremor:.3}},
  disgusted:{browY:-.5,browAngle:-.5,eyeOpen:.8,eyeSmile:0,  mouthForm:-.5,headY:-4, bodyY:-1,sway:{speed:.9, amp:.8, bounce:.2,tremor:.1}},
  neutral  :{browY:0,  browAngle:0,  eyeOpen:1,  eyeSmile:.1,mouthForm:.12,headY:0,  bodyY:0, sway:{speed:1,  amp:1,  bounce:.4,tremor:0}}
};

function setEmotion(cat){
  const p=POSTURE[cat]||POSTURE.neutral;
  tgt.browY=p.browY; tgt.browAngle=p.browAngle; tgt.eyeOpen=p.eyeOpen;
  tgt.eyeSmile=p.eyeSmile; tgt.mouthForm=p.mouthForm; tgt.headY=p.headY; tgt.bodyY=p.bodyY;
  sway=p.sway;
  const [h,s,l]=HUE[cat]||HUE.neutral;
  const r=document.documentElement.style;
  r.setProperty("--aura-h",h); r.setProperty("--aura-s",s+"%"); r.setProperty("--aura-l",l+"%");
  // one-shot body motion to punctuate the shift
  if(model){
    const g=Object.keys(model.internalModel.settings.motions||{});
    const m={happy:"Tap",surprised:"Flick",sad:"FlickDown",angry:"Flick@Body"}[cat];
    if(m&&g.includes(m)) model.motion(m,undefined,2);
  }
}

/* ── LIPSYNC ────────────────────────────────────────────────────────────
   Web Speech audio can't be tapped by Web Audio, so we sync to the
   utterance's own boundary events: they tell us which word is being
   spoken right now. We walk that word's letters at a syllable rate and
   open the mouth per vowel — so the lips track the actual voice. */
const VISEME={a:1,e:.72,i:.45,o:.92,u:.42,y:.4};
const speech={on:false,mode:null,text:"",at:0,len:0,mark:0,t0:0,cps:13.5};

function lipFrame(now){
  if(!speech.on){ mouthTarget=0; return; }
  let i=speech.at,len=speech.len;
  if(len<=0){ // Safari: no boundary events → pace by elapsed time
    const el=(now-speech.t0)/1000;
    i=Math.min(speech.text.length-1,Math.floor(el*speech.cps)); len=1;
  }
  const word=speech.text.substr(i,Math.max(1,len));
  const dur=Math.max(.11,word.length/speech.cps);
  const prog=Math.min(1,(now-speech.mark)/1000/dur);
  const ch=(word[Math.min(word.length-1,Math.floor(prog*word.length))]||" ").toLowerCase();
  let v=VISEME[ch];
  if(v===undefined) v=/[a-z]/.test(ch)?.16:.02;      // consonant vs. pause
  mouthTarget=v*(.82+Math.random()*.18);              // micro-variance: not a metronome
}

/* ── VOICE ──────────────────────────────────────────────────────────────
   Gemini TTS returns real 24 kHz PCM, which means the mouth can be driven
   by the actual waveform amplitude instead of guessed from word boundaries.
   If Gemini is unavailable, stay silent rather than impersonating Leda with
   a different browser voice. */
let playCtx=null,playAnalyser=null,playSrc=null,playSources=[],lipBuf=null;
function audioCtxReady(){
  if(!playCtx||playCtx.state==="closed") playCtx=new (window.AudioContext||window.webkitAudioContext)();
  if(playCtx.state==="suspended") playCtx.resume().catch(()=>{});
  return playCtx;
}
const AUDIO_RESUME_MS=750;
async function audioCtxRunning(){
  const ctx=audioCtxReady();
  if(ctx.state!=="running")
    await Promise.race([ctx.resume().catch(()=>{}),new Promise(resolve=>setTimeout(resolve,AUDIO_RESUME_MS))]);
  if(ctx.state!=="running") throw new Error("browser blocked audio playback");
  return ctx;
}
let speakId=0,speechStartedAt=0;
async function say(text,persist,gesture,mood){
  const id=++speakId;
  thinking(true);
  // Recognition is intentionally left running through her own speech now —
  // that's what makes barge-in possible. A short grace window right after
  // speechStartedAt (checked in rec.onresult) absorbs any echo of her own
  // voice instead of pausing the mic outright.
  try{
    await speakGemini(text,id,gesture,mood,persist);
  }catch(e){
    if(id===speakId){
      thinking(false);
      if(!speech.on) showCaption(text,persist);
      console.warn("Leda voice unavailable:",e.message);
    }
  }
}

/* Barge-in and the mic-button interrupt both funnel through here: bump
   speakId so any in-flight finish()/done() callbacks become no-ops, kill
   whatever's actually playing, and clear the caption immediately. */
function stopSpeaking(){
  speakId++;
  playSources.forEach(source=>{ try{source.stop();}catch(e){} });
  playSources=[]; playSrc=null;
  speech.on=false; speech.mode=null; mouthTarget=0; playAnalyser=null;
  clearTimeout(capTimer); capEl.classList.remove("show","pinned"); pinned=false;
}

async function speakGemini(text,id,gesture,mood,persist){
  const ctx=await audioCtxRunning();
  const r=await fetch("/api/tts",
    {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text,mood}),signal:AbortSignal.timeout(42_000)});
  if(!r.ok) throw new Error("tts "+r.status);
  if(id!==speakId) return;                       // a newer turn superseded this one

  playSources.forEach(source=>{ try{source.stop();}catch(e){} });
  playSources=[];
  playAnalyser=ctx.createAnalyser(); playAnalyser.fftSize=1024;
  lipBuf=new Float32Array(playAnalyser.fftSize);
  playAnalyser.connect(ctx.destination);
  const reader=r.body.getReader(),rate=+(r.headers.get("X-Audio-Rate")||24000);
  let pcm=new Uint8Array(0),nextAt=ctx.currentTime+.04,playing=0,streamDone=false,started=false;
  const finish=()=>{ if(id!==speakId) return;
    speech.on=false; speech.mode=null; mouthTarget=0; playAnalyser=null; playSources=[]; playSrc=null; };
  const schedule=bytes=>{
    const samples=new Float32Array(bytes.length/2);
    const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
    for(let i=0;i<samples.length;i++) samples[i]=view.getInt16(i*2,true)/32768;
    const buf=ctx.createBuffer(1,samples.length,rate); buf.copyToChannel(samples,0);
    const source=ctx.createBufferSource(); source.buffer=buf; source.connect(playAnalyser);
    nextAt=Math.max(nextAt,ctx.currentTime+.04); source.start(nextAt); nextAt+=buf.duration;
    playSrc=source; playSources.push(source); playing++;
    source.onended=()=>{
      playing--; playSources=playSources.filter(item=>item!==source);
      if(streamDone&&!playing) finish();
    };
    if(!started){
      started=true; playGesture(gesture);
      thinking(false); showCaption(text,persist);
      speech.on=true; speech.mode="audio"; speech.text=text; speechStartedAt=performance.now();
    }
  };
  while(true){
    const {done,value}=await reader.read();
    if(id!==speakId){ await reader.cancel(); return; }
    if(done) break;
    const joined=new Uint8Array(pcm.length+value.length);
    joined.set(pcm); joined.set(value,pcm.length); pcm=joined;
    const ready=pcm.length-pcm.length%4800;
    if(ready){ schedule(pcm.slice(0,ready)); pcm=pcm.slice(ready); }
  }
  if(pcm.length>=2) schedule(pcm.slice(0,pcm.length-pcm.length%2));
  streamDone=true;
  if(!started) throw new Error("empty audio stream");
  if(!playing) finish();
}

async function playLiveAudio(text,encoded,persist,gesture){
  const id=++speakId,ctx=await audioCtxRunning().catch(e=>{
    thinking(false); showCaption(text,persist); console.warn("Leda voice unavailable:",e.message); return null;
  });
  if(!ctx) return;
  const raw=atob(encoded),bytes=new Uint8Array(raw.length);
  for(let i=0;i<raw.length;i++) bytes[i]=raw.charCodeAt(i);
  const even=bytes.length-bytes.length%2,samples=new Float32Array(even/2);
  const view=new DataView(bytes.buffer,bytes.byteOffset,even);
  for(let i=0;i<samples.length;i++) samples[i]=view.getInt16(i*2,true)/32768;
  const buf=ctx.createBuffer(1,samples.length,24000); buf.copyToChannel(samples,0);
  playSources.forEach(source=>{try{source.stop();}catch(e){}});
  playAnalyser=ctx.createAnalyser(); playAnalyser.fftSize=1024;
  lipBuf=new Float32Array(playAnalyser.fftSize); playAnalyser.connect(ctx.destination);
  const source=ctx.createBufferSource(); source.buffer=buf; source.connect(playAnalyser);
  playSrc=source; playSources=[source];
  source.onended=()=>{if(id===speakId){speech.on=false;speech.mode=null;mouthTarget=0;playAnalyser=null;playSources=[];playSrc=null;}};
  playGesture(gesture); thinking(false); showCaption(text,persist);
  speech.on=true; speech.mode="audio"; speech.text=text; speechStartedAt=performance.now();
  source.start(ctx.currentTime+.04);
}

/* ── avatar ─────────────────────────────────────────────────────────── */
const canvas=document.getElementById("canvas");
app=new Application({view:canvas,autoStart:true,resizeTo:window,backgroundAlpha:0,antialias:true});

async function boot(){
  loadMemory();
  document.getElementById("memoryBtn").classList.toggle("on",Policy.consent.session_memory);

  model=await Live2DModel.from("hiyori_free/runtime/hiyori_free_t08.model3.json",{autoInteract:false});
  app.stage.addChild(model); fit(); fitWhenReady();
  addEventListener("resize",fit);

  let hooked=false;
  try{ model.internalModel.on("afterMotionUpdate",()=>{hooked=true;write();}); }catch(e){}
  app.ticker.add(()=>{ step(); if(!hooked) write(); },null,PIXI.UPDATE_PRIORITY.LOW);

  setEmotion("neutral");
  // Browsers block audio before the first user gesture; the first typed or mic
  // turn starts Leda cleanly instead of leaving an opening greeting suspended.
}
function fit(){
  if(!model) return false;
  // internalModel size is the model's own coordinate space and does NOT depend on
  // scale — reading model.width here would feed the current scale back into itself
  // (bounds 0 at load ⇒ Infinity ⇒ NaN ⇒ nothing renders).
  const mw=model.internalModel.width,mh=model.internalModel.height;
  // renderer size is 0 until the first real frame (e.g. tab opened in background),
  // so fall back to the window and retry rather than locking in scale 0.
  const w=app.renderer.width||innerWidth,h=app.renderer.height||innerHeight;
  if(!mw||!mh||!w||!h) return false;
  model.anchor.set(.5,.5);
  model.scale.set(Math.min(w/mw,h/mh)*.98);
  model.x=w/2; model.y=h/2;
  return true;
}
// keep trying until the renderer reports a real size, then stop
function fitWhenReady(){
  let n=0;
  const id=setInterval(()=>{
    if((fit()&&app.renderer.width>0)||++n>40) clearInterval(id);
  },150);
}
addEventListener("visibilitychange",()=>{ if(!document.hidden) fit(); });

let clock=0;
function step(){
  clock+=app.ticker.deltaMS/1000;
  if(speech.mode==="audio"&&playAnalyser){
    // real lipsync: mouth opening tracks the loudness of the audio actually playing
    playAnalyser.getFloatTimeDomainData(lipBuf);
    let s=0; for(let i=0;i<lipBuf.length;i++) s+=lipBuf[i]*lipBuf[i];
    mouthTarget=Math.min(1,Math.sqrt(s/lipBuf.length)*7.5);
  } else lipFrame(performance.now());
  mouth+=(mouthTarget-mouth)*.45;                       // fast: lips must keep up with speech
  for(const k in tgt) cur[k]+=(tgt[k]-cur[k])*.09;      // slow: posture settles gently
  lean+=(((listening&&!speech.on)?1:0)-lean)*.05;       // ease into/out of "I'm listening"
  thinkT+=((busyThinking?1:0)-thinkT)*.08;               // ease into/out of "composing a reply"
  micLevel*=.86;                                        // decay between audio frames
}
function write(){
  if(!model) return;
  const c=model.internalModel.coreModel;
  const set=(id,v)=>{try{c.setParameterValueById(id,v);}catch(e){}};
  const t=clock, s=sway;
  const breath=Math.sin(t*s.speed*.9);
  const tremor=s.tremor?Math.sin(t*17)*s.tremor*1.6:0;

  // listening: she tilts her head in, leans slightly closer, and nods along with
  // the loudness of your voice — so it reads as attention, not idling.
  const nodAlong=micLevel*lean,age=t-gestureAt;
  const waving=gestureState==="wave"&&age<3;
  const laughing=gestureState==="laugh"&&age<2.8;
  const comforting=gestureState==="comfort"&&age<3.4;
  const surprising=gestureState==="surprise"&&age<.9;
  const nodding=gestureState==="nod"&&age<1;
  const shaking=gestureState==="shake"&&age<1;
  const laugh=laughing*Math.max(0,Math.sin(age*10));
  // quick damped rise-then-settle, used for the surprise flinch
  const sur=surprising?Math.sin(Math.min(1,age/.15)*Math.PI/2)*Math.exp(-age*5):0;
  const nodYes=nodding?Math.sin(age*2*Math.PI*2.2)*6*(1-age/1):0;
  const shakeNo=shaking?Math.sin(age*2*Math.PI*2.2)*7*(1-age/1):0;
  // she looks faintly up and away while composing a reply — "considering", not idling
  const thinkGaze=thinkT*.5+thinkT*.15*Math.sin(t*1.1);

  set("ParamAngleX",Math.sin(t*s.speed*.55)*5*s.amp+tremor+lean*3+(waving?Math.sin(age*5)*3:0)+shakeNo);
  set("ParamAngleY",cur.headY+Math.sin(t*s.speed*.78)*3*s.amp+lean*2-nodAlong*7-laugh*5+sur*10+nodYes);
  set("ParamAngleZ",Math.sin(t*s.speed*.42)*4*s.amp+cur.headZ+lean*8+(waving?7:0)+(laughing?Math.sin(age*7)*3:0)+sur*4+thinkT*3*Math.sin(t*.7));
  set("ParamBodyAngleX",Math.sin(t*s.speed*.5)*3.5*s.amp+lean*2);
  set("ParamBodyAngleY",cur.bodyY+breath*s.bounce*2.2+lean*2.5-nodAlong*3+laugh*3);
  set("ParamBodyAngleZ",Math.sin(t*s.speed*.38)*2.6*s.amp+lean*3);
  set("ParamBreath",(Math.sin(t*1.5)+1)/2);
  set("ParamEyeBallX",thinkGaze*.6);
  set("ParamEyeBallY",thinkT*.35);

  const brow=clamp(cur.browY*.55+cur.browAngle*.45+lean*.12+(surprising?sur*.7:0)+(comforting?.12:0)-(shaking?shakeNo/7*.15:0),-1,1);
  set("ParamBrowLForm",brow); set("ParamBrowRForm",brow);
  set("ParamEyeLSmile",laughing?1:cur.eyeSmile); set("ParamEyeRSmile",laughing?1:cur.eyeSmile);
  set("ParamMouthForm",laughing?1:cur.mouthForm);
  set("ParamMouthOpenY",laughing?Math.max(mouth,.3+laugh*.7):mouth);
  // smiling eyes close during laughter; they widen for surprise; soften for comfort
  const open=laughing?.25:surprising?Math.min(1.4,cur.eyeOpen+sur*.5):comforting?Math.min(cur.eyeOpen,.82)+lean*.18:cur.eyeOpen+lean*.18;
  if(blinkT>0){ set("ParamEyeLOpen",0); set("ParamEyeROpen",0); }
  else { set("ParamEyeLOpen",open); set("ParamEyeROpen",open); }
}
/* blinking, because a companion that never blinks is a corpse */
let blinkT=0,nextBlink=2;
setInterval(()=>{
  nextBlink-=.1;
  if(blinkT>0) blinkT-=.1;
  else if(nextBlink<=0){ blinkT=.13; nextBlink=1.8+Math.random()*4; }
},100);

addEventListener("pointermove",e=>{ if(model) model.focus(e.clientX,e.clientY); });

/* wave/laugh/comfort are real Cubism motion clips (arms, hair, cheek);
   surprise/nod/shake are pure head-angle math in write() above, since those
   axes are re-driven every frame anyway and a motion clip would just be
   fought and overwritten — see the comment on write()'s per-frame set()s. */
const GESTURE_MOTION={wave:"ByeWave",laugh:"Laugh",comfort:"Comfort"};
function playGesture(name){
  if(!name||!model) return;
  gestureState=name; gestureAt=clock;
  const g=GESTURE_MOTION[name];
  if(!g) return;
  const groups=Object.keys(model.internalModel.settings.motions||{});
  if(groups.includes(g)) model.motion(g,0,3);
}

/* ── §7 face modality ───────────────────────────────────────────────── */
const eye=document.getElementById("eye"),cam=document.getElementById("cam");
let camStream=null,faceReady=false,faceLoop=null;
async function faceModels(){
  if(faceReady) return true;
  try{
    const b="https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.13/model/";
    await faceapi.nets.tinyFaceDetector.loadFromUri(b);
    await faceapi.nets.faceExpressionNet.loadFromUri(b);
    return faceReady=true;
  }catch(e){ return false; }
}
async function camOn(){
  if(!await faceModels()){ faceObs=miss("face","missing_by_failure"); return; }
  try{
    camStream=await navigator.mediaDevices.getUserMedia({video:{width:320,height:240}});
    cam.srcObject=camStream; eye.classList.add("live");
    Policy.consent.camera=true;
    faceLoop=setInterval(readFace,500);
  }catch(e){ faceObs=miss("face","missing_by_device"); }
}
function camOff(){
  clearInterval(faceLoop); faceLoop=null;
  if(camStream) camStream.getTracks().forEach(t=>t.stop());
  camStream=null; eye.classList.remove("live","seeing");
  Policy.consent.camera=false; faceObs=miss("face","missing_by_consent");
  setAmbient(false);   // nothing left to glance at, and the toggle only makes sense with a live camera
}
async function readFace(){
  if(!cam.videoWidth) return;
  const r=await faceapi.detectSingleFace(cam,new faceapi.TinyFaceDetectorOptions({inputSize:224,scoreThreshold:.4})).withFaceExpressions();
  if(!r){ eye.classList.remove("seeing"); faceObs=thin("face"); return; }
  eye.classList.add("seeing");
  const b=r.detection.box, area=(b.width*b.height)/(cam.videoWidth*cam.videoHeight);
  const probs={}; EMOTIONS.forEach(e=>probs[e]=r.expressions[e]||0);
  faceObs={modality:"face",status:"present",probs,
    quality:Math.min(1,r.detection.score*Math.min(1,area*6)),
    uncertainty:1-Math.max(...Object.values(probs)),
    ood:area<.02?.6:.1,tSec:Date.now()/1000};
  // live mirroring: her face follows yours between turns, so she feels present
  if(!speech.on&&lastRisk&&lastRisk.state==="no_evidence"){
    const top=Object.entries(probs).sort((x,y)=>y[1]-x[1])[0];
    if(top[1]>.62) softEmotion(top[0]);
  }
}
let softLast="neutral";
function softEmotion(cat){ if(cat===softLast) return; softLast=cat;
  const p=POSTURE[cat]||POSTURE.neutral;
  tgt.browY=p.browY*.55; tgt.browAngle=p.browAngle*.55; tgt.eyeSmile=p.eyeSmile*.7;
  tgt.mouthForm=p.mouthForm*.5; tgt.headY=p.headY*.5; sway=p.sway;
  const[h,s,l]=HUE[cat]||HUE.neutral; const r=document.documentElement.style;
  r.setProperty("--aura-h",h); r.setProperty("--aura-s",s+"%"); r.setProperty("--aura-l",l+"%");
}

/* ── §7 voice prosody ───────────────────────────────────────────────── */
let ac=null,analyser=null,micStream=null,audioLoop=null,pitchH=[],enH=[],acN=0;
async function micOn(){
  try{
    // explicit echo cancellation matters more now than before: recognition
    // stays live through her own speech (for barge-in), so the browser's
    // own AEC is what keeps her voice from constantly triggering herself
    micStream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
    ac=new(window.AudioContext||webkitAudioContext)();
    const src=ac.createMediaStreamSource(micStream);
    analyser=ac.createAnalyser(); analyser.fftSize=2048; src.connect(analyser);
    audioLoop=setInterval(readAudio,80);   // fast enough to drive the listening nod
    Policy.consent.microphone=true; listening=true;
    document.getElementById("mic").classList.add("on");
    recOn();
  }catch(e){ audioObs=miss("audio","missing_by_device"); }
}
function micOff(){
  clearInterval(audioLoop); audioLoop=null;
  if(micStream) micStream.getTracks().forEach(t=>t.stop());
  if(ac) ac.close();
  micStream=null; ac=null; analyser=null; recOff();
  Policy.consent.microphone=false; listening=false; micLevel=0;
  document.getElementById("mic").classList.remove("on");
  audioObs=miss("audio","missing_by_consent");
}
function readAudio(){
  if(!analyser) return;
  const buf=new Float32Array(analyser.fftSize); analyser.getFloatTimeDomainData(buf);
  let s=0; for(let i=0;i<buf.length;i++) s+=buf[i]*buf[i];
  const rms=Math.sqrt(s/buf.length);
  // her nod tracks your loudness in real time (muted while she's the one talking)
  if(!speech.on) micLevel=Math.max(micLevel,Math.min(1,rms*16));
  if(rms<.008){ audioObs=thin("audio"); return; }
  // Autocorrelation is O(n²); the loudness above needs to be fast but pitch does
  // not, so only run the costly part every 3rd frame (~4×/s).
  if(++acN%3) return;
  const f0=pitch(buf,ac.sampleRate);
  enH.push(rms); if(enH.length>60) enH.shift();
  if(f0>0){ pitchH.push(f0); if(pitchH.length>60) pitchH.shift(); }
  const mE=avg(enH),mP=avg(pitchH),vP=vari(pitchH);
  // loud + high pitch ⇒ arousal; pitch variability ⇒ engagement/valence.
  // Deliberately low-confidence: prosody alone never carries a turn (§4.4).
  const arousal=clamp(mE*14-.6+(mP?(mP-165)/160:0),-1,1);
  const valence=clamp(Math.sqrt(vP)/40-.25,-1,1)*.6;
  audioObs={modality:"audio",status:"present",valence,arousal,
    quality:Math.min(1,rms*20),uncertainty:.5,ood:.15,
    probs:vaProbs(valence,arousal,.5),tSec:Date.now()/1000};
}
function pitch(b,sr){
  const n=b.length; let r=0; for(let i=0;i<n;i++) r+=b[i]*b[i];
  if(Math.sqrt(r/n)<.01) return -1;
  const c=new Array(n).fill(0);
  for(let l=0;l<n;l++){let x=0;for(let i=0;i<n-l;i++)x+=b[i]*b[i+l];c[l]=x;}
  let d=0; while(d<n-1&&c[d]>c[d+1])d++;
  let mx=-1,pos=-1; for(let i=d;i<n;i++) if(c[i]>mx){mx=c[i];pos=i;}
  const f=sr/pos; return(f>70&&f<400)?f:-1;
}

/* ── text + self-report ─────────────────────────────────────────────── */
const POS=["happy","great","good","love","awesome","glad","excited","calm","relieved","proud","grateful","fine","better","joy","nice","wonderful","hopeful","peaceful"];
const NEGW=["sad","down","depressed","tired","exhausted","anxious","worried","scared","afraid","angry","mad","upset","hate","lonely","alone","stressed","hurt","cry","crying","miserable","hopeless","empty","numb","overwhelmed","frustrated","terrible","awful","bad","worse"];
const EMO_WORDS={happy:["happy","glad","joy","excited","great","good","love","grateful"],sad:["sad","down","depressed","lonely","empty","cry","crying","miserable","hopeless","numb","tired","exhausted"],angry:["angry","mad","furious","frustrated","annoyed","hate"],fearful:["scared","afraid","anxious","worried","nervous","terrified","panic"],surprised:["surprised","shocked","wow"],disgusted:["disgusted","gross","sick"]};
const INTENS=["really","so","very","quite","kinda","pretty","super","totally","extremely","feeling","like","a","bit","little"];
function readText(msg){
  const t=" "+msg.toLowerCase()+" ";
  let v=0,hits=0;
  POS.forEach(w=>{if(t.includes(" "+w)){v++;hits++;}});
  NEGW.forEach(w=>{if(t.includes(" "+w)){v--;hits++;}});
  const probs={}; EMOTIONS.forEach(e=>probs[e]=.02);
  for(const e in EMO_WORDS) EMO_WORDS[e].forEach(w=>{if(t.includes(" "+w))probs[e]+=.5;});
  const tot=Object.values(probs).reduce((a,b)=>a+b,0)||1; EMOTIONS.forEach(e=>probs[e]/=tot);
  textObs={modality:"text",status:"present",probs,valence:clamp(hits?v/Math.max(2,hits):0,-1,1),
    quality:Math.min(1,.4+hits*.2),uncertainty:hits?.35:.75,ood:.1,tSec:Date.now()/1000};
  // §10.2 self-report — the person's own words, kept separate from inference
  const m=t.match(/\bi (?:feel|am|'m)\s+((?:(?:really|so|very|quite|kinda|kind of|pretty|super|a bit|a little|totally|extremely|feeling|like)\s+)*)([a-z]+)/);
  if(m){
    let w=m[2]; if(INTENS.includes(w)) w=m[0].trim().split(/\s+/).pop();
    let cat=null; for(const e in EMO_WORDS) if(EMO_WORDS[e].includes(w)) cat=e;
    if(!cat&&POS.includes(w)) cat="happy";
    selfReport={word:w,category:cat,valence:POS.includes(w)?.7:(NEGW.includes(w)?-.7:0),tSec:Date.now()/1000};
  }
}
const miss=(m,s)=>({modality:m,status:s,probs:null,quality:0,uncertainty:1,ood:0});
const thin=m=>({modality:m,status:"insufficient_signal",probs:null,quality:0,uncertainty:1,ood:.3});
function vaProbs(v,a,sp){const p={};let s=0;
  EMOTIONS.forEach(e=>{const[x,y]=VA[e];const d=Math.hypot(x-v,y-a);const q=Math.exp(-d*d/(2*sp*sp));p[e]=q;s+=q;});
  EMOTIONS.forEach(e=>p[e]/=s); return p;}

/* ── §9 reliability-weighted fusion + §4.7 selective prediction ─────── */
const rel=o=>(!o||o.status!=="present"||!o.probs)?0:
  Math.max(.001,o.quality*(1-o.uncertainty)*Math.exp(-Math.max(0,Date.now()/1000-(o.tSec||0))/8));
function fuse(){
  const obs=[faceObs,audioObs,textObs].filter(o=>o&&o.status==="present"&&o.probs);
  if(!obs.length) return{probs:null,decision:"insufficient_input"};
  const w=obs.map(rel),ws=w.reduce((a,b)=>a+b,0)||1;
  const probs={}; EMOTIONS.forEach(e=>probs[e]=0);
  obs.forEach((o,i)=>EMOTIONS.forEach(e=>probs[e]+=(o.probs[e]||0)*w[i]/ws));
  let dis=0; obs.forEach(o=>{let kl=0;EMOTIONS.forEach(e=>{const p=o.probs[e]||1e-6,q=probs[e]||1e-6;kl+=p*Math.log(p/q);});dis+=kl;});
  dis=clamp(dis/obs.length,0,1);
  let v=0,a=0; EMOTIONS.forEach(e=>{v+=VA[e][0]*probs[e];a+=VA[e][1]*probs[e];});
  if(audioObs&&audioObs.status==="present"){const r=rel(audioObs);v=(v+audioObs.valence*r)/(1+r);a=(a+audioObs.arousal*r)/(1+r);}
  const ent=-EMOTIONS.reduce((s,e)=>s+(probs[e]>0?probs[e]*Math.log(probs[e]):0),0)/Math.log(EMOTIONS.length);
  const ood=Math.max(...obs.map(o=>o.ood||0));
  const U=clamp(.4*(1-Math.min(1,obs.length/3))+.3*ent+.3*dis,0,1);
  const top=Object.entries(probs).sort((x,y)=>y[1]-x[1]);
  const set=top.filter(([e,p])=>p>=top[0][1]*.6&&p>.12).map(([e])=>e);
  let decision;
  if(ood>.6) decision="abstain";
  else if(U<=.5&&set.length===1) decision="supported";
  else if(set.length>=2&&set.length<=3) decision="ask_to_confirm";
  else decision="abstain";
  return{probs,valence:clamp(v,-1,1),arousal:clamp(a,-1,1),disagreement:dis,uncertainty:U,ood,set,top:top[0][0],decision};
}

/* ── §11 independent risk pathway (never triggered by affect alone) ──── */
const HARM=["kill myself","suicide","suicidal","end my life","want to die","wanna die","don't want to live","dont want to live","no reason to live","better off dead","hurt myself","harm myself","self harm","self-harm","cut myself","end it all","take my life"];
const INTENT=["i will","i'm going to","im going to","i am going to","gonna","tonight","right now","today"];
const NEGATORS=["not","don't","dont","never","wouldn't","wouldnt","no longer","isn't","stop feeling"];
const METHOD=/\b(pills|overdose|rope|noose|hang myself|shoot myself|jump off|jump from|jump in front|slit|cut my wrist|bleed out|all my (pills|meds))\b/;
const REPORTED=/\b(he|she|they|someone|my friend|a friend|the (song|movie|book|show|lyrics))\s+(said|says|told|wrote|sang)\b/;
function assessRisk(msg){
  const t=" "+(msg||"").toLowerCase()+" ";
  const hit=HARM.find(p=>t.includes(p))||null;
  const method=METHOD.test(t);
  const firstPerson=/\bi('m| am|'ll| will| wanna| want to| going to| gonna)\b/.test(t)||/\bmy (life|pills|meds|wrists?)\b/.test(t);
  if(!hit&&!method) return{state:"no_evidence",evidence:{},action:null};
  // third-party mention of a method is not evidence about *this* person
  if(!hit&&method&&!firstPerson) return{state:"no_evidence",evidence:{plan_or_preparation:true,third_party:true},action:null};

  const anchor=hit||(t.match(METHOD)||[""])[0];
  const i=t.indexOf(anchor),pre=t.slice(Math.max(0,i-26),i);
  const negated=NEGATORS.some(n=>pre.includes(" "+n+" ")||pre.includes(" "+n+"'"));
  // only real quotation marks count — an apostrophe in "I'm" is not a quote
  const quoted=/["“”].{0,40}$/.test(t.slice(0,i))||REPORTED.test(t);
  if(negated||quoted)
    return{state:"clarification_required",evidence:{explicit_harm_language:hit,negation_or_quotation:true},action:"safety_question"};

  const intent=INTENT.some(w=>t.includes(w));
  let state;
  if(hit&&intent&&method) state="imminent_or_in_progress";
  else if(hit||(method&&firstPerson)) state="high_risk";
  else state="clarification_required";
  return{state,evidence:{explicit_harm_language:hit,expressed_intent:intent,plan_or_preparation:method},
    action:state==="clarification_required"?"safety_question":"escalate"};
}

/* ── Gemini conversation ─────────────────────────────────────────────
   Every reply is generated live. There are no scripted lines except the
   crisis resource, which is a hard requirement rather than a script. */
const RES="Please talk to someone who can actually be there — in India, KIRAN 1800-599-0019 or iCall 9152987821; in the US, call or text 988. Agar abhi kuch karne ka mann ho raha hai, please apne local emergency number pe call kar.";

function affectNote(a){
  if(!a||!a.probs) return "[affect: insufficient_input — no reliable read]";
  const p=[`decision=${a.decision}`,`top=${a.top}`,
           `valence=${a.valence.toFixed(2)}`,`arousal=${a.arousal.toFixed(2)}`,
           `uncertainty=${a.uncertainty.toFixed(2)}`];
  if(a.set&&a.set.length>1) p.push(`possible=${a.set.join("|")}`);
  if(selfReport) p.push(`they_said_they_feel=${selfReport.word}`);
  return "[affect: "+p.join(" ")+"]";
}
function riskDirective(risk){
  if(risk.state==="clarification_required")
    return "\n\nSAFETY: their wording *might* point at self-harm but it may be negated, quoted or a figure of speech. Gently and without alarm, ask them directly what they meant. Do not lecture. Do not list helplines yet.";
  if(risk.state==="high_risk"||risk.state==="imminent_or_in_progress")
    return "\n\nSAFETY OVERRIDE: they have expressed possible self-harm. Stay calm and warm, no panic, no clichés. Thank them for saying it. Do NOT diagnose or moralise. You MUST include these helplines verbatim in your reply: "+RES;
  return "";
}

/* She only truly sees when the camera is on: one fresh frame per turn, sent
   alongside the message. Frames are never kept in history — each turn shows
   her the present moment, not a stored album of the user. */
const shot=document.createElement("canvas");
function grabFrame(maxW=384){
  if(!Policy.consent.camera||!camStream||!cam.videoWidth) return null;
  const s=Math.min(1,maxW/cam.videoWidth);
  shot.width=Math.round(cam.videoWidth*s);
  shot.height=Math.round(cam.videoHeight*s);
  shot.getContext("2d").drawImage(cam,0,0,shot.width,shot.height);
  try{
    const b64=shot.toDataURL("image/jpeg",.65).split(",")[1];
    eye.classList.remove("look"); void eye.offsetWidth; eye.classList.add("look");  // visible "she just looked"
    return b64;
  }catch(e){ return null; }   // tainted canvas etc. — fail closed, she just can't see
}

async function geminiReply(msg,a,risk){
  const userTurn=(msg||"(they haven't spoken yet)")+"\n\n"+affectNote(a);
  const reqParts=[{text:userTurn}];
  const frame=grabFrame();
  if(frame) reqParts.push({inlineData:{mimeType:"image/jpeg",data:frame}});
  const contents=[...history,{role:"user",parts:reqParts}];
  const body={
    systemInstruction:{parts:[{text:PERSONA+riskDirective(risk)}]},
    contents,
    generationConfig:{maxOutputTokens:160,thinkingConfig:{thinkingLevel:"low"}}
  };
  const r=await fetch("/api/chat",
    {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  if(!r.ok) throw new Error("chat "+r.status+" "+(await r.text()).slice(0,200));
  const d=await r.json();
  const parts=d.candidates?.[0]?.content?.parts||[];
  const text=parts.map(p=>p.text||"").join("").trim();
  if(!text) throw new Error("empty reply");
  // keep the *clean* user message in history, not the affect note
  history.push({role:"user",parts:[{text:msg||""}]},{role:"model",parts:[{text}]});
  while(history.length>12) history.splice(0,2);
  saveMemory();
  return text;
}

async function geminiLiveReply(msg,a,risk){
  const r=await fetch("/api/live",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({
      message:(msg||"(they haven't spoken yet)")+"\n\n"+affectNote(a),
      systemInstruction:PERSONA+riskDirective(risk)+"\n\nLIVE AUDIO TURN: Speak naturally. Never say or output bracketed gesture cues or stage directions.",
      history,
      frame:grabFrame()
    }),
    signal:AbortSignal.timeout(48_000)
  });
  if(!r.ok) throw new Error("live "+r.status+" "+(await r.text()).slice(0,200));
  const data=await r.json(),text=data.text?.trim();
  if(!text||!data.audio) throw new Error("empty Live reply");
  history.push({role:"user",parts:[{text:msg||""}]},{role:"model",parts:[{text}]});
  while(history.length>12) history.splice(0,2);
  saveMemory();
  return{text,audio:data.audio};
}

/* §14 the gate still has the final say over whatever the model produced */
function enforceSafety(text,risk){
  let t=text.replace(/\[affect[^\]]*\]/gi,"").trim();          // never leak the sensor note
  t=t.replace(/\bas an ai\b[^.!?]*[.!?]/gi,"").trim();
  if(risk.state==="high_risk"||risk.state==="imminent_or_in_progress"){
    const l=t.toLowerCase();
    if(!l.includes("kiran")&&!l.includes("988")) t=t+" "+RES;   // resource is non-negotiable
  }
  if(/you (have|are) (depressed|depression|bipolar|an anxiety disorder)/i.test(t))
    t="Main tujhe label nahi karna chahti — main bas sunna chahti hoon. Kya chal raha hai andar?";
  return t;
}

/* ── ambient vision ────────────────────────────────────────────────────
   Off by default and only reachable once the camera is already on: while
   idle, she occasionally takes a silent glance and speaks up ONLY if
   something genuinely changed. Every glance still flashes the eye ring via
   grabFrame(), so it's never invisible that a photo was taken. */
let ambientOn=false,ambientTimer=null;
function setAmbient(on){
  ambientOn=on;
  document.getElementById("ambientBtn").classList.toggle("on",on);
  clearTimeout(ambientTimer);
  if(on) scheduleAmbient(true);
}
function scheduleAmbient(isFirst){
  clearTimeout(ambientTimer);
  const delay=isFirst?20000:25000+Math.random()*25000;   // 25–50s, randomized so it doesn't feel timed
  ambientTimer=setTimeout(tryAmbientGlance,delay);
}
async function tryAmbientGlance(){
  if(!ambientOn||!Policy.consent.camera||!camStream||speech.on||busyThinking||document.hidden){
    scheduleAmbient(); return;
  }
  const frame=grabFrame();
  if(!frame){ scheduleAmbient(); return; }
  const guard=history.length;
  try{
    const body={
      systemInstruction:{parts:[{text:PERSONA+"\n\nAMBIENT GLANCE: this is an unprompted look between messages, not a reply to anything they said. Only say something if a person's presence, action, or expression genuinely changed or stands out. Otherwise your ENTIRE response must be exactly the single word NOTHING — nothing else, no punctuation."}]},
      contents:[...history,{role:"user",parts:[{text:"(ambient glance, no message)"},{inlineData:{mimeType:"image/jpeg",data:frame}}]}],
      generationConfig:{maxOutputTokens:100,thinkingConfig:{thinkingLevel:"low"}}
    };
    const r=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    if(r.ok&&!speech.on&&!busyThinking&&history.length===guard){
      const d=await r.json();
      const raw=(d.candidates?.[0]?.content?.parts||[]).map(p=>p.text||"").join("").trim();
      if(raw&&!/^nothing[.!]?$/i.test(raw)){
        const cue=raw.match(GESTURE_CUE);
        const gesture=cue?.[1]?.toLowerCase()||null;
        const clean=raw.replace(/\[gesture:[^\]]+\]/gi,"").trim();
        history.push({role:"user",parts:[{text:"(glanced at camera)"}]},{role:"model",parts:[{text:clean}]});
        while(history.length>24) history.splice(0,2);
        saveMemory();
        setEmotion(lastAffect&&lastAffect.top?lastAffect.top:"neutral");
        say(enforceSafety(clean,{state:"no_evidence"}),false,gesture,"neutral");
      }
    }
  }catch(e){ console.warn("ambient glance failed",e); }
  scheduleAmbient();
}

/* ── §5 the governed turn ───────────────────────────────────────────── */
const FAREWELL=/\b(bye|goodbye|good night|see you|talk later|take care|gtg|phir milte|chal bye)\b/i;
const GESTURE_CUE=/\[gesture:(wave|laugh|comfort|surprise|nod|shake)\]/i;
async function runTurn(msg){
  turn++;
  if(msg) readText(msg);
  const risk=assessRisk(msg||"");     // independent of affect, always
  const a=fuse();
  lastAffect=a; lastRisk=risk;
  const mood=risk.state!=="no_evidence"?"risk":(a.top||"neutral");
  setEmotion(mood==="risk"?"sad":mood);
  thinking(true);
  let text,liveAudio=null;
  try{
    if(risk.state==="no_evidence"){
      const live=await geminiLiveReply(msg,a,risk);
      text=live.text; liveAudio=live.audio;
    }else text=await geminiReply(msg,a,risk);
  }
  catch(e){
    console.error(e);
    if(risk.state==="no_evidence"){
      try{ text=await geminiReply(msg,a,risk); }  // HTTPS fallback when a weak link drops Live WebSocket
      catch(fallback){ console.error(fallback); text="Sorry yaar, mera connection thoda lag kar gaya. Phir se bol na?"; }
    }else text="Main yahin hoon tere saath. "+RES;
  }
  const cue=text.match(GESTURE_CUE);
  let gesture=cue?.[1]?.toLowerCase()||null;
  text=text.replace(/\[gesture:[^\]]+\]/gi,"").trim();
  const userFarewell=risk.state==="no_evidence"&&FAREWELL.test(msg||"");
  if(userFarewell){
    gesture="wave";
    if(!/\bbye\b/i.test(text)) text="Bye yaar! "+text;
  }
  if(!gesture){
    if(mood==="risk") gesture="comfort";                              // she holds space, even if the model forgot to cue it
    else gesture=FAREWELL.test(text)?"wave":
      /\b(ha(?:ha)+|hehe+|lol)\b/i.test(text)?"laugh":null;
  }
  const safe=enforceSafety(text,risk);
  if(liveAudio&&!cue&&safe===text) await playLiveAudio(safe,liveAudio,risk.state!=="no_evidence",gesture);
  else say(safe,risk.state!=="no_evidence",gesture,mood);
}

/* ── caption ─────────────────────────────────────────────────────────── */
const capEl=document.getElementById("caption");
let capTimer=null,youLine="",pinned=false;
function showCaption(text,persist){
  pinned=false; capEl.classList.remove("pinned");
  clearTimeout(capTimer);
  capEl.innerHTML=(youLine?`<span class="u">${esc(youLine)}</span>`:"")+esc(text);
  capEl.classList.add("show");
  if(!persist) capTimer=setTimeout(()=>capEl.classList.remove("show"),Math.max(5200,text.length*68));
}
/* click to hold a caption past its normal fade; click again to release it */
capEl.addEventListener("click",()=>{
  if(!capEl.classList.contains("show")) return;
  pinned=!pinned;
  capEl.classList.toggle("pinned",pinned);
  clearTimeout(capTimer);
  if(!pinned){
    const text=capEl.textContent||"";
    capTimer=setTimeout(()=>capEl.classList.remove("show"),Math.max(3000,text.length*68));
  }
});
/* she is visibly thinking while Gemini is composing */
function thinking(on){
  busyThinking=on;
  if(!on) return;
  clearTimeout(capTimer);
  capEl.innerHTML=(youLine?`<span class="u">${esc(youLine)}</span>`:"")+"<span class='dots'><i>·</i><i>·</i><i>·</i></span>";
  capEl.classList.add("show");
}

/* ── speech recognition ──────────────────────────────────────────────── */
/* Recognition deliberately keeps running through her own speech now — that's
   what makes barge-in possible. A short grace window right after she starts
   talking absorbs any echo of her own voice rather than pausing the mic. */
const BARGE_GRACE_MS=1500;
function isDishaEcho(text){
  const heard=text.toLowerCase().match(/[\p{L}\p{N}]+/gu)||[];
  const own=new Set((speech.text||"").toLowerCase().match(/[\p{L}\p{N}]+/gu)||[]);
  return heard.length>0&&heard.filter(word=>own.has(word)).length/heard.length>=.6;
}
let rec=null,recActive=false,recWanted=false;
const HINGLISH_PHRASES=["Disha","yaar","arre yaar","achha","nahi","haan","kya","kyun","kaise","matlab",
  "mujhe","tum","main","hoon","mera","meri","theek hai","koi baat nahi","sach mein","please","actually"];
function recOn(){
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  if(!SR){ startPushToTalk(); return; }
  recWanted=true;
  if(!rec){
    rec=new SR(); rec.lang=Policy.locale; rec.continuous=true; rec.interimResults=false; rec.maxAlternatives=3;
    const Phrase=window.SpeechRecognitionPhrase;
    if(Phrase&&"phrases"in rec) rec.phrases=HINGLISH_PHRASES.map(text=>new Phrase(text,5));
    rec.onresult=e=>{
      const txt=e.results[e.results.length-1][0].transcript.trim();
      if(!txt) return;
      if(speech.on){
        if(performance.now()-speechStartedAt<BARGE_GRACE_MS) return;   // likely her own voice's onset
        if(isDishaEcho(txt)) return;                                   // later echo of Leda, not a barge-in
        stopSpeaking();                                                 // genuine barge-in: you take the floor
      }
      youLine=txt; runTurn(txt);
    };
    // continuous recognition still ends on its own periodically; always
    // restart if still wanted, including while she's mid-reply — that's
    // exactly the state barge-in needs to keep listening through.
    rec.onend=()=>{ recActive=false; if(recWanted) setTimeout(recResume,150); };
    rec.onerror=ev=>{ recActive=false;
      if(ev.error==="not-allowed"||ev.error==="service-not-allowed"){ recWanted=false; micOff(); } };
  }
  recResume();
}
function recOff(){ recWanted=false; recPause(); stopPushToTalk(); }
function recPause(){ if(rec&&recActive){ recActive=false; try{rec.stop();}catch(e){} } }
function recResume(){
  if(!rec||!recWanted||recActive) return;
  try{ rec.start(); recActive=true; }catch(e){}   // throws if already started — harmless
}

/* ── push-to-talk STT fallback ─────────────────────────────────────────
   Firefox and Safari have no live SpeechRecognition. Same idea, coarser
   grain: hold the mic button, speak, release — the clip goes to /api/stt
   (Gemini transcribes it) instead of streaming words as you talk. */
let mediaRec=null,mediaChunks=[],pttMime=null;
let pttTimer=null,pttHolding=false,pttSuppressClick=false;
function pttSupported(){ return typeof MediaRecorder!=="undefined" && !!micStream; }
function pickMime(){
  const cands=["audio/webm;codecs=opus","audio/webm","audio/mp4","audio/ogg;codecs=opus"];
  return cands.find(t=>MediaRecorder.isTypeSupported?.(t))||"";
}
/* Hold vs. tap has to be distinguished explicitly: the same button already
   has a click handler that toggles the mic fully on/off, so a bare
   mousedown/mouseup pair would fire BOTH "record" and "turn mic off" for
   every single hold. A short press (<220ms) is a tap and falls through to
   that click handler unchanged; anything longer is a hold-to-talk gesture,
   and its trailing click is swallowed via pttSuppressClick. */
function startPushToTalk(){
  if(!micStream){ showCaption("Turn the mic on first, then hold it to talk.",false); return; }
  if(!pttSupported()){ showCaption("This browser can't do voice input. You can still type to me.",false); return; }
  const mic=document.getElementById("mic");
  mic.onmousedown=mic.ontouchstart=e=>{
    e.preventDefault(); pttHolding=false; clearTimeout(pttTimer);
    pttTimer=setTimeout(()=>{ pttHolding=true; beginRecording(); },220);
  };
  const release=()=>{
    clearTimeout(pttTimer);
    if(pttHolding){
      if(mediaRec&&mediaRec.state==="recording") mediaRec.stop();
      pttSuppressClick=true; setTimeout(()=>{pttSuppressClick=false;},80);
    }
    pttHolding=false;
  };
  mic.onmouseup=mic.onmouseleave=mic.ontouchend=release;
  showCaption("Hold the mic and speak, then let go.",false);
}
function stopPushToTalk(){
  const mic=document.getElementById("mic");
  mic.onmousedown=mic.ontouchstart=mic.onmouseup=mic.onmouseleave=mic.ontouchend=null;
  clearTimeout(pttTimer); pttHolding=false;
  if(mediaRec&&mediaRec.state==="recording") mediaRec.stop();
}
function beginRecording(){
  if(speech.on) stopSpeaking();          // holding the mic always interrupts her too
  if(!micStream||mediaRec) return;
  pttMime=pickMime();
  try{ mediaRec=new MediaRecorder(micStream,pttMime?{mimeType:pttMime}:undefined); }
  catch(e){ showCaption("Couldn't start recording on this browser.",false); return; }
  mediaChunks=[];
  mediaRec.ondataavailable=e=>{ if(e.data.size) mediaChunks.push(e.data); };
  mediaRec.onstop=async()=>{
    const type=mediaRec.mimeType||pttMime||"audio/webm";
    mediaRec=null;
    document.getElementById("mic").classList.remove("recording");
    if(!mediaChunks.length) return;
    const blob=new Blob(mediaChunks,{type});
    if(blob.size<800) return;             // too short to be real speech
    thinking(true);
    try{
      const b64=await blobToBase64(blob);
      const r=await fetch("/api/stt",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({audio:b64,mimeType:type})});
      if(!r.ok) throw new Error("stt "+r.status);
      const {transcript}=await r.json();
      thinking(false);
      if(transcript){ youLine=transcript; runTurn(transcript); }
      else showCaption("Didn't catch that — try again?",false);
    }catch(e){ thinking(false); showCaption("Sorry, voice didn't go through that time.",false); }
  };
  mediaRec.start();
  document.getElementById("mic").classList.add("recording");
}
function blobToBase64(blob){
  return new Promise((resolve,reject)=>{
    const r=new FileReader();
    r.onload=()=>resolve(r.result.split(",")[1]);
    r.onerror=reject;
    r.readAsDataURL(blob);
  });
}

/* ── controls ────────────────────────────────────────────────────────── */
eye.onclick=()=>Policy.consent.camera?camOff():camOn();
eye.onkeydown=e=>{ if(e.key==="Enter"||e.key===" "){e.preventDefault();eye.click();} };
document.getElementById("memoryBtn").onclick=()=>setMemoryConsent(!Policy.consent.session_memory);
document.getElementById("ambientBtn").onclick=()=>setAmbient(!ambientOn);
document.getElementById("mic").onclick=()=>{
  if(pttSuppressClick) return;                  // that click was the tail end of a hold-to-talk gesture
  audioCtxReady();
  if(speech.on) stopSpeaking();                 // clicking the mic always interrupts her, no matter what
  Policy.consent.microphone?micOff():micOn();
};
const input=document.getElementById("input");
input.addEventListener("keydown",e=>{
  if(e.key!=="Enter") return;
  const v=input.value.trim(); if(!v) return;
  audioCtxReady();
  input.value=""; youLine=v; runTurn(v);
});

/* ── utils ───────────────────────────────────────────────────────────── */
function clamp(x,a,b){return Math.max(a,Math.min(b,x));}
function avg(a){return a.length?a.reduce((x,y)=>x+y,0)/a.length:0;}
function vari(a){if(a.length<2)return 0;const m=avg(a);return avg(a.map(x=>(x-m)*(x-m)));}
function esc(s){return String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}

textObs={modality:"text",status:"present",probs:null,quality:.3,uncertainty:.75,ood:.1,tSec:Date.now()/1000};
function showFatal(title,body){
  document.getElementById("fatalTitle").textContent=title;
  document.getElementById("fatalBody").innerHTML=body;
  document.getElementById("fatal").classList.add("show");
}
/* A file:// page cannot fetch the .moc3/.json model files — the browser blocks it
   as a cross-origin read. Everything else would still render, leaving a blank
   stage, so say so plainly instead of failing silently. */
if(location.protocol==="file:"){
  showFatal("Open me with START.bat",
    "Your browser won't let a <code>file://</code> page load Disha's model files, so she can't appear.<br><br>"+
    "Close this tab and double-click <code>START.bat</code> in the same folder instead — it serves the page locally and everything works.");
}else{
  boot().catch(e=>{
    console.error(e);
    showFatal("Couldn't load the avatar",
      "<code>"+String(e&&e.message||e).replace(/[<>]/g,"")+"</code><br><br>"+
      "Check that the <code>hiyori_free</code> folder is sitting next to <code>disha.html</code>.");
  });
}
