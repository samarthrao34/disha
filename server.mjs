import "dotenv/config";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";
import {GoogleGenAI,Modality} from "@google/genai";

const ROOT=path.dirname(fileURLToPath(import.meta.url));
const DIST=path.join(ROOT,"dist"),MODEL_DIR=path.join(ROOT,"hiyori_free");
const PORT=8777,API="https://generativelanguage.googleapis.com/v1beta/models";
const LIVE_MODEL="gemini-3.1-flash-live-preview";
const CHAT_MODEL="gemini-3.5-flash-lite",TTS_MODEL="gemini-3.1-flash-tts-preview";
const TTS_FALLBACK_MODEL="gemini-2.5-flash-preview-tts";
const key=process.env.GEMINI_API_KEY;
if(!key) throw new Error("GEMINI_API_KEY is missing from .env");
const ai=new GoogleGenAI({apiKey:key});
const recent=[];
const mime={
  ".html":"text/html; charset=utf-8",".js":"text/javascript; charset=utf-8",".css":"text/css; charset=utf-8",
  ".json":"application/json",".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg",".webp":"image/webp",
  ".svg":"image/svg+xml",".wasm":"application/wasm",".moc3":"application/octet-stream"
};

function reply(res,status,body,type="text/plain; charset=utf-8"){
  res.writeHead(status,{"Content-Type":type,"Cache-Control":"no-store","X-Content-Type-Options":"nosniff"});
  res.end(body);
}
function localFile(base,relative){
  const file=path.resolve(base,relative),back=path.relative(base,file);
  return back&&back!==".."&&!back.startsWith(`..${path.sep}`)&&!path.isAbsolute(back)?file:null;
}
function allowed(req,res){
  const origin=req.headers.origin;
  if(origin&&!/^http:\/\/(localhost|127\.0\.0\.1):8777$/.test(origin)){reply(res,403,"Forbidden");return false;}
  const now=Date.now();
  while(recent[0]<now-60000) recent.shift();
  if(recent.length>=60){reply(res,429,"Please wait before making more requests");return false;}
  recent.push(now);
  return true;
}
async function json(req,res,max){
  let raw="";
  for await(const chunk of req){
    raw+=chunk;
    if(raw.length>max){reply(res,413,"Request too large");return null;}
  }
  try{return JSON.parse(raw);}catch{reply(res,400,"Invalid JSON");return null;}
}
async function gemini(model,body){
  const request=()=>fetch(`${API}/${model}:generateContent`,{
    method:"POST",headers:{"Content-Type":"application/json","x-goog-api-key":key},body:JSON.stringify(body),
    signal:AbortSignal.timeout(8_000)
  });
  let response;
  try{response=await request();}catch{response=await request();}
  if(response.status===500) response=await request();
  if(!response.ok){
    const detail=await response.json().catch(()=>null);
    throw Object.assign(new Error(detail?.error?.message||"Gemini request failed"),{status:response.status});
  }
  return response.json();
}
async function chat(req,res){
  if(!allowed(req,res)) return;
  const body=await json(req,res,3_000_000);   // camera frames ride along in contents
  if(!body||!Array.isArray(body.contents)) return body&&reply(res,400,"Invalid chat request");
  try{reply(res,200,JSON.stringify(await gemini(CHAT_MODEL,body)),"application/json");}
  catch(error){console.error("Gemini chat failed:",error.status||error.name,error.message);reply(res,502,"Chat service unavailable");}
}

async function live(req,res){
  if(!allowed(req,res)) return;
  const body=await json(req,res,3_000_000);
  const message=body?.message?.trim(),instruction=body?.systemInstruction?.trim();
  if(!message||message.length>5000||!instruction||instruction.length>20_000)
    return body&&reply(res,400,"Invalid Live request");
  const history=Array.isArray(body.history)?body.history.slice(-12):[];
  const context=history.map(turn=>{
    const role=turn?.role==="model"?"Disha":"Friend";
    const text=Array.isArray(turn?.parts)?turn.parts.map(part=>part?.text||"").join("").trim():"";
    return text?`${role}: ${text}`:"";
  }).filter(Boolean).join("\n");
  let session,timer,settled=false,resolveTurn,rejectTurn;
  const turn=new Promise((resolve,reject)=>{resolveTurn=resolve;rejectTurn=reject;});
  turn.catch(()=>{}); // a setup error may arrive before execution reaches `await turn`
  const audio=[],transcript=[];
  const finish=(fn,value)=>{
    if(settled) return;
    settled=true; clearTimeout(timer); fn(value);
  };
  try{
    const config={
        responseModalities:[Modality.AUDIO],
        outputAudioTranscription:{},
        speechConfig:{voiceConfig:{prebuiltVoiceConfig:{voiceName:"Leda"}}},
        thinkingConfig:{thinkingLevel:"minimal"},
        // Native audio uses roughly 25 tokens/second. 160 could cut a normal
        // sentence mid-word; 512 leaves room to finish while the persona stays short.
        maxOutputTokens:512,
        systemInstruction:instruction+(context?`\n\nRECENT CONVERSATION:\n${context}`:"")
      };
    const connect=async()=>{
      let ready=false;
      const controller=new AbortController(),setupTimer=setTimeout(()=>controller.abort(),8_000);
      try{
        const candidate=await ai.live.connect({
          model:LIVE_MODEL,config:{...config,abortSignal:controller.signal},
          callbacks:{
            onmessage(event){
              const content=event.serverContent;
              for(const part of content?.modelTurn?.parts||[]){
                if(part.inlineData?.data) audio.push(Buffer.from(part.inlineData.data,"base64"));
              }
              if(content?.outputTranscription?.text) transcript.push(content.outputTranscription.text);
              if(content?.turnComplete) finish(resolveTurn);
            },
          onerror(event){if(ready) finish(rejectTurn,event.error||new Error(event.message||"Live API error"));},
          onclose(event){if(ready&&!settled) finish(rejectTurn,new Error(event.reason||"Live session closed"));},
          }
        });
        ready=true;
        return candidate;
      }finally{clearTimeout(setupTimer);}
    };
    try{session=await connect();}
    catch{await new Promise(resolve=>setTimeout(resolve,300));session=await connect();}
    timer=setTimeout(()=>finish(rejectTurn,new Error("Live response timed out")),30_000);
    res.on("close",()=>{if(!res.writableEnded){session?.close();finish(rejectTurn,new Error("Client closed"));}});
    if(typeof body.frame==="string"&&body.frame)
      session.sendRealtimeInput({video:{data:body.frame,mimeType:"image/jpeg"}});
    session.sendRealtimeInput({text:message});
    await turn;
    const text=transcript.join("").trim();
    if(!text||!audio.length) throw new Error("Live API returned an empty turn");
    reply(res,200,JSON.stringify({text,audio:Buffer.concat(audio).toString("base64")}),"application/json");
  }catch(error){
    console.error("Gemini Live failed:",error.status||error.name,error.message);
    if(!res.headersSent) reply(res,502,"Live service unavailable"); else res.end();
  }finally{
    clearTimeout(timer);
    session?.close();
  }
}

// Delivery style per emotional context — the same line for every mood was flat;
// this is advisory only, the actual words still come from the model's reply.
const MOOD_STYLE={
  happy:"Sound warmly upbeat and lightly playful, like sharing good news with a close friend.",
  surprised:"Sound genuinely surprised and animated, without exaggerating.",
  sad:"Speak gently, warmly and a little slower, like comforting a close friend.",
  fearful:"Speak slowly, softly and reassuringly, like calming a worried friend.",
  angry:"Speak steadily and calmly with warmth, like grounding a friend who's upset.",
  disgusted:"Use mild playful distaste, never harsh or exaggerated.",
  risk:"Speak slowly, gently and steadily, taking a friend's crisis seriously without sounding clinical or alarmed.",
  neutral:"Speak warmly, naturally, and casually like a close Indian friend."
};
let ttsPrimaryAfter=0;
async function tts(req,res){
  if(!allowed(req,res)) return;
  const body=await json(req,res,5000),text=body?.text?.trim();
  if(!text||text.length>1200) return body&&reply(res,400,"Text must be between 1 and 1200 characters");
  const style=MOOD_STYLE[body.mood]||MOOD_STYLE.neutral;
  const prompt=`Synthesize natural speech. ${style} Use a relaxed Indian conversational cadence and switch naturally between Hindi and English when the transcript does. Read only the transcript exactly as written.\n\nTRANSCRIPT:\n${text}`;
  const controller=new AbortController();
  res.on("close",()=>{ if(!res.writableEnded) controller.abort(); });
  try{
    const request=(model,stream)=>fetch(`${API}/${model}:${stream?"streamGenerateContent?alt=sse":"generateContent"}`,{
      method:"POST",
      headers:{"Content-Type":"application/json","x-goog-api-key":key},
      body:JSON.stringify({
        contents:[{parts:[{text:prompt}]}],
        generationConfig:{responseModalities:["AUDIO"],speechConfig:{voiceConfig:{prebuiltVoiceConfig:{voiceName:"Leda"}}}}
      }),
      signal:AbortSignal.any([controller.signal,AbortSignal.timeout(20_000)])
    });
    let model=Date.now()>=ttsPrimaryAfter?TTS_MODEL:TTS_FALLBACK_MODEL;
    let stream=model===TTS_MODEL,upstream;
    try{ upstream=await request(model,stream); }
    catch(error){
      if(controller.signal.aborted) throw error;
      upstream=await request(model,stream); // one transient network failure should not replace Leda
    }
    if(upstream.status===500) upstream=await request(model,stream); // documented rare TTS text-token failure
    if([429,500,503].includes(upstream.status)){
      ttsPrimaryAfter=Date.now()+300_000;
      model=TTS_FALLBACK_MODEL; stream=false;
      upstream=await request(model,stream);
    }
    if(!upstream.ok){
      const detail=await upstream.json().catch(()=>null);
      throw Object.assign(new Error(detail?.error?.message?.split("\n")[0]||"Gemini request failed"),{status:upstream.status});
    }
    res.writeHead(200,{"Content-Type":"application/octet-stream","X-Audio-Rate":"24000","Cache-Control":"no-store","X-Content-Type-Options":"nosniff"});
    if(!stream){
      const data=await upstream.json();
      const audio=data.candidates?.[0]?.content?.parts?.find(part=>part.inlineData?.data)?.inlineData?.data;
      if(!audio) throw new Error("No audio returned");
      return res.end(Buffer.from(audio,"base64"));
    }
    const decoder=new TextDecoder();
    let pending="",sent=false;
    const emit=event=>{
      const line=event.split(/\r?\n/).find(value=>value.startsWith("data:"));
      if(!line||line.slice(5).trim()==="[DONE]") return;
      const data=JSON.parse(line.slice(5));
      for(const part of data.candidates?.[0]?.content?.parts||[]){
        if(part.inlineData?.data){res.write(Buffer.from(part.inlineData.data,"base64"));sent=true;}
      }
    };
    for await(const chunk of upstream.body){
      pending+=decoder.decode(chunk,{stream:true});
      const events=pending.split(/\r?\n\r?\n/); pending=events.pop();
      events.forEach(emit);
    }
    pending+=decoder.decode();
    if(pending.trim()) emit(pending);
    if(!sent) throw new Error("No audio returned");
    res.end();
  }catch(error){
    if(error.name!=="AbortError") console.error("Gemini TTS failed:",error.status||error.name,error.message);
    if(!res.headersSent) reply(res,502,`Voice service unavailable: ${error.status||error.name}`); else res.end();
  }
}

// Speech-to-text for browsers without SpeechRecognition (Firefox, Safari): the
// client records a short clip and posts it here instead of streaming words live.
async function stt(req,res){
  if(!allowed(req,res)) return;
  const body=await json(req,res,8_000_000);
  const audio=body?.audio,mimeType=body?.mimeType;
  if(!audio||typeof audio!=="string"||!mimeType) return reply(res,400,"Missing audio");
  try{
    const data=await gemini(CHAT_MODEL,{
      contents:[{role:"user",parts:[
        {text:"Transcribe verbatim only what is spoken aloud in this clip. Output ONLY the transcript — no quotes, no preamble, no commentary. If nothing intelligible was said, output exactly: (silence)"},
        {inlineData:{mimeType,data:audio}}
      ]}],
      generationConfig:{maxOutputTokens:300,thinkingConfig:{thinkingLevel:"low"}}
    });
    const text=(data.candidates?.[0]?.content?.parts||[]).map(p=>p.text||"").join("").trim();
    reply(res,200,JSON.stringify({transcript:/^\(silence\)$/i.test(text)?"":text}),"application/json");
  }catch(error){console.error("Gemini STT failed:",error.status||error.name);reply(res,502,"Voice transcription unavailable");}
}

http.createServer(async(req,res)=>{
  const url=new URL(req.url,"http://localhost");
  if(url.pathname==="/api/live") return req.method==="POST"?live(req,res):reply(res,405,"Method not allowed");
  if(url.pathname==="/api/chat") return req.method==="POST"?chat(req,res):reply(res,405,"Method not allowed");
  if(url.pathname==="/api/tts") return req.method==="POST"?tts(req,res):reply(res,405,"Method not allowed");
  if(url.pathname==="/api/stt") return req.method==="POST"?stt(req,res):reply(res,405,"Method not allowed");
  if(req.method!=="GET"&&req.method!=="HEAD") return reply(res,405,"Method not allowed");

  let pathname;
  try{pathname=decodeURIComponent(url.pathname);}catch{return reply(res,400,"Invalid URL");}
  let file;
  if(pathname==="/"||pathname==="/disha.html") file=localFile(DIST,"index.html");
  else if(pathname.startsWith("/assets/")) file=localFile(DIST,pathname.slice(1));
  else if(pathname.startsWith("/hiyori_free/")) file=localFile(MODEL_DIR,pathname.slice("/hiyori_free/".length));
  if(!file) return reply(res,404,"Not found");
  fs.stat(file,(error,stat)=>{
    if(error||!stat.isFile()) return reply(res,404,"Not found");
    res.writeHead(200,{"Content-Type":mime[path.extname(file).toLowerCase()]||"application/octet-stream","Cache-Control":"no-cache","X-Content-Type-Options":"nosniff"});
    if(req.method==="HEAD") return res.end();
    fs.createReadStream(file).pipe(res);
  });
}).listen(PORT,"127.0.0.1",()=>{
  console.log(`Disha is running at http://localhost:${PORT}/`);
  fetch(`${API}?pageSize=1`,{headers:{"x-goog-api-key":key},signal:AbortSignal.timeout(10_000)}).catch(()=>{});
});
