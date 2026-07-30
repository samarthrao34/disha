import "dotenv/config";
import {GoogleGenAI,Modality} from "@google/genai";

const key=process.env.GEMINI_API_KEY;
if(!key) throw new Error("GEMINI_API_KEY is missing from .env");

const started=performance.now();
let firstAudio=0,audioBytes=0,transcript="",finish,fail;
const done=new Promise((resolve,reject)=>{finish=resolve;fail=reject;});
const timeout=setTimeout(()=>fail(new Error("Live response timed out")),30_000);
const ai=new GoogleGenAI({apiKey:key});
const phrase="Hello yaar, this is Disha speaking clearly and naturally. I am testing every word from the beginning, through the calm middle of this sentence, and all the way to the very end, where I will finish by saying the special words, blue lantern complete.";
const session=await ai.live.connect({
  model:"gemini-3.1-flash-live-preview",
  config:{
    responseModalities:[Modality.AUDIO],
    outputAudioTranscription:{},
    speechConfig:{voiceConfig:{prebuiltVoiceConfig:{voiceName:"Leda"}}},
    thinkingConfig:{thinkingLevel:"minimal"},
    maxOutputTokens:512,
    systemInstruction:"Read the user's sentence exactly. Do not add, remove, or change words."
  },
  callbacks:{
    onmessage(message){
      const content=message.serverContent;
      for(const part of content?.modelTurn?.parts||[]){
        if(part.inlineData?.data){
          if(!firstAudio) firstAudio=performance.now()-started;
          audioBytes+=Buffer.from(part.inlineData.data,"base64").length;
        }
      }
      if(content?.outputTranscription?.text) transcript+=content.outputTranscription.text;
      if(content?.turnComplete) finish();
    },
    onerror(event){fail(event.error||new Error(event.message||"Live API error"));},
    onclose(event){if(!audioBytes) fail(new Error(event.reason||"Live session closed"));},
  }
});

session.sendRealtimeInput({text:phrase});
try{
  await done;
  console.log(`LIVE_MODEL=gemini-3.1-flash-live-preview firstAudioMs=${Math.round(firstAudio)} audioBytes=${audioBytes}`);
  console.log(`transcript=${transcript.trim()}`);
  if(!audioBytes||!/blue lantern complete/i.test(transcript)) process.exitCode=1;
}finally{
  clearTimeout(timeout);
  session.close();
}
