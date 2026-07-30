import "dotenv/config";

const API="https://generativelanguage.googleapis.com/v1beta/models";
const key=process.env.GEMINI_API_KEY;
if(!key) throw new Error("GEMINI_API_KEY is missing from .env");

async function request(model,body){
  const response=await fetch(`${API}/${model}:generateContent`,{
    method:"POST",
    headers:{"Content-Type":"application/json","x-goog-api-key":key},
    body:JSON.stringify(body),
    signal:AbortSignal.timeout(30_000)
  });
  const data=await response.json().catch(()=>({}));
  if(!response.ok) throw Object.assign(new Error(data.error?.message?.split("\n")[0]||response.statusText),{status:response.status});
  return data;
}

async function choose(kind,models,body,valid){
  for(const model of models){
    try{
      if(valid(await request(model,body))){
        console.log(`${kind}_MODEL=${model}`);
        return model;
      }
      console.log(`${model}: empty ${kind.toLowerCase()} response`);
    }catch(error){
      console.log(`${model}: ${error.status||error.name} ${error.message}`);
    }
  }
  throw new Error(`No working ${kind.toLowerCase()} model found`);
}

await choose("CHAT",["gemini-3.6-flash","gemini-3.5-flash","gemini-3.5-flash-lite"],{
  contents:[{parts:[{text:"Reply with exactly: OK"}]}],
  generationConfig:{maxOutputTokens:20}
},data=>data.candidates?.[0]?.content?.parts?.some(part=>part.text));

await choose("TTS",["gemini-3.1-flash-tts-preview","gemini-2.5-flash-preview-tts","gemini-2.5-pro-preview-tts"],{
  contents:[{parts:[{text:"Synthesize natural speech. TRANSCRIPT: Hi."}]}],
  generationConfig:{responseModalities:["AUDIO"],speechConfig:{voiceConfig:{prebuiltVoiceConfig:{voiceName:"Leda"}}}}
},data=>data.candidates?.[0]?.content?.parts?.some(part=>part.inlineData?.data));
