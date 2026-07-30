import {useEffect} from "react";

export default function App(){
  useEffect(()=>{
    void import("./disha").catch(error=>{
      const title=document.getElementById("fatalTitle");
      const body=document.getElementById("fatalBody");
      document.getElementById("fatal")?.classList.add("show");
      if(title) title.textContent="Couldn't start Disha";
      if(body) body.textContent=error instanceof Error?error.message:String(error);
    });
  },[]);

  return <>
    <div className="room">
      <div className="aura a"></div><div className="aura b"></div><div className="aura c"></div>
    </div>
    <div className="grain"></div>
    <div className="vignette"></div>

    <canvas id="canvas"></canvas>

    <div className="eye" id="eye" role="button" tabIndex={0} aria-label="Toggle camera so Disha can see you">
      <video id="cam" autoPlay muted playsInline></video>
      <span className="off">let me<br />see you</span>
    </div>
    <button className="ambient" id="ambientBtn" aria-label="Let her glance on her own sometimes, without you saying anything" title="Let her glance on her own sometimes">
      <svg viewBox="0 0 24 24"><path d="M12 5c-5 0-9 4-10 7 1 3 5 7 10 7s9-4 10-7c-1-3-5-7-10-7zm0 11a4 4 0 1 1 0-8 4 4 0 0 1 0 8z" /></svg>
    </button>

    <button className="memory" id="memoryBtn" aria-label="Remember this conversation next time" title="Remember this conversation next time">
      <svg viewBox="0 0 24 24"><path d="M13 3a9 9 0 1 0 8.94 10h-2.02A7 7 0 1 1 13 5v4l5-4.5L13 0v3z" /></svg>
    </button>

    <div className="caption" id="caption"></div>

    <div className="fatal" id="fatal">
      <div>
        <h2 id="fatalTitle"></h2>
        <p id="fatalBody"></p>
      </div>
    </div>

    <div className="bar">
      <input id="input" placeholder="talk to me…" autoComplete="off" aria-label="Message Disha" />
      <button className="mic" id="mic" aria-label="Toggle voice input">
        <svg viewBox="0 0 24 24"><path d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2z" /></svg>
      </button>
    </div>
  </>;
}
