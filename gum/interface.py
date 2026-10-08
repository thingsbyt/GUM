"""Dependency-free local GUM Studio: interaction, evidence, memory, and replays."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
from urllib.parse import unquote, urlparse
import webbrowser

from .harness import GUMHarness
from .teaching_lab import TeachingLab


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GUM Studio</title>
<style>
:root{--ink:#edf4f7;--muted:#91a7af;--line:#274149;--panel:#0e2026;--panel2:#132a31;--bg:#071317;--mint:#6ee7b7;--gold:#f6c667;--blue:#72c7ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 10% 0,#163139 0,transparent 28%),var(--bg);color:var(--ink);font:15px/1.55 Inter,Segoe UI,system-ui,sans-serif}
header{padding:26px clamp(20px,4vw,58px) 18px;border-bottom:1px solid var(--line);display:flex;gap:22px;align-items:end;justify-content:space-between}.eyebrow{color:var(--mint);letter-spacing:.15em;text-transform:uppercase;font-size:11px;font-weight:800}h1{margin:3px 0 0;font-size:clamp(28px,4vw,48px);line-height:1}.tagline{color:var(--muted);max-width:560px}.badges{display:flex;gap:8px;flex-wrap:wrap}.badge{border:1px solid var(--line);border-radius:999px;padding:5px 9px;color:var(--muted);font-size:11px}.badge.good{color:var(--mint);border-color:#2e6655}
nav{padding:13px clamp(20px,4vw,58px);display:flex;gap:8px;border-bottom:1px solid var(--line);position:sticky;top:0;background:#071317e8;backdrop-filter:blur(14px);z-index:4}button{font:inherit;color:var(--ink);background:#17313a;border:1px solid #31515b;border-radius:9px;padding:9px 13px;cursor:pointer}button:hover{border-color:var(--mint)}button.primary{background:var(--mint);color:#052018;border-color:var(--mint);font-weight:800}.tab.active{background:#21434c;color:var(--mint)}
main{padding:24px clamp(20px,4vw,58px) 52px;max-width:1480px;margin:auto}.view{display:none}.view.active{display:block}.hero-grid,.two{display:grid;grid-template-columns:1.25fr .75fr;gap:18px}.card{background:linear-gradient(145deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:0 18px 50px #0003}.card h2,.card h3{margin:0 0 8px}.muted{color:var(--muted)}.metric-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}.metric{padding:15px;border-left:3px solid var(--mint);background:#0e2026;border-radius:5px 12px 12px 5px}.metric strong{display:block;font-size:23px}.metric span{color:var(--muted);font-size:12px}.chat{height:330px;overflow:auto;padding:8px;background:#09171b;border:1px solid var(--line);border-radius:11px}.bubble{max-width:86%;padding:10px 12px;border-radius:11px;margin:8px 0;white-space:pre-wrap}.bubble.you{margin-left:auto;background:#24444d}.bubble.gum{background:#17352e;border-left:3px solid var(--mint)}textarea,input{width:100%;font:inherit;color:var(--ink);background:#08171b;border:1px solid #38535b;border-radius:9px;padding:10px}textarea{height:72px;resize:vertical}.composer{display:grid;grid-template-columns:1fr auto;gap:8px;margin-top:9px}.quick{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0}.quick button{font-size:12px;padding:6px 9px}.world{border-top:1px solid var(--line);padding:11px 0}.world:first-child{border:0}.actions{display:flex;gap:8px;flex-wrap:wrap}.notice{padding:11px 13px;border:1px solid #6b5626;background:#31260e;border-radius:10px;color:#f7d998}.claims{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}.claim .status{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--mint)}.claim.mixed .status{color:var(--gold)}.claim h3{font-size:19px}.claim ul{color:var(--muted);padding-left:18px}.scope{font-size:12px;color:#bed0d6;border-top:1px solid var(--line);padding-top:9px}.replays{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:18px}.replay img{width:100%;aspect-ratio:16/10;object-fit:contain;background:#061014;border-radius:10px}.brain-grid{display:grid;grid-template-columns:.8fr 1.2fr;gap:18px}pre{white-space:pre-wrap;word-break:break-word;color:#bcd0d7;background:#08171b;border:1px solid var(--line);border-radius:11px;padding:14px;max-height:620px;overflow:auto}.docs{display:grid;grid-template-columns:270px 1fr;gap:18px}.doc-list button{width:100%;text-align:left;margin:0 0 7px}.doc-body{min-height:650px}.doc-body pre{max-height:none;white-space:pre-wrap}.spinner{display:none;color:var(--gold)}.spinner.on{display:inline}.footer{margin-top:30px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:16px}
.teaching-grid{display:grid;grid-template-columns:minmax(320px,.9fr) minmax(340px,1.1fr);gap:18px}.scene{width:min(100%,480px);image-rendering:pixelated;border:1px solid var(--line);border-radius:14px;cursor:crosshair;background:#080b12}.scene.selected{outline:3px solid var(--gold)}.steps{counter-reset:lesson}.steps p{padding-left:34px;position:relative}.steps p:before{counter-increment:lesson;content:counter(lesson);position:absolute;left:0;top:0;background:var(--mint);color:#052018;font-weight:900;border-radius:50%;width:24px;height:24px;text-align:center}.lexicon{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0}.word{border:1px solid #2e6655;border-radius:999px;padding:4px 8px;color:var(--mint);font-size:12px}
@media(max-width:900px){.hero-grid,.two,.brain-grid,.docs,.teaching-grid{grid-template-columns:1fr}.metric-row{grid-template-columns:repeat(2,1fr)}.claims,.replays{grid-template-columns:1fr}header{align-items:start;flex-direction:column}.composer{grid-template-columns:1fr}}
</style></head>
<body><header><div><div class="eyebrow">Growing Understanding Machine</div><h1>GUM Studio</h1><div class="tagline">Play with a persistent learner, teach visual words, test what changed, replay prior trials, and trace public claims back to evidence.</div></div><div class="badges"><span class="badge good">LOCAL ONLY</span><span class="badge">NO CLOUD REQUIRED</span><span class="badge">HASH-LINKED LOGS</span></div></header>
<nav><button class="tab active" data-view="live">Live lab</button><button class="tab" data-view="teach">Teaching lab</button><button class="tab" data-view="evidence">Evidence</button><button class="tab" data-view="brain">Brain & lineage</button><button class="tab" data-view="docs">Documentation</button></nav>
<main>
<section id="live" class="view active">
 <div class="metric-row"><div class="metric"><strong id="states">—</strong><span>learned visual states</span></div><div class="metric"><strong id="worldCount">—</strong><span>worlds retained</span></div><div class="metric"><strong id="records">—</strong><span>verified lineage records</span></div><div class="metric"><strong id="activeName">none</strong><span>active world</span></div></div>
 <div class="hero-grid"><div class="card"><h2>Speak to GUM</h2><p class="muted">This is the bounded grounded interface—not an LLM pretending to be the learner.</p><div id="chat" class="chat"></div><div class="quick"><button data-say="what do you know?">What do you know?</button><button data-say="create a world">Create a world</button><button data-say="create a harder world">Create a harder world</button></div><div class="composer"><textarea id="msg" placeholder="Ask for status, create a world, or use a grounded command…"></textarea><button class="primary" id="send">Send</button></div></div>
 <div class="card"><h2>Explicit learning controls</h2><p class="notice">Nothing trains automatically. A run begins only when you press the button below.</p><label>Training episodes <input id="episodes" type="number" min="20" max="600" value="120"></label><div class="actions" style="margin-top:10px"><button class="primary" id="learn">Run bounded learning</button><button id="refresh">Refresh</button></div><p id="busy" class="spinner">Learning now… the interface will update when complete.</p><pre id="runResult">Create or load a one-agent grid world first.</pre></div></div>
 <div class="two" style="margin-top:18px"><div class="card"><h2>World library</h2><div id="worlds"></div></div><div class="card"><h2>What “drop in a world” means</h2><p>A world is a folder containing a public contract and a private audit genome. Its adapter exposes pixels, anonymous actions, reward, and termination. Hidden coordinates and control meanings remain outside the mind.</p><p class="muted">Trusted adapters are registered in code; a dropped folder cannot execute arbitrary code.</p></div></div>
</section>
<section id="teach" class="view">
 <div class="teaching-grid">
  <div class="card"><h2>Teach by showing</h2><p class="muted">Type what the object means, then click it. GUM receives the sentence, the pointing location, and pixels—not a hidden color/shape answer table.</p><img id="teachScene" class="scene" src="/api/teaching/image" alt="Three colored shapes used for a pointing lesson"><label>Your teaching phrase <input id="teachPhrase" value="please choose the red circle"></label><div class="actions" style="margin-top:10px"><button id="newLesson">New lesson scene</button><button id="learnWords" class="primary">Learn meanings</button></div><p id="pointHint" class="muted">Click an object to store this phrase as a pointing example. Reliable words need repeated, varied examples.</p><div class="actions"><button id="starter">Load starter lessons</button><button id="resetTeaching">Reset teaching memory</button></div></div>
  <div class="card"><h2>Test what it understood</h2><div class="steps"><p>Load the starter lessons—or teach repeated examples yourself.</p><p>Open the ambiguity test and ask it to approach the red object.</p><p>When it asks which one, answer “the square” or “the circle.”</p></div><div class="actions"><button id="ambiguity">Open ambiguity test</button><button id="auditTeaching">Run fresh 45-trial audit</button></div><div class="composer"><input id="teachMessage" value="approach the red object"><button id="sendTeaching" class="primary">Send</button></div><div id="teachChat" class="chat" style="height:210px;margin-top:10px"></div><h3>Learned vocabulary</h3><div id="lexicon" class="lexicon"></div><pre id="teachState"></pre></div>
 </div>
</section>
<section id="evidence" class="view"><div class="notice" id="globalWarning"></div><h2>Claim ledger</h2><p class="muted">Green means a bounded internal test passed. Gold marks a mixed result. Neither means independent scientific replication.</p><div id="claims" class="claims"></div><h2 style="margin-top:28px">Replay theater</h2><div id="replays" class="replays"></div></section>
<section id="brain" class="view"><div class="brain-grid"><div class="card"><h2>How it grows</h2><p>GUM adds learned state, causal mappings, strategies, and experience records. Reopening the same workspace reloads those memories. The ledger is append-only and hash-linked, so edits or missing records are detectable.</p><p>Different specialists still use different learning machinery. The harness unifies their lifecycle and evidence boundary; it does not magically turn them into one universal neural brain.</p><h3>Legend</h3><p><span style="color:var(--mint)">Persistent:</span> survives restart.</p><p><span style="color:var(--blue)">Grounded:</span> learned through observed consequences.</p><p><span style="color:var(--gold)">Engineered:</span> supplied algorithm or tool.</p></div><div class="card"><h2>Current machine state</h2><pre id="state"></pre></div></div></section>
<section id="docs" class="view"><div class="docs"><div class="card doc-list"><h2>Research package</h2><div id="docList"></div></div><div class="card doc-body"><pre id="docBody">Choose a document.</pre></div></div></section>
<div class="footer">GUM Studio is an evidence viewer and bounded experimental harness. It is not a claim of consciousness, unrestricted general intelligence, or medical capability.</div>
</main>
<script>
const $=id=>document.getElementById(id); let index={};
function bubble(who,text){const d=document.createElement('div');d.className='bubble '+who;d.textContent=(who==='you'?'You: ':'GUM: ')+text;$('chat').appendChild(d);$('chat').scrollTop=$('chat').scrollHeight}
async function api(url,opt){const r=await fetch(url,opt);const j=await r.json();if(!r.ok)throw Error(j.error||r.statusText);return j}
async function refresh(){const s=await api('/api/state');$('states').textContent=s.mind.learned_states;$('worldCount').textContent=s.mind.worlds_retained;$('records').textContent=s.evolution.records;$('activeName').textContent=s.active_world?s.active_world.world_id:'none';$('state').textContent=JSON.stringify(s,null,2);const w=await api('/api/worlds');$('worlds').replaceChildren(...w.worlds.map(x=>{const d=document.createElement('div');d.className='world';const t=document.createElement('strong');t.textContent=x.world_id;const p=document.createElement('div');p.className='muted';p.textContent=`${x.family} · ${x.agents} agent${x.agents===1?'':'s'} · ${x.action_count} anonymous actions`;const b=document.createElement('button');b.textContent=x.active?'Active':'Load';b.disabled=x.active;b.onclick=()=>loadWorld(x.world_id);d.append(t,p,b);return d}))}
async function send(text){text=(text||$('msg').value).trim();if(!text)return;bubble('you',text);$('msg').value='';try{const j=await api('/api/message',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({text})});bubble('gum',j.response);await refresh()}catch(e){bubble('gum','Error: '+e.message)}}
async function loadWorld(id){try{await api('/api/world',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({world_id:id})});await refresh()}catch(e){alert(e.message)}}
async function learn(){const episodes=Number($('episodes').value);$('busy').classList.add('on');$('learn').disabled=true;try{const j=await api('/api/practice',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({episodes})});$('runResult').textContent=JSON.stringify({learned:j.learned,before:j.before,after:j.after,trace:j.transition_trace},null,2);bubble('gum',`Run complete. Success changed from ${(100*j.before.success_rate).toFixed(1)}% to ${(100*j.after.success_rate).toFixed(1)}%.`);await refresh()}catch(e){$('runResult').textContent='Run stopped: '+e.message}finally{$('busy').classList.remove('on');$('learn').disabled=false}}
function teachingBubble(who,text){const d=document.createElement('div');d.className='bubble '+who;d.textContent=(who==='you'?'You: ':'GUM: ')+text;$('teachChat').appendChild(d);$('teachChat').scrollTop=$('teachChat').scrollHeight}
function renderTeaching(s){$('teachState').textContent=JSON.stringify({examples:s.examples,learned_word_count:s.learned_word_count,scene_mode:s.scene_mode,pending_clarification:s.pending_clarification,last_turn:s.last_turn},null,2);$('lexicon').replaceChildren(...Object.keys(s.learned_words||{}).map(word=>{const d=document.createElement('span');d.className='word';d.textContent=word;return d}));if(s.message)teachingBubble('gum',s.message);$('teachScene').src='/api/teaching/image?scene='+s.scene_number+'&t='+Date.now()}
async function teachingAction(action,value={}){try{const s=await api('/api/teaching/'+action,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(value)});renderTeaching(s);return s}catch(e){teachingBubble('gum','Error: '+e.message)}}
async function refreshTeaching(){try{renderTeaching(await api('/api/teaching/state'))}catch(e){teachingBubble('gum','Startup error: '+e.message)}}
async function sendTeaching(){const text=$('teachMessage').value.trim();if(!text)return;teachingBubble('you',text);const state=await api('/api/teaching/state');await teachingAction(state.pending_clarification?'answer':'begin',{text})}
function renderIndex(){$('globalWarning').textContent=index.global_warning||'';$('claims').replaceChildren(...(index.claims||[]).map(c=>{const d=document.createElement('article');d.className='card claim '+(c.status==='mixed-result'?'mixed':'');const st=document.createElement('div');st.className='status';st.textContent=c.status.replaceAll('-',' ');const h=document.createElement('h3');h.textContent=c.title;const lead=document.createElement('strong');lead.textContent=c.headline;const ul=document.createElement('ul');c.metrics.forEach(m=>{const li=document.createElement('li');li.textContent=m;ul.appendChild(li)});const scope=document.createElement('div');scope.className='scope';scope.textContent=c.scope;d.append(st,h,lead,ul,scope);return d}));$('replays').replaceChildren(...(index.replays||[]).map(r=>{const d=document.createElement('article');d.className='card replay';const h=document.createElement('h3');h.textContent=r.title;const img=document.createElement('img');img.loading='lazy';img.src='/api/media/'+encodeURIComponent(r.id);img.alt=r.title;d.append(h,img);return d}));$('docList').replaceChildren(...(index.documents||[]).map(doc=>{const b=document.createElement('button');b.textContent=doc.title;b.onclick=()=>loadDoc(doc.id);return b}))}
async function loadDoc(id){const r=await fetch('/api/document/'+encodeURIComponent(id));$('docBody').textContent=await r.text()}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab,.view').forEach(x=>x.classList.remove('active'));b.classList.add('active');$(b.dataset.view).classList.add('active')});document.querySelectorAll('[data-say]').forEach(b=>b.onclick=()=>send(b.dataset.say));$('send').onclick=()=>send();$('msg').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}};$('learn').onclick=learn;$('refresh').onclick=refresh;
$('starter').onclick=()=>teachingAction('starter');$('resetTeaching').onclick=()=>teachingAction('reset');$('newLesson').onclick=()=>teachingAction('scene',{mode:'lesson'});$('ambiguity').onclick=()=>teachingAction('scene',{mode:'ambiguity'});$('learnWords').onclick=()=>teachingAction('learn');$('sendTeaching').onclick=sendTeaching;$('teachMessage').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();sendTeaching()}};$('auditTeaching').onclick=async()=>{teachingBubble('gum','Running the fresh-learner audit…');const result=await teachingAction('audit');if(result&&result.audit){const a=result.audit;teachingBubble('gum',`Audit ${a.passed?'passed':'failed'}: ${a.results.overall_successes}/${a.results.overall_trials} completed successfully.`)}};$('teachScene').onclick=e=>{const r=e.currentTarget.getBoundingClientRect();const x=Math.round((e.clientX-r.left)*e.currentTarget.naturalWidth/r.width);const y=Math.round((e.clientY-r.top)*e.currentTarget.naturalHeight/r.height);teachingAction('demonstrate',{phrase:$('teachPhrase').value,x,y})};
(async()=>{try{index=await api('/api/evidence');renderIndex();await refresh();await refreshTeaching();bubble('gum','Studio ready. Create a world, teach visual words, inspect evidence, or start an explicit bounded learning run.')}catch(e){bubble('gum','Startup error: '+e.message)}})();
</script></body></html>'''


class StudioFiles:
    def __init__(self, release_root: Path | None):
        self.root = Path(release_root).resolve() if release_root else None
        self.index = self._load_index()

    def _load_index(self) -> dict:
        if self.root and (self.root / "EVIDENCE_INDEX.json").exists():
            return json.loads((self.root / "EVIDENCE_INDEX.json").read_text(encoding="utf-8"))
        return {"format": "gum-evidence-index-v1", "claims": [], "replays": [], "documents": [],
                "global_warning": "No research package was attached. Live harness controls remain available."}

    def _declared(self, collection: str, item_id: str) -> Path | None:
        if not self.root:
            return None
        for item in self.index.get(collection, []):
            if item.get("id") == item_id:
                path = (self.root / item["file"]).resolve()
                try: path.relative_to(self.root)
                except ValueError: return None
                return path if path.is_file() else None
        return None

    def media(self, item_id: str) -> Path | None: return self._declared("replays", item_id)
    def document(self, item_id: str) -> Path | None: return self._declared("documents", item_id)


def serve(workspace: Path, host="127.0.0.1", port=8765, release_root: Path | None = None,
          open_browser: bool = False):
    harness = GUMHarness(workspace); studio = StudioFiles(release_root); run_lock = threading.Lock()
    teaching = TeachingLab(Path(workspace) / "teaching-lab"); teaching_lock = threading.Lock()

    def worlds():
        rows = []; active = None if harness.active_world is None else harness.active_world.resolve()
        for folder in sorted(harness.world_root.iterdir()) if harness.world_root.exists() else []:
            public, private = folder / "world.json", folder / "genome.private.json"
            if not public.is_file() or not private.is_file(): continue
            try:
                row = json.loads(public.read_text(encoding="utf-8")); row["active"] = active == folder.resolve(); rows.append(row)
            except (OSError, json.JSONDecodeError): continue
        return rows

    class Handler(BaseHTTPRequestHandler):
        def _json(self, value, status=200):
            data = json.dumps(value).encode("utf-8"); self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def _body(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000: raise ValueError("request too large")
            return json.loads(self.rfile.read(length) or b"{}")
        def _file(self, path: Path, content_type: str | None = None):
            data = path.read_bytes(); self.send_response(200)
            self.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            route = urlparse(self.path).path
            try:
                if route == "/":
                    data = HTML.encode("utf-8"); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
                elif route == "/api/state": self._json(harness.status())
                elif route == "/api/worlds": self._json({"worlds": worlds()})
                elif route == "/api/evidence": self._json(studio.index)
                elif route == "/api/teaching/state": self._json(teaching.state())
                elif route == "/api/teaching/image": self._file(teaching.image_path, "image/png")
                elif route.startswith("/api/media/"):
                    path = studio.media(unquote(route.removeprefix("/api/media/")))
                    self._json({"error": "unknown replay"}, 404) if path is None else self._file(path)
                elif route.startswith("/api/document/"):
                    path = studio.document(unquote(route.removeprefix("/api/document/")))
                    self._json({"error": "unknown document"}, 404) if path is None else self._file(path, "text/plain; charset=utf-8")
                else: self._json({"error": "not found"}, 404)
            except Exception as error: self._json({"error": str(error)}, 500)
        def do_POST(self):
            route = urlparse(self.path).path
            try:
                value = self._body()
                if route == "/api/message":
                    text = str(value.get("text", "")).strip()
                    response = ("Learning is an explicit operation in Studio. Choose an episode budget and press Run bounded learning."
                                if "practice" in text.lower() or "learn this world" in text.lower() else harness.communicate(text))
                    self._json({"response": response, "state": harness.status()})
                elif route == "/api/world":
                    requested = str(value.get("world_id", "")); options = {row["world_id"]: harness.world_root / row["world_id"] for row in worlds()}
                    if requested not in options: self._json({"error": "world is not in the local library"}, 404); return
                    self._json({"world": harness.load_world(options[requested])})
                elif route == "/api/practice":
                    if harness.active_world is None: self._json({"error": "create or load a world first"}, 409); return
                    public = json.loads((harness.active_world / "world.json").read_text(encoding="utf-8"))
                    if int(public.get("agents", 1)) != 1:
                        self._json({"error": "The bounded learning button is for one-agent grid worlds; cooperative replays are in Evidence."}, 409); return
                    episodes = max(20, min(600, int(value.get("episodes", 120))))
                    if not run_lock.acquire(blocking=False): self._json({"error": "a learning run is already active"}, 409); return
                    try: self._json(harness.practice(training_episodes=episodes, evaluation_episodes=max(10, min(60, episodes // 3))))
                    finally: run_lock.release()
                elif route.startswith("/api/teaching/"):
                    action = route.removeprefix("/api/teaching/")
                    if not teaching_lock.acquire(blocking=False): self._json({"error": "a teaching operation is already active"}, 409); return
                    try:
                        if action == "starter": result = teaching.starter_curriculum()
                        elif action == "reset": result = teaching.reset()
                        elif action == "scene": result = teaching.new_scene(str(value.get("mode", "lesson")))
                        elif action == "demonstrate": result = teaching.demonstrate(str(value.get("phrase", "")), int(value.get("x", -1)), int(value.get("y", -1)))
                        elif action == "learn": result = teaching.learn()
                        elif action == "begin": result = teaching.begin(str(value.get("text", "")))
                        elif action == "answer": result = teaching.answer(str(value.get("text", "")))
                        elif action == "audit":
                            audit = teaching.audit(); result = teaching.state(audit["note"]); result["audit"] = audit
                        else: self._json({"error": "unknown teaching operation"}, 404); return
                        self._json(result)
                    finally: teaching_lock.release()
                else: self._json({"error": "not found"}, 404)
            except (ValueError, TypeError, json.JSONDecodeError) as error: self._json({"error": str(error)}, 400)
            except Exception as error: self._json({"error": str(error)}, 500)
        def log_message(self, format, *args): pass

    server = ThreadingHTTPServer((host, int(port)), Handler)
    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    print(f"GUM Studio: {url}"); server.serve_forever()
