"""Read-only, loopback live view and complete episode playback for a study."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import secrets
import threading
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from gum.lineage import HashLedger, file_sha256
from .escape_chamber import RewardTable, make_escape_chamber


HTML = '''<!doctype html><html><head><meta charset="utf-8"><title>GUM / Independent PPO study</title>
<style>body{background:#0b1118;color:#e9f0f5;font:17px Segoe UI;margin:25px}small{color:#9db1bf}img{max-width:100%;background:#070b10}#eyes{display:flex}#eyes img{width:25%}button,select{padding:10px;background:#23364b;color:white;border:1px solid #567}pre{white-space:pre-wrap;font-size:14px}</style></head>
<body><h1>Cooperative learning: GUM / Independent PPO</h1><div id="label">Connecting…</div>
<small>Authoritative records · all episodes retained · sharing disabled</small><br><img id="room">
<div id="eyes"></div><pre id="state"></pre><h2>Full archive playback</h2>
<button onclick="list()">Load every recorded episode</button><select id="episodes"></select>
<button onclick="startReplay()">Play selected episode</button><button onclick="live=true">Return to live view</button>
<script>const token=__TOKEN__;let live=true,replayTick=0,replayLength=0,busy=false,lastFrame='';const q='?token='+token;
async function fetchJSON(path){const r=await fetch(path);if(!r.ok)throw Error(await r.text());return r.json()}
for(let i=0;i<4;i++)document.querySelector('#eyes').innerHTML+='<img id="eye'+i+'">';
async function refresh(){if(busy)return;busy=true;try{let s;if(live){s=await fetchJSON('/state'+q)}else{if(replayTick<replayLength){s=await fetchJSON('/replay'+q+'&tick='+replayTick);replayTick++}else{return}}
document.querySelector('#label').textContent=s.method+' · '+s.phase+' · team '+s.team+' / seed '+s.team_seed+' · episode '+s.episode+' · tick '+s.tick;
document.querySelector('#state').textContent=JSON.stringify(s,null,2);const serial=s.frame_serial||s.episode+'/'+s.tick;let extra='&serial='+encodeURIComponent(serial);if(lastFrame!==serial){lastFrame=serial;document.querySelector('#room').src='/frame'+q+extra;
for(let i=0;i<4;i++)document.querySelector('#eye'+i).src='/frame'+q+'&member='+i+extra;}
}catch(e){document.querySelector('#label').textContent=e.message}finally{busy=false}}
async function list(){const rows=await fetchJSON('/episodes'+q);let el=document.querySelector('#episodes');el.replaceChildren();for(const row of rows){let o=document.createElement('option');o.value=row.key;o.textContent=row.label;el.appendChild(o)}}
async function startReplay(){const r=await fetchJSON('/replay'+q+'&key='+encodeURIComponent(document.querySelector('#episodes').value));live=false;replayTick=0;replayLength=r.length}
setInterval(refresh,250);refresh();</script></body></html>'''


class StudyViewer:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.lock = threading.Lock()
        self.state = {"method":"waiting","phase":"idle","team":0,"team_seed":0,"episode":0,"tick":0}
        self.frames = None
        self.replay_frames = None
        self.replaying = False
        self.last_displayed = 0
        self.replay_world = None
        self.replay_arrays = None
        self.frame_serial = 0

    def update(self, world, step, metadata):
        with self.lock:
            self.state = {**metadata,"tick":world.steps,"episode_id":step["episode_id"],
                          "state_sha256":step["state_sha256"],"escapes":len(world.escape_order)}
            if world.steps % 8 == 0 or world.steps == 1 or world._done:
                self.frames = [world.spectator_frame(), *world.observations()]
                self.frame_serial += 1
                self.frame_tick = world.steps
            self.state["frame_serial"] = self.frame_serial
            self.state["frame_tick"] = getattr(self,"frame_tick",0)

    def episodes(self):
        rows = []
        for path in sorted(self.root.rglob("EPISODE_LEDGER.jsonl")):
            if not HashLedger(path).verify()["valid"]: raise ValueError("invalid ledger")
            cumulative = {"training_joint_ticks":0,"evaluation_joint_ticks":0}
            for line in path.read_text().splitlines():
                record = json.loads(line)
                if record["event"] != "episode-completed": continue
                row = record["payload"]
                key = path.parent.relative_to(self.root).as_posix()+"/"+row["episode_id"]
                rows.append({"key":key,"label":f"{path.parent.parent.name} {path.parent.name.upper()} · {'training' if row['training'] else 'frozen evaluation'} · seed {row['seed']} · {row['escaped_count']}/3 · {row['episode_id']}","row":row,"root":str(path.parent),"cumulative_before":dict(cumulative)})
                prefix = "training" if row["training"] else "evaluation"
                cumulative[prefix+"_joint_ticks"] += row["steps"]
        return rows

    def replay(self, key=None, tick=None):
        if key is not None:
            entry = next(r for r in self.episodes() if r["key"]==key)
            row, root = entry["row"], Path(entry["root"])
            archive = root / row["archive"]["path"]
            if file_sha256(archive) != row["archive"]["sha256"].removeprefix("sha256:"):
                raise ValueError("archive hash mismatch")
            with np.load(archive,allow_pickle=False) as saved: self.replay_arrays = {k:saved[k] for k in saved.files}
            self.replay_world = make_escape_chamber(row["environment_adapter"],seed=row["seed"],horizon=row["steps"],reward_table=RewardTable())
            self.replay_world.reset()
            self.replay_row = row
            self.replay_key = key
            self.replay_cumulative_before = entry["cumulative_before"]
            pointer=json.loads((root/"ESCAPE_TEAM.json").read_text())
            manifest=json.loads((root/pointer["manifest"]).read_text())
            self.replay_method = "PPO" if manifest.get("initialization",{}).get("learning_method","").startswith("independent-ppo") else "GUM"
            parts = key.split('/')
            self.replay_team_label = parts[-3] if len(parts)>=3 else root.name
            self.replay_team_seed=manifest["members"][0]["identity"]["seed"]-101
            return {"length":row["steps"]}
        world, arrays = self.replay_world, self.replay_arrays
        t = int(tick)
        if t != world.steps: raise ValueError("replay must advance sequentially")
        if not np.array_equal(np.stack(world.observations()),arrays["observations"][t]): raise ValueError("replay pixels mismatch")
        step = world.step([None if a<0 else int(a) for a in arrays["actions"][t]])
        if world.audit_state()["state_sha256"].removeprefix("sha256:") != arrays["state_hashes"][t].decode(): raise ValueError("replay state mismatch")
        if not (np.array_equal(np.stack(step.observations),arrays["next_observations"][t]) and
                np.allclose(step.rewards,arrays["rewards"][t],atol=1e-7) and
                step.terminated==bool(arrays["terminated"][t]) and step.truncated==bool(arrays["truncated"][t])):
            raise ValueError("replay consequences mismatch")
        with self.lock:
            self.replay_frames = [world.spectator_frame(),*world.observations()]
            self.replaying = True
        cumulative = dict(self.replay_cumulative_before)
        prefix = "training" if self.replay_row["training"] else "evaluation"
        cumulative[prefix+"_joint_ticks"] += world.steps
        cumulative["current_episode_individual_actions"] = int((arrays["actions"][:world.steps]>=0).sum())
        return {"method":self.replay_method,"phase":"replay of "+("training" if self.replay_row["training"] else "frozen evaluation"),
                "team":self.replay_team_label,"team_seed":self.replay_team_seed,"environment_seed":world.seed,
                "episode":self.replay_row["episode_id"],"tick":world.steps,"escapes":len(world.escape_order),"recorded_actions":arrays["actions"][t].tolist(),"cumulative":cumulative}


def start_viewer(root, port=8786):
    viewer = StudyViewer(root)
    token = secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            query = parse_qs(urlparse(self.path).query)
            hosts = {f"127.0.0.1:{self.server.server_port}",f"localhost:{self.server.server_port}"}
            if (self.headers.get("Host","").lower() not in hosts or
                not secrets.compare_digest(query.get("token",[""])[0],token) or
                self.headers.get("Origin") not in {None,*["http://"+h for h in hosts]}):
                self.send_error(403); return
            try:
                path = urlparse(self.path).path
                kind = "application/json"
                if path == "/": data=HTML.replace("__TOKEN__",json.dumps(token)).encode();kind="text/html; charset=utf-8"
                elif path == "/state":
                    with viewer.lock: data=json.dumps(viewer.state).encode();viewer.replaying=False
                elif path == "/episodes": data=json.dumps([{k:r[k] for k in ("key","label")} for r in viewer.episodes()]).encode()
                elif path == "/replay": data=json.dumps(viewer.replay(query.get("key",[None])[0],query.get("tick",[None])[0])).encode()
                elif path == "/frame":
                    with viewer.lock:
                        frames = viewer.replay_frames if viewer.replaying else viewer.frames
                        if frames is None: self.send_error(404); return
                        index = int(query["member"][0])+1 if "member" in query else 0
                        frame=frames[index].copy()
                    buffer=io.BytesIO();Image.fromarray(frame).save(buffer,format="PNG");data=buffer.getvalue();kind="image/png"
                else: self.send_error(404);return
                self.send_response(200);self.send_header("Content-Type",kind);self.send_header("Content-Length",str(len(data)))
                self.send_header("Cache-Control","no-store");self.send_header("X-Content-Type-Options","nosniff")
                self.send_header("Referrer-Policy","no-referrer");self.send_header("X-Frame-Options","DENY")
                self.send_header("Content-Security-Policy","default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
                self.end_headers();self.wfile.write(data)
            except (BrokenPipeError, ConnectionError): pass
            except Exception as error: self.send_error(400,str(error))
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(("127.0.0.1",port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    return viewer,server,f"http://127.0.0.1:{server.server_port}/?token={token}"
