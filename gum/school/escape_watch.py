"""Loopback-only live watch interface for genuine escape-team learning."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import ipaddress
import json
from pathlib import Path
import secrets
import socket
import threading
import time
from typing import Any
from urllib.parse import parse_qs, quote, urlparse
import uuid

from PIL import Image

from .escape_chamber import EscapeChamberRoomA, MEMBER_COLORS, TEAM_SIZE
from .escape_team import EscapeTeam


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GUM · Cooperative Escape Chamber</title><style>
:root{--bg:#090e14;--panel:#111923;--line:#263443;--ink:#f0ebda;--muted:#8ea0ad;--gold:#e1b75d;--good:#b7dc9b;--warn:#e47f69}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% 0,#14202b 0,var(--bg) 42%);color:var(--ink);font:14px/1.45 Inter,Segoe UI,sans-serif}button,select,input{font:inherit}
header{height:74px;padding:17px 28px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;background:#0b1118e8;position:sticky;top:0;z-index:3}.brand small{display:block;color:#73e4d0;letter-spacing:.16em;font-weight:700}.brand strong{font-size:22px}.badge{border:1px solid #315064;border-radius:999px;padding:8px 13px;color:#9dc3d4}
main{max-width:1600px;margin:auto;padding:22px;display:grid;grid-template-columns:minmax(760px,1fr) 350px;gap:18px}.panel{background:linear-gradient(160deg,#111a24,#0e161f);border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:0 18px 60px #0005}.panel-head{padding:13px 16px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between}.panel-head b{font-size:15px}.mode{color:#73e4d0;font-size:12px;letter-spacing:.09em}.hero{width:100%;display:block;aspect-ratio:16/9;object-fit:contain;background:#080d12}.controls{display:flex;gap:9px;flex-wrap:wrap;padding:13px 16px;border-top:1px solid var(--line)}button{background:#172433;color:var(--ink);border:1px solid #344b61;border-radius:8px;padding:9px 13px;cursor:pointer}button.primary{background:#315b55;border-color:#4d8278}button.stop{margin-left:auto;background:#49282b;border-color:#7c4145}button:disabled{opacity:.38;cursor:not-allowed}select,input{background:#0d1620;color:var(--ink);border:1px solid #344b61;border-radius:7px;padding:8px}.readout{padding:0 16px 14px;color:var(--muted);font-size:12px}
.side{display:flex;flex-direction:column;gap:18px}.summary{padding:16px}.summary h2{margin:0 0 12px;font-size:17px}.metric{display:flex;justify-content:space-between;border-bottom:1px solid #22303d;padding:9px 0}.metric span{color:var(--muted)}.metric b{font-variant-numeric:tabular-nums}.truth{margin-top:13px;border-left:3px solid var(--gold);padding:9px 11px;background:#151c21;color:#cdd4d4}.members{padding:10px;display:grid;gap:8px}.member{border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:9px;padding:10px 11px;background:#0b1219}.member b{color:var(--accent)}.member .row{display:flex;justify-content:space-between;color:var(--muted);margin-top:5px}.eyes{grid-column:1/-1}.eye-grid{padding:12px;display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.eye{background:#090f15;border:1px solid var(--line);border-radius:10px;overflow:hidden}.eye img{width:100%;display:block;image-rendering:auto}.eye div{padding:8px 10px;color:var(--muted);display:flex;justify-content:space-between}.outcomes{grid-column:1/-1;padding:14px 16px}.outcome-list{display:flex;gap:7px;min-height:48px;align-items:end}.bar{width:25px;background:#344656;border-radius:5px 5px 2px 2px;min-height:6px;position:relative}.bar.max{background:#83a96d}.bar:hover:after{content:attr(data-tip);position:absolute;bottom:110%;left:50%;transform:translateX(-50%);white-space:nowrap;background:#05080b;padding:5px 7px;border-radius:5px;color:var(--ink)}.foot{color:var(--muted);padding:4px 1px 20px;grid-column:1/-1}
@media(max-width:1100px){main{grid-template-columns:1fr}.side{display:grid;grid-template-columns:1fr 1fr}.eye-grid{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<header><div class="brand"><small>GUM / LIVE LEARNING EXPERIMENT</small><strong>Cooperative Escape Chamber</strong></div><div class="badge" id="connection">connecting…</div></header>
<main>
<section class="panel"><div class="panel-head"><b>Authoritative room</b><span class="mode" id="mode">PREVIEW · NOT TRAINING</span></div><img class="hero" id="hero" alt="Authoritative spectator view"><div class="controls">
<button class="primary" id="start">Start learning</button><button id="pause">Pause safely</button><button id="step">Single step</button><button id="resume">Resume</button><label>speed <select id="speed"><option>5</option><option selected>10</option><option>30</option><option>120</option></select> ticks/s</label><label>episodes <input id="episodes" type="number" value="12" min="1" max="500" style="width:72px"></label><button class="stop" id="stop">Stop + checkpoint</button>
</div><div class="readout" id="frameRef">No authoritative training frame yet.</div></section>
<aside class="side"><section class="panel summary"><h2>Run truth</h2><div class="metric"><span>Run ID</span><b id="run">—</b></div><div class="metric"><span>Episode</span><b id="episode">0 / 0</b></div><div class="metric"><span>Joint tick</span><b id="tick">0</b></div><div class="metric"><span>Escaped</span><b id="escaped">0 / 3</b></div><div class="metric"><span>Weights</span><b id="weights">not updating</b></div><div class="metric"><span>Sharing</span><b>OFF</b></div><div class="truth">Three escapes are the legal maximum. A successful episode leaves one living member visibly inside.</div></section><section class="panel"><div class="panel-head"><b>Independent bodies</b></div><div class="members" id="members"></div></section></aside>
<section class="panel eyes"><div class="panel-head"><b>Exact learner rasters</b><span class="mode">WHITE RING = SELF · NO SPECTATOR LABELS</span></div><div class="eye-grid" id="eyes"></div></section>
<section class="panel outcomes"><div class="panel-head"><b>Completed outcomes</b><span class="mode">EVERY EPISODE · FAILURES INCLUDED</span></div><div class="outcome-list" id="outcomes"></div></section>
<div class="foot">The display follows simulation timestamps. Closing this page does not stop training. No LLM, scripted route, captain, or browser timing selects learner actions. “Experience sent” will appear only after a real sharing treatment is implemented; protocol v1 enforces sharing off.</div>
</main><script>
const token=__ESCAPE_TOKEN__, $=id=>document.getElementById(id);let lastFrame=-1;
async function api(path,opt={}){opt.headers=new Headers(opt.headers||{});opt.headers.set('X-GUM-Escape-Token',token);if(opt.body)opt.headers.set('Content-Type','application/json');const r=await fetch(path,opt);const j=await r.json();if(!r.ok)throw Error(j.error||r.statusText);return j}
function media(path){return path+(path.includes('?')?'&':'?')+'token='+encodeURIComponent(token)+'&t='+Date.now()}
function memberCard(m){return `<div class="member" style="--accent:rgb(${m.color.join(',')})"><b>${m.name} / ${m.shape}</b><div class="row"><span>${m.status}</span><span>${m.action===null?'-':'slot '+m.action}</span></div><div class="row"><span>local reward</span><span>${m.reward.toFixed(3)}</span></div></div>`}
async function refresh(){try{const s=await api('/api/state');$('connection').textContent='local · authoritative';$('mode').textContent=s.mode.toUpperCase();$('run').textContent=s.run_id.slice(0,8);$('episode').textContent=`${s.episode_index} / ${s.episode_budget}`;$('tick').textContent=s.tick;$('escaped').textContent=`${s.escaped_count} / 3`;$('weights').textContent=s.weights_updating?'UPDATING':'frozen / idle';$('frameRef').textContent=s.frame_reference||'No authoritative training frame yet.';$('members').innerHTML=s.members.map(memberCard).join('');$('start').disabled=s.running;$('pause').disabled=!s.running||s.paused;$('step').disabled=!s.running||!s.paused;$('resume').disabled=!s.running||!s.paused;$('stop').disabled=!s.running;if(s.frame_serial!==lastFrame){lastFrame=s.frame_serial;$('hero').src=media('/api/frame/spectator.png');for(let i=0;i<4;i++)$('eye'+i).src=media('/api/frame/member/'+i+'.png')}const out=$('outcomes');out.replaceChildren(...s.outcomes.map(o=>{const d=document.createElement('div');d.className='bar '+(o.escaped_count===3?'max':'');d.style.height=(8+o.escaped_count*12)+'px';d.dataset.tip=`episode ${o.index}: ${o.escaped_count}/3, ${o.steps} ticks`;return d}))}catch(e){$('connection').textContent='error · '+e.message}}
async function post(path,value={}){try{await api(path,{method:'POST',body:JSON.stringify(value)});await refresh()}catch(e){alert(e.message)}}
$('start').onclick=()=>post('/api/start',{episodes:Number($('episodes').value),speed:Number($('speed').value)});$('pause').onclick=()=>post('/api/pause');$('step').onclick=()=>post('/api/step');$('resume').onclick=()=>post('/api/resume');$('stop').onclick=()=>post('/api/stop');$('speed').onchange=()=>post('/api/speed',{speed:Number($('speed').value)});
$('eyes').innerHTML=[0,1,2,3].map(i=>`<div class="eye"><img id="eye${i}" alt="Fortis-${i+1} learner raster"><div><span>Fortis-${i+1}</span><span>policy input</span></div></div>`).join('');refresh();setInterval(refresh,180);
</script></body></html>'''


class EscapeWatchSession:
    def __init__(self, team: EscapeTeam):
        self.team = team
        self.run_id = str(uuid.uuid4())
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._running = False
        self._paused = False
        self._step_permits = 0
        self._stop_after_episode = False
        self._speed = 10.0
        self._episode_budget = 0
        self._episode_index = 0
        self._tick = 0
        self._error: str | None = None
        self._outcomes: list[dict[str, Any]] = []
        self._frame_serial = 0
        self._frame_reference: str | None = None
        preview = EscapeChamberRoomA(seed=0, reward_table=team.reward_table)
        self._member_frames = preview.reset()
        self._spectator_frame = preview.spectator_frame()
        self._world = preview

    def start(self, *, episodes: int, speed: float, seed_start: int = 1_930_000) -> None:
        with self._condition:
            if self._running:
                raise RuntimeError("learning is already running")
            if not 1 <= int(episodes) <= 500:
                raise ValueError("episodes must be in [1, 500]")
            self._set_speed(speed)
            self.run_id = str(uuid.uuid4())
            self._episode_budget = int(episodes)
            self._episode_index = 0
            self._tick = 0
            self._error = None
            self._outcomes = []
            self._paused = False
            self._step_permits = 0
            self._stop_after_episode = False
            self._running = True
            self._thread = threading.Thread(
                target=self._run,
                args=(int(seed_start),),
                daemon=True,
                name="gum-escape-learning",
            )
            self._thread.start()

    def _run(self, seed_start: int) -> None:
        try:
            for offset in range(self._episode_budget):
                with self._condition:
                    self._episode_index = offset + 1
                capsule = self.team.run_episode(
                    seed=seed_start + offset,
                    training=True,
                    on_step=self._on_step,
                )
                with self._condition:
                    self._outcomes.append({
                        "index": self._episode_index,
                        "episode_id": capsule["episode_id"],
                        "escaped_count": capsule["escaped_count"],
                        "steps": capsule["steps"],
                        "legal_maximum_reached": capsule["legal_maximum_reached"],
                    })
                    if self._stop_after_episode:
                        break
        except Exception as error:
            with self._condition:
                self._error = f"{type(error).__name__}: {error}"
        finally:
            with self._condition:
                self._running = False
                self._paused = False
                self._condition.notify_all()

    def _on_step(self, world: EscapeChamberRoomA, row: dict[str, Any]) -> None:
        with self._condition:
            self._world = world
            self._tick = world.steps
            self._spectator_frame = world.spectator_frame()
            self._member_frames = world.observations()
            self._frame_serial += 1
            self._frame_reference = (
                f"run {self.run_id} · episode {self._episode_index} · joint tick "
                f"{world.steps} · state sha256:{row['state_sha256']}"
            )
            while self._paused and self._step_permits == 0:
                self._condition.wait()
            if self._paused and self._step_permits > 0:
                self._step_permits -= 1
            delay = 1.0 / self._speed
        if delay > 0:
            time.sleep(delay)

    def pause(self) -> None:
        with self._condition:
            if not self._running:
                raise RuntimeError("learning is not running")
            self._paused = True

    def step_once(self) -> None:
        with self._condition:
            if not self._running or not self._paused:
                raise RuntimeError("single-step requires paused live learning")
            self._step_permits += 1
            self._condition.notify_all()

    def resume(self) -> None:
        with self._condition:
            if not self._running:
                raise RuntimeError("learning is not running")
            self._paused = False
            self._step_permits = 0
            self._condition.notify_all()

    def stop_after_episode(self) -> None:
        with self._condition:
            if not self._running:
                raise RuntimeError("learning is not running")
            self._stop_after_episode = True
            self._paused = False
            self._condition.notify_all()

    def set_speed(self, speed: float) -> None:
        with self._condition:
            self._set_speed(speed)

    def _set_speed(self, speed: float) -> None:
        value = float(speed)
        if not 1.0 <= value <= 240.0:
            raise ValueError("speed must be in [1, 240] ticks per second")
        self._speed = value

    def state(self) -> dict[str, Any]:
        with self._condition:
            world = self._world
            members = []
            for index, member in enumerate(self.team.members):
                escaped = bool(world.escaped[index])
                remaining = (
                    world._done and len(world.escape_order) == TEAM_SIZE - 1 and not escaped
                )
                members.append({
                    "member_id": member.identity.member_id,
                    "name": member.identity.name,
                    "shape": member.identity.shape,
                    "color": list(MEMBER_COLORS[index]),
                    "status": "remaining inside" if remaining else ("escaped" if escaped else "active"),
                    "action": world.last_actions[index],
                    "reward": float(world.last_rewards[index]),
                })
            if self._error:
                mode = "error · " + self._error
            elif self._running and self._paused:
                mode = "live training paused at safe joint tick"
            elif self._running:
                mode = f"live training · {self._speed:g} joint ticks/s target"
            elif self._outcomes:
                mode = "training stopped · coherent episode checkpoint saved"
            else:
                mode = "preview · not training"
            return {
                "run_id": self.run_id,
                "mode": mode,
                "running": self._running,
                "paused": self._paused,
                # A paused joint tick must not be presented as if optimizer
                # state were changing behind the viewer's back.
                "weights_updating": self._running and not self._paused,
                "episode_index": self._episode_index,
                "episode_budget": self._episode_budget,
                "tick": self._tick,
                "escaped_count": len(world.escape_order),
                "legal_maximum": 3,
                "sharing_mode": "off",
                "frame_serial": self._frame_serial,
                "frame_reference": self._frame_reference,
                "members": members,
                "outcomes": list(self._outcomes),
                "error": self._error,
            }

    def frame_png(self, member: int | None = None) -> bytes:
        with self._condition:
            array = (
                self._spectator_frame.copy()
                if member is None else self._member_frames[member].copy()
            )
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()


def build_escape_watch_server(
    team: EscapeTeam,
    *,
    host: str = "127.0.0.1",
    port: int = 8783,
    token: str | None = None,
) -> tuple[ThreadingHTTPServer, str, EscapeWatchSession]:
    try:
        loopback = host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        raise ValueError("escape watch only binds to a loopback address")
    session_token = token or secrets.token_urlsafe(32)
    session = EscapeWatchSession(team)

    class Handler(BaseHTTPRequestHandler):
        server_version = "GUMEscapeWatch/1"

        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def _allowed(self) -> bool:
            actual_port = self.server.server_address[1]
            hosts = {f"127.0.0.1:{actual_port}", f"localhost:{actual_port}"}
            if str(self.headers.get("Host", "")).lower() not in hosts:
                return False
            origin = self.headers.get("Origin")
            if origin is not None and origin.lower() not in {f"http://{value}" for value in hosts}:
                return False
            query_token = parse_qs(urlparse(self.path).query).get("token", [""])[0]
            supplied = self.headers.get("X-GUM-Escape-Token") or query_token
            return secrets.compare_digest(str(supplied), session_token)

        def _reply(self, body: bytes, status: int, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                "img-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value: Any, status: int = 200) -> None:
            self._reply(json.dumps(value).encode(), status, "application/json; charset=utf-8")

        def _body(self) -> dict[str, Any]:
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                raise ValueError("Content-Type must be application/json")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65_536:
                raise ValueError("invalid request size")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("request body must be an object")
            return value

        def do_GET(self):
            if not self._allowed():
                self._json({"error": "local token and same-origin Host are required"}, 403)
                return
            route = urlparse(self.path).path
            try:
                if route == "/":
                    body = HTML.replace("__ESCAPE_TOKEN__", json.dumps(session_token)).encode()
                    self._reply(body, 200, "text/html; charset=utf-8")
                elif route == "/api/state":
                    self._json(session.state())
                elif route == "/api/frame/spectator.png":
                    self._reply(session.frame_png(), 200, "image/png")
                elif route.startswith("/api/frame/member/") and route.endswith(".png"):
                    member = int(route.removeprefix("/api/frame/member/").removesuffix(".png"))
                    if not 0 <= member < TEAM_SIZE:
                        raise ValueError("member index is outside the team")
                    self._reply(session.frame_png(member), 200, "image/png")
                else:
                    self._json({"error": "not found"}, 404)
            except socket.timeout:
                self._json({"error": "request timed out"}, 408)
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, 400)
            except Exception as error:
                self._json({"error": str(error)}, 500)

        def do_POST(self):
            if not self._allowed():
                self._json({"error": "local token and same-origin Host are required"}, 403)
                return
            route = urlparse(self.path).path
            try:
                value = self._body()
                if route == "/api/start":
                    session.start(episodes=int(value.get("episodes", 12)), speed=float(value.get("speed", 10)))
                elif route == "/api/pause":
                    session.pause()
                elif route == "/api/step":
                    session.step_once()
                elif route == "/api/resume":
                    session.resume()
                elif route == "/api/stop":
                    session.stop_after_episode()
                elif route == "/api/speed":
                    session.set_speed(float(value.get("speed", 10)))
                else:
                    self._json({"error": "not found"}, 404)
                    return
                self._json(session.state())
            except (ValueError, RuntimeError, TypeError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, 409)
            except Exception as error:
                self._json({"error": str(error)}, 500)

        def do_OPTIONS(self):
            self._json({"error": "cross-origin requests are not supported"}, 403)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer((host, int(port)), Handler)
    server.daemon_threads = True
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/?token={quote(session_token)}"
    server.gum_session_token = session_token
    server.gum_url = url
    return server, url, session
