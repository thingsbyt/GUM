"""Small localhost-only dashboard. One bounded/cancellable worker at a time."""
from __future__ import annotations
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import parse_qs,urlparse
import webbrowser
from .config import Config
from .runtime import ROOT,atomic_json

ACTIONS={'verify':['verify'],'collect':['collect'],'encode':['encode'],'train':['train'],
         'trial_train':['train','--max-steps','10'],'evaluate':['evaluate','--trials','8'],
         'start_fresh':['learn','--episodes','1000000','--fresh'],
         'continue':['learn','--episodes','1000000'],
         'watch':['watch','--episodes','1','--replay-delay','.03']}

class Controller:
    def __init__(self,cfg:Config,config_path:Path,workspace:Path):
        self.cfg=cfg;self.config_path=config_path;self.base_workspace=workspace
        self.base_workspace.mkdir(parents=True,exist_ok=True)
        self.workspace=self.base_workspace
        registry=self.base_workspace/'active_brain.json'
        try:
            candidate=Path(json.loads(registry.read_text(encoding='utf-8'))['workspace']).resolve()
            if candidate.is_relative_to((self.base_workspace/'brains').resolve()):self.workspace=candidate
        except (OSError,KeyError,ValueError,json.JSONDecodeError):pass
        self.token=secrets.token_urlsafe(24);self.process=None;self.log=None
        self.lock=threading.Lock();self.action=None
    def start(self,action:str):
        if action not in ACTIONS:raise ValueError('Unknown action.')
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError('A job is already running. Stop it before starting another.')
            if self.log:self.log.close()
            if action=='start_fresh':
                identity=time.strftime('brain_%Y%m%d_%H%M%S_')+secrets.token_hex(3)
                self.workspace=self.base_workspace/'brains'/identity
                self.workspace.mkdir(parents=True,exist_ok=False)
                atomic_json(self.base_workspace/'active_brain.json',{'workspace':str(self.workspace)})
            elif action in ('continue','watch') and not (self.workspace/'stable_brain.pt').exists():
                raise RuntimeError('No saved brain is selected. Start fresh first.')
            (self.workspace/'STOP').unlink(missing_ok=True)
            self.log=(self.workspace/'job.log').open('w',encoding='utf-8')
            command=[sys.executable,'-u','-m','jepa_asteroids','--config',str(self.config_path),
                     '--workspace',str(self.workspace),'--keep-stop',*ACTIONS[action]]
            flags={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
            self.process=subprocess.Popen(command,cwd=ROOT,stdout=self.log,stderr=subprocess.STDOUT,**flags)
            self.action=action
    def stop(self):
        (self.workspace/'STOP').touch()
    def state(self)->dict:
        try:progress=json.loads((self.workspace/'status.json').read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError):progress={'stage':'ready','message':'Set up the official DINOv3 weights, then record and encode.'}
        try:
            with (self.workspace/'job.log').open('rb') as handle:
                handle.seek(0,2);size=handle.tell();handle.seek(max(0,size-16000))
                log=handle.read().decode('utf-8',errors='replace')
        except OSError:log=''
        code=self.process.poll() if self.process else None
        return {'running':self.process is not None and code is None,'returncode':code,'action':self.action,
            'progress':progress,'log':log,'config':{'name':self.cfg.name,'batch_size':self.cfg.stable_batch_size,
            'epochs':0,'predictor_dim':self.cfg.stable_latent_dim,'predictor_depth':2},
            'workspace':str(self.workspace),'stop_requested':(self.workspace/'STOP').exists()}


def make_handler(controller:Controller):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def _allowed(self):
            port=self.server.server_address[1]
            hosts={f'127.0.0.1:{port}',f'localhost:{port}'}
            if self.headers.get('Host') not in hosts:return False
            origin=self.headers.get('Origin')
            if origin is not None and origin not in {f'http://{h}' for h in hosts}:return False
            token=self.headers.get('X-Lab-Token') or parse_qs(urlparse(self.path).query).get('token',[''])[0]
            return secrets.compare_digest(token,controller.token)
        def _reply(self,code:int,data:bytes,content_type:str):
            self.send_response(code);self.send_header('Content-Type',content_type)
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def _json(self,code:int,value:dict):self._reply(code,json.dumps(value).encode(),'application/json')
        def do_GET(self):
            if not self._allowed():return self._json(403,{'error':'Local dashboard token/origin required.'})
            path=urlparse(self.path).path
            if path=='/':
                html=(ROOT/'web'/'index.html').read_text(encoding='utf-8').replace('__LAB_TOKEN__',controller.token)
                return self._reply(200,html.encode(),'text/html; charset=utf-8')
            if path=='/api/state':return self._json(200,controller.state())
            if path in ('/current.png','/goal.png'):
                file=controller.workspace/path.lstrip('/')
                if file.exists():return self._reply(200,file.read_bytes(),'image/png')
                return self._json(404,{'error':'No frame yet.'})
            return self._json(404,{'error':'Not found.'})
        def do_POST(self):
            if not self._allowed():return self._json(403,{'error':'Local dashboard token/origin required.'})
            if urlparse(self.path).path!='/api/action':return self._json(404,{'error':'Not found.'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=2048:raise ValueError('Invalid request size.')
                value=json.loads(self.rfile.read(length));action=value['action']
                if action=='stop':controller.stop()
                else:controller.start(action)
                return self._json(200,{'ok':True})
            except (ValueError,KeyError,json.JSONDecodeError) as exc:return self._json(400,{'error':str(exc)})
            except RuntimeError as exc:return self._json(409,{'error':str(exc)})
    return Handler


def serve(cfg:Config,config_path:Path,workspace:Path,port:int=8779,open_browser:bool=True):
    controller=Controller(cfg,config_path,workspace)
    server=ThreadingHTTPServer(('127.0.0.1',port),make_handler(controller))
    url=f'http://127.0.0.1:{server.server_address[1]}/?token={controller.token}'
    print(f'Local dashboard: {url}',flush=True)
    print('Closing a browser tab does NOT stop a worker. Use Stop, or Ctrl+C here.',flush=True)
    if open_browser:webbrowser.open(url)
    try:server.serve_forever(poll_interval=.3)
    except KeyboardInterrupt:pass
    finally:
        controller.stop();server.server_close()
        if controller.log:controller.log.close()
