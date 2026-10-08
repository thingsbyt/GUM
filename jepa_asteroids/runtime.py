"""Local paths, atomic writes, cancellation, and monitored resource limits."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import psutil
import torch
from .config import Config

ROOT=Path(__file__).resolve().parents[1]


def privacy_environment(root:Path=ROOT)->None:
    # A user-specified HF_HOME is respected. Otherwise keep downloads on this drive.
    os.environ.setdefault('HF_HOME',str(root/'model_cache'))
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    os.environ['DO_NOT_TRACK']='1'
    os.environ['WANDB_MODE']='disabled'
    os.environ['TOKENIZERS_PARALLELISM']='false'


def replace_file(source:Path,destination:Path)->None:
    for attempt in range(8):
        try:
            os.replace(source,destination)
            return
        except PermissionError:
            if attempt==7:raise
            time.sleep(.02*(attempt+1))


def atomic_json(path:Path,value:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    replace_file(temp,path)


def atomic_bytes(path:Path,value:bytes)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_bytes(value);replace_file(temp,path)


class Stopped(RuntimeError):pass


class Guard:
    """Polling thresholds, not a hardware/temperature guarantee or system-wide cap."""
    def __init__(self,cfg:Config,workspace:Path,device:torch.device):
        self.cfg=cfg;self.workspace=workspace;self.device=device
        self.last=0.0;self.process=psutil.Process()
        if device.type=='cuda':
            total=torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(0.9,cfg.max_gpu_gib*2**30/total),device)

    def check(self,force:bool=False)->None:
        if (self.workspace/'STOP').exists():
            raise Stopped('Stop requested; current completed work is saved.')
        now=time.monotonic()
        if not force and now-self.last<2:return
        self.last=now
        used=self.process.memory_info().rss/2**30
        system=psutil.virtual_memory().used/2**30
        if used>self.cfg.max_process_ram_gib:
            raise Stopped(f'Process RAM threshold exceeded: {used:.1f} GiB.')
        if system>self.cfg.max_system_ram_gib:
            raise Stopped(f'System RAM threshold exceeded: {system:.1f} GiB. Close other workloads.')
        if self.device.type=='cuda':
            allocated=torch.cuda.memory_reserved(self.device)/2**30
            if allocated>self.cfg.max_gpu_gib:
                raise Stopped(f'PyTorch GPU memory threshold exceeded: {allocated:.1f} GiB.')
            smi=shutil.which('nvidia-smi')
            if smi:
                flags={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
                try:
                    result=subprocess.run([smi,'--query-gpu=memory.used,temperature.gpu',
                        '--format=csv,noheader,nounits','-i',str(self.device.index or 0)],
                        capture_output=True,text=True,timeout=3,**flags)
                    values=result.stdout.strip().split(',')
                    if result.returncode==0 and len(values)==2:
                        gpu_mib,temp=map(float,values)
                        if gpu_mib/1024>self.cfg.max_gpu_gib or temp>=self.cfg.max_gpu_temperature_c:
                            raise Stopped(f'GPU threshold reached: {gpu_mib/1024:.1f} GiB, {temp:.0f} C.')
                except (subprocess.SubprocessError,ValueError):
                    pass


def device_for(cfg:Config,override:str|None=None)->torch.device:
    torch.set_num_threads(cfg.cpu_threads)
    name=override or cfg.device
    if name=='auto':name='cuda:0' if torch.cuda.is_available() else 'cpu'
    device=torch.device(name)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but this Python has no usable CUDA PyTorch. No package was replaced.')
    return device


def disk_budget(folder:Path,extra_bytes:int,cfg:Config)->None:
    folder.mkdir(parents=True,exist_ok=True)
    used=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    if used+extra_bytes>cfg.max_cache_gib*2**30:
        raise Stopped(f'Dataset cache would exceed {cfg.max_cache_gib:g} GiB. Use fewer episodes or explicitly raise the cap.')
    if shutil.disk_usage(folder).free<extra_bytes+2**30:
        raise Stopped('Insufficient disk space; retaining 1 GiB free margin.')
