"""Deterministic RGB camera for the game. Numeric HUD is never in the camera image."""
from __future__ import annotations
import io
import math
import numpy as np
from PIL import Image, ImageDraw


def render_snapshot(state:dict,width:int=512,height:int=256)->np.ndarray:
    image=Image.new('RGB',(width,height),(8,12,20))
    draw=ImageDraw.Draw(image)
    sx,sy=width/state['size'][0],height/state['size'][1]
    def point(x,y):return x*sx,y*sy
    for x,y,vx,vy,r in state['rocks']:
        # Fixed shape derived only from visible radius. No hidden-state glyphs.
        pts=[]
        for k in range(9):
            theta=2*math.pi*k/9
            radius=r*(0.88 if k%2 else 1.0)
            pts.append(point(x+radius*math.cos(theta),y+radius*math.sin(theta)))
        draw.polygon(pts,fill=(63,73,92),outline=(182,191,206))
    for x,y,*_ in state['bullets']:
        px,py=point(x,y)
        draw.ellipse((px-2,py-2,px+2,py+2),fill=(255,221,126))
    x,y=state['ship'];a=state['angle']
    if not state['dead']:
        pts=[point(x+13*math.cos(a),y+13*math.sin(a)),
             point(x+10*math.cos(a+2.45),y+10*math.sin(a+2.45)),
             point(x+10*math.cos(a-2.45),y+10*math.sin(a-2.45))]
        draw.polygon(pts,fill=(93,215,207),outline=(221,255,249))
        if state['shield']>0:
            px,py=point(x,y);rx,ry=17*sx,17*sy
            draw.ellipse((px-rx,py-ry,px+rx,py+ry),outline=(67,125,146),width=1)
    else:
        px,py=point(x,y)
        draw.line((px-7,py-7,px+7,py+7),fill=(255,115,114),width=2)
        draw.line((px-7,py+7,px+7,py-7),fill=(255,115,114),width=2)
    return np.asarray(image,dtype=np.uint8).copy()


def png_bytes(frame:np.ndarray)->bytes:
    out=io.BytesIO();Image.fromarray(frame).save(out,format='PNG');return out.getvalue()
