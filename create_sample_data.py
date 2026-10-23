"""Create a tiny synthetic RGB dataset using only Python's standard library.

These procedural landscape illustrations are useful for a quick pipeline demo;
replace them with licensed photos for meaningful natural-image colorization.
"""
from __future__ import annotations
import argparse
import math
import random
import struct
from pathlib import Path

PALETTES = [
    ((112, 190, 235), (255, 207, 90), (74, 156, 90), (31, 112, 66), (54, 139, 190)),
    ((232, 157, 111), (255, 224, 144), (136, 133, 79), (71, 105, 65), (40, 105, 150)),
    ((150, 190, 225), (245, 233, 180), (134, 163, 109), (66, 117, 86), (72, 139, 168)),
    ((100, 177, 197), (255, 194, 127), (179, 131, 88), (91, 113, 82), (38, 111, 152)),
]

def bmp(path: Path, w: int, h: int, pixels: bytearray) -> None:
    stride = (w * 3 + 3) & ~3
    rows = bytearray()
    for y in range(h - 1, -1, -1):
        row = bytearray()
        for x in range(w):
            r, g, b = pixels[(y*w+x)*3:(y*w+x+1)*3]
            row.extend((b, g, r))
        row.extend(b"\0" * (stride - w*3)); rows.extend(row)
    size = 14 + 40 + len(rows)
    header = b"BM" + struct.pack("<IHHI", size, 0, 0, 54)
    dib = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, len(rows), 2835, 2835, 0, 0)
    path.write_bytes(header + dib + rows)

def make_image(seed: int, size: int = 128) -> bytearray:
    rng = random.Random(seed); w=h=size
    sky, sun, far, near, water = PALETTES[seed % len(PALETTES)]
    p = bytearray(w*h*3)
    horizon = rng.randint(66, 87)
    # Vertical sky gradient.
    for y in range(h):
        t=y/max(1,h-1)
        c=tuple(round(sky[k]*(1-t*.22)+water[k]*(t*.22)) for k in range(3))
        for x in range(w): p[(y*w+x)*3:(y*w+x+1)*3]=bytes(c)
    def put(x,y,c):
        if 0 <= x < w and 0 <= y < h: p[(y*w+x)*3:(y*w+x+1)*3]=bytes(c)
    def ellipse(cx,cy,rx,ry,c):
        for y in range(max(0,cy-ry),min(h,cy+ry+1)):
            for x in range(max(0,cx-rx),min(w,cx+rx+1)):
                if ((x-cx)/max(1,rx))**2+((y-cy)/max(1,ry))**2 <= 1: put(x,y,c)
    def polygon(points,c):
        ymin=max(0,min(y for x,y in points)); ymax=min(h-1,max(y for x,y in points))
        for y in range(ymin,ymax+1):
            xs=[]
            for i,(x1,y1) in enumerate(points):
                x2,y2=points[(i+1)%len(points)]
                if y1==y2: continue
                if min(y1,y2)<=y<max(y1,y2): xs.append(int(x1+(y-y1)*(x2-x1)/(y2-y1)))
            xs.sort()
            for a,b in zip(xs[::2],xs[1::2]):
                for x in range(max(0,a),min(w,b+1)): put(x,y,c)
    sx=rng.randint(18,110); sy=rng.randint(12,42); sr=rng.randint(8,15)
    ellipse(sx,sy,sr,sr,sun)
    # distant rolling mountain silhouettes
    peaks=[(0,horizon)]
    for x in range(0,w+1,16): peaks.append((x, rng.randint(horizon-28,horizon+8)))
    peaks.extend([(w,horizon),(w,h),(0,h)])
    polygon(peaks,far)
    # foreground rolling hills
    peaks=[(0,horizon+22)]
    for x in range(0,w+1,12): peaks.append((x,rng.randint(horizon+4,horizon+38)))
    peaks.extend([(w,horizon+30),(w,h),(0,h)])
    polygon(peaks,near)
    # foreground meadow base
    for y in range(horizon+45,h):
        for x in range(w): put(x,y,tuple(max(0,min(255,v + (y-horizon-45)//3)) for v in near))
    # small house
    hx=rng.randint(16,88); hy=horizon+rng.randint(10,26); hw=rng.randint(18,29); hh=rng.randint(15,23)
    wall=(232,205,154); roof=(145,67,55)
    polygon([(hx,hy+8),(hx+hw//2,hy-4),(hx+hw,hy+8)],roof)
    polygon([(hx+3,hy+8),(hx+hw-3,hy+8),(hx+hw-3,hy+hh),(hx+3,hy+hh)],wall)
    for yy in range(hy+12,hy+hh,5):
        for xx in (hx+6,hx+hw-10):
            for dy in range(3):
                for dx in range(4): put(xx+dx,yy+dy,(70,145,180))
    # trees and varied flowers add high-contrast color cues
    for _ in range(5):
        tx=rng.randint(4,w-5); ty=rng.randint(horizon+15,h-7); th=rng.randint(12,25)
        for y in range(ty-th,ty+2): put(tx,y,(94,61,42))
        green=(rng.randint(30,65),rng.randint(90,145),rng.randint(35,70))
        ellipse(tx,ty-th, rng.randint(6,10),rng.randint(7,12),green)
    for _ in range(25):
        x=rng.randrange(w); y=rng.randrange(horizon+30,h)
        ellipse(x,y,1,1,rng.choice([(255,223,74),(235,115,145),(240,240,224)]))
    # soft horizontal cloud strokes
    for _ in range(3):
        cx=rng.randint(8,110); cy=rng.randint(10,50)
        if abs(cx-sx)>sr+10: ellipse(cx,cy, rng.randint(5,10),2,(235,244,246))
    return p

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,default=Path("data/train/generated"))
    ap.add_argument("--count",type=int,default=24)
    ap.add_argument("--size",type=int,default=128)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    for i in range(args.count): bmp(args.output/f"synthetic-landscape-{i+1:03d}.bmp",args.size,args.size,make_image(i,args.size))
    print(f"Created {args.count} synthetic BMP training images in {args.output}")

if __name__=="__main__": main()
