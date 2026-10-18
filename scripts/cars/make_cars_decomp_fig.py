"""Clean decomposition showcase: diverse cars -> their natural 5-part lattice."""
import json, numpy as np, random
from PIL import Image
PARTS=["body","front-face","greenhouse","body-side","wheel"]
COL=[(140,140,140),(232,64,64),(66,120,238),(70,200,110),(248,208,48)]
seg=json.load(open("experiments/cars_seg.json"))
# pick diverse clean side/3-4 views, distinct classes
cand=[r for r in seg if r["has_front"] and 2.1<=r["aspect"]<=3.4 and r["coverage"]>=0.95]
random.seed(5); random.shuffle(cand)
seen=set(); pick=[]
for r in cand:
    if r["label"] in seen: continue
    seen.add(r["label"]); pick.append(r)
    if len(pick)==4: break
def pair(r):
    img=np.asarray(Image.open(r["image_path"]).convert("RGB"))
    pm=np.load(r["npz"])["partmap"]; H,W=pm.shape
    img=np.asarray(Image.fromarray(img).resize((W,H)))
    ov=img.copy()
    for i in range(5):
        m=pm==i
        if m.any(): ov[m]=(0.5*np.array(COL[i])+0.5*ov[m]).astype(np.uint8)
    ov[pm==-1]=(255,255,255)
    g=np.full((H,6,3),255,np.uint8)
    p=np.concatenate([img,g,ov],1)
    return Image.fromarray(p).resize((520,int(520*H/p.shape[1])))
tiles=[pair(r) for r in pick]
h=max(t.height for t in tiles)
tiles=[t.resize((520,h)) for t in tiles]
gap=Image.new("RGB",(14,h),(255,255,255)); vg=Image.new("RGB",(520*2+14,10),(255,255,255))
def row(a,b):
    c=Image.new("RGB",(520*2+14,h),(255,255,255)); c.paste(a,(0,0)); c.paste(gap,(520,0)); c.paste(b,(534,0)); return c
top=row(tiles[0],tiles[1]); bot=row(tiles[2],tiles[3])
canvas=Image.new("RGB",(520*2+14, h*2+10),(255,255,255))
canvas.paste(top,(0,0)); canvas.paste(vg,(0,h)); canvas.paste(bot,(0,h+10))
canvas.save("paper/figures/fig_cars.jpg",quality=93)
print("SAVED fig_cars.jpg", canvas.size, "| classes:", [r["label"] for r in pick])
