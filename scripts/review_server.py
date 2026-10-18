"""Local accept/reject GUI for hull/superstructure splits (active learning).

Shows each vessel with its hull (blue) / superstructure (red) overlay and lets
you Accept / Reject / Skip / Back. Accepting saves the shown region map as a
gold label; the accepted set is the segmenter's training data. Run it once on the
geometric bootstrap, train, predict, then run it again with --deck-source
predictions to review the model's output and collect more.

    python -m scripts.review_server --manifest data/pose_full.json \
        --accepted-dir data/accepted [--deck-source predictions \
        --predictions-dir data/predictions] [--port 8000]

Then open http://localhost:8000 . Keys: A accept, R reject, S skip, ArrowLeft back.
No external dependencies (stdlib http.server).
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import random
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from hullprobe.canonical import (
    estimate_deck_geometric,
    full_vessel_alpha,
    hull_superstructure_masks,
    overlay_split,
    region_map_from_masks,
)
from hullprobe.pose import transverse_coord
from hullprobe.segmenter import load_region_png, save_region_png

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>hull/superstructure review</title>
<style>
 body{background:#111;color:#eee;font-family:system-ui,sans-serif;margin:0;text-align:center}
 #stage{position:relative;display:inline-block;margin-top:10px}
 #img{max-width:96vw;max-height:74vh;border:1px solid #333;display:block;-webkit-user-drag:none;user-select:none}
 #band{position:absolute;border:2px dashed #ffd400;background:rgba(255,212,0,.12);display:none;pointer-events:none}
 .bar{margin:12px;font-size:15px}
 button{font-size:16px;padding:10px 20px;margin:0 6px;border:0;border-radius:8px;cursor:pointer;color:#fff;background:#3a4250}
 #acc{background:#1a7f37}#rej{background:#b62324}#skip{background:#555}#back{background:#333}
 #prog{color:#9fb}#counts{color:#aaa}
 .legend{font-size:13px;color:#bbb}.r{color:#ff6b6b}.b{color:#6ba7ff}
</style></head><body>
 <div class=bar><span id=prog></span> &nbsp; <span id=counts></span></div>
 <div class=legend><span class=b>&#9632; hull</span> &nbsp; <span class=r>&#9632; superstructure</span></div>
 <div id=stage><img id=img src=""><div id=band></div></div>
 <div class=bar>
  <button id=up>deck &#8593; (more hull)</button>
  <button id=down>deck &#8595; (more super)</button>
  <button id=rccw>&#8630; rotate</button>
  <button id=rcw>rotate &#8631;</button>
  <button id=clr>&#8635; Reset mask (Z)</button>
 </div>
 <div class=bar>
  <button id=back>&#9003; Back</button>
  <button id=rej>&#10007; Reject (R)</button>
  <button id=acc>&#10003; Accept (A)</button>
  <button id=skip>Skip (S)</button>
 </div>
 <div class=legend>keys: A accept &middot; R reject &middot; S skip &middot; Backspace back<br>
   arrows adjust the deck line: &#8595;/&#8593; move down/up, &#8592;/&#8594; rotate</div>
<script>
let i=0,total=0,mode='deck';
const img=document.getElementById('img');
async function load(){
 const s=await (await fetch('/state')).json();
 i=s.i;total=s.total;mode=s.mode;
 img.src='/overlay?i='+i+'&t='+Date.now();
 document.getElementById('prog').textContent='image '+(i+1)+' / '+total+
   (mode==='sam'?'  [SAM: drag a box — left=superstructure, right=hull; tiny click also works; Z reset — '+s.points+' ops]':'');
 document.getElementById('counts').textContent='accepted '+s.accepted+' | rejected '+s.rejected;
}
async function decide(d){
 await fetch('/decide',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({i:i,decision:d})});
 await load();
}
async function click(fx,fy,label){
 await fetch('/click',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({i:i,fx:fx,fy:fy,label:label})});
 await load();
}
const band=document.getElementById('band');
let drag=null;
function fracXY(cx,cy){const r=img.getBoundingClientRect();
 return [Math.max(0,Math.min(1,(cx-r.left)/r.width)),Math.max(0,Math.min(1,(cy-r.top)/r.height))];}
function showBand(x0,y0,x1,y1){const r=img.getBoundingClientRect();
 band.style.left=(Math.min(x0,x1)-r.left)+'px';band.style.top=(Math.min(y0,y1)-r.top)+'px';
 band.style.width=Math.abs(x1-x0)+'px';band.style.height=Math.abs(y1-y0)+'px';band.style.display='block';}
async function box(fx0,fy0,fx1,fy1,label){
 await fetch('/box',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({i:i,fx0:fx0,fy0:fy0,fx1:fx1,fy1:fy1,label:label})});
 await load();}
img.onmousedown=(e)=>{if(mode!=='sam')return;e.preventDefault();
 drag={x0:e.clientX,y0:e.clientY,btn:e.button};showBand(e.clientX,e.clientY,e.clientX,e.clientY);};
document.onmousemove=(e)=>{if(drag)showBand(drag.x0,drag.y0,e.clientX,e.clientY);};
document.onmouseup=(e)=>{if(!drag)return;const s=drag;drag=null;band.style.display='none';
 const label=(s.btn===2)?0:1;
 const dx=Math.abs(e.clientX-s.x0),dy=Math.abs(e.clientY-s.y0);
 if(dx<5&&dy<5){const[fx,fy]=fracXY(e.clientX,e.clientY);click(fx,fy,label);}
 else{const[fx0,fy0]=fracXY(Math.min(s.x0,e.clientX),Math.min(s.y0,e.clientY));
      const[fx1,fy1]=fracXY(Math.max(s.x0,e.clientX),Math.max(s.y0,e.clientY));
      box(fx0,fy0,fx1,fy1,label);}};
img.oncontextmenu=(e)=>{e.preventDefault();};
document.getElementById('acc').onclick=()=>decide('accept');
document.getElementById('rej').onclick=()=>decide('reject');
document.getElementById('skip').onclick=()=>decide('skip');
document.getElementById('back').onclick=()=>decide('back');
document.getElementById('down').onclick=()=>decide('down');
document.getElementById('up').onclick=()=>decide('up');
document.getElementById('rccw').onclick=()=>decide('rotate_ccw');
document.getElementById('rcw').onclick=()=>decide('rotate_cw');
document.getElementById('clr').onclick=()=>decide('clearpoints');
document.onkeydown=(e)=>{const k=e.key.toLowerCase();
 if(k==='a')decide('accept');else if(k==='r')decide('reject');
 else if(k==='s')decide('skip');else if(k==='z')decide('clearpoints');
 else if(e.key==='Backspace'){e.preventDefault();decide('back');}
 else if(e.key==='ArrowDown'){e.preventDefault();decide('down');}
 else if(e.key==='ArrowUp'){e.preventDefault();decide('up');}
 else if(e.key==='ArrowLeft'){e.preventDefault();decide('rotate_ccw');}
 else if(e.key==='ArrowRight'){e.preventDefault();decide('rotate_cw');}};
load();
</script></body></html>"""


class Reviewer:
    def __init__(self, records, accepted_dir, deck_source, predictions_dir,
                 rotate_step_deg=1.0, label_mode="deck", sam=None):
        self.records = records
        self.deck_source = deck_source
        self.predictions_dir = Path(predictions_dir) if predictions_dir else None
        self.dir = Path(accepted_dir)
        self.labels_dir = self.dir / "labels"
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.accepted_path = self.dir / "accepted.json"
        self.decisions_path = self.dir / "decisions.json"
        self.accepted = self._load(self.accepted_path, [])
        self.decisions = self._load(self.decisions_path, {})
        self.offsets = {}        # rec_id -> integer deck-translation steps
        self.angle_offsets = {}  # rec_id -> integer deck-rotation steps
        self.ANGLE_STEP_DEG = float(rotate_step_deg)
        self.label_mode = label_mode  # "deck" or "sam"
        self.sam = sam
        self.points = {}         # rec_id -> [(px, py, label), ...] (click count)
        self.sam_masks = {}      # rec_id -> accumulated superstructure bool mask
        self.lock = threading.Lock()
        self.i = self._first_undecided()

    @staticmethod
    def _load(path, default):
        return json.loads(path.read_text()) if path.exists() else default

    def _save(self):
        self.accepted_path.write_text(json.dumps(self.accepted, indent=1))
        self.decisions_path.write_text(json.dumps(self.decisions, indent=1))

    def rec_id(self, idx):
        return Path(self.records[idx]["image_path"]).stem

    def _first_undecided(self):
        for idx in range(len(self.records)):
            if self.rec_id(idx) not in self.decisions:
                return idx
        return min(len(self.records) - 1, len(self.decisions))

    def _next_undecided(self, after):
        """Next not-yet-decided image after `after` (wrap once). Skips anything
        already accepted or rejected, so annotated ones never reappear."""
        n = len(self.records)
        for j in range(after + 1, n):
            if self.rec_id(j) not in self.decisions:
                return j
        for j in range(0, after + 1):
            if self.rec_id(j) not in self.decisions:
                return j
        return after

    def _transverse_step(self, rec):
        """5% of the vessel's transverse extent — one nudge unit."""
        full_alpha, _w, _h = full_vessel_alpha(rec)
        ys, xs = np.where(full_alpha > 12)
        t = transverse_coord(xs.astype(float), ys.astype(float),
                             rec["pca_angle_deg"], rec["pca_centroid_x"], rec["pca_centroid_y"])
        return 0.05 * (float(t.max()) - float(t.min()))

    def add_point(self, idx, fx, fy, label):
        """One SAM click segments the structure under the cursor and UNIONs it
        into the accumulated superstructure (left) or subtracts it (right), so
        separate parts — bridge, cranes, masts — build up independently."""
        with self.lock:
            rec = self.records[idx]
            rid = self.rec_id(idx)
            image = Image.open(rec["image_path"])
            width, height = image.size
            px, py = float(fx) * width, float(fy) * height
            full_alpha, _w, _h = full_vessel_alpha(rec)
            vessel = full_alpha > 12
            acc = self.sam_masks.get(rid)
            if acc is None or acc.shape != vessel.shape:
                acc = np.zeros_like(vessel)
            self.sam.set_image(image, key=rid)
            piece = self.sam.predict([(px, py)], [1])  # object under the click
            if piece is not None and piece.shape == vessel.shape:
                piece = piece & vessel
                acc = (acc | piece) if int(label) == 1 else (acc & ~piece)
            self.sam_masks[rid] = acc
            self.points.setdefault(rid, []).append((px, py, int(label)))
            return idx

    def add_box(self, idx, fx0, fy0, fx1, fy1, label):
        """Drag-box prompt: SAM segments the object inside the box; left (label 1)
        assigns it to superstructure, right (label 0) assigns it to hull. Hull is
        always vessel minus superstructure, so the two stay touching & complete."""
        with self.lock:
            rec = self.records[idx]
            rid = self.rec_id(idx)
            image = Image.open(rec["image_path"])
            width, height = image.size
            box = [min(fx0, fx1) * width, min(fy0, fy1) * height,
                   max(fx0, fx1) * width, max(fy0, fy1) * height]
            full_alpha, _w, _h = full_vessel_alpha(rec)
            vessel = full_alpha > 12
            acc = self.sam_masks.get(rid)
            if acc is None or acc.shape != vessel.shape:
                acc = np.zeros_like(vessel)
            self.sam.set_image(image, key=rid)
            piece = self.sam.predict(box=box)
            if piece is not None and piece.shape == vessel.shape:
                piece = piece & vessel
                acc = (acc | piece) if int(label) == 1 else (acc & ~piece)
            self.sam_masks[rid] = acc
            self.points.setdefault(rid, []).append(("box", label))
            return idx

    def region_map(self, idx):
        rec = self.records[idx]
        rid = self.rec_id(idx)
        if self.label_mode == "sam" and self.sam is not None:
            # Start clean (all hull); clicks accumulate the superstructure union
            # in add_point, so this just reads the stored mask (no SAM call).
            full_alpha, _w, _h = full_vessel_alpha(rec)
            vessel = full_alpha > 12
            acc = self.sam_masks.get(rid)
            superstructure = (acc & vessel) if (acc is not None and acc.shape == vessel.shape) \
                else np.zeros_like(vessel)
            return region_map_from_masks(vessel & ~superstructure, superstructure)
        if self.deck_source == "predictions" and self.predictions_dir:
            path = self.predictions_dir / f"{rid}.png"
            if path.exists() and self.offsets.get(rid, 0) == 0 and self.angle_offsets.get(rid, 0) == 0:
                return load_region_png(path)
        # Geometric split, adjustable per image:
        #  * translation: +offset lowers the deck (more superstructure);
        #  * rotation: the deck line tilts off the PCA axis, pivoting around its
        #    anchor so translation stays consistent.
        angle_off = self.angle_offsets.get(rid, 0) * self.ANGLE_STEP_DEG
        phi = float(rec["pca_angle_deg"]) + angle_off
        base_deck = estimate_deck_geometric(rec) * float(np.cos(np.radians(angle_off)))
        deck = base_deck + self.offsets.get(rid, 0) * self._transverse_step(rec)
        hull, superstructure = hull_superstructure_masks(rec, deck=deck, angle=phi)
        return region_map_from_masks(hull, superstructure)

    def overlay_jpeg(self, idx):
        rec = self.records[idx]
        # SAM mode shows only the painted superstructure over a clean image, so
        # Reset leaves the vessel blank (no stray blue hull fill).
        image = overlay_split(rec, region_map=self.region_map(idx),
                              tint_hull=(self.label_mode != "sam"))
        image.thumbnail((900, 900))
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=88)
        return buf.getvalue()

    def decide(self, idx, decision):
        with self.lock:
            rid = self.rec_id(idx)
            if decision == "back":
                self.i = max(0, idx - 1)
                return self.i
            if decision in ("down", "up"):
                # down lowers the deck (more superstructure); up raises it.
                self.offsets[rid] = self.offsets.get(rid, 0) + (1 if decision == "down" else -1)
                return idx  # stay on this image, re-render
            if decision in ("rotate_cw", "rotate_ccw"):
                self.angle_offsets[rid] = self.angle_offsets.get(rid, 0) + (
                    1 if decision == "rotate_cw" else -1)
                return idx  # stay on this image, re-render
            if decision == "clearpoints":
                self.points[rid] = []
                self.sam_masks[rid] = None
                return idx
            if decision == "accept":
                rec = self.records[idx]
                label_path = self.labels_dir / f"{rid}.png"
                save_region_png(np.asarray(self.region_map(idx)), label_path)
                self.accepted = [a for a in self.accepted if a["id"] != rid]
                self.accepted.append({
                    "id": rid, "image_path": rec["image_path"],
                    "label_path": str(label_path), "source": self.deck_source,
                    "label_mode": self.label_mode,
                    "points": len(self.points.get(rid, [])),
                })
                self.decisions[rid] = "accept"
            elif decision == "reject":
                self.accepted = [a for a in self.accepted if a["id"] != rid]
                self.decisions[rid] = "reject"
            # skip: leave undecided, just advance past it
            self._save()
            self.i = self._next_undecided(idx)
            return self.i

    def state(self):
        rejected = sum(1 for v in self.decisions.values() if v == "reject")
        return {"i": self.i, "total": len(self.records), "mode": self.label_mode,
                "points": len(self.points.get(self.rec_id(self.i), [])),
                "accepted": len(self.accepted), "rejected": rejected}


def order_records(records, order, seed=42):
    """Reorder the review queue so accepts spread across classes."""
    if order == "manifest":
        return records
    if order == "shuffle":
        out = list(records)
        random.Random(seed).shuffle(out)
        return out
    # balanced: round-robin one image per class, so every class is reached early.
    groups = defaultdict(list)
    for rec in records:
        groups[rec.get("class")].append(rec)
    for g in groups.values():
        random.Random(seed).shuffle(g)
    out = []
    for tup in itertools.zip_longest(*groups.values()):
        out.extend(rec for rec in tup if rec is not None)
    return out


def make_handler(reviewer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # Client navigated away / canceled an in-flight request (common
                # when a slow CPU overlay is superseded by a new click). Harmless.
                pass

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(200, PAGE.encode(), "text/html")
            elif parsed.path == "/state":
                self._send(200, json.dumps(reviewer.state()).encode(), "application/json")
            elif parsed.path == "/overlay":
                idx = int(parse_qs(parsed.query).get("i", ["0"])[0])
                try:
                    self._send(200, reviewer.overlay_jpeg(idx), "image/jpeg")
                except Exception as exc:
                    self._send(500, str(exc).encode(), "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}") if length else {}
            if path == "/decide":
                nxt = reviewer.decide(int(payload["i"]), payload["decision"])
                self._send(200, json.dumps({"i": nxt}).encode(), "application/json")
            elif path == "/click":
                try:
                    reviewer.add_point(int(payload["i"]), float(payload["fx"]),
                                       float(payload["fy"]), int(payload.get("label", 1)))
                    self._send(200, b"{}", "application/json")
                except Exception as exc:
                    self._send(500, str(exc).encode(), "text/plain")
            elif path == "/box":
                try:
                    reviewer.add_box(int(payload["i"]), float(payload["fx0"]), float(payload["fy0"]),
                                     float(payload["fx1"]), float(payload["fy1"]),
                                     int(payload.get("label", 1)))
                    self._send(200, b"{}", "application/json")
                except Exception as exc:
                    self._send(500, str(exc).encode(), "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--accepted-dir", required=True)
    parser.add_argument("--deck-source", choices=["geometric", "predictions"], default="geometric")
    parser.add_argument("--predictions-dir", default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--rotate-step", type=float, default=1.0,
                        help="degrees per deck-rotation key press (default 1.0)")
    parser.add_argument("--order", choices=["balanced", "shuffle", "manifest"],
                        default="balanced",
                        help="review queue order; balanced = round-robin across classes")
    parser.add_argument("--label-mode", choices=["deck", "sam"], default="deck",
                        help="deck = adjust the straight deck line; sam = click the "
                             "superstructure and SAM2 segments it (freeform labels)")
    parser.add_argument("--sam-device", default="cpu",
                        help="device for SAM2 (default cpu to avoid disturbing a GPU "
                             "training job; use cuda when the GPU is free)")
    args = parser.parse_args()

    data = json.loads(Path(args.manifest).read_text())
    records = [r for r in (data if isinstance(data, list) else data.values())
               if r.get("full_vessel_box_xyxy")]
    records = order_records(records, args.order)
    sam = None
    if args.label_mode == "sam":
        from hullprobe.interactive_sam import InteractiveSAM
        print(f"loading SAM2 for interactive labeling on {args.sam_device} ...")
        sam = InteractiveSAM(device=args.sam_device)
    reviewer = Reviewer(records, args.accepted_dir, args.deck_source, args.predictions_dir,
                        rotate_step_deg=args.rotate_step, label_mode=args.label_mode, sam=sam)
    handler = make_handler(reviewer)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"review {len(records)} images at http://localhost:{args.port}  (source={args.deck_source})")
    print(f"accepted set -> {Path(args.accepted_dir).resolve()}  "
          f"({len(reviewer.accepted)} already accepted, resuming at image {reviewer.i + 1})")
    print("keys: A accept, R reject, S skip, <- back. Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped. accepted={len(reviewer.accepted)}")


if __name__ == "__main__":
    main()
