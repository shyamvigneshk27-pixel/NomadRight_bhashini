#!/usr/bin/env python3
"""
Qwen3-VL-2B (Q4_K_M) latency / memory benchmark: Ollama vs the same GGUF served
directly by the llama.cpp server Ollama bundles (/usr/local/lib/ollama/llama-server).

Both backends run the identical model files (Ollama's blob store), so every
difference measured here is the serving layer: Ollama fixes the vision
encoder's image_min_pixels at 1,048,576 (every photo -> ~1,230 tokens), while
llama-server exposes --image-min-tokens / --image-max-tokens.

What is measured per request: wall time, prompt tokens, generated tokens,
prompt-eval and generation time (from the backend's own timings), the runner
process RSS, system RAM used, GPU load, and the answer text.

usage:
  qwen_benchmark.py images            # synthesise the test document images
  qwen_benchmark.py ollama  [--tag x] # matrix against the running Ollama service
  qwen_benchmark.py llama   [--min-tokens 256 --ctx 2048 ...]   # starts llama-server itself
  qwen_benchmark.py report
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = f"{HERE}/results"
IMAGES = os.path.expanduser("~/benchmark_data/vision")
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "hf.co/Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M"
BLOBS = "/usr/share/ollama/.ollama/models/blobs"
GGUF = f"{BLOBS}/sha256-089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae"
MMPROJ = f"{BLOBS}/sha256-f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82"
LLAMA_SERVER = "/usr/local/lib/ollama/llama-server"
LLAMA_LIB = "/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_jetpack6"
LLAMA_PORT = 11435
SENTINEL = "[[NO_ANSWER_FOUND]]"

VISION_PROMPT = (
    "You are a helpful visual assistant at an offline welfare-rights kiosk for migrant workers in India. "
    "A worker has pointed the kiosk's camera at something and asked a question about it. Carefully look at the "
    "provided image and answer using ONLY what is visible in it. If the image shows a government form or document, "
    "read the relevant text or field precisely. Keep the answer under 40 words, plain factual sentences, no "
    f"speculation. Only if the image is too blurry, dark, or unclear to make out anything meaningful, or the requested "
    f"detail genuinely isn't visible, reply with exactly this text and nothing else: {SENTINEL}"
)
VISION_PROMPT_SHORT = (
    "You read Indian government documents for a welfare kiosk. Answer from the image only, in at most 20 words. "
    f"If the detail is not visible, reply exactly: {SENTINEL}"
)
DESCRIBE = ("What document is this? Read the issuing state or department, the card type or category, "
            "the holder's name and the card number if visible.")
QUESTIONS = {
    "ration_card": ["What state issued this ration card and what is the card type?", DESCRIBE],
    "eshram_card": ["What is the UAN number on this card?", DESCRIBE],
    "form_table": ["Which fields in this form are still blank?", "What is written in the Name and Age boxes?"],
    "camera_frame": ["Describe what is visible in this photo.", DESCRIBE],
}
TEXT_PROMPT = (
    "You are a factual assistant for a welfare-rights kiosk used by migrant workers in India. Answer the question "
    "using ONLY the information given in the context below. Keep the answer under 40 words, plain factual sentences, "
    f"no speculation. If the context does not contain the answer, reply with exactly this text and nothing else: {SENTINEL}\n\n"
    "Context:\n- About the person: construction worker, from Bihar, now in Chennai, has a BOCW card from Bihar\n"
    "- BOCW registration is with the state welfare board where the worker works; benefits are not automatically "
    "portable between states, though some boards allow transfer of registration.\n"
    "- Under ONORC a ration card holder can draw grains at any ePoS fair price shop in India.\n\n"
    "Question: Will my Bihar labour card work in Chennai?\nAnswer:"
)


# ----------------------------------------------------------------------------- images
def make_images():
    from PIL import Image, ImageDraw, ImageFont
    os.makedirs(IMAGES, exist_ok=True)
    fonts = [f for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf") if os.path.exists(f)]
    def font(sz, bold=False):
        try:
            return ImageFont.truetype(fonts[1 if bold and len(fonts) > 1 else 0], sz)
        except Exception:
            return ImageFont.load_default()

    # 1. Ration card (Bihar, PHH) - photographed at an angle-free close-up, 1280x960
    im = Image.new("RGB", (1280, 960), (236, 232, 214))
    d = ImageDraw.Draw(im)
    d.rectangle([40, 40, 1240, 920], outline=(90, 60, 30), width=6)
    d.text((110, 70), "GOVERNMENT OF BIHAR", fill=(120, 20, 20), font=font(44, True))
    d.text((110, 125), "Food & Consumer Protection Department", fill=(40, 40, 40), font=font(28))
    d.text((110, 170), "RATION CARD (National Food Security Act, 2013)", fill=(40, 40, 40), font=font(30, True))
    rows = [("Card Type", "PHH (Priority Household)"), ("Card No.", "BR 0412 3390 7781"), ("Head of Family", "RAM KUMAR YADAV"),
            ("Father / Husband", "SHYAM YADAV"), ("District", "Gaya"), ("Block", "Sherghati"), ("Members", "5"),
            ("Fair Price Shop", "FPS-1174, Ward 6"), ("Issue Date", "12-03-2022")]
    y = 240
    for k, v in rows:
        d.text((130, y), k, fill=(70, 70, 70), font=font(28))
        d.text((520, y), ":  " + v, fill=(10, 10, 10), font=font(30, True))
        y += 62
    d.rectangle([900, 250, 1180, 560], outline=(60, 60, 60), width=3); d.text((940, 380), "PHOTO", fill=(120, 120, 120), font=font(32))
    d.text((110, 840), "Aadhaar seeded: YES      ONORC portability: ENABLED", fill=(0, 90, 0), font=font(28, True))
    im.save(f"{IMAGES}/ration_card.jpg", quality=90)

    # 2. e-Shram card, 1280x800
    im = Image.new("RGB", (1280, 800), (222, 235, 245)); d = ImageDraw.Draw(im)
    d.rectangle([30, 30, 1250, 770], outline=(20, 60, 120), width=6)
    d.text((80, 60), "e-SHRAM", fill=(20, 60, 120), font=font(56, True)); d.text((80, 130), "Ministry of Labour & Employment, Government of India", fill=(30, 30, 30), font=font(26))
    d.text((80, 220), "UAN : 9142 3378 5610", fill=(0, 0, 0), font=font(46, True))
    d.text((80, 300), "Name : SUNITA DEVI", fill=(0, 0, 0), font=font(38, True)); d.text((80, 360), "Date of Birth : 05-07-1988", fill=(0, 0, 0), font=font(32))
    d.text((80, 420), "Occupation : Domestic worker", fill=(0, 0, 0), font=font(32)); d.text((80, 480), "State : Jharkhand", fill=(0, 0, 0), font=font(32))
    d.rectangle([950, 220, 1200, 520], outline=(60, 60, 60), width=3); d.text((1010, 350), "PHOTO", fill=(120, 120, 120), font=font(30))
    d.text((80, 680), "Helpline 14434 | eshram.gov.in", fill=(20, 60, 120), font=font(28, True))
    im.save(f"{IMAGES}/eshram_card.jpg", quality=90)

    # 3. Application form with a table and blank boxes, 1240x1400 (portrait page)
    im = Image.new("RGB", (1240, 1400), (250, 250, 250)); d = ImageDraw.Draw(im)
    d.text((300, 50), "APPLICATION FORM - BOCW REGISTRATION", fill=(0, 0, 0), font=font(36, True))
    fields = [("1. Name of worker", "RAMESH"), ("2. Age", "34"), ("3. Mobile number", ""), ("4. Aadhaar number", ""),
              ("5. Trade / occupation", "Mason"), ("6. Present address", ""), ("7. Bank account number", "")]
    y = 140
    for k, v in fields:
        d.text((80, y), k, fill=(0, 0, 0), font=font(28)); d.rectangle([520, y - 8, 1160, y + 44], outline=(0, 0, 0), width=2)
        if v: d.text((535, y), v, fill=(20, 20, 120), font=font(30))
        y += 80
    d.text((80, y + 20), "8. Employment in last 12 months", fill=(0, 0, 0), font=font(28, True)); y += 70
    cols = [80, 420, 760, 1160]; hdr = ["Employer", "Site / place", "Days worked"]
    for r in range(4):
        for c in range(3):
            d.rectangle([cols[c], y + r * 60, cols[c + 1], y + (r + 1) * 60], outline=(0, 0, 0), width=2)
            if r == 0: d.text((cols[c] + 12, y + 14), hdr[c], fill=(0, 0, 0), font=font(26, True))
    d.text((92, y + 74), "Sri Balaji Builders", fill=(20, 20, 120), font=font(26)); d.text((432, y + 74), "Porur, Chennai", fill=(20, 20, 120), font=font(26)); d.text((772, y + 74), "96", fill=(20, 20, 120), font=font(26))
    d.text((80, 1200), "Signature / thumb impression: ______________________", fill=(0, 0, 0), font=font(28))
    im.save(f"{IMAGES}/form_table.jpg", quality=90)

    # 4. A real frame from the USB camera, whatever it sees (latency reference for real photos)
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ok, frame = False, None
        for _ in range(15):
            ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            cv2.imwrite(f"{IMAGES}/camera_frame.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print("camera frame", frame.shape)
    except Exception as exc:
        print("camera frame skipped:", exc)
    print("images written to", IMAGES, os.listdir(IMAGES))


def downscale(path, max_side, quality=85):
    import cv2
    img = cv2.imread(path)
    h, w = img.shape[:2]
    s = max_side / float(max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes(), img.shape[1], img.shape[0]


# ----------------------------------------------------------------------------- sampling
def mem_used_mb():
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":"); m[k] = int(v.split()[0]) // 1024
    return m["MemTotal"] - m["MemAvailable"]


def runner_rss_mb(pattern):
    out = subprocess.run(["ps", "-eo", "rss,args"], capture_output=True, text=True).stdout
    tot = 0
    for line in out.splitlines()[1:]:
        rss, _, args = line.strip().partition(" ")
        if re.search(pattern, args):
            tot += int(rss) // 1024
    return tot


class GpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.stop = threading.Event(); self.loads = []
    def run(self):
        try:
            from jtop import jtop
            with jtop() as j:
                while not self.stop.is_set() and j.ok():
                    try: self.loads.append(float(j.gpu["gpu"]["status"]["load"]))
                    except Exception: pass
                    self.stop.wait(0.5)
        except Exception:
            pass


# ----------------------------------------------------------------------------- backends
def ollama_generate(prompt, image_b64, num_predict, num_ctx, temperature=0.2, keep_alive="30m"):
    payload = {"model": MODEL, "prompt": prompt, "stream": False, "keep_alive": keep_alive,
               "options": {"num_gpu": 999, "num_thread": 6, "num_ctx": num_ctx, "num_predict": num_predict, "temperature": temperature}}
    if image_b64:
        payload["images"] = [image_b64]
    t0 = time.time()
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
    wall = time.time() - t0
    r.raise_for_status(); d = r.json()
    return {"wall_s": round(wall, 3), "text": (d.get("response") or "").strip(), "prompt_tokens": d.get("prompt_eval_count"),
            "gen_tokens": d.get("eval_count"), "prompt_ms": round((d.get("prompt_eval_duration") or 0) / 1e6),
            "gen_ms": round((d.get("eval_duration") or 0) / 1e6), "load_ms": round((d.get("load_duration") or 0) / 1e6),
            "total_ms": round((d.get("total_duration") or 0) / 1e6)}


class LlamaServer:
    def __init__(self, ctx=2048, min_tokens=None, max_tokens=None, flash="auto", ctk=None, ctv=None, threads=6, extra=()):
        self.args = [LLAMA_SERVER, "-m", GGUF, "--mmproj", MMPROJ, "-ngl", "99", "-c", str(ctx), "-t", str(threads),
                     "--port", str(LLAMA_PORT), "--host", "127.0.0.1", "-np", "1", "--no-webui", "-fa", flash]
        if min_tokens: self.args += ["--image-min-tokens", str(min_tokens)]
        if max_tokens: self.args += ["--image-max-tokens", str(max_tokens)]
        if ctk: self.args += ["-ctk", ctk]
        if ctv: self.args += ["-ctv", ctv]
        self.args += list(extra)
        self.proc = None

    def __enter__(self):
        # ggml discovers its backends (libggml-cuda.so lives in cuda_jetpack6/) only through
        # GGML_BACKEND_PATH - with LD_LIBRARY_PATH alone the server silently runs on the CPU
        # (measured 2026-09-12: 336 % CPU, 10 tok/s, no GPU load).
        env = {**os.environ, "LD_LIBRARY_PATH": LLAMA_LIB, "GGML_BACKEND_PATH": LLAMA_LIB.split(":")[-1] + "/libggml-cuda.so"}
        self.log = open(f"{RESULTS}/llama_server.log", "ab")
        self.proc = subprocess.Popen(self.args, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 240:
            try:
                if requests.get(f"http://127.0.0.1:{LLAMA_PORT}/health", timeout=1).status_code == 200:
                    self.load_s = round(time.time() - t0, 1); return self
            except Exception:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode}; see {RESULTS}/llama_server.log")
            time.sleep(0.5)
        raise RuntimeError("llama-server did not become healthy in 240 s")

    def __exit__(self, *a):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try: self.proc.wait(10)
            except subprocess.TimeoutExpired: self.proc.kill()
        self.log.close()

    def generate(self, prompt, image_b64, num_predict, temperature=0.2):
        content = [{"type": "text", "text": prompt}]
        if image_b64:
            content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        payload = {"messages": [{"role": "user", "content": content}], "max_tokens": num_predict, "temperature": temperature,
                   "stream": False, "timings_per_token": False}
        t0 = time.time()
        r = requests.post(f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions", json=payload, timeout=180)
        wall = time.time() - t0
        r.raise_for_status(); d = r.json()
        tm = d.get("timings") or {}
        return {"wall_s": round(wall, 3), "text": (d["choices"][0]["message"]["content"] or "").strip(),
                "prompt_tokens": tm.get("prompt_n", d.get("usage", {}).get("prompt_tokens")),
                "gen_tokens": tm.get("predicted_n", d.get("usage", {}).get("completion_tokens")),
                "prompt_ms": round(tm.get("prompt_ms", 0)), "gen_ms": round(tm.get("predicted_ms", 0))}


# ----------------------------------------------------------------------------- matrix
def cases(sides, predicts, prompts):
    for img in ("ration_card", "eshram_card", "form_table", "camera_frame"):
        path = f"{IMAGES}/{img}.jpg"
        if not os.path.exists(path):
            continue
        for side in sides:
            for np_ in predicts:
                for pname in prompts:
                    for q in QUESTIONS[img][:1] if pname == "short" else QUESTIONS[img]:
                        yield img, path, side, np_, pname, q


def run_matrix(gen, backend_name, args, runner_pattern):
    os.makedirs(RESULTS, exist_ok=True)
    sides = [int(s) for s in args.sides.split(",")]
    predicts = [int(p) for p in args.predicts.split(",")]
    prompts = args.prompts.split(",")
    rows = []
    # text-only reference (the COMPLEX-question path)
    for rep in range(2):
        gs = GpuSampler(); gs.start(); m0 = mem_used_mb()
        try:
            r = gen(TEXT_PROMPT, None, 60)
        except Exception as exc:
            r = {"error": str(exc)[:200], "wall_s": None}
        gs.stop.set(); gs.join(1)
        rows.append({"backend": backend_name, "kind": "text", "rep": rep, **r, "sys_used_mb": mem_used_mb(), "sys_delta_mb": mem_used_mb() - m0,
                     "runner_rss_mb": runner_rss_mb(runner_pattern), "gpu_load_mean": round(float(np.mean(gs.loads)), 1) if gs.loads else None})
        print(f"[{backend_name}] text rep{rep}: {r.get('wall_s')}s tok={r.get('gen_tokens')} | {str(r.get('text'))[:80]!r}")
    for img, path, side, np_, pname, q in cases(sides, predicts, prompts):
        jpg, w, h = downscale(path, side)
        b64 = base64.b64encode(jpg).decode()
        prompt = (VISION_PROMPT if pname == "full" else VISION_PROMPT_SHORT) + f"\n\nQuestion: {q}\nAnswer:"
        for rep in range(args.reps):
            gs = GpuSampler(); gs.start(); m0 = mem_used_mb()
            try:
                r = gen(prompt, b64, np_)
            except Exception as exc:
                r = {"error": str(exc)[:200], "wall_s": None}
            gs.stop.set(); gs.join(1)
            row = {"backend": backend_name, "kind": "vision", "image": img, "side": side, "w": w, "h": h, "jpeg_kb": len(jpg) // 1024,
                   "num_predict": np_, "prompt": pname, "question": q, "rep": rep, **r, "sys_used_mb": mem_used_mb(),
                   "sys_delta_mb": mem_used_mb() - m0, "runner_rss_mb": runner_rss_mb(runner_pattern),
                   "gpu_load_mean": round(float(np.mean(gs.loads)), 1) if gs.loads else None}
            rows.append(row)
            print(f"[{backend_name}] {img} {w}x{h} np={np_} {pname} rep{rep}: {r.get('wall_s')}s prompt={r.get('prompt_tokens')} "
                  f"gen={r.get('gen_tokens')} | {str(r.get('text'))[:70]!r}")
    return rows


def cmd_ollama(args):
    # make sure the model is loaded exactly like the app loads it (same runner options)
    ollama_generate("hi", None, 1, args.ctx)
    time.sleep(1)
    rows = run_matrix(lambda p, i, n: ollama_generate(p, i, n, args.ctx), "ollama", args, r"ollama.*runner|llama-server")
    out = f"{RESULTS}/qwen_ollama{('_' + args.tag) if args.tag else ''}.json"
    json.dump({"config": {"num_ctx": args.ctx}, "rows": rows}, open(out, "w"), indent=1)
    print("written", out)


def cmd_llama(args):
    with LlamaServer(ctx=args.ctx, min_tokens=args.min_tokens, max_tokens=args.max_tokens, flash=args.flash,
                     ctk=args.ctk, ctv=args.ctv) as srv:
        print(f"llama-server up in {srv.load_s}s")
        srv.generate("hi", None, 1)
        rows = run_matrix(srv.generate, "llama-server", args, r"llama-server.*--port " + str(LLAMA_PORT))
        for r in rows:
            r["load_s"] = srv.load_s
    tag = args.tag or f"min{args.min_tokens or 'def'}_ctx{args.ctx}_fa{args.flash}" + (f"_{args.ctk}" if args.ctk else "")
    out = f"{RESULTS}/qwen_llama_{tag}.json"
    json.dump({"config": {k: v for k, v in vars(args).items() if k != "fn"}, "rows": rows}, open(out, "w"), indent=1)
    print("written", out)


def cmd_report(args):
    files = sorted(f for f in os.listdir(RESULTS) if f.startswith("qwen_") and f.endswith(".json"))
    print("file | kind | n | wall p50 | wall max | prompt tok | gen tok | prompt ms | gen ms | runner rss | sys used | gpu")
    for f in files:
        d = json.load(open(f"{RESULTS}/{f}"))
        for kind in ("text", "vision"):
            rows = [r for r in d["rows"] if r["kind"] == kind and r.get("wall_s")]
            if not rows: continue
            def med(k):
                v = [r[k] for r in rows if r.get(k) is not None]; return round(float(np.median(v)), 1) if v else "-"
            print(f"{f} | {kind} | {len(rows)} | {med('wall_s')} | {max(r['wall_s'] for r in rows)} | {med('prompt_tokens')} | {med('gen_tokens')} | "
                  f"{med('prompt_ms')} | {med('gen_ms')} | {med('runner_rss_mb')} | {med('sys_used_mb')} | {med('gpu_load_mean')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("images").set_defaults(fn=lambda a: make_images())
    for name, fn in (("ollama", cmd_ollama), ("llama", cmd_llama)):
        p = sub.add_parser(name)
        p.add_argument("--sides", default="1024,768,640,512"); p.add_argument("--predicts", default="100,40")
        p.add_argument("--prompts", default="full,short"); p.add_argument("--reps", type=int, default=2)
        p.add_argument("--ctx", type=int, default=2048); p.add_argument("--tag", default="")
        if name == "llama":
            p.add_argument("--min-tokens", type=int, default=0); p.add_argument("--max-tokens", type=int, default=0)
            p.add_argument("--flash", default="auto"); p.add_argument("--ctk", default=None); p.add_argument("--ctv", default=None)
        p.set_defaults(fn=fn)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = ap.parse_args(); a.fn(a)
