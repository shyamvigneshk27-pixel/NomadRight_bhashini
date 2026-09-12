#!/usr/bin/env python3
"""
System-wide memory / swap / GPU watcher for load tests on the Jetson.

Samples every second: MemTotal-MemAvailable, MemFree, swap used, per-process
RSS+swap of the interesting processes (bhashini_models, ollama runner,
pocketinfer app, kiosk WebKit, gnome-shell), GPU load and CPU %, and writes
them to a JSON file plus a one-line summary (peak used, peak swap, whether swap
grew during the run) - the evidence for "no swap during normal operation".

usage: memwatch.py --out file.json [--seconds 600] [--label "voice turns"]
"""
import argparse
import json
import os
import re
import time

import psutil

PATTERNS = {
    "bhashini": r"bhashini_models/main\.py",
    "ollama_runner": r"ollama.*runner|llama-server.*--port",
    "ollama": r"ollama serve",
    "app": r"pocketinfer-service|pocketinfer\.service|master\.py",
    "kiosk": r"webview_launcher|WebKitWebProcess|WebKitNetworkProcess",
    "gnome": r"gnome-shell",
}


def meminfo():
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":"); m[k] = int(v.split()[0]) // 1024
    return m


def procs():
    out = {k: {"rss_mb": 0, "swap_mb": 0, "n": 0} for k in PATTERNS}
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
            for k, rx in PATTERNS.items():
                if re.search(rx, cmd):
                    st = open(f"/proc/{p.info['pid']}/status").read()
                    rss = int(re.search(r"VmRSS:\s+(\d+)", st).group(1)) // 1024
                    swp = int(re.search(r"VmSwap:\s+(\d+)", st).group(1)) // 1024
                    out[k]["rss_mb"] += rss; out[k]["swap_mb"] += swp; out[k]["n"] += 1
                    break
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--seconds", type=int, default=600); ap.add_argument("--label", default="")
    ap.add_argument("--interval", type=float, default=1.0)
    a = ap.parse_args()
    jt = None
    try:
        from jtop import jtop
        jt = jtop(); jt.start()
    except Exception:
        jt = None
    psutil.cpu_percent(None)
    samples = []
    t0 = time.time()
    m0 = meminfo()
    swap0 = m0["SwapTotal"] - m0["SwapFree"]
    try:
        while time.time() - t0 < a.seconds:
            m = meminfo()
            s = {"t": round(time.time() - t0, 1), "used_mb": m["MemTotal"] - m["MemAvailable"], "free_mb": m["MemFree"],
                 "swap_mb": m["SwapTotal"] - m["SwapFree"], "cpu_pct": psutil.cpu_percent(None), "procs": procs()}
            if jt is not None and jt.ok():
                try: s["gpu_load"] = float(jt.gpu["gpu"]["status"]["load"])
                except Exception: pass
            samples.append(s)
            time.sleep(a.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if jt is not None: jt.close()
    used = [s["used_mb"] for s in samples]; swap = [s["swap_mb"] for s in samples]
    summary = {"label": a.label, "seconds": round(time.time() - t0), "samples": len(samples), "mem_total_mb": m0["MemTotal"],
               "used_peak_mb": max(used) if used else None, "used_mean_mb": round(sum(used) / len(used)) if used else None,
               "free_min_mb": min(s["free_mb"] for s in samples) if samples else None,
               "swap_start_mb": swap0, "swap_end_mb": swap[-1] if swap else None, "swap_peak_mb": max(swap) if swap else None,
               "swap_grew_mb": (max(swap) - swap0) if swap else None,
               "gpu_load_peak": max((s.get("gpu_load", 0) for s in samples), default=None),
               "proc_peak_rss_mb": {k: max(s["procs"][k]["rss_mb"] for s in samples) for k in PATTERNS} if samples else {},
               "proc_end_swap_mb": {k: samples[-1]["procs"][k]["swap_mb"] for k in PATTERNS} if samples else {}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"summary": summary, "samples": samples}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
