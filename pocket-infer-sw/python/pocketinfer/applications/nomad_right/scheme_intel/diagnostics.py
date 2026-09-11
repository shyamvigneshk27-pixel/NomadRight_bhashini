"""
Lightweight diagnostic mode - run on demand, never in the background.

    python -m pocketinfer.applications.nomad_right.scheme_intel.diagnostics            # snapshot
    python -m pocketinfer.applications.nomad_right.scheme_intel.diagnostics --selftest # + 6 reference queries
    python -m pocketinfer.applications.nomad_right.scheme_intel.diagnostics --speech   # + NMT/TTS/ASR round trip

Snapshot: memory / swap, RSS of pocketinfer / bhashini / ollama processes,
the Ollama runner (context length, memory), knowledge-base build info and
the live P50/P95/P99 latencies the running app last wrote.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from pocketinfer.applications.nomad_right.scheme_intel import si_constants as C
from pocketinfer.applications.nomad_right.scheme_intel.metrics import LATENCY, load_snapshot

# flite, measured on this device (si_constants.MAX_VOICE_WORDS): seconds per English word
_SEC_PER_WORD = {"hi": 0.392, "ta": 0.437}

REFERENCE_QUERIES = [
    ("T1", "What is One Nation One Ration Card?"),
    ("T2", "I am from Bihar and now working in Tamil Nadu. I have a ration card. Can I get my ration here?"),
    ("T3", "I am a construction worker. What government benefits can I get?"),
    ("T4", "I am 32 years old, from Bihar, working in Chennai as a construction worker. My monthly income is "
           "around 12000. What schemes can help me?"),
    ("T5", "What documents are required for this scheme?"),
    ("T6", "I don't know what government scheme I can get. I am working in Tamil Nadu."),
]


def memory() -> Dict[str, int]:
    vals = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            vals[k] = int(v.split()[0]) // 1024
    return {"total_mb": vals["MemTotal"], "available_mb": vals["MemAvailable"],
            "used_mb": vals["MemTotal"] - vals["MemAvailable"], "swap_used_mb": vals["SwapTotal"] - vals["SwapFree"]}


def processes() -> List[Dict[str, Any]]:
    wanted = ("pocketinfer", "bhashini", "ollama", "webview_launcher")
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="ignore").strip()
            if not any(w in cmd for w in wanted) or "diagnostics" in cmd:
                continue
            rss = swap = 0
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) // 1024
                    elif line.startswith("VmSwap:"):
                        swap = int(line.split()[1]) // 1024
            out.append({"pid": int(pid), "rss_mb": rss, "swap_mb": swap, "cmd": cmd[:90]})
        except OSError:
            continue
    return sorted(out, key=lambda p: -p["rss_mb"])


def ollama() -> Any:
    try:
        import requests
        models = requests.get("http://localhost:11434/api/ps", timeout=3).json().get("models", [])
        return [{"model": m.get("name"), "context": m.get("context_length"),
                 "mem_mb": (m.get("size_vram") or m.get("size") or 0) // 2 ** 20, "until": m.get("expires_at")}
                for m in models]
    except Exception as exc:
        return f"unreachable ({exc.__class__.__name__})"


def kb() -> Dict[str, Any]:
    try:
        with open(C.MANIFEST_PATH, encoding="utf-8") as f:
            m = json.load(f)
        return {"built_at": m.get("built_at"), "counts": m.get("counts"), "warnings": len(m.get("warnings", []))}
    except (OSError, ValueError):
        return {"error": f"{C.MANIFEST_PATH} missing - run build_kb"}


def print_snapshot(as_json: bool) -> Dict[str, Any]:
    snap = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "memory": memory(), "processes": processes(),
            "ollama": ollama(), "knowledge_base": kb(), "latency": load_snapshot()}
    if as_json:
        print(json.dumps(snap, indent=1))
        return snap
    m = snap["memory"]
    print(f"== NomadRight diagnostics {snap['time']}")
    print(f"memory  used {m['used_mb']} MB / {m['total_mb']} MB, available {m['available_mb']} MB, swap used {m['swap_used_mb']} MB")
    for p in snap["processes"]:
        print(f"  pid {p['pid']:>6}  rss {p['rss_mb']:>5} MB  swap {p['swap_mb']:>4} MB  {p['cmd']}")
    print(f"ollama  {snap['ollama']}")
    print(f"kb      {snap['knowledge_base']}")
    lat = snap["latency"]
    if lat:
        age = time.time() - lat.get("written_at", 0)
        print(f"latency (last written by pid {lat.get('pid')}, {age:.0f}s ago) ms:")
        for stage, p in lat.get("stages", {}).items():
            if p.get("n"):
                print(f"  {stage:<16} n={p['n']:<4} p50={p['p50']:<8} p95={p['p95']:<8} p99={p['p99']:<8} max={p['max']}")
    else:
        print("latency no snapshot yet (the app writes one every few turns)")
    return snap


def selftest(use_embedder: bool) -> int:
    from pocketinfer.applications.nomad_right.scheme_intel.service import SchemeIntelligence
    provider = None
    if use_embedder:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        holder = {}

        def provider():
            if "m" not in holder:
                from sentence_transformers import SentenceTransformer
                holder["m"] = SentenceTransformer(C.EMBEDDING_MODEL_NAME, device="cpu")
            return holder["m"]
    si = SchemeIntelligence.create(embedder_provider=provider, qwen_client=None)
    if si is None:
        print("scheme intelligence unavailable (disabled or KB missing)")
        return 1
    print(f"== selftest ({'with' if use_embedder else 'without'} embedder, Qwen off; T1-T5 one conversation, T6 a new visitor)")
    for tid, q in REFERENCE_QUERIES:
        if tid == "T6":
            si.reset_session()
        t0 = time.perf_counter()
        ans = si.handle(q)
        ms = (time.perf_counter() - t0) * 1000
        if ans is None:
            print(f"{tid} {ms:7.1f} ms  DEFER (legacy pipeline)")
            continue
        n = len(ans.voice.split())
        est = " ".join(f"{k}~{v * n:.1f}s" for k, v in _SEC_PER_WORD.items())
        print(f"{tid} {ms:7.1f} ms  {ans.trace['intent']} -> {ans.route.value} {ans.scheme_id or '-'} "
              f"{ans.status}  {n} words ({est})")
        print(f"     voice: {ans.voice}")
        print(f"     lcd  : {ans.top} | {ans.bottom}")
    print("session:", json.dumps(si.session.snapshot()["profile"]))
    return 0


def speech_roundtrip() -> int:
    from pocketinfer.applications.nomad_right.bhashini_bridge import BhashiniBridge
    from pocketinfer.applications.nomad_right.scheme_intel.speech_adapters import (
        ExistingASRAdapter, ExistingNMTAdapter, ExistingTTSAdapter,
    )
    bridge = BhashiniBridge()
    nmt, tts, asr = ExistingNMTAdapter(bridge), ExistingTTSAdapter(bridge), ExistingASRAdapter(bridge)
    sample = "Under One Nation One Ration Card, you can use your ration card at any Fair Price Shop."
    for lang in ("hi", "ta"):
        try:
            native = nmt.from_english(sample, lang)
            wav = tts.synthesize(native, lang)
            heard = asr.transcribe(wav, lang) if wav else ""
            back = nmt.to_english(heard, lang) if heard else ""
            print(f"[{lang}] NMT: {native[:60]}\n     ASR: {heard[:60]}\n     back: {back[:80]}")
        except Exception as exc:
            print(f"[{lang}] speech round trip failed: {exc}")
    for stage, p in LATENCY.snapshot()["stages"].items():
        print(f"  {stage:<12} {p}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="NomadRight diagnostic snapshot")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="run the 6 reference queries through the layer")
    ap.add_argument("--no-embedder", action="store_true", help="selftest without loading e5 (rules only)")
    ap.add_argument("--speech", action="store_true", help="NMT -> TTS -> ASR round trip via bhashini_models")
    args = ap.parse_args(argv)
    # This process only prints its own timings; the running app's live
    # snapshot file (read just below) is never overwritten by a selftest.
    LATENCY._path = None
    print_snapshot(args.json)
    rc = 0
    if args.selftest:
        rc |= selftest(use_embedder=not args.no_embedder)
    if args.speech:
        rc |= speech_roundtrip()
    return rc


if __name__ == "__main__":
    sys.exit(main())
