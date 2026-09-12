#!/usr/bin/env python3
"""
Repeated assisted-form sessions (fake TTS/ASR answers, real OCR on a fixed photo and
synthetic documents): per-session identification and document-OCR latency, the
process RSS after every session, system free RAM / swap, and a check that no
language-model process appears. usage: form_session_benchmark.py --sessions 10 [--json out]
"""
import argparse, json, os, resource, subprocess, sys, time
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
from pocketinfer.applications.nomad_right.formfill.catalog import FormCatalog
from pocketinfer.applications.nomad_right.formfill.flow import FormFlow
from pocketinfer.applications.nomad_right.formfill import flow_test as FT


def meminfo():
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":"); m[k] = int(v.split()[0]) // 1024
    return {"free_mb": m["MemAvailable"], "swap_used_mb": m["SwapTotal"] - m["SwapFree"]}


def llm_running():
    out = subprocess.run(["pgrep", "-f", "llama-server|ollama runner"], capture_output=True, text=True).stdout.split()
    return len(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sessions", type=int, default=10); ap.add_argument("--json")
    a = ap.parse_args()
    cat = FormCatalog()
    photo = cv2.imread(os.path.expanduser("~/benchmark_data/forms/photos/pmsby_hi__photo1.jpg"))
    rows, m0 = [], meminfo()
    for i in range(1, a.sessions + 1):
        io = FT.FakeIO(FT.PMSBY_ANSWERS_HI, documents=[FT.PASSBOOK, FT.PASSBOOK, FT.AADHAAR_CARD])
        ob = FT.FakeOutbox()
        t0 = time.perf_counter()
        res = FormFlow(cat, io.io(), "hi", outbox=ob).run(photo, prior_scheme_id="SCH_PMSBY")
        secs = time.perf_counter() - t0
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        mi = meminfo()
        rows.append({"session": i, "outcome": res.outcome, "prompts": len(io.spoken), "identify_s": res.timings.get("identify"), "doc_ocr_s": res.timings.get("doc_ocr"),
                     "flow_s": round(secs, 2), "rss_mb": rss, **mi, "llm_procs": llm_running()})
        print(f"[{i:2}] {res.outcome:9} {len(io.spoken):3} prompts identify {res.timings.get('identify')}s doc-ocr {res.timings.get('doc_ocr')}s flow {secs:5.1f}s rss {rss} MB free {mi['free_mb']} swap {mi['swap_used_mb']} llm {llm_running()}", flush=True)
    ok = sum(r["outcome"] == "sent" for r in rows)
    summary = {"sessions": a.sessions, "completed": ok, "rss_first_mb": rows[0]["rss_mb"], "rss_last_mb": rows[-1]["rss_mb"], "rss_growth_mb": rows[-1]["rss_mb"] - rows[min(1, len(rows) - 1)]["rss_mb"],
               "swap_start_mb": m0["swap_used_mb"], "swap_end_mb": rows[-1]["swap_used_mb"], "free_min_mb": min(r["free_mb"] for r in rows),
               "identify_p50_s": sorted(r["identify_s"] or 0 for r in rows)[len(rows) // 2], "doc_ocr_p50_s": sorted(r["doc_ocr_s"] or 0 for r in rows)[len(rows) // 2],
               "llm_procs_max": max(r["llm_procs"] for r in rows)}
    print(json.dumps(summary, indent=1))
    if a.json:
        json.dump({"summary": summary, "rows": rows}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
