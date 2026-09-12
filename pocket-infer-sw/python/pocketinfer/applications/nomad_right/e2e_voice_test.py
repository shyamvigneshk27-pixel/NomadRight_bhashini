#!/usr/bin/env python3
"""
End-to-end voice test without a microphone: spoken questions (WAV files) go
through exactly the calls app.py makes per turn -

  BHASHINI ASR (/asr) -> NMT to English (/nmt) -> WorkflowController.process()
  -> NMT back to the worker's language -> BHASHINI TTS (/tts)

- and every stage is timed. The question audio is the kiosk-domain set from
bhashini_models/benchmarks (synthesised with the device's own flite voices,
reference text known), so the ASR transcript can be scored (CER) and the
answer checked against what the question is about.

Requires bhashini_models.service on :11400 (and Ollama for the few complex
questions). Usage (repo root):
  python python/pocketinfer/applications/nomad_right/e2e_voice_test.py [--langs ta,hi,en] [--json out.json]
"""
import argparse
import json
import logging
import os
import re
import sys
import time
import unicodedata
import wave
from io import BytesIO

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))
os.chdir(_REPO_ROOT)
logging.basicConfig(level=logging.WARNING)

SETS = os.path.expanduser("~/benchmark_data/asr_sets")

# What each kiosk-domain sentence (bhashini_models/benchmarks/asr_benchmark.py's
# DOMAIN_SENTENCES, same order) must produce: a regex on the ENGLISH answer.
EXPECT = {
    "ta": [r"ration", r"aadhaar|document|card", r"bocw|construction|labour|register|scheme", r"ration|fair price|any",
           r"e-?shram|register|csc|portal", r"work|job|mgnrega|employment|scheme|which", r"wage|labour|complain|shram|contractor|helpline",
           r"kisan|6,?000|farmer", r"job card|mgnrega|100 days|gram|work", r"pension|60|old age|scheme",
           r"free|hospital|ayushman|treatment|health", r"aadhaar", r"ration|wife|family", r"labour|bocw|board|construction",
           r"which|scheme|tell|eligib", r"which|scheme|depends|tell", r"apply|which|scheme", r"helpline|14445|1800|number|which",
           r"chennai|tamil|scheme|which|help", r"housing|pmay|house|awas"],
    "hi": [r"ration", r"aadhaar|document|card", r"bihar|worker|scheme|which|bocw|shram", r"ration|fair price|any",
           r"e-?shram|register|csc|portal", r"wage|labour|complain|shram|contractor|helpline", r"kisan|6,?000|farmer",
           r"job card|mgnrega|100 days|gram|work", r"pension|60|old age|scheme", r"free|hospital|ayushman|treatment|health",
           r"aadhaar", r"which|scheme|tell|eligib", r"which|scheme|depends|tell", r"helpline|14445|1800|number|which",
           r"chennai|scheme|which|help"],
    "en": [r"ration", r"aadhaar|document|card", r"bihar|worker|scheme|which|bocw|shram", r"ration|fair price|any",
           r"e-?shram|register|csc|portal", r"wage|labour|complain|shram|contractor|helpline", r"kisan|6,?000|farmer",
           r"job card|mgnrega|100 days|gram|work", r"pension|60|old age|scheme", r"free|hospital|ayushman|treatment|health",
           r"aadhaar", r"which|scheme|tell|eligib", r"which|scheme|depends|tell", r"helpline|14445|1800|number|which",
           r"chennai|scheme|which|help"],
}


def norm(t):
    t = unicodedata.normalize("NFC", t or "").lower()
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


def cer(ref, hyp):
    r, h = norm(ref).replace(" ", ""), norm(hyp).replace(" ", "")
    prev = list(range(len(h) + 1))
    for i, x in enumerate(r, 1):
        cur = [i]
        for j, y in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] / max(1, len(r))


def wav_seconds(b):
    with wave.open(BytesIO(b)) as w:
        return w.getnframes() / w.getframerate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="ta,hi,en"); ap.add_argument("--json"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from pocketinfer.applications.nomad_right.config import NomadRightConfig
    from pocketinfer.applications.nomad_right.workflow import WorkflowController
    from pocketinfer.applications.nomad_right.bhashini_bridge import BhashiniBridge
    cfg = NomadRightConfig()
    bridge = BhashiniBridge(config=cfg)
    wf = WorkflowController(config=cfg)
    rows = []
    for lang in a.langs.split(","):
        items = [json.loads(l) for l in open(f"{SETS}/domain_{lang}.jsonl", encoding="utf-8") if l.strip()]
        if a.limit:
            items = items[:a.limit]
        for i, it in enumerate(items):
            wav = open(it["wav"], "rb").read()
            # Each clip is a different person at the kiosk (app.py resets on Home).
            if getattr(wf, "scheme_intel", None) is not None:
                wf.scheme_intel.reset_session()
            t = {}
            t0 = time.time()
            try:
                native = bridge.listen(wav, lang); t["asr"] = time.time() - t0
                t1 = time.time(); en = bridge.to_pipeline_language(native, lang); t["nmt_in"] = time.time() - t1
                t2 = time.time(); pkg = wf.process(en, original_query=native, response_language=lang); t["decide"] = time.time() - t2
                t3 = time.time(); ans_native = bridge.from_pipeline_language(pkg.voice_text, lang); t["nmt_out"] = time.time() - t3
                t4 = time.time(); tts = bridge.speak(ans_native, lang); t["tts"] = time.time() - t4
                err = None
            except Exception as exc:
                native, en, pkg, ans_native, tts, err = "", "", None, "", b"", f"{type(exc).__name__}: {str(exc)[:120]}"
            total = time.time() - t0
            answer_en = pkg.voice_text if pkg else ""
            exp = EXPECT.get(lang, [None] * 99)[i] if i < len(EXPECT.get(lang, [])) else None
            problems = []
            if err: problems.append(err)
            c = cer(it["text"], native)
            if c > 0.35: problems.append(f"ASR CER {c:.2f}")
            if not answer_en.strip(): problems.append("empty answer")
            if exp and not re.search(exp, answer_en, re.I): problems.append(f"answer lacks /{exp}/")
            speech_s = wav_seconds(tts) if tts else 0.0
            if speech_s > 20: problems.append(f"TTS {speech_s:.0f}s > 20s")
            if len(answer_en.split()) > 60: problems.append("answer > 60 words")
            rows.append({"lang": lang, "i": i, "ref": it["text"], "asr": native, "cer": round(c, 3), "query_en": en,
                         "answer_en": answer_en, "answer_native": ans_native, "scheme": pkg.scheme_code if pkg else None,
                         "timings_s": {k: round(v, 2) for k, v in t.items()}, "total_s": round(total, 2), "tts_s": round(speech_s, 1),
                         "problems": problems})
            flag = "PASS" if not problems else "FAIL"
            print(f"{flag} [{lang}{i:02d}] {total:5.1f}s (asr {t.get('asr',0):.1f} nmt {t.get('nmt_in',0):.1f} decide {t.get('decide',0):.1f} "
                  f"nmt {t.get('nmt_out',0):.1f} tts {t.get('tts',0):.1f}) cer={c:.2f} -> {pkg.scheme_code if pkg else '-'} | {answer_en[:70]!r}"
                  + (f"\n      !! {'; '.join(problems)}" if problems else ""))
    fails = [r for r in rows if r["problems"]]
    tot = [r["total_s"] for r in rows if not r["problems"] or "CER" in "".join(r["problems"])]
    print(f"\n{len(rows) - len(fails)}/{len(rows)} passed; turn total median {sorted(tot)[len(tot)//2] if tot else '-'} s")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1, ensure_ascii=False)
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
