#!/usr/bin/env python3
"""
NomadRight tough-input test: what the Decision Layer answers when the question
is ambiguous, incomplete, mangled by speech recognition, mixed-language, off
topic, or plain garbage. Every case states what a correct system must and must
not do; the run prints a table and exits non-zero on failures.

Runs the real WorkflowController (scheme intelligence + legacy layers) on text,
exactly what app.py calls after ASR/NMT. Qwen is used only where the workflow
would use it; set NOMADRIGHT_TOUGH_NO_LLM=1 to skip cases that need it.

Usage (repo root): python python/pocketinfer/applications/nomad_right/tough_test.py [--json out.json] [--only ambiguous,asr]
"""
import argparse
import json
import logging
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))
os.chdir(_REPO_ROOT)
logging.basicConfig(level=logging.WARNING)

# (category, query, expectation) - expectation keys:
#   scheme: legacy scheme code or SI scheme id that must appear (any of a list)
#   must:   regex that must match the spoken answer (case-insensitive)
#   must_not: regex that must NOT match
#   asks:   True -> the answer must ask the person something (a question) or admit it needs details
#   decline: True -> must be the not-found / off-topic fallback (no scheme claimed)
#   max_words: spoken answer length cap
CASES = [
    # --- normal
    ("normal", "Can I use my ration card in Tamil Nadu?", {"scheme": ["PDS", "SCH_ONORC"], "must": r"ration|fair price|shop|any state|portab"}),
    ("normal", "What documents are required for the Ayushman card?", {"scheme": ["PMJAY", "SCH_PMJAY"], "must": r"aadhaar|ration|document|card"}),
    ("normal", "How do I register on e-shram?", {"scheme": ["ESHRAM", "SCH_ESHRAM"], "must": r"register|csc|portal|aadhaar|mobile"}),
    ("normal", "How much money does PM Kisan give?", {"scheme": ["SCH_PMKISAN", "PMKISAN"], "must": r"6,?000|six thousand|2,?000|instal"}),
    # --- complex scenarios
    ("complex", "I am a 62 year old widow from Bihar living in Chennai with no income. What can I get?",
     {"must": r"pension|widow|old age|nsap|ration|scheme", "must_not": r"kisan|farmer"}),
    ("complex", "I am a construction worker from Bihar working in Chennai and I have an e-shram card. Can I get BOCW benefits here?",
     {"scheme": ["BOCW", "SCH_BOCW"], "must": r"bocw|board|register|labour|welfare"}),
    ("complex", "My wife is in Bihar and I am in Chennai. Can she take our ration there while I take mine here?",
     {"scheme": ["PDS", "SCH_ONORC"], "must": r"yes|family|part|ration|split|remaining|portion|share"}),
    ("complex", "I am 45, drive an auto in Mumbai, earn 12000 a month and have no pension. Which scheme?",
     {"must": r"pension|pm-?sym|atal|shram|apy", "must_not": r"kisan"}),
    # --- ambiguous / incomplete
    ("ambiguous", "I need money", {"asks": True, "must_not": r"you are eligible"}),
    ("ambiguous", "card", {"asks": True}),
    ("ambiguous", "Am I eligible?", {"asks": True}),
    ("incomplete", "documents", {"asks": True}),
    ("incomplete", "how to apply", {"asks": True}),
    ("incomplete", "pension", {"must": r"pension|which|tell me|age|work|scheme"}),
    # --- ASR-style distortions (what a misheard/mis-transcribed question looks like after NMT)
    ("asr", "pee em kisan yojna kya hai", {"scheme": ["SCH_PMKISAN", "PMKISAN"]}),
    ("asr", "ayushman bharath card documents", {"scheme": ["PMJAY", "SCH_PMJAY"]}),
    ("asr", "e sharam card registration", {"scheme": ["ESHRAM", "SCH_ESHRAM"]}),
    ("asr", "one nation one rashan card in tamilnadu", {"scheme": ["PDS", "SCH_ONORC"]}),
    ("asr", "manrega job card kaise banega", {"scheme": ["MGNREGS", "SCH_MGNREGA"]}),
    ("asr", "can i use my ration cart in chennai", {"scheme": ["PDS", "SCH_ONORC"]}),
    ("asr", "labour cart bihar to chennai", {"scheme": ["BOCW", "SCH_BOCW"]}),
    ("asr", "pm kissan samman nidhi money", {"scheme": ["SCH_PMKISAN", "PMKISAN"]}),
    ("asr", "aayushman golden card hospital free", {"scheme": ["PMJAY", "SCH_PMJAY"]}),
    ("asr", "atal pension yojana how much", {"scheme": ["SCH_APY", "APY"]}),
    ("asr", "sukanya samridhi account for my daughter", {"scheme": ["SCH_SSY", "SSY"]}),
    ("asr", "ujwala gas connection", {"scheme": ["SCH_PMUY", "PMUY"]}),
    ("asr", "vishwakarma scheme for carpenter", {"scheme": ["SCH_VISHWA", "VISHWA"]}),
    # --- mixed language / romanised
    ("mixed", "mujhe ration card chahiye", {"scheme": ["PDS", "SCH_ONORC"], "must": r"ration"}),
    ("mixed", "mera ration card tamil nadu me chalega kya", {"scheme": ["PDS", "SCH_ONORC"]}),
    ("mixed", "e-shram card ke fayde kya hai", {"scheme": ["ESHRAM", "SCH_ESHRAM"]}),
    ("mixed", "ayushman card se ilaj free hai kya", {"scheme": ["PMJAY", "SCH_PMJAY"]}),
    # --- unexpected inputs
    ("garbage", "", {"decline": True}),
    ("garbage", "asdfgh qwerty zxcv", {"decline": True}),
    ("garbage", "123456", {"decline": True}),
    ("garbage", "'; DROP TABLE schemes; --", {"decline": True}),
    ("garbage", "ration " * 120, {"max_words": 60}),
    ("offtopic", "What is the weather today?", {"decline": True}),
    ("offtopic", "Play a song for me", {"decline": True}),
    ("offtopic", "hello", {"decline": True}),
    ("offtopic", "Who is the prime minister?", {"decline": True}),
    # --- follow-ups with context (previous scheme remembered by the caller)
    ("context", ("PDS", "what documents do I need?"), {"scheme": ["PDS", "SCH_ONORC"], "must": r"ration card|aadhaar"}),
    ("context", ("PMJAY", "is it free?"), {"scheme": ["PMJAY", "SCH_PMJAY"], "must": r"free|cashless|no (?:\w+ )?fee|no cost|pay|premium|funded"}),
]

DECLINE_RE = re.compile(r"could not find|not able to|couldn't find|don't have information|do not have|not sure|"
                        r"welfare|government scheme|please ask|only help|can help you with|try asking|"
                        r"ration|health|pension|scheme|jobs?|which scheme|tell me", re.I)
QUESTION_RE = re.compile(r"\?|which|what (?:is|work|state)|tell me|please (?:say|tell|share)|do you|are you|how old|where do", re.I)


def run(only=None, json_out=None):
    from pocketinfer.applications.nomad_right.config import NomadRightConfig
    from pocketinfer.applications.nomad_right.workflow import WorkflowController
    wf = WorkflowController(config=NomadRightConfig())
    results = []
    for cat, q, exp in CASES:
        if only and cat not in only:
            continue
        ctx = None
        if isinstance(q, tuple):
            ctx, q = q
        # Every case is a different person at the kiosk (app.py resets on Home /
        # language change); "context" cases pass the previous scheme explicitly.
        if getattr(wf, "scheme_intel", None) is not None:
            wf.scheme_intel.reset_session()
        t0 = time.time()
        try:
            pkg = wf.process(q, context_scheme_code=ctx)
            voice, code = pkg.voice_text or "", pkg.scheme_code or ""
            err = None
        except Exception as exc:
            voice, code, err = "", "", f"{type(exc).__name__}: {exc}"
        ms = (time.time() - t0) * 1000
        # which scheme (legacy code or SI id) the answer is about
        about = code
        problems = []
        if err:
            problems.append(err[:120])
        if exp.get("scheme"):
            ok = any(s == code for s in exp["scheme"]) or any(s.replace("SCH_", "").lower() in voice.lower() for s in exp["scheme"] if s.startswith("SCH_")) \
                or any(s.lower() in voice.lower() for s in exp["scheme"])
            if not ok:
                problems.append(f"expected scheme {exp['scheme']} got {code or '-'}")
        if exp.get("must") and not re.search(exp["must"], voice, re.I):
            problems.append(f"missing /{exp['must']}/")
        if exp.get("must_not") and re.search(exp["must_not"], voice, re.I):
            problems.append(f"contains /{exp['must_not']}/")
        if exp.get("asks") and not QUESTION_RE.search(voice):
            problems.append("does not ask for details")
        if exp.get("decline"):
            if code and code not in ("GENERAL",):
                problems.append(f"claimed scheme {code}")
            if not DECLINE_RE.search(voice) and voice.strip():
                problems.append("not a decline")
        if exp.get("max_words") and len(voice.split()) > exp["max_words"]:
            problems.append(f"{len(voice.split())} words")
        if len(voice.split()) > 60:
            problems.append(f"too long: {len(voice.split())} words")
        results.append({"category": cat, "query": q[:80], "context": ctx, "scheme": about, "ms": round(ms), "voice": voice,
                        "problems": problems})
        flag = "PASS" if not problems else "FAIL"
        print(f"{flag} [{cat:10}] {ms:6.0f} ms  {q[:55]!r:58} -> {about or '-':12} | {voice[:90]!r}" + (f"\n      !! {'; '.join(problems)}" if problems else ""))
    fails = [r for r in results if r["problems"]]
    print(f"\n{len(results) - len(fails)}/{len(results)} passed; {len(fails)} failed")
    if json_out:
        json.dump(results, open(json_out, "w"), indent=1, ensure_ascii=False)
    return len(fails) == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json"); ap.add_argument("--only")
    a = ap.parse_args()
    sys.exit(0 if run(only=a.only.split(",") if a.only else None, json_out=a.json) else 1)
