#!/usr/bin/env python3
"""
Form identification benchmark: every synthetic photo (make_form_photos.py) through
the real pipeline - document crop -> header-band OCR (English, then the selected
language's script when needed) -> FormIdentifier - with per-image timings, the
tesseract child's peak RSS, and the decision. usage: form_id_benchmark.py [--json out]
"""
import argparse, json, os, resource, statistics as st, sys, time
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
from pocketinfer.applications.nomad_right.formfill import ocr as ocrmod
from pocketinfer.applications.nomad_right.formfill.catalog import FormCatalog
from pocketinfer.applications.nomad_right.formfill.identifier import FormIdentifier, ACCEPT_MIN

PHOTOS = os.path.expanduser("~/benchmark_data/forms/photos")
EXCLUDED = ("apy_ta__", "pmsby_ta__")   # sources whose PDF text is unreadable even before OCR


def identify_image(img, lang, ident, prior=None):
    """The flow's actual strategy: English pass first; add the selected language's
    script only when English alone is not conclusive."""
    t0 = time.perf_counter()
    passes = ocrmod.read_header(img, langs=("en",))
    res = ident.identify(passes, prior)
    if res.status != "accept" and lang in ("hi", "ta"):
        passes.update(ocrmod.read_header(img, langs=(lang,)))
        res = ident.identify(passes, prior)
    if res.status != "accept" and (res.status in ("unsupported", "not_readable") or (res.candidates and res.candidates[0].score < 0.6)):
        # titles printed below the header band (APY, the PMJJBY enrolment template): one full-page pass
        passes["en_full"] = ocrmod.read_document(img, "en", frac=0.7, max_width=1400)
        res = ident.identify(passes, prior)
    return res, passes, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); ap.add_argument("--prior", action="store_true", help="give the identifier the true scheme as prior")
    a = ap.parse_args()
    cat = FormCatalog(); ident = FormIdentifier(cat)
    manifest = json.load(open(f"{PHOTOS}/manifest.json"))
    rows = []
    for m in manifest:
        img = cv2.imread(f"{PHOTOS}/{m['file']}")
        prior = cat.forms[m["expected"]]["scheme_id"] if (a.prior and m["expected"]) else None
        r0 = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        res, passes, secs = identify_image(img, m["lang"], ident, prior)
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024
        top = res.candidates[0] if res.candidates else None
        correct = (res.best == m["expected"]) if m["expected"] else (res.status in ("unsupported", "not_readable"))
        if m["file"].startswith(EXCLUDED):
            outcome = "EXCLUDED"          # the source PDF itself renders its Tamil text as garbage (fonts) - not the pipeline's fault
        elif m["expected"]:
            outcome = ("OK-accept" if res.status == "accept" and correct else "OK-confirm" if res.status in ("confirm", "choose") and top and top.form_id == m["expected"]
                       else "MISS" if res.status in ("unsupported", "not_readable") else "WRONG")
        else:   # a negative may be rejected or asked about - only a straight accept is wrong
            outcome = "OK-reject" if res.status in ("unsupported", "not_readable") else "OK-asked" if res.status in ("confirm", "choose") else "WRONG"
        rows.append({"file": m["file"], "expected": m["expected"], "variant": m["variant"], "lang": m["lang"], "status": res.status, "best": res.best,
                     "top": top.form_id if top else None, "score": top.score if top else 0, "margin": round(res.margin, 3), "words": res.words_read,
                     "passes": list(passes), "ocr_s": round(sum(p.seconds for p in passes.values()), 2), "band_words": res.words_read, "total_s": round(secs, 2), "child_rss_mb": rss, "outcome": outcome,
                     "evidence": top.evidence[:6] if top else []})
        print(f"{outcome:10} {m['file']:34} {res.status:12} best={res.best or '-':30} score={top.score if top else 0:.2f} margin={res.margin:.2f} words={res.words_read:3} {secs:5.2f}s passes={'+'.join(passes)}", flush=True)
    excluded = [r for r in rows if r["outcome"] == "EXCLUDED"]
    pos = [r for r in rows if r["expected"] and r["outcome"] != "EXCLUDED"]; neg = [r for r in rows if not r["expected"]]
    ok_pos = sum(r["outcome"].startswith("OK") for r in pos); wrong = sum(r["outcome"] == "WRONG" for r in rows)
    lat = sorted(r["total_s"] for r in rows)
    summary = {"images": len(rows), "excluded_bad_sources": len(excluded), "positives": len(pos), "positives_ok": ok_pos, "accepted_directly": sum(r["outcome"] == "OK-accept" for r in pos),
               "negatives": len(neg), "negatives_ok": sum(r["outcome"].startswith("OK") for r in neg), "negatives_asked": sum(r["outcome"] == "OK-asked" for r in neg), "wrong": wrong,
               "latency_p50_s": lat[len(lat) // 2], "latency_p95_s": lat[int(len(lat) * 0.95)], "child_rss_peak_mb": max(r["child_rss_mb"] for r in rows)}
    print(json.dumps(summary, indent=1))
    if a.json:
        json.dump({"summary": summary, "rows": rows}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
