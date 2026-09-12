#!/usr/bin/env python3
"""
Builds nomadright/scheme_intel/scheme_catalog.json - the list the kiosk's
Schemes page shows (name, one-line summary, category, top benefits, documents,
helpline, website) in English, Hindi and Tamil.

English comes straight from the knowledge base (scheme_intel.db, listed
schemes only). Hindi and Tamil are produced once, here, by the device's own
IndicTrans2 NMT (bhashini_models.service, :11400/nmt) and cached in
scheme_catalog.translations.json so a rebuild only translates new or changed
strings. Nothing is translated at runtime: the page is a static file served by
ui/hdmi/server.py's /api/schemes.

usage (bhashini_models.service must be running for hi/ta):
  python -m pocketinfer.applications.nomad_right.scheme_intel.build_catalog [--langs hi,ta] [--en-only]
"""
import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time

import requests

from pocketinfer.applications.nomad_right.scheme_intel import si_constants as C

CATALOG_PATH = os.path.join(C.KB_DIR, "scheme_catalog.json")
CACHE_PATH = os.path.join(C.KB_DIR, "scheme_catalog.translations.json")
NMT_URL = "http://127.0.0.1:11400/nmt"
MAX_BENEFITS, MAX_DOCUMENTS = 3, 4


def _loads(v, default):
    try:
        out = json.loads(v) if isinstance(v, str) else v
        return out if out is not None else default
    except (TypeError, ValueError):
        return default


def _first_sentence(text: str, max_words: int = 32) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    s = m.group(1) if m else text
    w = s.split()
    return s if len(w) <= max_words else " ".join(w[:max_words]) + "..."


def english_catalog(db_path: str):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    out = []
    for r in con.execute("SELECT * FROM schemes WHERE is_listed = '1' OR is_listed = 1 ORDER BY category, short_name"):
        helpline = re.split(r"\s*/\s*", r["helpline"] or "")[0]
        out.append({
            "id": r["scheme_id"], "short_name": r["short_name"], "name": r["scheme_name"],
            "spoken_name": r["spoken_name"], "category": r["category"] or "Other",
            "summary": r["voice_summary"] or _first_sentence(r["description"]),
            "benefits": [_first_sentence(b, 24) for b in _loads(r["benefits"], [])[:MAX_BENEFITS]],
            "documents": _loads(r["documents"], [])[:MAX_DOCUMENTS],
            "helpline": helpline, "url": r["official_url"] or "",
            "application_mode": r["application_mode"] or "",
        })
    con.close()
    return out


TRANSLATED_FIELDS = ("name", "category", "summary", "benefits", "documents")


def translate_all(en_catalog, langs, cache):
    t0 = time.time()
    hits = misses = 0
    catalog = {"en": en_catalog}
    for lang in langs:
        rows = []
        for e in en_catalog:
            t = dict(e)
            for field in TRANSLATED_FIELDS:
                val = e[field]
                if isinstance(val, list):
                    t[field] = [translate(v, lang, cache) for v in val]
                    misses += sum(1 for v in val if _key(v, lang) not in cache)
                else:
                    t[field] = translate(val, lang, cache)
            rows.append(t)
        catalog[lang] = rows
        print(f"[catalog] {lang}: {len(rows)} schemes translated ({time.time() - t0:.0f}s)", flush=True)
    return catalog


def _key(text, lang):
    return hashlib.sha1(f"{lang}\x00{text}".encode()).hexdigest()


def translate(text: str, lang: str, cache: dict) -> str:
    if not text or not text.strip():
        return text
    k = _key(text, lang)
    if k in cache:
        return cache[k]["text"]
    r = requests.post(NMT_URL, json={"text": text, "src_lang": "EN", "tgt_lang": lang}, timeout=120)
    r.raise_for_status()
    out = (r.json().get("translated_text") or "").strip() or text
    cache[k] = {"lang": lang, "en": text, "text": out}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="hi,ta")
    ap.add_argument("--en-only", action="store_true")
    ap.add_argument("--db", default=C.DB_PATH)
    args = ap.parse_args()
    en = english_catalog(args.db)
    print(f"[catalog] {len(en)} listed schemes from {args.db}")
    cache = json.load(open(CACHE_PATH, encoding="utf-8")) if os.path.exists(CACHE_PATH) else {}
    langs = [] if args.en_only else [l.strip() for l in args.langs.split(",") if l.strip()]
    if langs:
        try:
            requests.get("http://127.0.0.1:11400/health", timeout=3).raise_for_status()
        except Exception as exc:
            print(f"[catalog] bhashini_models not reachable ({exc}); run again with the service up, or --en-only")
            sys.exit(2)
    catalog = translate_all(en, langs, cache)
    catalog["_meta"] = {"built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "schemes": len(en), "languages": ["en"] + langs}
    tmp = CATALOG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=1)
    os.replace(tmp, CATALOG_PATH)
    with open(CACHE_PATH + ".tmp", "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=0)
    os.replace(CACHE_PATH + ".tmp", CACHE_PATH)
    print(f"[catalog] written {CATALOG_PATH} ({os.path.getsize(CATALOG_PATH) // 1024} KB, {len(cache)} cached strings)")


if __name__ == "__main__":
    main()
