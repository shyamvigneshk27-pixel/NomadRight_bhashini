"""
Scheme-intelligence layer constants.

Everything tunable for the intent / scenario / rules / RAG layer lives here,
kept apart from the legacy constants.py so the existing Decision Layer's
behaviour cannot change by accident when these are tuned.
"""

import os

from pocketinfer.applications.nomad_right import constants as nr_constants


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


# ============================================================================
# Kill switch
# ============================================================================
#
# True: WorkflowController.process() asks the scheme-intelligence layer first
# and only falls back to the legacy Intent/Rules/RAG/Qwen pipeline when this
# layer declines (returns None) or raises.
# False (or env NOMADRIGHT_SCHEME_INTEL=0): the legacy pipeline runs exactly
# as it did before this layer existed - nothing new is even loaded.
SCHEME_INTEL_ENABLED = _env_flag("NOMADRIGHT_SCHEME_INTEL", True)


# ============================================================================
# Knowledge-base locations (built offline by build_kb.py)
# ============================================================================

KB_DIR = os.path.join(nr_constants._REPO_ROOT, "nomadright", "scheme_intel")
SOURCES_DIR = os.path.join(KB_DIR, "sources")
CANONICAL_DIR = os.path.join(SOURCES_DIR, "gov_scheme_qa", "data")
INTENT_DATASET_PATH = os.path.join(SOURCES_DIR, "gov_scheme_qa", "dataset", "intent_dataset.json")
CURATED_OVERLAY_PATH = os.path.join(SOURCES_DIR, "curated_overlay.json")
INTENT_EXAMPLES_PATH = os.path.join(SOURCES_DIR, "intent_examples.json")

# The existing scheme JSON files (pds.json, pmjay.json, ...). Read-only.
LEGACY_SCHEME_DIR = os.path.join(nr_constants._REPO_ROOT, "nomadright")

DB_PATH = os.path.join(KB_DIR, "scheme_intel.db")
RAG_EMBEDDINGS_PATH = os.path.join(KB_DIR, "rag_embeddings.npy")
INTENT_PROTOTYPES_PATH = os.path.join(KB_DIR, "intent_prototypes.npz")
MANIFEST_PATH = os.path.join(KB_DIR, "manifest.json")
FAQ_QUESTIONS_PATH = os.path.join(KB_DIR, "faq_questions.npz")

# Same embedder the legacy RAGRetriever already loads - shared, never loaded twice.
EMBEDDING_MODEL_NAME = nr_constants.EMBEDDING_MODEL_NAME
QUERY_PREFIX = nr_constants.RAG_QUERY_PREFIX
PASSAGE_PREFIX = nr_constants.RAG_PASSAGE_PREFIX


# ============================================================================
# Spoken-answer budget
# ============================================================================
#
# Requirement: every answer must be spoken by TTS within 20 seconds.
# Measured on this device (flite, the exact voices/flags bhashini_models uses,
# 2026-09-11): a 38-word English answer became 14.9 s of Hindi speech
# (0.392 s per English word) and 16.6 s of Tamil speech (0.437 s per English
# word). Tamil is the slower of the two: 20 s / 0.437 = 45 English words.
# 40 leaves ~10% headroom for NMT output that is wordier than the reference.
MAX_VOICE_WORDS = 40

# LCD/UI line limits - identical to the legacy ones.
DISPLAY_TOP_MAX = nr_constants.DISPLAY_HEADER_MAX_LEN
DISPLAY_BOTTOM_MAX = nr_constants.DISPLAY_BODY_MAX_LEN


# ============================================================================
# Session
# ============================================================================

# A kiosk serves one person after another. Facts are forgotten after this
# much inactivity so the next visitor never inherits the last one's profile.
SESSION_TTL_S = 300.0  # 5 min idle at a kiosk = a different person; Home / language change reset immediately


# ============================================================================
# Retrieval
# ============================================================================

RAG_TOP_K = 3
# multilingual-e5 cosine scores are compressed into a narrow band; these were
# tuned on this knowledge base (see scheme_intel_test.py / diagnostics).
RAG_MIN_SCORE = 0.82
# An FAQ chunk this close to the question is answered directly (no LLM).
RAG_DIRECT_ANSWER_SCORE = 0.90
# Question-to-question similarity (faq_questions.npz) at which a knowledge-base
# FAQ is read out as it is instead of the overview / Qwen. Calibrated 2026-09-11:
# rewordings of an FAQ scored 0.866-0.981, questions no FAQ answers 0.818-0.914,
# and two different FAQ questions up to 0.957 - so only near-verbatim rewordings
# pass ("When does the Sukanya Samriddhi account mature?" 0.938).
FAQ_MATCH_MIN_SCORE = 0.93
QUERY_EMBED_CACHE_SIZE = 64


# ============================================================================
# Intent classifier (second stage, after the rules)
# ============================================================================

INTENT_KNN_K = 7
INTENT_MIN_SIMILARITY = 0.86
INTENT_MIN_VOTE_SHARE = 0.55


# ============================================================================
# Qwen (only for COMPLEX_SCENARIO and vision)
# ============================================================================

LLM_COMPLEX_ENABLED = True
LLM_CONTEXT_MAX_SNIPPETS = 4
LLM_SNIPPET_MAX_WORDS = 45


# ============================================================================
# Vision
# ============================================================================
#
# Measured 2026-09-11 against the running Qwen3-VL-2B (num_ctx 2048): a
# 1920x1080 frame is 2230 prompt tokens and Ollama rejects the request
# ("exceeds the available context size (2048 tokens)") - the camera path
# silently fell back to "I'm sorry". 1280x720 = 1265 tokens, 8.7 s;
# 960x540 = 1222 tokens, 5.8 s. Frames are downscaled to this longest side
# before they reach Qwen.
VISION_MAX_SIDE = 1024
VISION_JPEG_QUALITY = 85


# ============================================================================
# Latency instrumentation
# ============================================================================

METRICS_WINDOW = 200
METRICS_SNAPSHOT_PATH = os.path.join(nr_constants.DEFAULT_LOG_DIR, "latency_snapshot.json")
METRICS_SNAPSHOT_EVERY = 5
