"""
NomadRight Application Constants
"""

import os


APP_NAME = "NomadRight"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = (
    "Offline, voice-first, multilingual rights navigator "
    "for interstate migrant workers."
)
APP_AUTHOR = "STARK-X"


# ============================================================================
# BHASHINI Model Service
# ============================================================================

BHASHINI_HOST = "localhost"
BHASHINI_PORT = 11400


# ============================================================================
# Storage & Path Constants
# ============================================================================

# Anchored to this file's own on-disk location:
#
# repo_root/
# └── python/
#     └── pocketinfer/
#         └── applications/
#             └── nomad_right/
#                 └── constants.py
#
# Therefore:
# constants.py
#   -> ../../../../
#   -> repo_root

_HERE = os.path.dirname(os.path.abspath(__file__))

_REPO_ROOT = os.path.normpath(
    os.path.join(
        _HERE,
        "..",
        "..",
        "..",
        "..",
    )
)

DEFAULT_LOG_DIR = "/tmp/nomad_right_logs"

DEFAULT_DB_PATH = os.path.join(
    _REPO_ROOT,
    "nomadright",
    "nomadright_kb.db",
)

DEFAULT_RULES_DIR = os.path.join(
    _REPO_ROOT,
    "nomadright",
)

DEFAULT_CHROMA_PERSIST_DIR = os.path.join(
    _REPO_ROOT,
    "nomadright",
    "chroma_db",
)


# ============================================================================
# UI & Display Limits
# ============================================================================

# 320x240 LCD screen - LEGACY context only. The physical 2.4" ILI9341
# touchscreen this refers to (ui/handheld.py) has been superseded by the
# HDMI browser UI (ui/hdmi/ + boards/hdmi.py + /home/stark-x/UI), which
# has no such fixed-pixel budget. These limits still apply when running
# with `--legacy-lcd`.

DISPLAY_HEADER_MAX_LEN = 30
DISPLAY_BODY_MAX_LEN = 60


# ============================================================================
# Audio & Hardware Thresholds
# ============================================================================

MAX_AUDIO_RECORD_SECONDS = 15

DEFAULT_SAMPLE_RATE = 16000

# Keep spoken answers short enough for comfortable listening.
MAX_VOICE_WORDS = 60


# ============================================================================
# TTS VOICE STYLING
# ============================================================================
#
# The voice-processing pipeline is intentionally gentle.
#
# Previous implementation:
#
#   BHASHINI TTS
#       ↓
#   Pitch shift using asetrate
#       ↓
#   Tempo modification
#       ↓
#   Aggressive EQ
#       ↓
#   Aggressive compression
#       ↓
#   Loudness normalization
#
# This could make the Flite/BHASHINI voice sound:
#
#   - robotic
#   - metallic
#   - deep/heavy
#   - harsh
#   - "Megatron-like"
#
#
# Current implementation:
#
#   BHASHINI TTS
#       ↓
#   Original pitch preserved
#       ↓
#   Gentle high-pass filter
#       ↓
#   Mild speech-presence EQ
#       ↓
#   Gentle compression
#       ↓
#   Comfortable loudness normalization
#       ↓
#   AudioPlayer
#
#
# IMPORTANT:
#
# We intentionally DO NOT use TTS_PITCH_FACTOR or TTS_ENERGY_BOOST anymore.
#
# The actual FFmpeg filter chain is implemented in:
#
#   bhashini_bridge.py
#
# These constants allow the audio characteristics to be tuned without
# changing the orchestration logic.
# ============================================================================


# Enable/disable FFmpeg voice processing.
#
# True:
#   BHASHINI TTS → gentle DSP → speaker
#
# False:
#   BHASHINI TTS → speaker
#
TTS_VOICE_STYLE_ENABLED = True


# ---------------------------------------------------------------------------
# High-pass filter
# ---------------------------------------------------------------------------
#
# Removes very low-frequency rumble while preserving voice warmth.

TTS_HIGHPASS_HZ = 70


# ---------------------------------------------------------------------------
# Speech presence EQ
# ---------------------------------------------------------------------------
#
# A small boost around 2.5 kHz improves consonant intelligibility.
#
# Keep this small. Large boosts can make synthetic voices sound harsh.
#

TTS_PRESENCE_FREQUENCY_HZ = 2500

TTS_PRESENCE_GAIN_DB = 1.5


# ---------------------------------------------------------------------------
# High-frequency air
# ---------------------------------------------------------------------------
#
# Adds a tiny amount of crispness.
#
# Deliberately kept below 1 dB to avoid excessive sibilance.

TTS_AIR_FREQUENCY_HZ = 7000

TTS_AIR_GAIN_DB = 0.8


# ---------------------------------------------------------------------------
# Gentle dynamic compression
# ---------------------------------------------------------------------------
#
# Compression keeps quiet syllables audible without making the entire voice
# sound aggressive or crushed.

TTS_COMPRESSOR_THRESHOLD_DB = -20

TTS_COMPRESSOR_RATIO = 1.8

TTS_COMPRESSOR_ATTACK_MS = 10

TTS_COMPRESSOR_RELEASE_MS = 100

TTS_COMPRESSOR_MAKEUP_DB = 1.0


# ---------------------------------------------------------------------------
# Loudness normalization
# ---------------------------------------------------------------------------
#
# -18 LUFS is intentionally softer than the previous -16 LUFS.
#
# This is more comfortable on a small embedded speaker.

TTS_TARGET_LUFS = -18

TTS_TRUE_PEAK_DB = -2

TTS_LOUDNESS_RANGE = 7


# ============================================================================
# Generative Fallback Model (qwen, via Ollama) - see qwen_client.py
# ============================================================================
#
# These were missing from this file (an incomplete merge dropped them
# while qwen_client.py, workflow.py, and app.py's RegisterApplication
# metadata already expected them, leaving pocketinfer-service unable to
# even import). Restored here to match qwen_client.py's actual usage and
# this device's actually-installed Ollama model (`ollama list`).

# Master switch for the intent/RAG generative fallback (workflow.py Step
# 6.5) - only ever the last resort when rules AND thresholded RAG both
# find nothing; never the primary answer path (see app.py's module
# docstring).
LLM_FALLBACK_ENABLED = True

# Must match a model actually pulled in Ollama on this device (`ollama
# list`) - matches master.py's --model default.
LLM_FALLBACK_MODEL = "hf.co/Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M"

# How long Ollama keeps this model resident after the last call before
# unloading it. Deliberately NOT -1 (permanently resident, which
# pocketinfer.models.ollama.Ollama uses) - this module's own docstring
# explains why: the fallback is only occasionally needed, so tying up
# ~2.4GB of RAM for the model's whole lifetime would be wasteful (see
# master.py's app_needs_ollama()/pre-warm skip comment).
#
# Was "5m" - measured live on-device (2026-09-09) that a cold Ollama
# load of this model takes ~43-45s here (llama-server's own
# "load_duration" in the /api/generate response). A 5-minute TTL means
# any two real citizen queries more than 5 minutes apart - completely
# normal at a walk-up kiosk - forces that ~45s reload on the second
# query, which then raced LLM_REQUEST_TIMEOUT_S below and frequently
# lost outright (falls to the wrong-topic template fallback instead of
# a real answer). 30 minutes covers realistic gaps between visitors
# during operating hours while still releasing the ~2.4GB during a
# genuinely idle stretch (overnight, etc).
LLM_KEEP_ALIVE = "30m"

# Forced high on every call rather than left to Ollama's own auto-fit
# heuristic: the Jetson's GPU/CPU share one physical RAM pool, and that
# heuristic offloads 0 layers whenever BHASHINI is already resident
# (which is always) - see qwen_client.py's module docstring. 999 is the
# conventional llama.cpp/Ollama "as many layers as fit" sentinel.
LLM_NUM_GPU = 999

# Jetson Orin Nano has 6 CPU cores - matches
# pocketinfer.models.ollama.Ollama's own default.
LLM_NUM_THREAD = 6

# Context window - generous enough for the grounded prompt (system
# prompt + a handful of RAG snippets + the question) without paying for
# more KV-cache than this edge device needs.
LLM_NUM_CTX = 2048

LLM_TEMPERATURE = 0.2

# Vision analysis in particular can take a while on this hardware - see
# app.py's "[ANALYZING PHOTO] This may take a moment" status text.
#
# Was 45.0 - too tight a margin above the ~43-45s cold-load time
# measured above (LLM_KEEP_ALIVE's comment): a cold-load call and this
# timeout were essentially racing to the same finish line, so a cold
# load frequently got cut off just short of finishing and returned
# nothing usable instead of the real (if slow) answer. Raised well
# above the observed cold-load time so a cold-load call actually
# completes instead of aborting right at the line.
LLM_REQUEST_TIMEOUT_S = 90.0

# Every qwen prompt in qwen_client.py caps answers at "under 40 words" -
# these leave comfortable headroom above that without letting a run-on
# response burn extra latency.
LLM_NUM_PREDICT_TEXT = 100
LLM_NUM_PREDICT_VISION = 100

# Exact string the model is instructed to return verbatim when it has no
# grounded answer (see qwen_client.py's system prompts) - distinctive
# enough that it will never appear in a genuine answer.
LLM_SENTINEL_NOT_FOUND = "[[NO_ANSWER_FOUND]]"

# How long a Camera-button photo stays valid awaiting its follow-up
# question before being silently discarded (see app.py's
# pending_form_image) - long enough for a worker to think of their
# question, short enough that a much later unrelated question can't
# accidentally attach to a stale photo.
LLM_VISION_PENDING_TTL_S = 90.0


# ============================================================================
# Supported Welfare Scheme Identifiers
# ============================================================================

SCHEME_CODE_PDS = "PDS"

SCHEME_CODE_PMJAY = "PMJAY"

SCHEME_CODE_ESHRAM = "ESHRAM"

SCHEME_CODE_BOCW = "BOCW"

SCHEME_CODE_MGNREGS = "MGNREGS"


SUPPORTED_SCHEME_CODES = [
    SCHEME_CODE_PDS,
    SCHEME_CODE_PMJAY,
    SCHEME_CODE_ESHRAM,
    SCHEME_CODE_BOCW,
    SCHEME_CODE_MGNREGS,
]


# ============================================================================
# Database Scheme IDs
# ============================================================================

DB_SCHEME_ID_PDS = "PDS_ONORC_001"

DB_SCHEME_ID_PMJAY = "PMJAY_001"

DB_SCHEME_ID_ESHRAM = "ESHRAM_OSH_001"

DB_SCHEME_ID_BOCW = "BOCW_001"

DB_SCHEME_ID_MGNREGS = "MGNREGS_001"


# ============================================================================
# Scheme Code -> Database ID
# ============================================================================

SCHEME_CODE_TO_DB_ID = {
    SCHEME_CODE_PDS: DB_SCHEME_ID_PDS,
    SCHEME_CODE_PMJAY: DB_SCHEME_ID_PMJAY,
    SCHEME_CODE_ESHRAM: DB_SCHEME_ID_ESHRAM,
    SCHEME_CODE_BOCW: DB_SCHEME_ID_BOCW,
    SCHEME_CODE_MGNREGS: DB_SCHEME_ID_MGNREGS,
}


# ============================================================================
# Scheme Helplines
# ============================================================================

SCHEME_HELPLINES = {
    SCHEME_CODE_PDS: "14445",
    SCHEME_CODE_PMJAY: "14555",
    SCHEME_CODE_ESHRAM: "14434",
    SCHEME_CODE_BOCW: "1800-891-8888",
    SCHEME_CODE_MGNREGS: "UNKNOWN",
}


# ============================================================================
# Scheme JSON Source Files
# ============================================================================

SCHEME_JSON_FILES = [
    "pds.json",
    "pmjay.json",
    "eshram_osh.json",
    "bocw.json",
    "mgnregs.json",
]


# ============================================================================
# Language Configuration
# ============================================================================

# Source languages used for ASR input.

SOURCE_LANGUAGES = {
    "hi": "Hindi",
    "ta": "Tamil",
    "or": "Odia",
    "bho": "Bhojpuri",
    "mai": "Maithili",
    "sat": "Santali",
    "hne": "Chhattisgarhi",
}

DEFAULT_SOURCE_LANGUAGE = "hi"

# Of SOURCE_LANGUAGES above, only these have an actual ASR checkpoint on the
# BHASHINI side (~/bhashini_models/asr/checkpoints/{hi,ta}-conformer.onnx -
# see infer.py's `self.sessions` dict). The rest are UI/roadmap entries the
# touchscreen still renders buttons for; selecting one used to be accepted
# silently and made every subsequent turn's /asr call 500 forever with no
# on-screen explanation. app.py's ui_cb() gates on this set before honouring
# a selection - add a code here only once its .onnx checkpoint actually
# exists on disk.
ASR_SUPPORTED_LANGUAGES = {"hi", "ta"}

# ============================================================================
# Voice Bridge Languages
# ============================================================================

BRIDGE_LANGUAGES = {
    "ta": "Tamil",
    "gu": "Gujarati",
    "mr": "Marathi",
    "kn": "Kannada",
}

DEFAULT_BRIDGE_LANGUAGE = "ta"


# ============================================================================
# Decision Layer Language
# ============================================================================

# The Decision Layer always operates in English.
#
# Worker language
#       ↓
# ASR
#       ↓
# Worker language text
#       ↓
# NMT
#       ↓
# English
#       ↓
# Decision Layer / Rules / RAG
#       ↓
# NMT
#       ↓
# Worker language
#       ↓
# TTS

PIPELINE_LANGUAGE = "EN"


# ============================================================================
# RAG Pipeline
# ============================================================================

CHROMA_COLLECTION_NAME = "nomadright_schemes"

EMBEDDING_MODEL_NAME = (
    "intfloat/multilingual-e5-small"
)

RAG_TOP_K = 3


# multilingual-e5-small expects:
#
#   query: ...
#
# and:
#
#   passage: ...
#

RAG_QUERY_PREFIX = "query: "

RAG_PASSAGE_PREFIX = "passage: "


# ChromaDB always returns k nearest neighbours even if none are actually
# relevant. Therefore a minimum similarity threshold is used to prevent
# unrelated or garbled ASR input from receiving a confident scheme answer.

RAG_MIN_SCORE = 0.83