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

# 320x240 LCD screen

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