"""
Adapters around the EXISTING speech models (served by bhashini_models via
BhashiniBridge). They add latency recording and a uniform interface for the
diagnostics command; they do not replace, reconfigure or re-load anything.
The application's own run() loop keeps calling BhashiniBridge directly.
"""

import time
from typing import Optional

from pocketinfer.applications.nomad_right.scheme_intel.metrics import LATENCY


class _Timed:
    def __init__(self, bridge, recorder=LATENCY):
        self.bridge = bridge
        self.recorder = recorder

    def _timed(self, stage: str, fn, *args):
        t0 = time.perf_counter()
        try:
            return fn(*args)
        finally:
            self.recorder.record(stage, (time.perf_counter() - t0) * 1000.0)


class ExistingASRAdapter(_Timed):
    """Hindi/Tamil Conformer (TensorRT FP16 engines) behind bhashini_models /asr."""

    def transcribe(self, wav_bytes: bytes, lang: str) -> str:
        return self._timed(f"asr_{lang}", self.bridge.listen, wav_bytes, lang)


class ExistingNMTAdapter(_Timed):
    """IndicTrans2 CTranslate2 int8 behind bhashini_models /nmt."""

    def to_english(self, text: str, lang: str) -> str:
        return self._timed(f"nmt_{lang}_en", self.bridge.to_pipeline_language, text, lang)

    def from_english(self, text: str, lang: str) -> str:
        return self._timed(f"nmt_en_{lang}", self.bridge.from_pipeline_language, text, lang)


class ExistingTTSAdapter(_Timed):
    """Flite Indic voices behind bhashini_models /tts."""

    def synthesize(self, text: str, lang: str) -> Optional[bytes]:
        return self._timed(f"tts_{lang}", self.bridge.speak, text, lang)
