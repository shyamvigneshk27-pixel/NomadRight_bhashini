"""
QwenAdapter - the only way this layer reaches Qwen3-VL, and only for
COMPLEX_SCENARIO questions that no structured answer or FAQ chunk covers.

Reuses the application's existing QwenClient (same Ollama runner, same
options: num_ctx 2048, num_gpu, keep_alive - so the model is never reloaded)
and its grounded, sentinel-gated prompt. The context handed to Qwen is kept
minimal: one line of the person's facts, one line per rule outcome, and at
most LLM_CONTEXT_MAX_SNIPPETS retrieved chunks of <= LLM_SNIPPET_MAX_WORDS
words each. Prompts are never logged.
"""

import logging
from typing import List, Optional, Tuple

from pocketinfer.applications.nomad_right.scheme_intel import si_constants as C

logger = logging.getLogger(__name__)


def _clip_words(text: str, n: int) -> str:
    w = text.split()
    return text if len(w) <= n else " ".join(w[:n]) + " ..."


class QwenAdapter:
    def __init__(self, qwen_client):
        self._client = qwen_client

    @property
    def available(self) -> bool:
        return self._client is not None and C.LLM_COMPLEX_ENABLED

    def complex_answer(self, question: str, facts_line: str, rule_lines: List[str],
                       chunks: List[str]) -> Tuple[Optional[str], bool]:
        """Returns (answer or None, declined). declined=True means Qwen was reached and said it cannot answer."""
        if not self.available:
            return None, False
        context: List[str] = []
        if facts_line:
            context.append(f"About the person: {facts_line}")
        context += rule_lines[:3]
        context += [_clip_words(c, C.LLM_SNIPPET_MAX_WORDS) for c in chunks[:C.LLM_CONTEXT_MAX_SNIPPETS]]
        try:
            return self._client.answer_text_with_decline(question, context)
        except Exception as exc:  # QwenClient already swallows transport errors; belt and braces
            logger.warning(f"[SI] Qwen complex-answer failed: {exc}")
            return None, False
