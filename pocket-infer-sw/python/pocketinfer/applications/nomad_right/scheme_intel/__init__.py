"""
NomadRight scheme-intelligence layer.

Sits between the existing speech front-end (ASR -> NMT -> English) and the
existing response back-end (NMT -> TTS -> speaker / UI):

    English query
      -> ScenarioEngine   (who is asking: age, work, origin/current state ...)
      -> IntentEngine     (rules first, then a small e5 nearest-neighbour classifier)
      -> QueryRouter      (DIRECT / ELIGIBILITY / DISCOVERY / COMPLEX / CLARIFY / DEFER)
      -> SchemeRepository (SQLite, metadata filters)
      -> RulesEngine      (deterministic, six eligibility states)
      -> RAGEngine        (numpy index over pre-computed e5 embeddings, filtered first)
      -> QwenAdapter      (only for complex scenarios - never for simple questions)
      -> ResponseComposer (short grounded templates, <= 20 s of speech)

WorkflowController.process() calls this first and falls back to the legacy
pipeline whenever it declines or fails; SCHEME_INTEL_ENABLED (or env
NOMADRIGHT_SCHEME_INTEL=0) switches it off entirely.

Heavy modules are imported lazily - importing this package costs nothing.
"""

from pocketinfer.applications.nomad_right.scheme_intel.si_constants import SCHEME_INTEL_ENABLED

__all__ = ["SCHEME_INTEL_ENABLED", "create_scheme_intelligence"]


def create_scheme_intelligence(**kwargs):
    """Builds the layer, or returns None (with a logged warning) if it cannot start."""
    from pocketinfer.applications.nomad_right.scheme_intel.service import SchemeIntelligence
    return SchemeIntelligence.create(**kwargs)
