# Source - https://stackoverflow.com/a/1057534
# Posted by Anurag Uniyal, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-30, License - CC BY-SA 4.0

from os.path import dirname, basename, isfile, join
import glob
import importlib
import logging

modules = glob.glob(join(dirname(__file__), "*.py"))
_candidate_names = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

# Each name above is imported individually (rather than relying on `from
# pocketinfer.applications import *` to do it) so one legacy application
# module with an unmet dependency - e.g. hear_the_world.py's
# pocketinfer.models.asr/nmt/tts, removed from this checkout in favor of
# the HTTP-based bhashini_bridge.py - logs a warning and is skipped
# instead of crashing every application (including the one actually in
# use, NomadRight) at process startup.
__all__ = []
for _name in _candidate_names:
    try:
        importlib.import_module(f"pocketinfer.applications.{_name}")
        __all__.append(_name)
    except ImportError:
        logging.getLogger(__name__).warning(
            f"Skipping application module 'pocketinfer.applications.{_name}': missing dependency",
            exc_info=True,
        )

try:
    from pocketinfer.applications.nomad_right.app import NomadRightApplication
    __all__.append("NomadRightApplication")
except ImportError:
    pass

