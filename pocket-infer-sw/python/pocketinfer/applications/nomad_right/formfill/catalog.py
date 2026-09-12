"""The form catalogue: forms.json loaded once, with helpers for prompts, words and fields."""
import json
import os
from typing import Dict, List, Optional

from pocketinfer.applications.nomad_right import constants

DEFAULT_PATH = os.path.join(constants._REPO_ROOT, "nomadright", "forms", "forms.json") if hasattr(constants, "_REPO_ROOT") else None


class FormCatalog:
    def __init__(self, path: Optional[str] = None):
        path = path or os.environ.get("NOMADRIGHT_FORMS") or DEFAULT_PATH
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"forms.json not found: {path}")
        self.path = path
        data = json.load(open(path, encoding="utf-8"))
        self.version = data.get("version")
        self.languages: List[str] = data["languages"]
        self.prompts: Dict[str, Dict[str, str]] = data["prompts"]
        self.words: Dict[str, Dict[str, List[str]]] = data["words"]
        self.forms: Dict[str, dict] = {f["form_id"]: f for f in data["forms"]}
        self.by_scheme: Dict[str, List[str]] = {}
        for fid, f in self.forms.items():
            self.by_scheme.setdefault(f["scheme_id"], []).append(fid)

    def form(self, form_id: str) -> dict:
        return self.forms[form_id]

    def prompt(self, key: str, lang: str, **kw) -> str:
        text = self.prompts[key].get(lang) or self.prompts[key]["en"]
        return text.format(**kw) if kw else text

    def spoken_name(self, form_id: str, lang: str) -> str:
        f = self.forms[form_id]
        return f["spoken_name"].get(lang) or f["spoken_name"]["en"]

    def word_set(self, kind: str, lang: str) -> List[str]:
        return list(self.words[kind].get(lang, [])) + list(self.words[kind].get("en", []))

    def supported_names(self, lang: str) -> str:
        return ", ".join(self.spoken_name(fid, lang) for fid in self.forms)
