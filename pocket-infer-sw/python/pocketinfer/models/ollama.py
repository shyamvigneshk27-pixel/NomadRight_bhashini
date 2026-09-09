import ollama
import logging
import threading
from subprocess import check_output
import requests


class Ollama:
    def __init__(self, model_name: str):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.preload()

    def preload(self):
        """Pre-load model into memory with keep_alive=-1 to guarantee sub-200ms initial response time."""
        try:
            self.logger.info(f"Pre-warming Ollama model '{self.model_name}' in memory...")
            requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.model_name,
                    'prompt': '',
                    'keep_alive': -1,
                    'options': {'num_thread': 6}
                },
                timeout=10.0
            )
        except Exception as e:
            self.logger.warning(f"Failed to pre-warm Ollama model '{self.model_name}': {e}")

    def chat(self, messages: list, options: dict = None) -> ollama.ChatResponse:
        default_opts = {
            "num_thread": 6,
            "num_predict": 25,
            "temperature": 0.2,
            "top_k": 20,
            "top_p": 0.8,
        }
        if options:
            default_opts.update(options)
        return ollama.chat(model=self.model_name, messages=messages, options=default_opts, keep_alive=-1)

    def generate(self, prompt: str, images: list = None, options: dict = None):
        default_opts = {
            "num_thread": 6,
            "num_predict": 25,
            "temperature": 0.2,
            "top_k": 20,
            "top_p": 0.8,
        }
        if options:
            default_opts.update(options)
        
        kwargs = {
            "model": self.model_name,
            "prompt": prompt,
            "options": default_opts,
            "keep_alive": -1
        }
        if images and len(images) > 0:
            formatted_images = []
            for img in images:
                if isinstance(img, bytearray):
                    formatted_images.append(bytes(img))
                else:
                    formatted_images.append(img)
            kwargs["images"] = formatted_images

        return ollama.generate(**kwargs)

    def restart(self):
        print(check_output('systemctl restart ollama', shell=True))
        self.preload()
    
    @classmethod
    def verify(cls, args):
        """
        Reports whether `args['model_name']` is actually pulled and Ollama
        is reachable - nothing more. Used to gate whether to fall through
        to update() (ollama.pull(), a real internet download this 100%-
        offline device can't make - see master.py's app docstring).

        The pre-warm POST below used to be inside the same try/except as
        the availability check, and unbounded (no `timeout=`) - a slow
        cold GPU load (or any transient hiccup) on that warm-up call made
        this whole method report the model *unavailable* even though it
        was genuinely present, which then triggered update()'s doomed
        pull() attempt and could crash startup entirely
        (verify_dependencies() raises if the post-update re-verify also
        fails). The warm-up is now fire-and-forget on its own thread,
        exactly like master.py's own verify_ollama_model() does the same
        thing - it can never affect this method's return value.
        """
        try:
            ret = ollama.list()
            model_found = any(model.model == args["model_name"] for model in ret.models)
        except Exception as e:
            return False, str(e)
        if not model_found:
            return False, f"Model '{args['model_name']}' not found."

        def _warm():
            # No keep_alive here on purpose - omitting it lets Ollama's
            # own default TTL (~5min) apply instead of the old
            # keep_alive=-1 (never unload). Confirmed on-device via
            # `ollama ps`: this device's own verify() calls (this one and
            # master.py's) were the reason Qwen sat "Forever" resident
            # (~2GB) after every single service launch, regardless of
            # whether a real query ever followed - a real, measured
            # contributor to the memory pressure that was slowing down
            # everything else (RAG cold-load, even the kiosk browser).
            try:
                requests.post(
                    'http://localhost:11434/api/generate',
                    json={'model': args['model_name'], 'prompt': '', 'options': {'num_thread': 6}},
                    timeout=60.0,
                )
            except Exception as e:
                logging.getLogger(__name__).debug(f"Ollama pre-warm (verify) skipped: {e}")
        threading.Thread(target=_warm, daemon=True).start()
        return True, "Ollama service is available."

    @classmethod
    def update(cls, args):
        try:
            logging.info(f"Pulling Ollama model '{args['model_name']}'")
            ollama.pull(model=args["model_name"])
        except Exception as e:
            raise RuntimeError(f"Failed to update Ollama model '{args['model_name']}': {str(e)}")

