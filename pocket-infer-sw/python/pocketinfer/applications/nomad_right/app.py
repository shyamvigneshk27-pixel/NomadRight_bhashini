"""
NomadRight Application Entry Point

Pipeline: press & hold the trigger button -> record audio -> BHASHINI ASR
(worker's own language) -> BHASHINI NMT (-> English) -> Decision Layer
(Intent / Entity / Rules / RAG) -> BHASHINI NMT (English -> worker's
language) -> BHASHINI TTS -> LCD + speaker.

The worker's source language and the "voice bridge" destination-official
language are both selected from the touchscreen Settings page (see
ui/handheld.py) using the same `ASR <lang>` / `Bridge <lang>` message
convention already established by the HearTheWorld reference application.

100% offline: BHASHINI ASR/NMT/TTS all talk to localhost:11400 only.
qwen2.5vl:3b (via Ollama, localhost:11434) is the only generative model in
the loop, and only in two narrow, explicitly-gated cases: a grounded text
fallback when intent/RAG matching finds nothing (see workflow.py Step 6.5),
and the Camera-button form-reading flow below - never as the primary answer
path. See qwen_client.py for the grounding/sentinel guarantees.
"""

import os
import time
import wave
import logging
import threading
from io import BytesIO
from typing import Optional, Dict, Any

import cv2

from pocketinfer.applications.nomad_right import document_crop

from pocketinfer.applications.base import BaseApplication
from pocketinfer.applications.registry import RegisterApplication
from pocketinfer.audio import AudioPlayer

from pocketinfer.applications.nomad_right import constants
from pocketinfer.applications.nomad_right.config import NomadRightConfig
from pocketinfer.applications.nomad_right.workflow import WorkflowController
from pocketinfer.applications.nomad_right.response import StructuredResponsePackage
from pocketinfer.applications.nomad_right.bhashini_bridge import BhashiniBridge
from pocketinfer.applications.nomad_right.intent import IntentType
from pocketinfer.applications.nomad_right.scheme_intel.metrics import LATENCY
from pocketinfer.applications.nomad_right.formfill.flow import FormFlow, FlowIO, HOME as FORM_HOME, TIMEOUT as FORM_TIMEOUT

logger = logging.getLogger(__name__)


@RegisterApplication({
    "name": "NomadRight",
    "description": "Offline, voice-first, multilingual rights navigator for interstate migrant workers.",
    "author": "STARK-X",
    "version": constants.APP_VERSION,
    "models": {
        "asr": {},
        "nmt": {},
        "tts": {},
        # Only when Ollama serves Qwen: with the on-demand llama-server backend
        # (constants.LLM_BACKEND) master.py must neither start nor pre-warm Ollama.
        **({"ollama": {"model_name": constants.LLM_FALLBACK_MODEL}} if constants.LLM_BACKEND == "ollama" else {}),
    },
    "default_settings": {
        "input_language": constants.DEFAULT_SOURCE_LANGUAGE,
        "bridge_language": constants.DEFAULT_BRIDGE_LANGUAGE,
        "log_directory": constants.DEFAULT_LOG_DIR,
    },
    "service_dependencies": ["bhashini_models"],
    # Runner options master.py pre-warms Qwen with. They MUST equal
    # QwenClient's per-request options (qwen_client._call): Ollama reloads the
    # model whenever they differ. Measured 2026-09-11: a pre-warm with only
    # {"num_thread": 6} loaded a 4096-token context (1884 MB), then the first
    # real query paid a second 34.9 s load to switch to num_ctx 2048 (1658 MB).
    # Kept outside "models" so BaseApplication.verify_dependencies() is unaffected.
    "ollama_runner_options": {
        "num_gpu": constants.LLM_NUM_GPU,
        "num_thread": constants.LLM_NUM_THREAD,
        "num_ctx": constants.LLM_NUM_CTX,
    },
})
class NomadRightApplication(BaseApplication):
    """
    NomadRight application wrapper integrated into the Suno Sutra application framework.
    Coordinates hardware inputs (trigger button, microphone), invokes the BHASHINI
    model adapters via BhashiniBridge, and drives display & audio playout through
    the WorkflowController Decision Layer.
    """

    # Shared Home/ready-screen hint, kept short and reused everywhere it's
    # shown (run()'s idle screen, _on_home_pressed()'s two branches) so
    # there's exactly one string to keep within the display's safe line
    # length rather than several independently-drifting copies.
    HOME_HINT = "Hold: Voice Q&A  |  Camera: Scan Document"

    def __init__(self, board: Any, settings: Optional[Dict[str, Any]] = None):
        super().__init__(board, settings)
        # BaseApplication.__init__ sets self.logger to the generic
        # "pocketinfer.applications.base" logger (since that's evaluated in
        # base.py, not here) - override so NomadRight's own log lines are
        # identifiable instead of blending into that shared bucket.
        self.logger = logging.getLogger(__name__)
        self.app_config = NomadRightConfig.from_settings(self.settings)
        # Remembers the last answer so a follow-up "translate this for the
        # officer" request (TRANSLATION_REQUEST intent) has something to bridge.
        self.last_answer_en: str = ""
        # Remembers the scheme discussed in the previous turn (e.g. "PDS")
        # so a generic follow-up like "what documents do I need?" resolves
        # without the worker having to repeat the scheme name - see
        # EntityExtractor._CONTEXT_INHERITABLE_INTENTS.
        self.last_scheme_code: Optional[str] = None
        # Set when the worker presses the touchscreen Camera button (see
        # ui_cb below). The *next* voice query is answered against this
        # photo via workflow.process_vision_query() instead of the normal
        # Decision Layer, then cleared (one photo -> one follow-up question;
        # asking a second question about the same form means pressing
        # Camera again). Cleared without use if constants.LLM_VISION_PENDING_TTL_S
        # elapses first, so a stale photo can't attach to an unrelated
        # later question.
        self.pending_form_image: Optional[bytes] = None
        self.pending_form_image_ts: float = 0.0
        self._image_lock = threading.Lock()
        # Set for the duration of _on_camera_pressed() (which runs on the UI
        # callback thread, concurrently with the main run() loop below
        # blocking on wait_for_trigger_button_down()). Without this, a
        # worker who presses the trigger button again fast enough - before
        # the capture finishes and "Photo captured" is shown - could have
        # their follow-up question race ahead of pending_form_image being
        # set, and get answered as a normal voice query instead of a vision
        # one. run() waits on this before it starts listening, so the mic
        # is never armed until the capture is actually confirmed on screen.
        self._camera_busy = threading.Event()

        # Text-query entry point for the HDMI UI's on-screen keyboard (see
        # ui/hdmi/protocol.py's CMD_ASK_DOCUMENT_TEXT and submit_text_query()
        # below) - an alternative to holding the trigger button and
        # speaking, primarily for the Document Scanner Q&A view. Wakes
        # run()'s _wait_for_next_turn() the same way a physical press does;
        # the submitted text is used directly as that turn's native_query,
        # skipping the mic/ASR stage entirely.
        self._text_query_pending: Optional[str] = None
        self._text_query_event = threading.Event()
        # Assisted form filling (formfill/): the Camera button captures the form and
        # hands it to run() through this event; the flow itself runs on the turn
        # thread so hold-to-talk, Home and the language setting behave as in a
        # normal turn. "Ask Chatbot" keeps the previous photo+question flow.
        self._form_image: Optional[Any] = None
        self._form_event = threading.Event()
        self._form_cmd: Optional[str] = None          # on-screen Repeat/Change/Skip/Cancel while a form runs
        self._form_catalog = None
        self.form_outbox = None

        # ── Mode / navigation state (Home <-> Voice Translation <-> ────────
        # Document Scanner) - see module docstring's "UI modes" section.
        # Display-only label reflecting what the LCD's mode_text should
        # currently read; not a hard state machine gate - the real branch
        # logic still runs entirely on pending_form_image, exactly as
        # before. "HOME" is the idle/ready screen (both options available);
        # it switches to "VOICE TRANSLATION" or "DOCUMENT SCANNER" only for
        # the duration of one active turn, then reverts to "HOME".
        self._mode: str = "HOME"

        # Cancellable audio playback (Home button, see _on_home_pressed/
        # _stop_audio): the AudioPlayer currently on-air, if any. Written
        # by the main thread (_play()) and read/killed by the UI callback
        # thread, so access is lock-guarded even though CPython's GIL would
        # likely make the bare reference swap safe anyway - explicit is
        # cheap here and matches the existing _image_lock pattern.
        self._current_player = None
        self._player_lock = threading.Lock()
        # Set by _stop_audio() right before killing the player, cleared at
        # the start of every _play() call - lets _play() tell its caller
        # apart a user-initiated stop from a normal finish or a genuine
        # playback error (see _play()'s return value).
        self._audio_stop_requested = threading.Event()
        # Set by _on_home_pressed() (UI callback thread); polled by run()
        # (main thread) at safe checkpoints - never mid-inference, since a
        # live Qwen/RAG HTTP call can't be safely aborted once sent (see
        # module docstring) - to drop back to the Home/ready screen instead
        # of speaking out an answer the worker already tried to back out of.
        self._home_requested = threading.Event()

        # A "screen update" (mode/top/bottom/status together, describing one
        # coherent state) used to be 2-4 separate RPC calls to the UI
        # subprocess, each individually lock-serialized but not atomic as a
        # *sequence* - a Camera/Home press from the UI callback thread (see
        # jetson.py's _process_ui_events) landing mid-sequence could
        # interleave with run()'s own update and leave one field from the
        # old state next to one from the new state on screen at once (the
        # "overlay" glitch). board.update_screen() (boards/jetson.py) now
        # sends every field in ONE RPC call instead, which
        # multiprocess_launch()'s serial dispatch loop (ui/handheld.py)
        # applies as a single unit - atomic by construction, so no lock is
        # needed here any more. Note: this only covers the app's own screen
        # updates; the framework's memory_text() stats thread (service.py)
        # is a separate single-field RPC call that isn't part of any
        # update_screen() sequence, so it can't tear one.

    # ── On-screen pipeline log ─────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        """
        Emit one line to BOTH the Python logger and the LCD's pipeline-log
        page (topbar terminal icon, see ui/handheld.py).

        This is what makes the kiosk usable with no terminal attached: every
        stage boundary that previously only existed in the console scrollback
        is now readable on the device itself, including whether the trigger
        button actually registered and how long each stage took.

        Lines are forced to ASCII because the log page renders in
        terminalio.FONT, which has no Devanagari glyphs - a native-script
        character would draw as nothing while still consuming a column. Log
        *facts about* native text (its length, its language) here; the text
        itself belongs on the main screen, which uses a font that can render
        it.
        """
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        line = line.encode("ascii", "replace").decode("ascii")
        self.logger.info("[NomadRight] %s", msg)
        try:
            self.board.log_line(line)
        except Exception:
            # Diagnostics must never be able to break a real turn - a failed
            # log push is strictly less important than the answer in flight.
            self.logger.debug("[NomadRight] on-screen log push failed", exc_info=True)

    def _sync_language_buttons(self) -> None:
        """
        Make the Settings page's highlighted language match the language the
        pipeline is actually configured to use.

        The page's constructor default highlights 'ASR En' (correct for
        HearTheWorld, whose default input_language really is "en"), but
        NomadRight's SOURCE_LANGUAGES has no English entry at all - so on
        this app the screen claimed English while every turn ran in Hindi,
        and pressing that highlighted button did nothing whatsoever, because
        _lang_name_to_code('En', SOURCE_LANGUAGES) returns None and the
        setting is left untouched. Declaring the truth here fixes both the
        wrong highlight and the "that button is dead" symptom.
        """
        lang = self.settings.get("input_language", constants.DEFAULT_SOURCE_LANGUAGE)
        bridge_lang = self.settings.get("bridge_language", constants.DEFAULT_BRIDGE_LANGUAGE)
        try:
            # Button labels are the language code title-cased ('hi' ->
            # 'ASR Hi', 'bho' -> 'ASR Bho'), matching ui/handheld.py.
            self.board.select_radio("ASR ", f"ASR {lang.capitalize()}")
            self.board.select_radio("Bridge ", f"Bridge {bridge_lang.capitalize()}")
        except Exception:
            self.logger.debug("[NomadRight] language button sync failed", exc_info=True)

    def _startup_selfcheck(self) -> None:
        """
        Probe every subsystem a turn depends on and report each on the LCD
        log, so "is this thing ready?" is answerable by looking at the
        device instead of by pressing the button and waiting to see.

        Deliberately synchronous (before the ready screen appears) and
        tightly timed out: reaching "[READY]" should mean the checks
        actually passed, not that they haven't run yet. Nothing here can
        abort startup - a subsystem reported DOWN still surfaces its real
        error later through the existing per-turn error handling, exactly as
        it did before this existed.
        """
        import requests

        degraded = []

        def _probe(name: str, url: str) -> None:
            try:
                resp = requests.get(url, timeout=1.5)
                if resp.status_code == 200:
                    self._log(f"{name:<9} OK")
                    return
                self._log(f"{name:<9} DOWN (HTTP {resp.status_code})")
            except Exception:
                self._log(f"{name:<9} DOWN (no response)")
            degraded.append(name)

        _probe("BHASHINI", f"http://{self.app_config.bhashini_host}:"
                           f"{self.app_config.bhashini_port}/health")
        _probe("OLLAMA", "http://localhost:11434/api/tags")

        mic = getattr(self.board, "alsa_capture_card", None)
        spk = getattr(self.board, "alsa_playback_card", None)
        self._log(f"MIC       {'card ' + str(mic) if mic is not None else 'NOT FOUND'}")
        self._log(f"SPEAKER   {'card ' + str(spk) if spk is not None else 'NOT FOUND'}")
        if mic is None:
            degraded.append("MIC")
        if spk is None:
            degraded.append("SPEAKER")

        try:
            schemes = len(constants.SUPPORTED_SCHEME_CODES)
            self._log(f"KB        {schemes} schemes  lang={self.settings.get('input_language')}")
        except Exception:
            self._log("KB        unavailable")

        if degraded:
            self._log(f"WARNING: degraded -> {','.join(degraded)}")
        else:
            self._log("All subsystems OK")

    def _prewarm_fastpath(self) -> None:
        """
        Pay the one-time warm-up costs here, before the ready screen, so the
        first worker of a session isn't the one who pays them.

        Measured on-device, cold vs. steady state:
            first scenario-cache lookup   4.94s  ->  0.19-0.30s
            first ASR call                5.7s   ->  0.3s

        Left unwarmed those two alone turned a 4.8s turn into a 16.9s one.
        Done synchronously (not on a thread) because "[READY]" should mean
        genuinely ready - the boot this is added to already takes ~30s, and a
        worker who presses the button the instant READY appears should get
        the fast path, not a race with a background warm-up. Any failure here
        is swallowed: it only costs the warm-up, and the real error surfaces
        normally on the first actual turn.
        """

        stage_start = time.time()
        try:
            # Half a second of silence is enough to force the ASR model
            # through its first inference; the empty transcript is discarded.
            buf = BytesIO()
            with wave.open(buf, "wb") as silent:
                silent.setnchannels(1)
                silent.setsampwidth(2)
                silent.setframerate(constants.DEFAULT_SAMPLE_RATE)
                silent.writeframes(b"\x00" * (constants.DEFAULT_SAMPLE_RATE))
            lang = self.settings.get("input_language", constants.DEFAULT_SOURCE_LANGUAGE)
            self.bridge.listen(buf.getvalue(), lang)
            self._log(f"ASR warmed   {time.time() - stage_start:.1f}s")
        except Exception:
            self.logger.debug("[NomadRight] ASR warm-up skipped", exc_info=True)

    def start(self) -> None:
        """Application start hook. Instantiates the BHASHINI bridge and pipeline controller."""
        self._log(f"NomadRight v{constants.APP_VERSION} starting")

        load_start = time.time()
        self.bridge = BhashiniBridge(config=self.app_config)
        self.workflow = WorkflowController(config=self.app_config)
        self._log(f"Pipeline loaded  {time.time() - load_start:.1f}s")


        self.board.subscribe_to_ui(self.ui_cb)

        if not os.path.exists(self.app_config.log_dir):
            os.makedirs(self.app_config.log_dir, exist_ok=True)

        self._sync_language_buttons()
        try:
            self.board.set_asr_languages(sorted(constants.ASR_SUPPORTED_LANGUAGES))
        except Exception:
            self.logger.debug("[NomadRight] ASR language list push skipped", exc_info=True)
        self._startup_selfcheck()
        self._prewarm_fastpath()
        if constants.FORM_FILLING_ENABLED:
            self._start_outbox()

        # Pre-warm the camera (device open + first-frame negotiation) at
        # startup rather than leaving it lazy until the worker's first
        # Camera press - CameraReader (boards/base.py) keeps streaming in
        # the background once started, so only the *first* call ever pays
        # this cost; every capture after it is already fast. Measured
        # on-device: ~1.8s for that first open+negotiate. Backgrounded so
        # a slow/unavailable camera can never delay reaching the ready
        # screen, and any failure here is silently swallowed - a real
        # failure surfaces normally (with its own error screen) the first
        # time the worker actually presses Camera, exactly as before this
        # existed; this is purely a latency head start, not new error
        # handling.
        threading.Thread(target=self._prewarm_camera, daemon=True).start()

        super().start()

    def _prewarm_camera(self) -> None:
        try:
            warm_start = time.time()
            frame = self.board.camera_frame_jpg()
            if frame:
                self._log(f"CAMERA    OK  {time.time() - warm_start:.1f}s")
            else:
                self._log("CAMERA    NO FRAME")
        except Exception as exc:
            self._log("CAMERA    UNAVAILABLE")
            self.logger.debug(f"[NomadRight] Camera pre-warm skipped: {exc}")

    # ── Touchscreen Settings page: language selection ──────────────────────

    def ui_cb(self, msg: str) -> None:
        """
        Handles touchscreen Settings page button presses. Reuses the exact
        `ASR <lang>` message convention from hear_the_world.py for the
        worker's spoken language, plus a NomadRight-specific `Bridge <lang>`
        convention for the voice-bridge destination-official language.
        """
        if msg.startswith("ASR "):
            code = self._lang_name_to_code(msg[4:], constants.SOURCE_LANGUAGES)
            if code and code in constants.ASR_SUPPORTED_LANGUAGES:
                if self.settings.get("input_language") != code:
                    # A new language on the start page is a new person.
                    self._reset_conversation("language change")
                    self._home_requested.set()            # also ends a form in progress (answers wiped)
                self.settings["input_language"] = code
                self.logger.info(f"[NomadRight] Worker language set to {code}")
            elif code:
                # SOURCE_LANGUAGES lists this code, but BHASHINI has no ASR
                # checkpoint for it (see constants.ASR_SUPPORTED_LANGUAGES) -
                # accepting it used to leave every future turn's /asr call
                # 500-ing with the screen still confidently highlighting the
                # dead button. Refuse the switch and snap the highlight back
                # to what's actually active instead.
                current = self.settings.get("input_language", constants.DEFAULT_SOURCE_LANGUAGE)
                self.logger.warning(
                    f"[NomadRight] ASR {code} has no BHASHINI model - staying on {current}")
                self._log(f"ASR {code} unavailable - staying on {current}")
                try:
                    self.board.select_radio("ASR ", f"ASR {current.capitalize()}")
                except Exception:
                    self.logger.debug("[NomadRight] language button re-sync failed", exc_info=True)
        elif msg.startswith("Bridge "):
            code = self._lang_name_to_code(msg[7:], constants.BRIDGE_LANGUAGES)
            if code:
                self.settings["bridge_language"] = code
                self.logger.info(f"[NomadRight] Voice bridge language set to {code}")
        elif msg == "Camera":
            self._on_camera_pressed()
        elif msg == "Chatbot":
            self._on_chatbot_pressed()
        elif msg.startswith("FormCmd "):
            self._form_cmd = msg[8:].strip().lower()
        elif msg == "Replay":
            self._replay_last_answer()
        elif msg == "Home":
            self._on_home_pressed()
        elif msg.startswith("DocText "):
            self.submit_text_query(msg[8:])

    def submit_text_query(self, text: str) -> None:
        """
        Entry point for the HDMI UI's on-screen-keyboard question - lets a
        worker type a follow-up instead of holding the trigger button and
        speaking (see the Document Scanner Q&A view). Safe to call from any
        thread (the UI callback thread, via ui_cb above).
        """
        text = (text or "").strip()
        if not text:
            return
        self._text_query_pending = text
        self._text_query_event.set()

    def _replay_last_answer(self) -> None:
        """'Listen again' on the answer screen: plays the last spoken answer once
        more from memory (no new synthesis), only while the kiosk is idle."""
        wav = getattr(self, "_last_answer_wav", None)
        if not wav or self._mode != "HOME" or self._camera_busy.is_set():
            self.logger.info(f"[NomadRight] replay refused (answer={'yes' if wav else 'no'}, mode={self._mode})")
            return
        self._log("REPLAY    last answer")
        self.board.statusbar("[SPEAKING] Home=stop")
        self._play(wav)
        self.board.statusbar("[READY]")

    def _prewarm_qwen(self) -> None:
        client = getattr(getattr(self, "workflow", None), "qwen_client", None)
        if client is None or not hasattr(client, "prewarm"):
            return
        try:
            secs = client.prewarm()
            if secs > 1.0:
                self._log(f"QWEN      loaded {secs:.1f}s")
        except Exception as exc:
            self.logger.debug(f"[NomadRight] Qwen pre-warm skipped: {exc}")

    def _on_camera_pressed(self) -> None:
        """
        Camera button = assisted form filling (constants.FORM_FILLING_ENABLED):
        capture the form the person is holding up and let run() drive the flow
        (formfill/flow.py). Falls back to the chatbot flow when the feature is
        off. Runs on the UI callback thread.
        """
        if not constants.FORM_FILLING_ENABLED:
            return self._on_chatbot_pressed()
        lang = self.settings.get("input_language", constants.DEFAULT_SOURCE_LANGUAGE)
        if lang not in constants.FORM_LANGUAGES:
            cat = self._catalog()
            msg = cat.prompt("need_language", lang) if cat else "Please choose Hindi or Tamil to fill a form by voice."
            self.board.update_screen(mode="FORM FILLING", top="Choose Hindi or Tamil", bottom=msg, status="[FORM] Language not supported")
            self._log("FORM: language not supported")
            return
        self._stop_audio()
        self._camera_busy.set()
        try:
            self.board.update_screen(mode="FORM FILLING", top="Capturing...", bottom="Hold the form flat and steady", status="[CAPTURING]")
            self._log("CAMERA pressed - form capture")
            try:
                frame = self.board.camera_frame() if hasattr(self.board, "camera_frame") else None
                test_image = os.environ.get("NOMADRIGHT_FORM_TEST_IMAGE")
                if test_image:
                    # test hook: a file stands in for the webcam so the form flow can be
                    # driven end to end without a printed form (never set on a real kiosk)
                    import cv2
                    frame = cv2.imread(test_image)
                    self._log(f"FORM test image {os.path.basename(test_image)}")
            except Exception as exc:
                self.logger.error(f"[NomadRight] Camera capture failed: {exc}")
                frame = None
            if frame is None:
                self.board.update_screen(top="CAMERA ERROR", bottom="No frame captured. Camera=retry, Home=cancel", status="[ERROR] Camera unavailable")
                self._mode = "HOME"
                return
            with self._image_lock:
                self._form_image = frame
            self._form_event.set()
            self._mode = "FORM FILLING"
            self.board.update_screen(top="FORM CAPTURED", bottom="Reading the form...", status="[FORM] Reading")
        finally:
            self._camera_busy.clear()

    def _on_chatbot_pressed(self) -> None:
        """
        Touchscreen Camera button handler: captures a form photo via the
        board's existing camera_frame_jpg() (boards/base.py - USB webcam,
        currently a SunplusIT "ABWB1002 PC WebCam", see boards/jetson.py's
        V4L_CAMERA_NAME comment for the hardware history/probing fallback;
        no new camera-driver code needed here) and puts the app into
        "waiting for your query" state -
        this IS the Document Scanner mode's entry point (see module
        docstring). Runs on the UI subprocess's callback thread, not the
        main run() loop thread - board.top_text/bottom_text/statusbar calls
        are thread-safe (see ui/handheld.py's RemoteUI rpc_lock).

        Single-frame only, deliberately: neither camera_frame_jpg() nor
        qwen_client.answer_vision() support multiple images per call, and
        this device's memory budget doesn't comfortably support buffering
        several full-resolution frames - see the multi-frame note in the
        final report for why this was scoped out rather than half-built.
        """
        # Any audio still playing from a *previous* turn must not keep
        # talking over a fresh scan - matches Home's own "stop audio first"
        # behavior for the same reason (see _on_home_pressed).
        self._stop_audio()
        self._camera_busy.set()
        # Qwen is only resident while someone is using the camera (LLM_KEEP_ALIVE):
        # start loading it now, in the background - framing the document and asking
        # the question takes longer than the load, so the answer still comes warm.
        threading.Thread(target=self._prewarm_qwen, name="qwen-prewarm", daemon=True).start()
        try:
            self.board.update_screen(mode="DOCUMENT SCANNER", top="Capturing...",
                                      bottom="Hold camera steady", status="[CAPTURING]")
            self._log("CAMERA pressed - capturing")
            capture_start = time.time()
            try:
                # Raw frame (not camera_frame_jpg()'s pre-encoded JPEG) so
                # auto_crop_document() can run on the actual pixel data
                # before compression - camera_frame_jpg() stays untouched
                # for the live preview stream (boards/hdmi.py), which must
                # keep showing the worker's real, uncropped framing while
                # they position the document. Boards without the raw-frame
                # API (older board classes, the ui_navigation/lcd_log test
                # boards) keep the original camera_frame_jpg() path.
                if hasattr(self.board, "camera_frame"):
                    frame = self.board.camera_frame()
                    if frame is None:
                        image_jpg = None
                    else:
                        cropped = document_crop.auto_crop_document(frame)
                        ok, buffer = cv2.imencode(".jpg", cropped)
                        image_jpg = bytearray(buffer) if ok else None
                else:
                    image_jpg = self.board.camera_frame_jpg()
            except Exception as exc:
                self.logger.error(f"[NomadRight] Camera capture failed: {exc}")
                self._log(f"CAMERA FAILED: {exc}"[:52])
                self.board.update_screen(top="CAMERA ERROR", bottom="Capture failed. Camera=retry, Home=cancel",
                                          status="[ERROR] Camera unavailable")
                self._mode = "HOME"
                return
            if not image_jpg:
                self.logger.warning("[NomadRight] Camera returned no frame.")
                self._log("CAMERA returned no frame")
                self.board.update_screen(top="CAMERA ERROR", bottom="No frame captured. Camera=retry, Home=cancel",
                                          status="[ERROR] Camera unavailable")
                self._mode = "HOME"
                return

            with self._image_lock:
                self.pending_form_image = bytes(image_jpg)
                self.pending_form_image_ts = time.time()
            self.logger.info(
                f"[NomadRight] Photo captured ({len(image_jpg)} bytes JPEG), "
                f"awaiting follow-up query."
            )
            self._log(
                f"PHOTO CAPTURED {len(image_jpg) // 1024} KB  "
                f"{time.time() - capture_start:.1f}s"
            )
            self._mode = "DOCUMENT SCANNER"
            # No checkmark/symbol here on purpose: the LCD's bitmap font
            # (NotoSansDevanagari-Regular-12.pcf) doesn't include a glyph
            # for U+2713 (checkmark) - confirmed via get_glyph() returning
            # None - an earlier version of this used one and it silently
            # contributed 0 width to the text-wrap calculation while still
            # occupying a character slot, which is the kind of thing that
            # can desync wrapping from what's actually drawn. Plain ASCII
            # only, matching every other screen in this app.
            self.board.update_screen(top="PHOTO CAPTURED", bottom="Hold button to ask, or Home to cancel",
                                      status="[WAITING FOR QUERY] Ask about the photo")
        finally:
            # Cleared last, after pending_form_image and the "Photo
            # captured" screen text are both fully in place (or on any
            # early-return error path above) - run() only starts listening
            # once this clears.
            self._camera_busy.clear()

    def _reset_conversation(self, reason: str) -> None:
        """Forget the previous person's facts and the scheme being discussed."""
        self.last_scheme_code = None
        self.last_answer_en = ""
        si = getattr(self.workflow, "scheme_intel", None)
        if si is not None:
            try:
                si.reset_session()
            except Exception:
                self.logger.debug("[NomadRight] scheme-intel session reset failed", exc_info=True)
        self.logger.info(f"[NomadRight] Conversation reset ({reason}).")

    def _on_home_pressed(self) -> None:
        """
        Touchscreen Home button handler - the app's single, always-
        available cancel/back/stop control (see module docstring's UI
        modes section). Runs on the UI callback thread, concurrently with
        whatever the main run() loop is doing. Two jobs, in priority order:

          1. If audio is currently playing, stop it immediately (see
             _stop_audio()) - the most likely reason a worker presses Home
             mid-turn is not wanting to hear the rest of an answer.
          2. Either way, clear any pending document-scanner photo and
             request a return to the Home/ready screen - run() checks
             self._home_requested at safe checkpoints (never mid-inference;
             a live Qwen/RAG call can't be safely aborted once sent) and
             drops back to Home there instead of speaking a stale answer.

        Screen is updated immediately in every case where a turn was
        actually in flight (not just the audio/photo cases) - this thread
        is the only one that can respond instantly. run()'s main loop
        remains synchronously blocked inside whatever ASR/NMT/RAG/Qwen
        call was already in progress and won't notice self._home_requested
        until that call returns on its own, however long that takes; the
        worker seeing this screen flip to READY right away - not the
        eventual silent discard once the stale call finishes - is what
        makes Cancel feel instant instead of "did that even do anything?".
        """
        was_already_home = self._mode == "HOME"
        stopped_audio = self._stop_audio()
        self._home_requested.set()
        # Home is the boundary between two people at the kiosk: nothing said so far
        # (state, occupation, held cards, the scheme being discussed) may leak into
        # the next person's answers.
        self._reset_conversation("home")
        with self._image_lock:
            had_pending_photo = self.pending_form_image is not None
            self.pending_form_image = None
        self._mode = "HOME"
        if stopped_audio:
            self.logger.info("[NomadRight] Home pressed - audio playback stopped.")
            self.board.update_screen(top="AUDIO STOPPED", bottom=self.HOME_HINT, status="[READY]")
        elif had_pending_photo:
            self.logger.info("[NomadRight] Home pressed - Document Scanner cancelled.")
            self.board.update_screen(top="CANCELLED", bottom=self.HOME_HINT, status="[READY]")
        elif not was_already_home:
            self.logger.info("[NomadRight] Home pressed - cancelling in-flight turn.")
            self.board.update_screen(top="CANCELLED", bottom=self.HOME_HINT, status="[READY]")
        else:
            self.logger.info("[NomadRight] Home pressed - already at Home.")

    # ── Assisted form filling ────────────────────────────────────────────────

    def _catalog(self):
        if self._form_catalog is None:
            try:
                from pocketinfer.applications.nomad_right.formfill.catalog import FormCatalog
                self._form_catalog = FormCatalog()
            except Exception as exc:
                self.logger.error(f"[NomadRight] form catalogue unavailable: {exc}")
        return self._form_catalog

    def _start_outbox(self) -> None:
        """Sealed forms waiting for the office PC: retried in the background; the
        officer's screen gets the pending/failed counts with every attempt."""
        try:
            from pocketinfer.applications.nomad_right.formfill.outbox import PairingConfig
            cfg = PairingConfig()
            self.form_outbox = cfg.outbox()
            self._log("RECEIVER  paired" if cfg.ready else "RECEIVER  not paired (forms stay sealed locally)")
        except Exception as exc:
            self.logger.error(f"[NomadRight] outbox unavailable: {exc}")
            return

        def _flusher() -> None:
            while self.running:
                time.sleep(constants.FORM_OUTBOX_FLUSH_INTERVAL_S)
                try:
                    if self.form_outbox.entries():
                        r = self.form_outbox.flush()
                        if r.get("sent") or r.get("expired"):
                            self._log(f"OUTBOX    sent {r['sent']} expired {r['expired']}")
                    if self.form_outbox.transport is not None:
                        self.form_outbox.transport.send_status(self.form_outbox.status())
                except Exception:
                    self.logger.debug("[NomadRight] outbox flush failed", exc_info=True)
        threading.Thread(target=_flusher, name="form-outbox", daemon=True).start()

    def _form_prior_scheme(self) -> Optional[str]:
        """The scheme the conversation settled on (a prior for identification, never a decision)."""
        code = self.last_scheme_code
        if not code:
            return None
        if code.startswith("SCH_"):
            return code
        si = getattr(self.workflow, "scheme_intel", None)
        try:
            return si.repo.sid_for_legacy(code) if si is not None else None
        except Exception:
            return None

    def _form_listen(self, lang: str, timeout: float) -> str:
        """Hold-to-talk answer for the form flow: on-screen command buttons count as
        spoken command words; Home ends the session."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._home_requested.is_set():
                return FORM_HOME
            cmd_word = self._take_form_cmd(lang)
            if cmd_word:
                return cmd_word
            self.board.wait_for_trigger_button_down(timeout=0.2)
            if self.board.trigger_button:
                self.board.button_led(True)
                self.board.statusbar("[LISTENING]")
                self.board.audio.start(max_seconds=constants.MAX_AUDIO_RECORD_SECONDS)
                self.board.wait_for_trigger_button_up()
                self.board.button_led(False)
                self.board.audio.stop()
                self.board.statusbar("[PROCESSING] Recognizing speech")
                t0 = time.time()
                text = self.bridge.listen(self.board.audio.to_audio_data().get_wav_data(), lang)
                self._latency(f"form_asr_{lang}", t0)
                self._log(f"FORM ASR {len(text)} chars {time.time() - t0:.1f}s")
                return text
        return FORM_TIMEOUT

    def _take_form_cmd(self, lang: str) -> Optional[str]:
        """The on-screen Repeat / Change / Skip / Cancel button, as the spoken word the
        session understands in this language (None when nothing was pressed)."""
        if not self._form_cmd:
            return None
        cmd, self._form_cmd = self._form_cmd, None
        cat = self._catalog()
        words = (cat.words.get(cmd, {}).get(lang) if cat else None) or [cmd]
        self._log(f"FORM cmd {cmd}")
        return words[0]

    def _form_capture(self, lang: str, timeout: float):
        """The person shows a document and presses the button: one frame, cropped."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._home_requested.is_set():
                return None
            cmd_word = self._take_form_cmd(lang)
            if cmd_word:
                return cmd_word                # skip / repeat / change / cancel instead of a document
            self.board.wait_for_trigger_button_down(timeout=0.2)
            if self.board.trigger_button:
                self.board.wait_for_trigger_button_up()
                try:
                    frame = self.board.camera_frame()
                    return document_crop.auto_crop_document(frame) if frame is not None else None
                except Exception as exc:
                    self.logger.error(f"[NomadRight] document capture failed: {exc}")
                    return None
        return None

    def _form_speak(self, lang: str, text: str) -> bool:
        """Speaks a form prompt. A press on Hold-and-speak or on one of the command
        buttons while it is still speaking stops the speech at once, so the person
        does not have to wait for the end of a long sentence; the press is then
        picked up by _form_listen / _form_capture. Home cancels."""
        if self._home_requested.is_set():
            return False
        t0 = time.time()
        wav = self.bridge.speak(text, lang)
        self._latency(f"form_tts_{lang}", t0)
        result = {"ok": True}
        player = threading.Thread(target=lambda: result.__setitem__("ok", self._play(wav)), daemon=True)
        player.start()
        interrupted = False
        while player.is_alive():
            if self._home_requested.is_set() or self.board.trigger_button or self._form_cmd:
                # stop as soon as the player exists; keep polling until the playback
                # thread has really ended so the next prompt never overlaps this one
                if self._stop_audio():
                    interrupted = True
            player.join(timeout=0.05)
        if self._home_requested.is_set():
            return False
        if interrupted:
            self._log("FORM speech interrupted by the button")
        elif not result["ok"]:
            # a playback hiccup must not end the session; the text is on the screen
            self.logger.warning("[NomadRight] form prompt playback failed - continuing")
        return True

    def _run_form_flow(self, lang: str) -> None:
        with self._image_lock:
            image, self._form_image = self._form_image, None
        self._form_event.clear()
        self._form_cmd = None
        cat = self._catalog()
        if image is None or cat is None:
            self.board.update_screen(mode="HOME", status="[READY]")
            return
        self._mode = "FORM FILLING"
        self._set_answer(None)
        io = FlowIO(speak=lambda text: self._form_speak(lang, text),
                    listen=lambda timeout: self._form_listen(lang, timeout),
                    capture=lambda timeout: self._form_capture(lang, timeout),
                    ui=self.board.set_form_state,
                    status=self.board.statusbar)
        flow = FormFlow(cat, io, lang, outbox=self.form_outbox)
        t0 = time.time()
        self._log(f"FORM flow start ({lang})")
        try:
            result = flow.run(image, prior_scheme_id=self._form_prior_scheme())
        except Exception as exc:
            self.logger.error(f"[NomadRight] form flow failed: {exc}", exc_info=True)
            result = None
        finally:
            del image
            try:
                self.board.set_form_state(None)
            except Exception:
                pass
        if result is not None:
            self._latency("form_total", t0)
            self._log(f"FORM {result.outcome} {result.form_id or '-'} {time.time() - t0:.0f}s")
            self.logger.info(f"[FORM] outcome={result.outcome} form={result.form_id} timings={result.timings}")
        self._home_requested.clear()
        self._mode = "HOME"
        self.board.update_screen(mode="HOME", status="[READY]", top="NomadRight", bottom=self.HOME_HINT)

    def _wait_for_next_turn(self) -> str:
        """
        Blocks until either the physical trigger button is pressed or a
        text query is submitted from the HDMI UI's on-screen keyboard
        (submit_text_query()) - whichever comes first. Returns 'button',
        'text', or 'stop' (self.running went False while waiting).

        Polls wait_for_trigger_button_down() with a short timeout rather
        than blocking on it indefinitely, so a text query submitted while
        idle is picked up promptly instead of waiting on a button press
        that may never come. Board.wait_for_trigger_button_down()
        (boards/base.py) clears-then-checks-then-waits internally on every
        call specifically to make this kind of repeated short-timeout
        polling race-free (see its own docstring).
        """
        while self.running:
            if self._form_event.is_set():
                return "form"
            if self._text_query_event.is_set():
                return "text"
            self.board.wait_for_trigger_button_down(timeout=0.2)
            if self.board.trigger_button:
                return "button"
        return "stop"

    def _stop_audio(self) -> bool:
        """
        Immediately stops any in-progress speaker playback by killing the
        underlying ffplay subprocess (see audio.py's AudioPlayer.stop()).
        Safe to call from any thread, and safe to call when nothing is
        playing (no-op, returns False). Used by both _on_home_pressed()
        (Home button) and _on_camera_pressed() (a fresh scan shouldn't
        have the previous turn's answer still talking over it).
        """
        with self._player_lock:
            player = self._current_player
        if player is None:
            return False
        self._audio_stop_requested.set()
        player.stop()
        return True

    def _set_answer(self, native: Optional[str], en: str = "", label: str = "") -> None:
        """Publishes the answer in the worker's own language for the HDMI UI
        (boards/hdmi.py's answer_text()); None clears it at the start of a turn.
        Boards without the method (LCD, the test boards) are unaffected."""
        fn = getattr(self.board, "answer_text", None)
        if fn is None:
            return
        try:
            fn(native or "", en, label)
        except Exception:
            self.logger.debug("[NomadRight] answer publish failed", exc_info=True)

    @staticmethod
    def _looks_english(text: str) -> bool:
        """True for text typed on the Latin on-screen keyboard: the NMT
        indic->en model receives only Indic-script text; ASCII input is
        already in the Decision Layer's language."""
        return bool(text) and all(ord(ch) < 128 for ch in text)

    @staticmethod
    def _lang_name_to_code(name: str, table: Dict[str, str]) -> Optional[str]:
        """Resolves a button label ('Hindi', 'hi') to its BHASHINI language code."""
        name_lower = name.strip().lower()
        for code, display_name in table.items():
            if display_name.lower() == name_lower or code.lower() == name_lower:
                return code
        return None

    # ── Audio playout ───────────────────────────────────────────────────────

    def _play(self, wav_bytes: bytes) -> bool:
        """
        Plays raw WAV bytes (as returned by BhashiniBridge.speak) on the
        speaker. Cancellable: _stop_audio() (Home button, see ui_cb) can
        interrupt playback mid-clip from another thread by killing the
        underlying ffplay subprocess - sound stops right away, and
        AudioPlayer.__exit__'s own wait() returns immediately once the
        process is gone, so this never hangs.

        Returns True if playback completed normally, False if it was
        stopped by the user or failed outright - run() uses this to avoid
        overwriting the "AUDIO STOPPED" screen _on_home_pressed() already
        showed with a normal "[READY]" message.
        """
        if not wav_bytes:
            return True
        self._audio_stop_requested.clear()
        try:
            wave_obj = wave.open(BytesIO(wav_bytes), "rb")
            with AudioPlayer(wave_obj.getframerate(), self.board.alsa_playback_device) as player:
                with self._player_lock:
                    self._current_player = player
                player.play(wave_obj.readframes(wave_obj.getnframes()))
            return not self._audio_stop_requested.is_set()
        except Exception as exc:
            self.logger.error(f"[NomadRight] Audio playback failed: {exc}")
            return False
        finally:
            with self._player_lock:
                self._current_player = None

    # ── Main loop ────────────────────────────────────────────────────────────

    @staticmethod
    def _latency(stage: str, start: float) -> None:
        """Feeds the bounded P50/P95/P99 stage recorder (scheme_intel/metrics.py). Never raises."""
        try:
            LATENCY.record(stage, (time.time() - start) * 1000.0)
        except Exception:
            pass

    def run(self) -> None:
        """
        Main application thread execution loop blocking on trigger button
        events.

        UI modes (see also __init__'s mode/navigation state and ui_cb):
          HOME               - idle/ready screen; both actions available:
                                hold the trigger button for Voice
                                Translation, or press the Camera icon for
                                Document Scanner. This is the state at the
                                top of every loop iteration unless a photo
                                is still pending an answer.
          VOICE TRANSLATION  - one voice-query turn, exactly the original
                                pipeline (unchanged - see module docstring).
          DOCUMENT SCANNER   - a photo is pending (or was just answered),
                                entered via the Camera icon (_on_camera_pressed).
        The Home icon (_on_home_pressed) is the single always-available
        cancel/back/stop-audio control for both modes.
        """
        self.board.clear_screen()
        self.board.update_screen(mode="HOME", top="NomadRight", bottom=self.HOME_HINT)
        # The one line that answers "is it ready for me to press the
        # button?" without a terminal attached. Logged once here rather than
        # per loop iteration - every turn already ends with its own ANSWER
        # line, so repeating this each time would only push real history off
        # the top of the page.
        self._log("READY - hold button to speak")

        while self.running:
            lang = self.settings.get("input_language", constants.DEFAULT_SOURCE_LANGUAGE)
            bridge_lang = self.settings.get("bridge_language", constants.DEFAULT_BRIDGE_LANGUAGE)

            with self._image_lock:
                scanner_pending = self.pending_form_image is not None
            if scanner_pending:
                # A photo is still waiting on its follow-up question -
                # Home hasn't cancelled it, so stay visibly in Document
                # Scanner mode rather than reverting to the generic Home
                # screen (_on_camera_pressed already set this text; this
                # just keeps it correct if we looped back here via an
                # ASR-empty retry - see step 1 below).
                self._mode = "DOCUMENT SCANNER"
                self.board.statusbar("[WAITING FOR QUERY] Ask about the photo")
            else:
                self._mode = "HOME"
                lang_name = constants.SOURCE_LANGUAGES.get(lang, lang.upper())
                self.board.update_screen(mode="HOME", status=f"[READY] Lang:{lang_name}  Hold=ask")

            turn_kind = self._wait_for_next_turn()

            if turn_kind == "stop" or not self.running:
                break

            # The language is read again HERE, after the wait: the person chooses it
            # on the Home page while the kiosk is idle, i.e. exactly while this loop
            # was waiting - the value captured before the wait is the previous
            # person's (measured 2026-09-13: a Tamil question answered in Hindi,
            # an English one in Tamil, a form started in the wrong language).
            lang = self.settings.get("input_language", constants.DEFAULT_SOURCE_LANGUAGE)
            bridge_lang = self.settings.get("bridge_language", constants.DEFAULT_BRIDGE_LANGUAGE)

            # Cleared here - right as a real turn actually begins - not
            # before the wait above. _on_home_pressed() sets this flag
            # unconditionally on every Home press, even a harmless one
            # with nothing to cancel ("already at Home", logged when idle
            # between turns). Clearing before the wait left that flag set
            # for however long the worker then took to press the trigger
            # button next, silently discarding the following turn's
            # perfectly good answer for a Home tap that had nothing to do
            # with it - confirmed on-device: ASR/NMT/RAG/Qwen all
            # completed normally, discarded anyway at the final
            # self._home_requested check below.
            self._home_requested.clear()

            # Documented above (self._mode's own docstring) as switching to
            # "VOICE TRANSLATION" for the duration of one active turn, but
            # that assignment never actually existed - self._mode stayed
            # "HOME" for a plain voice turn's entire LISTENING/PROCESSING
            # duration (scanner_pending turns were unaffected: they already
            # got "DOCUMENT SCANNER" set above, before the wait). That gap
            # is what made _on_home_pressed()'s was_already_home check
            # silently useless for the most common case: cancelling a
            # voice turn set self._home_requested correctly, but
            # was_already_home read True (self._mode still "HOME"), so no
            # screen update ever fired - the worker stayed looking at
            # "PROCESSING" for however long the abandoned ASR/NMT/RAG/Qwen
            # call took to finish on its own, looking exactly like Cancel
            # did nothing.
            if turn_kind == "form":
                self._run_form_flow(lang)
                continue
            if not scanner_pending:
                self._mode = "VOICE TRANSLATION"

            # A Camera-button capture may still be in flight on the UI
            # callback thread (see _on_camera_pressed) - don't start
            # listening until "Photo captured" is actually confirmed on
            # screen, so a follow-up question asked too quickly can never
            # race ahead of its own photo and get answered as a normal
            # voice query instead of a vision one.
            if self._camera_busy.is_set():
                self.board.statusbar("[WAIT] Capturing photo...")
                self._camera_busy.wait(timeout=5.0)

            try:
                if turn_kind == "text":
                    # On-screen-keyboard question (Document Scanner Q&A) -
                    # skip the mic/ASR stage entirely and use the typed
                    # text directly as this turn's native_query.
                    native_query = (self._text_query_pending or "").strip()
                    self._text_query_pending = None
                    self._text_query_event.clear()
                    turn_start = time.time()
                    self._log(f"TEXT QUERY {len(native_query)} chars")

                    if not native_query:
                        self.board.update_screen(status="[ERROR]", top="Empty question",
                                                  bottom="Please try again")
                        time.sleep(1.0)
                        continue

                    self._set_answer(None)
                    self.board.top_text(f"You: {native_query}"[:80])
                    self.logger.info(f"[NomadRight] TEXT query: '{native_query}'")
                else:
                    self.board.button_led(True)
                    self._set_answer(None)
                    self.board.update_screen(status="[LISTENING]", top="", bottom="")
                    # Logged before audio.start() rather than after, so the log
                    # confirms the press registered even if opening the capture
                    # device is what goes wrong.
                    self._log("BTN DOWN - listening")

                    # Press & hold triggers the mic - matches the wired capacitive
                    # touch button on GPIO09 (see jetson_suno_sutra_expansion_pinout.png),
                    # or the HDMI UI's on-screen "tap to speak" button (see
                    # boards/hdmi.py's virtual_trigger_down()/_up()).
                    record_start = time.time()
                    self.board.audio.start(max_seconds=constants.MAX_AUDIO_RECORD_SECONDS)
                    self.board.wait_for_trigger_button_up()
                    self.board.button_led(False)
                    self.board.audio.stop()
                    record_s = time.time() - record_start
                    # Everything after the button is released counts against the
                    # end-to-end budget - the worker's own hold time doesn't.
                    turn_start = time.time()
                    self._log(f"BTN UP - recorded {record_s:.1f}s")

                    # ── 1. ASR: worker's spoken language -> native text ─────────
                    self.board.statusbar("[PROCESSING] Recognizing speech")
                    stage_start = time.time()
                    wav_bytes = self.board.audio.to_audio_data().get_wav_data()
                    native_query = self.bridge.listen(wav_bytes, lang)
                    self._log(f"ASR {lang} {len(native_query)} chars  {time.time() - stage_start:.1f}s")
                    self._latency(f"asr_{lang}", stage_start)

                    if not native_query.strip():
                        self.logger.warning("[NomadRight] ASR returned empty text.")
                        self._log("ASR empty - ask again")
                        self.board.update_screen(status="[ERROR]", top="Could not hear you",
                                                  bottom="Please try again")
                        time.sleep(1.5)
                        continue

                    # Keep the worker's own question visible on screen for the
                    # rest of this turn (not overwritten by the answer) so both
                    # sides of the exchange stay readable at a glance.
                    self.board.top_text(f"You: {native_query}"[:80])
                    self.logger.info(f"[NomadRight] ASR[{lang}] query: '{native_query}'")

                # ── 2. NMT: native language -> English for the Decision Layer ─
                self.board.statusbar("[TRANSLATING]")
                stage_start = time.time()
                query_lang = "en" if turn_kind == "text" and self._looks_english(native_query) else lang
                query_en = self.bridge.to_pipeline_language(native_query, query_lang)
                self._log(f"NMT {query_lang}->EN  {time.time() - stage_start:.1f}s")
                self._latency(f"nmt_{query_lang}_en", stage_start)

                # Cancelled during LISTENING/ASR/NMT (all fast/cheap stages,
                # a few seconds at most) - skip the expensive Decision
                # Layer entirely (RAG retrieval + a Qwen call - 2-45+s
                # observed on this device) rather than running the whole
                # thing just to discard it at the final checkpoint below.
                # ASR/NMT themselves can't be aborted mid-request (same
                # reasoning as the module docstring), but nothing has
                # committed to Qwen/RAG yet at this point, so this is real
                # saved latency and compute, not just an earlier check.
                if self._home_requested.is_set():
                    self.logger.info("[NomadRight] Home pressed - skipping Decision Layer, turn discarded")
                    self._log("HOME pressed - turn discarded")
                    continue

                # ── 3. Camera follow-up short-circuit ────────────────────────
                # A pending form photo (Camera button, see ui_cb/_on_camera_pressed)
                # claims this turn's query before anything else - the worker
                # was explicitly told "waiting for your query...." after
                # snapping the photo. A stale photo past the TTL is dropped
                # silently and this turn falls through to the normal flow.
                image_for_this_turn: Optional[bytes] = None
                with self._image_lock:
                    if self.pending_form_image is not None:
                        if time.time() - self.pending_form_image_ts <= constants.LLM_VISION_PENDING_TTL_S:
                            image_for_this_turn = self.pending_form_image
                        else:
                            self.logger.info("[NomadRight] Pending form photo expired, discarding.")
                        self.pending_form_image = None

                if image_for_this_turn is not None:
                    self.logger.info("[PIPELINE] CAMERA_FORM")
                    self.logger.info("[CAMERA] Image captured")
                    self.logger.info(f"[ASR] lang={lang} text='{native_query}'")
                    self.logger.info(f"[LANGUAGE] input={lang}")
                    self.logger.info(f"[TRANSLATION] EN: '{query_en}'")
                    self.logger.info("[QWEN_VISION] RAG NOT called - image + question sent directly to Qwen")
                    self.board.statusbar("[ANALYZING PHOTO] This may take a moment")
                    self._log("QWEN-VL analyzing photo...")
                    stage_start = time.time()
                    response_pkg: StructuredResponsePackage = self.workflow.process_vision_query(
                        query_en, image_for_this_turn, context_scheme_code=self.last_scheme_code
                    )
                    self._log(f"QWEN-VL done  {time.time() - stage_start:.1f}s")
                    self._latency("vision", stage_start)
                    self.last_answer_en = response_pkg.voice_text
                    if response_pkg.scheme_code:
                        self.last_scheme_code = response_pkg.scheme_code

                else:
                    # ── 3. Voice bridge short-circuit ────────────────────
                    # If the worker is asking to translate the last answer for a
                    # destination-state official, skip the Decision Layer entirely.
                    intent_res = self.workflow.intent_recognizer.recognize(query_en)
                    if intent_res.intent_type == IntentType.TRANSLATION_REQUEST and self.last_answer_en:
                        entities = self.workflow.entity_extractor.extract(query_en, intent_res)
                        target_lang = entities.language_code or bridge_lang
                        if target_lang not in constants.BRIDGE_LANGUAGES:
                            target_lang = bridge_lang
                        target_name = constants.BRIDGE_LANGUAGES.get(target_lang, target_lang)

                        self.board.statusbar(f"[TRANSLATING] for official ({target_name})")
                        bridged_text = self.bridge.bridge_translate(self.last_answer_en, "EN", target_lang)

                        self.board.update_screen(top="VOICE BRIDGE", bottom=bridged_text[:100],
                                                  status="[SPEAKING] Home=stop")
                        self._set_answer(bridged_text, self.last_answer_en, f"VOICE BRIDGE ({target_name})")
                        played_fully = self._play(self.bridge.speak(bridged_text, target_lang))
                        if played_fully:
                            self.board.update_screen(status="[READY]", mode="HOME")
                        # else: _on_home_pressed() already set the
                        # "AUDIO STOPPED" screen - don't overwrite it here.
                        continue

                    # ── 4. Decision Layer: Intent -> Entity -> Rules/RAG -> Response ─
                    self.logger.info("[PIPELINE] VOICE_SCHEME")
                    self.logger.info(f"[ASR] lang={lang} text='{native_query}'")
                    self.logger.info(f"[LANGUAGE] input={lang}")
                    self.logger.info(f"[TRANSLATION] EN: '{query_en}'")
                    self.board.update_screen(mode="VOICE TRANSLATION", status="[PROCESSING] Finding your answer")
                    stage_start = time.time()
                    response_pkg = self.workflow.process(
                        query_en, context_scheme_code=self.last_scheme_code,
                        original_query=native_query, response_language=lang,
                    )
                    self._log(
                        f"DECIDE {response_pkg.scheme_code or 'no-match'}  "
                        f"{time.time() - stage_start:.1f}s"
                    )
                    self._latency("decision", stage_start)
                    self.last_answer_en = response_pkg.voice_text
                    # Only update on an actual scheme match this turn - keep the
                    # previous scheme remembered across a genuinely unrelated/
                    # unmatched follow-up rather than losing it (see
                    # EntityExtractor._CONTEXT_INHERITABLE_INTENTS).
                    if response_pkg.scheme_code:
                        self.last_scheme_code = response_pkg.scheme_code

                # Home was pressed while the Qwen/RAG call above was in
                # flight - it can't be safely aborted mid-request (see
                # module docstring), but we can at least not speak an
                # answer the worker already tried to back out of, and not
                # clobber the "CANCELLED"/"AUDIO STOPPED" screen
                # _on_home_pressed() already put up.
                if self._home_requested.is_set():
                    self.logger.info("[NomadRight] Home was pressed during processing - discarding this turn's answer.")
                    self._log("HOME pressed - turn discarded")
                    continue

                # ── 5. NMT: English answer -> worker's own language ─────────
                self.board.statusbar("[TRANSLATING]")
                stage_start = time.time()
                sorry_native = constants.QUERY_SORRY_TEXT.get(lang) if getattr(response_pkg, "is_fallback", False) else None
                if sorry_native:
                    # the apology is predefined in every kiosk language - no translation
                    answer_native = sorry_native
                    self._log(f"APOLOGY   {lang} (predefined, QUERY_FALLBACK={constants.QUERY_FALLBACK})")
                else:
                    answer_native = self.bridge.from_pipeline_language(response_pkg.voice_text, lang)
                    self._log(f"NMT EN->{lang}  {time.time() - stage_start:.1f}s")
                self._latency(f"nmt_en_{lang}", stage_start)
                self.logger.info(f"[TRANSLATION_OUTPUT] {lang}: '{answer_native}'")
                self._set_answer(answer_native, response_pkg.voice_text, response_pkg.display_top_text)

                # ── 6. Display + Speak — conversation view ────────────────────
                # top_text keeps showing "You: <question>" from step 1 above
                # (deliberately not overwritten here) so the worker's own
                # question and the answer are both visible together, instead
                # of the answer replacing the question the moment it arrives.
                answer_line = f"{response_pkg.display_top_text}: {response_pkg.display_bottom_text}"
                bottom_hint = f"{answer_line}  |  Say 'translate' for the officer"
                self.board.update_screen(bottom=bottom_hint[:180],
                                          status=f"[SPEAKING] {response_pkg.display_top_text} Home=stop")
                self.logger.info("[TTS] Synthesizing and playing speaker output")
                stage_start = time.time()
                tts_wav = self.bridge.speak(answer_native, lang)
                self._last_answer_wav = tts_wav
                # Logged before playback, so the end-to-end number below
                # measures time-to-first-sound (what the worker actually
                # waits for) rather than including however long the answer
                # takes to read out loud.
                self._log(f"TTS {lang}  {time.time() - stage_start:.1f}s")
                self._log(f"ANSWER after {time.time() - turn_start:.1f}s - speaking")
                self._latency(f"tts_{lang}", stage_start)
                self._latency("end_to_end", turn_start)
                played_fully = self._play(tts_wav)

                if played_fully:
                    self.board.update_screen(status="[READY] Hold=ask  Camera=scan  Home=menu", mode="HOME")
                # else: _on_home_pressed() already put up the "AUDIO
                # STOPPED" screen - don't overwrite it here. The answer
                # text on bottom_text stays visible either way.

            except Exception as exc:
                self.logger.error(f"[NomadRight] Error in application processing loop: {exc}", exc_info=True)
                # Put the failure on the LCD log too - the whole point of
                # that page is that a failed turn is diagnosable from the
                # device. Truncated to fit one log line; the untruncated
                # traceback is in the logger call above.
                self._log(f"ERROR {type(exc).__name__}: {exc}"[:52])
                self.board.button_led(False)
                self.board.update_screen(status="[ERROR]", top="SYSTEM ERROR", bottom="Please try again")
                time.sleep(2.0)

    def stop(self) -> None:
        """Application shutdown hook."""
        super().stop()
