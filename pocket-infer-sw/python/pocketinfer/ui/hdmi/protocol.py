"""
WebSocket message protocol between the HDMI bridge server
(pocketinfer.ui.hdmi.server.HDMIBridgeServer) and the React UI
(/home/stark-x/UI, see its src/services/backendClient.ts).

Two independent WebSocket connections:
  /ws/ui        - full app state <-> touch/virtual-button commands
  /ws/terminal  - PIN-gated PTY shell (see server.py's _run_terminal_session)

Every message on both sockets is a JSON object with a "type" field taken
from the constants below. This module is the single source of truth for
those type strings and payload shapes - keep this file and
UI/src/services/backendClient.ts's TERM_*/MSG_*/CMD_* constants in sync by
hand, since there's no shared schema generation in this codebase.
"""

# ---- Server -> Client (/ws/ui) --------------------------------------------

# {type, state: UIState.to_dict()} - sent once, immediately on connect, so
# a (re)connecting browser tab never has to guess at what's currently on
# screen.
MSG_STATE_FULL = "state_full"

# {type, patch: {...}} - a partial UIState update; the client shallow-merges
# `patch` into its local copy of state. Every Board method that mutates
# UIState (see boards/hdmi.py) sends one of these.
MSG_STATE_PATCH = "state_patch"

# {type, text} - one line appended to the on-screen pipeline log (the HDMI
# equivalent of ui/handheld.py's log page).
MSG_LOG_LINE = "log_line"

# {type} - blank the pipeline log.
MSG_LOG_CLEAR = "log_clear"

# {type, prefix, name} - highlight `name` within Settings radio group
# `prefix` (mirrors Board.select_radio, e.g. prefix="ASR " name="ASR hi").
MSG_RADIO_SELECT = "radio_select"

# {type, status: {id: "PASSED"|"FAILED"}} - live system/hardware health,
# replaces mockDiagnosticsService's canned SystemStatus.
MSG_SYSTEM_STATUS = "system_status"

# {type, jpeg_b64} - one live camera preview frame (Document Scanner
# screen), only sent while a client has requested preview streaming (see
# CMD_CAMERA_PREVIEW).
MSG_CAMERA_FRAME = "camera_frame"

# {type, results: [{id, name, category, status, details}]} - result of an
# on-demand Settings > Diagnostics run.
MSG_DIAGNOSTICS_RESULT = "diagnostics_result"

# {type, message} - a server-side error worth surfacing to the UI.
MSG_ERROR = "error"

# ---- Client -> Server (/ws/ui) --------------------------------------------

# Touch/virtual equivalents of the physical trigger button - see
# boards/hdmi.py's virtual_trigger_down()/_up(). The real GPIO button
# keeps working unchanged alongside these.
CMD_TRIGGER_DOWN = "trigger_down"
CMD_TRIGGER_UP = "trigger_up"

# Touch equivalents of the old ILI9341 topbar's Camera/Home icons.
CMD_CAMERA_PRESS = "camera_press"
CMD_HOME_PRESS = "home_press"

# {type, code} - worker input (ASR) language / voice-bridge (for
# officials) language, code is a bare BHASHINI language code (e.g. "hi"),
# not a display name - see boards/hdmi.py's _dispatch_ui_cb.
CMD_SELECT_ASR_LANG = "select_asr_lang"
CMD_SELECT_BRIDGE_LANG = "select_bridge_lang"

# {type, pct} - live volume/brightness (0-100), see boards/hdmi.py's
# set_volume_live()/set_brightness().
CMD_SET_VOLUME = "set_volume"
CMD_SET_BRIGHTNESS = "set_brightness"

# {type} - Admin panel "Exit Application": closes the kiosk browser and
# stops pocketinfer-service entirely (see boards/hdmi.py's exit_app()).
# There is no other way to leave a fullscreen kiosk browser with no
# window chrome and no keyboard attached, so this - PIN-gated, same as
# the rest of the Admin tab - is the escape hatch. To run again, launch
# `pocketinfer-service` again from a terminal.
CMD_EXIT_APP = "exit_app"

# {type} - run the real hardware/service self-test on demand
# (Settings > Diagnostics "Run Full System Test").
CMD_RUN_DIAGNOSTICS = "run_diagnostics"

# {type, enabled} - start/stop the Document Scanner live camera preview
# stream (MSG_CAMERA_FRAME) - only streamed while the Document screen is
# actually open, so the camera isn't wastefully polled otherwise.
CMD_CAMERA_PREVIEW = "camera_preview"

# {type, text} - an on-screen-keyboard follow-up question about a
# captured document photo, as an alternative to holding the trigger
# button and speaking it (see app.py's pending_form_image flow).
CMD_ASK_DOCUMENT_TEXT = "ask_document_text"

# ---- /ws/terminal ----------------------------------------------------------
# Framed JSON (not raw PTY bytes) so the PIN handshake and resize events
# can share the one socket with the actual shell I/O.

# client->server {type, pin} - must be sent (and match the real admin PIN,
# see boards/hdmi.py's DEFAULT_ADMIN_PIN) before anything else; the server
# does not spawn a shell until this succeeds.
TERM_PIN = "pin"
TERM_AUTH_OK = "auth_ok"      # server->client {type}
TERM_AUTH_FAIL = "auth_fail"  # server->client {type}

TERM_INPUT = "input"    # client->server {type, data} - keystrokes
TERM_RESIZE = "resize"  # client->server {type, cols, rows}
TERM_DATA = "data"      # server->client {type, data} - shell stdout/stderr
TERM_CLOSED = "closed"  # server->client {type} - the shell process exited
