import pyaudio
import wave
import os
import time
import threading
import numpy as np
import shutil
import subprocess
import logging 
import re
from typing import Optional
from speech_recognition import AudioData
from contextlib import contextmanager
from ctypes import CFUNCTYPE, c_char_p, c_int, cdll


ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

# pyaudio, and it's ALSA implementation, tends to drop a lot of unnecessary log info on instantiation
# They're not actionable messages, and it would take considerable rootfs tweaking to remove otherwise
# So instead, we'll suppress all libasound error messages on pyaudio startup
@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)

def alsa_devices_filtered(record=False, playback=False, blacklist=None):
    ''' Using pyaudio backend, retrieve list of audio devices
    When record=True, these will be capture devices
    When playback=True, these will be playback devices,
    Blacklist can be provided as a list of names that will be skipped
    Note: only valid ALSA devices (hw:X,Y) will be retrieved '''
    if blacklist is None:
        blacklist = []
    with noalsaerr():
        p = pyaudio.PyAudio()
    devices = []
    regex = re.compile(r'.*\(hw:(\d+),(\d+)\)')
    # Iterate across all Pyaudio devices
    for i in range(p.get_device_count()):
        devinfo = p.get_device_info_by_index(i)
        # Only accept ALSA devices that match (hw:X,Y)
        match = regex.match(devinfo['name'])
        if match is None:
            continue
        devinfo['alsa_card'] = int(match.group(1))
        devinfo['alsa_device'] = int(match.group(2))
        blacklist_skip = False
        for blacklisted_name in blacklist:
            if blacklisted_name in devinfo['name']:
                blacklist_skip = True
                break
        if blacklist_skip:
            continue
        # Recording devices must have at least one InputChannel
        if record and devinfo['maxInputChannels'] > 0:
            devices.append(devinfo)
        # Playback devices must have at least one OutputChannel
        elif playback and devinfo['maxOutputChannels'] > 0:
            devices.append(devinfo)
    return devices

# Default list of amixer simple controls that will be set automatically by set_volume
CONTROLS = [
    'PCM', 'Speaker', 'Headphone', 'Line', 'Mic', 'Capture', 'Digital',
]

def set_volume(alsa_card, volume, controls=None):
    ''' Set volume of ALL inputs or outputs of alsa_card to volume
     Volume should be an integer 0-100
      controls may be a  list of controls to set, defaults to an expansive list of defaults.
      Returns True if at least one control was actually set, False otherwise (e.g. no
      matching control found) - used by boards/hdmi.py's set_volume_live() to report
      success/failure to the UI instead of assuming it always worked. '''
    if controls is None:
        controls = CONTROLS
    elif isinstance(controls, str):
        controls = controls
    controls_raw = subprocess.run(
        ['/usr/bin/amixer', '-c', str(alsa_card), 'scontrols'],
        capture_output=True, text=True).stdout
    found_controls = re.findall(r"^Simple mixer control '(.*)',(\d+)", controls_raw, flags=re.MULTILINE)
    # Always try to set Master level to 100%
    if 'Master' in [x[0] for x in found_controls]:
        ret = subprocess.run(['/usr/bin/amixer', '-c', str(alsa_card), 'sset', 'Master', '100%'], capture_output=True, text=True)
        if ret.returncode != 0:
            logging.error(f'Unable to set Master volume for card {alsa_card}: {ret.stderr}')
    set_one = False
    for control, idx in found_controls:
        if control in controls:
            set_one = True
            ret = subprocess.run(['/usr/bin/amixer', '-c', str(alsa_card), 'sset', control, f'{volume}%'], capture_output=True, text=True)
            if ret.returncode != 0:
                logging.error(f'Unable to set {control} volume for card {alsa_card}: {ret.stderr}')
    if not set_one:
        logging.warning(f'Volume not set: No controls for card {alsa_card} matched: {found_controls}')
    return set_one



CLIP_FRACTION_LIMIT = 0.001      # more than 0.1% of samples at full scale: the microphone gain is too high


def condition_speech(samples, rate, target_peak=0.891, max_gain=20.0, highpass_hz=70.0):
    """Speech clean-up for recognition, no clipping ever.

    1. The switch-on click: USB microphones produce a loud low-frequency thump
       ~150-300 ms after the device is opened. If a frame in the first 0.6 s is
       more than four times louder than the typical level of the rest, the audio
       up to 60 ms after that frame is dropped.
    2. DC offset and rumble below `highpass_hz` are removed (2nd-order Butterworth).
    3. The peak is scaled to `target_peak` (-1 dBFS), with at most `max_gain`
       amplification so a quiet room is not blown up into loud noise.
    Returns (int16 array, stats) - stats holds the raw peak, the fraction of raw
    samples at full scale, the milliseconds cut and the gain used."""
    a = np.asarray(samples, dtype=np.float64) / 32768.0
    stats = {"raw_peak": float(np.max(np.abs(a))) if len(a) else 0.0,
             "raw_clipped": float(np.mean(np.abs(a) >= 0.999)) if len(a) else 0.0,
             "cut_ms": 0, "gain": 1.0}
    if len(a) == 0:
        return np.zeros(0, dtype=np.int16), stats
    frame = max(1, int(0.02 * rate))
    head = int(0.6 * rate)
    if len(a) > head + int(0.4 * rate):
        nfr = len(a) // frame
        env = np.sqrt(np.mean((a[: nfr * frame] - np.mean(a)).reshape(nfr, frame) ** 2, axis=1))
        head_frames = head // frame
        rest = env[head_frames:]
        typical = float(np.median(rest)) if len(rest) else 0.0
        loud = np.where(env[:head_frames] > 4.0 * max(typical, 1e-4))[0]
        if len(loud):
            cut = min(head, (int(loud[-1]) + 1) * frame + int(0.06 * rate))
            a = a[cut:]
            stats["cut_ms"] = int(cut * 1000 / rate)
    a = a - np.mean(a)
    try:
        from scipy.signal import butter, sosfilt
        sos = butter(2, highpass_hz, btype="highpass", fs=rate, output="sos")
        a = sosfilt(sos, a)
    except Exception:
        # no scipy: a first-order DC blocker still removes offset and most rumble
        y = np.empty_like(a); prev_x = prev_y = 0.0; r = 0.995
        for i, x in enumerate(a):
            prev_y = x - prev_x + r * prev_y; prev_x = x; y[i] = prev_y
        a = y
    peak = float(np.max(np.abs(a))) if len(a) else 0.0
    gain = min(max_gain, target_peak / peak) if peak > 0 else 1.0
    stats["gain"] = round(gain, 2)
    out = np.clip(a * gain, -1.0, 1.0)
    return (out * 32767.0).astype(np.int16), stats


class AudioRecorder:
    def __init__(self, device_idx=0, channels=1, frames_per_buffer=1024):
        self.logger = logging.getLogger(__name__)
        self.channels = channels
        self.frames_per_buffer = frames_per_buffer
        with noalsaerr():
            self.p = pyaudio.PyAudio()
        self.device_idx = device_idx
        self.stream = None
        self.frames = []
        self.thread = threading.Thread()
        self.recording = False
        self.last_stats = {}
        self.clip_callback = None     # board hook: called with the stats when a recording clipped

        self.rate = 16000  # Default rate, will be updated to a supported rate
        # List of common rates to test 
        # Hardware usually natively supports multiples of 44100 or 48000
        test_rates = [16000, 22050, 32000, 44100, 48000, 88200, 96000, 192000]

        # Automatically select smallest supported sample rate
        # pyaudio doesn't provide a way to do this upfront, it only reports a default sample rate
        for rate in test_rates:
            try:
                # Change 'input_channels' and 'input_format' if doing stereo or float recording
                if self.p.is_format_supported(
                    rate=rate, 
                    input_device=device_idx, 
                    input_channels=channels,
                    input_format=pyaudio.paInt16
                ):
                    self.rate = rate
                    break
            except ValueError:
                continue

    def start(self, max_seconds=None):
        ''' max_seconds, if given, hard-caps how long the background thread
        keeps appending captured frames, regardless of how long the caller
        keeps `recording` True (e.g. a hold-to-speak button held down for
        an unusually long time). Without this the buffer - and eventually
        the ASR request built from it - grows unbounded; NomadRight passes
        constants.MAX_AUDIO_RECORD_SECONDS here (a limit that existed as a
        constant but was never actually wired to anything). '''
        self.logger.debug('Opening device index %s for recording', self.device_idx)
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  input_device_index=self.device_idx,
                                  frames_per_buffer=self.frames_per_buffer)
        self.frames = []
        self.max_seconds = max_seconds
        self.thread = threading.Thread(target=self._record)
        self.thread.daemon = True
        self.recording = True
        self.thread.start()

    def _record(self):
        if self.stream is None:
            return
        start_time = time.monotonic()
        try:
            while self.recording:
                if self.max_seconds is not None and (time.monotonic() - start_time) >= self.max_seconds:
                    self.logger.warning(
                        f'Recording hit max_seconds={self.max_seconds} cap - stopping capture '
                        f'(caller is still holding the trigger; only what was captured up to '
                        f'the cap will be transcribed)'
                    )
                    break
                data = self.stream.read(self.frames_per_buffer)
                self.logger.debug(f'Read {len(data)} bytes from audio stream, {len(self.frames)} frames collected so far')
                self.frames.append(data)
        finally:
            self.recording = False
        self.logger.debug("Audio recording thread exiting")


    def stop(self):
        self.logger.debug('Stopping audio recording')
        self.recording = False
        time.sleep(0.15) # Experimentlaly determined to allow last buffer to be read
        self.thread.join()
        if self.stream is not None:
            self.logger.debug("Closing audio stream")
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.logger.debug("Shutdown complete")

    def save_to_file(self, filename):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

    def to_audio_data(self):
        """The recording as 16-bit mono AudioData, conditioned for speech recognition
        (condition_speech): the microphone's switch-on click is cut, DC and rumble
        are filtered out and the level is normalised WITHOUT clipping. The old
        version scaled the loudest sample to full scale and then multiplied by 1.2
        on purpose ("hard-clip a little"); on this kiosk's webcam microphone, whose
        switch-on click and room noise already reach full scale, that clipped 14% of
        every recording and turned "I am 40 years old" into nonsense for Whisper."""
        byte_data = b''.join(self.frames)
        arr = np.frombuffer(byte_data, dtype=np.int16)
        if len(arr) == 0:
            return AudioData(byte_data, self.rate, 2)
        out, stats = condition_speech(arr, self.rate)
        self.last_stats = stats
        if self.clip_callback is not None and stats.get("raw_clipped", 0.0) > CLIP_FRACTION_LIMIT:
            try:
                self.clip_callback(stats)
            except Exception:
                self.logger.debug("clip callback failed", exc_info=True)
        return AudioData(out.tobytes(), self.rate, 2)

    def terminate(self):
        self.p.terminate()

class DummyAudioRecorder(AudioRecorder):
    def __init__(self, filename):
        self.filename = filename
        self.channels = 1
        self.frames_per_buffer = 1024
        self.frames = []

    def start(self, max_seconds=None):
        # max_seconds accepted for interface parity with AudioRecorder.start()
        # (app.py always passes it) - meaningless here since a dummy board
        # reads a fixed pre-recorded file, not a live unbounded stream.
        # open the file for reading.
        self.frames = []

    def stop(self):
        with wave.open(self.filename, 'rb') as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                raise ValueError("Audio file must be mono, 16-bit")
            self.rate = wf.getframerate() 
            self.frames = [wf.readframes(wf.getnframes())]


# The following was modified from piper's audio_playback.py to add device selection
class AudioPlayer:
    """Plays raw audio using ffplay."""

    def __init__(self, sample_rate: int, device: str = "default") -> None:
        """Initialzes audio player."""
        self.sample_rate = sample_rate
        self.device = device
        self._proc: Optional[subprocess.Popen] = None

    def __enter__(self):
        """Starts ffplay subprocess and returns player."""
        my_env = os.environ.copy()
        my_env["SDL_AUDIODRIVER"] = "alsa"
        my_env["AUDIODEV"] = self.device
        self._proc = subprocess.Popen(
            [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-f",
                "s16le",
                "-ar",
                str(self.sample_rate),
                "-ac",
                "1",
                "-",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=my_env
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops ffplay subprocess."""
        if self._proc:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except Exception:
                pass
            self._proc.wait(timeout=60)

    def play(self, audio_bytes: bytes) -> None:
        """Plays raw audio using ffplay."""
        assert self._proc is not None
        assert self._proc.stdin is not None

        self._proc.stdin.write(audio_bytes)
        self._proc.stdin.flush()

    def stop(self) -> None:
        """
        Immediately kills the underlying ffplay process, stopping audio
        output right away. Safe to call from a different thread than the
        one running play()/the `with` block (e.g. a UI button-press
        callback interrupting mid-playback) and safe to call more than
        once or after playback already finished on its own - both are
        no-ops. __exit__'s own proc.wait() then returns immediately since
        the process is already gone, so this never causes a hang.
        """
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    @staticmethod
    def is_available() -> bool:
        """Returns true if ffplay is available."""
        return bool(shutil.which("ffplay"))
