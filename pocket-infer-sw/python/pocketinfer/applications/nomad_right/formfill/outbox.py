"""
Outbox: completed forms sealed for the office PC and handed over safely.

A finished form is sealed the moment it is complete (NaCl Box: the kiosk's
private key + the receiver's public key, both created at pairing - see
tools/pair_receiver.py) and written as one small file; the plaintext dict is
dropped right after. Sending goes over mTLS to the fixed receiver
(transport.py); only the receiver's acknowledgement (the blob's SHA-256) deletes
the file. Failures keep the sealed file and retry with backoff, up to
constants.FORM_OUTBOX_MAX_AGE_S (24 h), then the entry is moved to failed/ and
counted for the officer's screen. No personal data is ever logged - only ids,
sizes and states.
"""
import base64
import hashlib
import json
import logging
import os
import threading
import time
from typing import Dict, List, Optional

from pocketinfer.applications.nomad_right import constants

logger = logging.getLogger(__name__)


def _now() -> float:
    return time.time()


class Outbox:
    def __init__(self, directory: Optional[str] = None, transport=None, sealer=None):
        self.dir = directory or constants.FORM_OUTBOX_DIR
        self.failed_dir = os.path.join(self.dir, "failed")
        os.makedirs(self.dir, mode=0o700, exist_ok=True)
        os.makedirs(self.failed_dir, mode=0o700, exist_ok=True)
        self.transport = transport
        self.sealer = sealer
        self.lock = threading.RLock()
        self.wake = threading.Event()          # set when something new is queued: the background sender goes at once
        self.last_sent_at: Optional[float] = None
        self.last_error: Optional[str] = None
        self.last_flush_at: Optional[float] = None

    # ── sealing ────────────────────────────────────────────────────────────
    def seal(self, payload: Dict) -> bytes:
        if self.sealer is None:
            raise RuntimeError("no sealer configured (pairing not done)")
        return self.sealer.seal(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    def _write(self, blob: bytes, meta: Dict) -> str:
        pid = meta["payload_id"]
        path = os.path.join(self.dir, f"{pid}.sealed")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(blob)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        with open(path + ".meta", "w") as f:
            json.dump(meta, f)
        os.chmod(path + ".meta", 0o600)
        return path

    # ── submit / retry ─────────────────────────────────────────────────────
    def _prepare(self, payload: Dict):
        """Seal and store one payload. A numbered picture of a session (seq > 0) replaces
        every older unsent picture of the same session: the newest carries everything."""
        sid = payload.get("session_id") or hashlib.sha256(os.urandom(16)).hexdigest()[:12]
        seq = int(payload.get("seq") or 0)
        pid = f"{sid}-{seq:03d}" if seq else sid
        blob = self.seal(payload)
        meta = {"payload_id": pid, "session_id": sid, "seq": seq, "status": payload.get("status", "complete"),
                "form_id": payload.get("form_id"), "created_at": _now(), "attempts": 0, "next_try": 0.0,
                "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
        with self.lock:
            if seq:
                for old in self.entries():
                    if old.get("session_id") == sid and int(old.get("seq") or 0) < seq:
                        self._delete(old["path"])
            path = self._write(blob, meta)
        return path, blob, meta

    def submit(self, payload: Dict) -> str:
        """Seal, store, try once. Returns 'sent' or 'pending'. The caller must drop
        its own reference to the payload afterwards."""
        path, blob, meta = self._prepare(payload)
        logger.info(f"[OUTBOX] sealed {meta['payload_id']} form={meta['form_id']} {meta['status']} {len(blob)} bytes")
        return "sent" if self._try_send(path, blob, meta) else "pending"

    def queue(self, payload: Dict) -> str:
        """Seal and store now, send in the background: the session never waits for the
        network. The kiosk's sender thread wakes up at once (see app._flusher)."""
        path, blob, meta = self._prepare(payload)
        logger.info(f"[OUTBOX] queued {meta['payload_id']} form={meta['form_id']} {meta['status']} {len(blob)} bytes")
        self.wake.set()
        return "queued"

    def _try_send(self, path: str, blob: bytes, meta: Dict) -> bool:
        if self.transport is None:
            self.last_error = "no transport configured"
            return False
        meta["attempts"] += 1
        try:
            ack = self.transport.send(blob, meta["payload_id"], meta["form_id"])
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {str(exc)[:120]}"
            logger.warning(f"[OUTBOX] send failed for {meta['payload_id']} (attempt {meta['attempts']}): {self.last_error}")
            self._schedule_retry(path, meta)
            return False
        if ack.get("status") in ("stored", "duplicate") and ack.get("sha256") == meta["sha256"]:
            self._delete(path)
            self.last_sent_at = _now(); self.last_error = None
            logger.info(f"[OUTBOX] acknowledged {meta['payload_id']} ({ack.get('status')}) - local copy removed")
            return True
        self.last_error = f"bad acknowledgement: {str(ack)[:100]}"
        logger.warning(f"[OUTBOX] {self.last_error}")
        self._schedule_retry(path, meta)
        return False

    def _schedule_retry(self, path: str, meta: Dict) -> None:
        backoff = min(constants.FORM_OUTBOX_RETRY_MAX_S, constants.FORM_OUTBOX_RETRY_BASE_S * (2 ** min(meta["attempts"] - 1, 8)))
        meta["next_try"] = _now() + backoff
        with self.lock:
            with open(path + ".meta", "w") as f:
                json.dump(meta, f)

    def _delete(self, path: str) -> None:
        with self.lock:
            for p in (path, path + ".meta"):
                try:
                    size = os.path.getsize(p)
                    with open(p, "r+b") as f:       # overwrite before unlinking
                        f.write(b"\0" * size)
                    os.unlink(p)
                except OSError:
                    pass

    def entries(self) -> List[Dict]:
        out = []
        for fn in sorted(os.listdir(self.dir)):
            if fn.endswith(".sealed"):
                try:
                    meta = json.load(open(os.path.join(self.dir, fn + ".meta")))
                except (OSError, ValueError):
                    meta = {"payload_id": fn[:-7], "attempts": 0, "next_try": 0, "created_at": os.path.getmtime(os.path.join(self.dir, fn))}
                meta["path"] = os.path.join(self.dir, fn)
                out.append(meta)
        return out

    def flush(self) -> Dict[str, int]:
        """Retry every pending entry whose backoff has elapsed; expire the old ones."""
        self.last_flush_at = _now()
        sent = expired = skipped = 0
        for meta in self.entries():
            age = _now() - meta.get("created_at", _now())
            if age > constants.FORM_OUTBOX_MAX_AGE_S:
                self._expire(meta); expired += 1; continue
            if _now() < meta.get("next_try", 0):
                skipped += 1; continue
            try:
                blob = open(meta["path"], "rb").read()
            except OSError:
                continue
            if self._try_send(meta["path"], blob, meta):
                sent += 1
        return {"sent": sent, "expired": expired, "waiting": skipped}

    def _expire(self, meta: Dict) -> None:
        with self.lock:
            for suffix in ("", ".meta"):
                src = meta["path"] + suffix
                if os.path.exists(src):
                    os.replace(src, os.path.join(self.failed_dir, os.path.basename(src)))
        logger.error(f"[OUTBOX] {meta['payload_id']} not delivered within {constants.FORM_OUTBOX_MAX_AGE_S / 3600:.0f} h - moved to failed/ (still sealed)")

    def status(self) -> Dict:
        pending = self.entries()
        failed = [f for f in os.listdir(self.failed_dir) if f.endswith(".sealed")]
        return {"pending": len(pending), "failed": len(failed), "oldest_pending_s": (round(_now() - min(m.get("created_at", _now()) for m in pending)) if pending else 0),
                "last_sent_at": self.last_sent_at, "last_error": self.last_error}


class NaclSealer:
    """Box(kiosk private key, receiver public key): only the receiver can open it."""

    def __init__(self, kiosk_private_key_path: str, receiver_public_key_path: str):
        from nacl.public import Box, PrivateKey, PublicKey
        sk = PrivateKey(base64.b64decode(open(kiosk_private_key_path).read().strip()))
        pk = PublicKey(base64.b64decode(open(receiver_public_key_path).read().strip()))
        self.box = Box(sk, pk)

    def seal(self, data: bytes) -> bytes:
        return bytes(self.box.encrypt(data))       # nonce + ciphertext + MAC


class PairingConfig:
    """Everything the kiosk needs to talk to its one receiver; created by tools/pair_receiver.py."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or constants.FORM_RECEIVER_CONFIG
        self.data = json.load(open(self.path)) if os.path.exists(self.path) else None

    @property
    def ready(self) -> bool:
        return bool(self.data)

    def outbox(self) -> Outbox:
        if not self.ready:
            return Outbox(transport=None, sealer=None)
        from pocketinfer.applications.nomad_right.formfill.transport import Sender
        d = self.data
        sealer = NaclSealer(d["kiosk_private_key"], d["receiver_public_key"])
        sender = Sender(d["host"], d["port"], d["ca_cert"], d["client_cert"], d["client_key"], d.get("server_fingerprint"),
                        device_id=d.get("device_id", "kiosk"), timeout=constants.FORM_SEND_TIMEOUT_S)
        return Outbox(transport=sender, sealer=sealer)
