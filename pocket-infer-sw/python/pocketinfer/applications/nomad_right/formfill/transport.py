"""
Sender: one sealed form to the fixed receiver over the local network.

mTLS (Python ssl, OpenSSL 3): the receiver's certificate must chain to the
pairing CA and match the pinned fingerprint; the kiosk presents its client
certificate. No internet, no DNS: the receiver is an IP address on the LAN.
"""
import hashlib
import http.client
import json
import socket
import ssl
from typing import Dict, Optional


class Sender:
    def __init__(self, host: str, port: int, ca_cert: str, client_cert: str, client_key: str,
                 server_fingerprint: Optional[str] = None, device_id: str = "kiosk", timeout: float = 15.0):
        self.host, self.port, self.timeout, self.device_id = host, int(port), timeout, device_id
        self.server_fingerprint = (server_fingerprint or "").replace(":", "").lower() or None
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.check_hostname = False               # the receiver is addressed by IP; identity = CA + pinned fingerprint
        ctx.load_verify_locations(ca_cert)
        ctx.load_cert_chain(client_cert, client_key)
        self.ctx = ctx

    def _connect(self) -> http.client.HTTPSConnection:
        conn = http.client.HTTPSConnection(self.host, self.port, timeout=self.timeout, context=self.ctx)
        conn.connect()
        if self.server_fingerprint:
            der = conn.sock.getpeercert(binary_form=True)
            fp = hashlib.sha256(der).hexdigest()
            if fp != self.server_fingerprint:
                conn.close()
                raise ssl.SSLError(f"receiver certificate fingerprint mismatch ({fp[:16]}...)")
        return conn

    def send(self, blob: bytes, payload_id: str, form_id: Optional[str]) -> Dict:
        conn = self._connect()
        try:
            headers = {"Content-Type": "application/octet-stream", "X-Payload-Id": payload_id, "X-Form-Id": form_id or "",
                       "X-Device-Id": self.device_id, "X-Payload-Sha256": hashlib.sha256(blob).hexdigest()}
            conn.request("POST", "/submit", body=blob, headers=headers)
            resp = conn.getresponse()
            body = resp.read()
            if resp.status != 200:
                raise RuntimeError(f"receiver returned {resp.status}: {body[:100]!r}")
            return json.loads(body.decode("utf-8"))
        finally:
            conn.close()

    def send_status(self, status: Dict) -> bool:
        """The kiosk's outbox counts for the officer's screen; best effort."""
        try:
            conn = self._connect()
            try:
                conn.request("POST", "/status", body=json.dumps({"device_id": self.device_id, **status}).encode("utf-8"),
                             headers={"Content-Type": "application/json", "X-Device-Id": self.device_id})
                return conn.getresponse().status == 200
            finally:
                conn.close()
        except (OSError, ssl.SSLError, socket.timeout):
            return False

    def reachable(self) -> bool:
        try:
            self._connect().close()
            return True
        except Exception:
            return False
