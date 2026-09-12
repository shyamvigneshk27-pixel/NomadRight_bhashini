#!/usr/bin/env python3
"""
NomadRight receiver - runs on the fixed office PC (Windows/Linux/macOS, Python 3.8+,
`pip install pynacl`). Accepts sealed forms from the paired kiosk over mTLS on the
local network, opens them with the receiver's private key, stores each as a JSON
file, acknowledges with the blob's SHA-256, and shows the officer what arrived and
whether the kiosk has forms waiting. No internet is used or needed.

    python receiver.py            (in the folder copied from the kiosk: receiver.json + keys)
    then open http://localhost:8443/ ... no: the dashboard is served on plain HTTP at
    http://localhost:8080/ for the officer's browser (local machine only).
"""
import base64
import datetime
import hashlib
import html
import json
import os
import ssl
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "receiver.json")))
STORE = os.path.join(HERE, CFG.get("store_dir", "received"))
os.makedirs(STORE, exist_ok=True)
STATE = {"received": [], "kiosk": {}, "started": datetime.datetime.now().isoformat(timespec="seconds"), "errors": []}
LOCK = threading.Lock()
INDEX = os.path.join(STORE, "index.json")
if os.path.exists(INDEX):
    try:
        STATE["received"] = json.load(open(INDEX))
    except ValueError:
        pass
SEEN = {r["sha256"] for r in STATE["received"]}

from nacl.public import Box, PrivateKey, PublicKey
BOX = Box(PrivateKey(base64.b64decode(open(os.path.join(HERE, CFG["receiver_private_key"])).read().strip())),
          PublicKey(base64.b64decode(open(os.path.join(HERE, CFG["kiosk_public_key"])).read().strip())))


def log(msg):
    print(f"{datetime.datetime.now().strftime('%H:%M:%S')}  {msg}", flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):        # quiet
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        peer = self.connection.getpeercert() or {}
        cn = next((v for rdn in peer.get("subject", ()) for k, v in rdn if k == "commonName"), "?")
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n)
        if self.path == "/status":
            try:
                with LOCK:
                    STATE["kiosk"] = {**json.loads(body.decode()), "seen": datetime.datetime.now().isoformat(timespec="seconds"), "cn": cn}
                return self._json(200, {"status": "ok"})
            except Exception as exc:
                return self._json(400, {"status": "bad status", "error": str(exc)})
        if self.path != "/submit":
            return self._json(404, {"status": "not found"})
        sha = hashlib.sha256(body).hexdigest()
        pid = self.headers.get("X-Payload-Id", "?"); form_id = self.headers.get("X-Form-Id", "?")
        if self.headers.get("X-Payload-Sha256", sha) != sha:
            return self._json(400, {"status": "corrupt", "sha256": sha})
        if sha in SEEN:
            log(f"duplicate {pid} ({form_id}) from {cn} - already stored")
            return self._json(200, {"status": "duplicate", "sha256": sha})
        try:
            plain = BOX.decrypt(body)
            data = json.loads(plain.decode("utf-8"))
        except Exception as exc:
            log(f"REJECTED {pid} from {cn}: cannot open ({type(exc).__name__})")
            with LOCK:
                STATE["errors"].append({"at": datetime.datetime.now().isoformat(timespec="seconds"), "payload_id": pid, "error": type(exc).__name__})
            return self._json(400, {"status": "cannot open", "sha256": sha})
        day = datetime.date.today().isoformat()
        os.makedirs(os.path.join(STORE, day), exist_ok=True)
        fn = os.path.join(STORE, day, f"{data.get('form_id', form_id)}_{data.get('session_id', pid)}.json")
        with open(fn, "w", encoding="utf-8") as f:
            json.dump({**data, "received_at": datetime.datetime.now().isoformat(timespec="seconds"), "from": cn}, f, ensure_ascii=False, indent=1)
        rec = {"at": datetime.datetime.now().isoformat(timespec="seconds"), "form_id": data.get("form_id"), "scheme_id": data.get("scheme_id"), "language": data.get("language"),
               "session_id": data.get("session_id"), "fields": len(data.get("fields", {})), "unanswered": len(data.get("unanswered_required", [])), "file": os.path.relpath(fn, HERE), "sha256": sha, "from": cn}
        with LOCK:
            STATE["received"].append(rec); SEEN.add(sha)
            json.dump(STATE["received"], open(INDEX, "w"), indent=1)
        log(f"STORED {rec['form_id']} from {cn} -> {rec['file']} ({rec['fields']} fields, {rec['unanswered']} left for the officer)")
        self._json(200, {"status": "stored", "sha256": sha})


class Dashboard(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        with LOCK:
            recs = list(reversed(STATE["received"][-100:])); kiosk = dict(STATE["kiosk"]); errors = list(STATE["errors"][-10:])
        rows = "".join(f"<tr><td>{html.escape(r['at'])}</td><td>{html.escape(str(r['form_id']))}</td><td>{html.escape(str(r['language']))}</td><td>{r['fields']}</td>"
                       f"<td>{r['unanswered']}</td><td><code>{html.escape(r['file'])}</code></td></tr>" for r in recs)
        k = (f"last contact {html.escape(kiosk.get('seen', ''))} - <b>{kiosk.get('pending', 0)} waiting on the kiosk</b>, {kiosk.get('failed', 0)} failed"
             + (f", last error: {html.escape(str(kiosk.get('last_error')))}" if kiosk.get("last_error") else "")) if kiosk else "no contact from the kiosk yet"
        err = "".join(f"<li>{html.escape(e['at'])} {html.escape(e['payload_id'])}: {html.escape(e['error'])}</li>" for e in errors)
        page = f"""<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="10"><title>NomadRight receiver</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;max-width:1100px}} table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ccc;padding:6px 8px;font-size:14px;text-align:left}} th{{background:#eef}} .k{{padding:10px;background:#f6f6f6;border-left:4px solid #2f3e9e;margin:12px 0}}</style>
<h1>NomadRight receiver</h1><div class="k">Kiosk: {k}</div><p>Running since {STATE['started']}; {len(STATE['received'])} forms stored under <code>{html.escape(STORE)}</code>.</p>
<table><tr><th>Received</th><th>Form</th><th>Lang</th><th>Fields</th><th>For officer</th><th>File</th></tr>{rows or '<tr><td colspan=6>none yet</td></tr>'}</table>
{('<h3>Rejected</h3><ul>' + err + '</ul>') if err else ''}"""
        body = page.encode("utf-8")
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)


def main():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.load_verify_locations(os.path.join(HERE, CFG["ca_cert"]))
    ctx.load_cert_chain(os.path.join(HERE, CFG["server_cert"]), os.path.join(HERE, CFG["server_key"]))
    port = int(CFG.get("port", 8443))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    dash_port = int(CFG.get("dashboard_port", 8080))
    dash = ThreadingHTTPServer(("127.0.0.1", dash_port), Dashboard)
    threading.Thread(target=dash.serve_forever, daemon=True).start()
    log(f"receiving on port {port} (mTLS, kiosk certificate required); officer dashboard: http://localhost:{dash_port}/ ; storing in {STORE}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
