"""End-to-end test of pairing -> sealed outbox -> mTLS transport -> receiver -> ACK -> delete,
with the receiver run as a real subprocess from a throwaway pairing (HOME redirected)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", ".."))
PORT = 18443


class TestTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.home = tempfile.mkdtemp(prefix="nr_pair_")
        env = {**os.environ, "HOME": cls.home}
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "pair_receiver.py"), "--receiver-ip", "127.0.0.1", "--port", str(PORT), "--device-id", "kiosk-test"],
                           env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        cls.bundle = os.path.join(cls.home, "nomadright-receiver-bundle")
        cfg = json.load(open(os.path.join(cls.bundle, "receiver.json"))); cfg["dashboard_port"] = 18080
        json.dump(cfg, open(os.path.join(cls.bundle, "receiver.json"), "w"))
        cls.config_path = os.path.join(cls.home, ".config", "nomadright", "receiver.json")
        cls.receiver = None
        cls.start_receiver()

    @classmethod
    def start_receiver(cls):
        cls.receiver = subprocess.Popen([sys.executable, "receiver.py"], cwd=cls.bundle, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(1.5)
        assert cls.receiver.poll() is None, cls.receiver.stdout.read()

    @classmethod
    def stop_receiver(cls):
        if cls.receiver and cls.receiver.poll() is None:
            cls.receiver.terminate(); cls.receiver.wait(5)

    @classmethod
    def tearDownClass(cls):
        cls.stop_receiver()
        shutil.rmtree(cls.home, ignore_errors=True)

    def outbox(self):
        from pocketinfer.applications.nomad_right.formfill.outbox import PairingConfig
        from pocketinfer.applications.nomad_right import constants
        constants.FORM_OUTBOX_DIR = os.path.join(self.home, "outbox")
        cfg = PairingConfig(self.config_path)
        self.assertTrue(cfg.ready)
        ob = cfg.outbox(); ob.dir = constants.FORM_OUTBOX_DIR; ob.failed_dir = os.path.join(ob.dir, "failed")
        os.makedirs(ob.failed_dir, exist_ok=True)
        return ob

    def test_1_send_ack_delete_and_duplicate(self):
        ob = self.outbox()
        payload = {"version": 1, "session_id": "abc123def456", "form_id": "FORM_PMSBY_CONSENT", "scheme_id": "SCH_PMSBY", "language": "hi",
                   "fields": {"full_name": "रवि कुमार", "mobile": "9876543210"}, "skipped": [], "unanswered_required": []}
        self.assertEqual(ob.submit(dict(payload)), "sent")
        self.assertEqual(ob.entries(), [])                                     # deleted after the ACK
        stored = [f for d, _, fs in os.walk(os.path.join(self.bundle, "received")) for f in fs if f.endswith(".json") and f != "index.json"]
        self.assertEqual(len(stored), 1)
        rec = json.load(open(os.path.join(self.bundle, "received", [d for d in os.listdir(os.path.join(self.bundle, "received")) if os.path.isdir(os.path.join(self.bundle, "received", d))][0], stored[0])))
        self.assertEqual(rec["fields"]["full_name"], "रवि कुमार"); self.assertEqual(rec["from"], "kiosk-test")
        # the same form again (same session id) is acknowledged as a duplicate and not stored twice
        self.assertEqual(ob.submit(dict(payload)), "sent")
        stored2 = [f for d, _, fs in os.walk(os.path.join(self.bundle, "received")) for f in fs if f.endswith(".json") and f != "index.json"]
        self.assertEqual(len(stored2), 1)

    def test_2_receiver_down_keeps_sealed_copy_then_retries(self):
        ob = self.outbox()
        self.stop_receiver()
        payload = {"version": 1, "session_id": "pending000001", "form_id": "FORM_APY_REGISTRATION", "scheme_id": "SCH_APY", "language": "ta", "fields": {"full_name": "லட்சுமி"}}
        self.assertEqual(ob.submit(dict(payload)), "pending")
        entries = ob.entries(); self.assertEqual(len(entries), 1); self.assertEqual(entries[0]["attempts"], 1)
        sealed = open(entries[0]["path"], "rb").read()
        self.assertNotIn("லட்சுமி".encode("utf-8"), sealed)                     # sealed, not plaintext
        self.assertNotIn(b"full_name", sealed)
        r = ob.flush(); self.assertEqual(r["sent"], 0)                            # backoff not elapsed / receiver down
        self.start_receiver()
        for m in ob.entries():
            json.dump({**{k: v for k, v in m.items() if k != "path"}, "next_try": 0}, open(m["path"] + ".meta", "w"))
        r = ob.flush(); self.assertEqual(r["sent"], 1); self.assertEqual(ob.entries(), [])
        st = ob.status(); self.assertEqual(st["pending"], 0); self.assertIsNotNone(st["last_sent_at"])

    def test_3_wrong_client_is_rejected(self):
        from pocketinfer.applications.nomad_right.formfill.transport import Sender
        cfg = json.load(open(self.config_path))
        pki = os.path.dirname(cfg["client_cert"])
        subprocess.run(["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes", "-keyout", f"{pki}/rogue.key", "-out", f"{pki}/rogue.crt",
                        "-days", "2", "-subj", "/CN=rogue"], check=True, capture_output=True)
        rogue = Sender(cfg["host"], cfg["port"], cfg["ca_cert"], f"{pki}/rogue.crt", f"{pki}/rogue.key", cfg["server_fingerprint"])
        with self.assertRaises(Exception):
            rogue.send(b"x" * 40, "rogue1", "FORM_X")
        # right client, wrong pinned fingerprint -> refused before sending
        wrong = Sender(cfg["host"], cfg["port"], cfg["ca_cert"], cfg["client_cert"], cfg["client_key"], "00" * 32)
        with self.assertRaises(Exception):
            wrong.send(b"x" * 40, "fp1", "FORM_X")
        # right client, wrong receiver port (nothing listening) -> connection error, nothing lost
        down = Sender(cfg["host"], cfg["port"] + 1, cfg["ca_cert"], cfg["client_cert"], cfg["client_key"], cfg["server_fingerprint"], timeout=2)
        self.assertFalse(down.reachable())

    def test_4_status_reaches_dashboard(self):
        ob = self.outbox()
        self.assertTrue(ob.transport.send_status({"pending": 2, "failed": 0, "last_error": "test"}))
        import urllib.request
        page = urllib.request.urlopen("http://127.0.0.1:18080/", timeout=5).read().decode()
        self.assertIn("2 waiting on the kiosk", page)


if __name__ == "__main__":
    unittest.main(verbosity=1)
