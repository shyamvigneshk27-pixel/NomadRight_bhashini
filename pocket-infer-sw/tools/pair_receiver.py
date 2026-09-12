#!/usr/bin/env python3
"""
One-time pairing of this kiosk with its fixed receiving PC. Run ONCE on the Jetson:

    python3 tools/pair_receiver.py --receiver-ip 172.16.104.50 [--port 8443] [--device-id kiosk-01]

It creates, with nothing sent anywhere:
  ~/.config/nomadright/pki/       a private CA, a receiver (server) certificate, a kiosk
                                  (client) certificate, and NaCl keypairs for both sides
  ~/.config/nomadright/receiver.json   what the kiosk uses (paths, IP, port, pinned fingerprint)
  ~/nomadright-receiver-bundle/   the folder to copy to the PC (USB stick / scp) - it holds
                                  the receiver's private key, its certificate, the CA, the
                                  kiosk's public key and receiver.py
Re-running replaces everything (a new pairing); the old bundle stops working.
"""
import argparse
import base64
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys

HOME = os.path.expanduser("~")
PKI = os.path.join(HOME, ".config", "nomadright", "pki")
CONFIG = os.path.join(HOME, ".config", "nomadright", "receiver.json")
BUNDLE = os.path.join(HOME, "nomadright-receiver-bundle")
HERE = os.path.dirname(os.path.abspath(__file__))


def sh(*args):
    subprocess.run(args, check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receiver-ip", required=True, help="fixed LAN IP of the receiving PC")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--device-id", default="kiosk-01")
    ap.add_argument("--days", type=int, default=3650)
    a = ap.parse_args()
    from nacl.public import PrivateKey
    os.makedirs(PKI, mode=0o700, exist_ok=True)
    os.chmod(PKI, 0o700)
    ca_key, ca_crt = f"{PKI}/ca.key", f"{PKI}/ca.crt"
    srv_key, srv_csr, srv_crt = f"{PKI}/receiver.key", f"{PKI}/receiver.csr", f"{PKI}/receiver.crt"
    cli_key, cli_csr, cli_crt = f"{PKI}/kiosk.key", f"{PKI}/kiosk.csr", f"{PKI}/kiosk.crt"
    ext = f"{PKI}/receiver.ext"
    # 1. private CA
    sh("openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes", "-keyout", ca_key, "-out", ca_crt,
       "-days", str(a.days), "-subj", "/CN=NomadRight pairing CA")
    # 2. receiver (server) cert with the IP as SAN
    open(ext, "w").write(f"subjectAltName=IP:{a.receiver_ip},DNS:nomadright-receiver\nextendedKeyUsage=serverAuth\n")
    sh("openssl", "req", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes", "-keyout", srv_key, "-out", srv_csr, "-subj", "/CN=nomadright-receiver")
    sh("openssl", "x509", "-req", "-in", srv_csr, "-CA", ca_crt, "-CAkey", ca_key, "-CAcreateserial", "-out", srv_crt, "-days", str(a.days), "-extfile", ext)
    # 3. kiosk (client) cert
    open(ext, "w").write("extendedKeyUsage=clientAuth\n")
    sh("openssl", "req", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes", "-keyout", cli_key, "-out", cli_csr, "-subj", f"/CN={a.device_id}")
    sh("openssl", "x509", "-req", "-in", cli_csr, "-CA", ca_crt, "-CAkey", ca_key, "-CAcreateserial", "-out", cli_crt, "-days", str(a.days), "-extfile", ext)
    # 4. NaCl keypairs (end-to-end sealing of every form)
    k_sk, r_sk = PrivateKey.generate(), PrivateKey.generate()
    for path, key in ((f"{PKI}/kiosk.nacl.key", k_sk), (f"{PKI}/receiver.nacl.key", r_sk)):
        open(path, "w").write(base64.b64encode(bytes(key)).decode())
    open(f"{PKI}/kiosk.nacl.pub", "w").write(base64.b64encode(bytes(k_sk.public_key)).decode())
    open(f"{PKI}/receiver.nacl.pub", "w").write(base64.b64encode(bytes(r_sk.public_key)).decode())
    for fn in os.listdir(PKI):
        os.chmod(os.path.join(PKI, fn), 0o600)
    der = subprocess.run(["openssl", "x509", "-in", srv_crt, "-outform", "DER"], check=True, capture_output=True).stdout
    fingerprint = hashlib.sha256(der).hexdigest()
    cfg = {"host": a.receiver_ip, "port": a.port, "device_id": a.device_id, "ca_cert": ca_crt, "client_cert": cli_crt, "client_key": cli_key,
           "kiosk_private_key": f"{PKI}/kiosk.nacl.key", "receiver_public_key": f"{PKI}/receiver.nacl.pub", "server_fingerprint": fingerprint}
    os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
    json.dump(cfg, open(CONFIG, "w"), indent=1)
    os.chmod(CONFIG, 0o600)
    # 5. the receiver's bundle
    if os.path.exists(BUNDLE):
        shutil.rmtree(BUNDLE)
    os.makedirs(BUNDLE, mode=0o700)
    for src, dst in ((srv_key, "receiver.key"), (srv_crt, "receiver.crt"), (ca_crt, "ca.crt"), (f"{PKI}/receiver.nacl.key", "receiver.nacl.key"),
                     (f"{PKI}/kiosk.nacl.pub", "kiosk.nacl.pub"), (os.path.join(HERE, "..", "receiver", "receiver.py"), "receiver.py")):
        shutil.copy(src, os.path.join(BUNDLE, dst))
    json.dump({"port": a.port, "device_id": a.device_id, "kiosk_public_key": "kiosk.nacl.pub", "server_cert": "receiver.crt", "server_key": "receiver.key",
               "ca_cert": "ca.crt", "receiver_private_key": "receiver.nacl.key", "store_dir": "received"}, open(os.path.join(BUNDLE, "receiver.json"), "w"), indent=1)
    for fn in os.listdir(BUNDLE):
        os.chmod(os.path.join(BUNDLE, fn), 0o600)
    print(f"paired: receiver {a.receiver_ip}:{a.port}, device {a.device_id}\n  kiosk config : {CONFIG}\n  copy to PC   : {BUNDLE}/  (then on the PC: pip install pynacl; python receiver.py)\n  fingerprint  : {fingerprint}")


if __name__ == "__main__":
    main()
