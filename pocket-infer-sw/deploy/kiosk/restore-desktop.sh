#!/bin/bash
# sudo bash restore-desktop.sh   (from a TTY: Ctrl+Alt+F3, log in, run it, then: sudo systemctl restart gdm3)
# Puts the normal Ubuntu desktop session back for the automatic-login user.
set -e
USER_NAME="${1:-stark-x}"
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
F="/var/lib/AccountsService/users/$USER_NAME"
python3 - "$F" <<'PY'
import configparser, sys, os
p = sys.argv[1]
c = configparser.RawConfigParser(); c.optionxform = str
if os.path.exists(p): c.read(p)
if not c.has_section("User"): c.add_section("User")
c.set("User", "Session", "ubuntu"); c.set("User", "XSession", "ubuntu")
with open(p, "w") as f: c.write(f, space_around_delimiters=False)
PY
echo "Ubuntu desktop session restored for $USER_NAME (takes effect at the next login: sudo systemctl restart gdm3, or reboot)."
echo "The kiosk service still starts inside the desktop session; stop that with: systemctl --user disable --now nomadright-kiosk nomadright-health.timer"
