#!/bin/bash
# sudo bash install-kiosk-session.sh
# Makes the automatic-login user boot straight into the NomadRight kiosk session
# (no Ubuntu desktop). Undo with: sudo bash restore-desktop.sh
set -e
USER_NAME="${1:-stark-x}"
HERE="$(cd "$(dirname "$0")" && pwd)"
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
[ -x "/home/$USER_NAME/.local/bin/nomadright-session" ] || { echo "first run install-kiosk-service.sh as $USER_NAME (it installs ~/.local/bin/nomadright-session)"; exit 1; }
install -m 755 "$HERE/nomadright-session-wrapper" /usr/local/bin/nomadright-session
install -m 644 "$HERE/nomadright.desktop" /usr/share/xsessions/nomadright.desktop
F="/var/lib/AccountsService/users/$USER_NAME"
mkdir -p /var/lib/AccountsService/users
python3 - "$F" <<'PY'
import configparser, sys, os
p = sys.argv[1]
c = configparser.RawConfigParser(); c.optionxform = str
if os.path.exists(p): c.read(p)
if not c.has_section("User"): c.add_section("User")
c.set("User", "Session", "nomadright"); c.set("User", "XSession", "nomadright")
if not c.has_option("User", "SystemAccount"): c.set("User", "SystemAccount", "false")
with open(p, "w") as f: c.write(f, space_around_delimiters=False)
PY
chmod 644 "$F"
grep -q '^AutomaticLoginEnable=true' /etc/gdm3/custom.conf || echo "NOTE: automatic login is not enabled in /etc/gdm3/custom.conf"
grep -q '^WaylandEnable=false' /etc/gdm3/custom.conf || echo "NOTE: WaylandEnable=false is not set in /etc/gdm3/custom.conf; the kiosk session is an X session"
echo "Installed. Next boot: NomadRight Kiosk session (no desktop). Undo: sudo bash $HERE/restore-desktop.sh"
