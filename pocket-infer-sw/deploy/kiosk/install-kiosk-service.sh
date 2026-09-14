#!/bin/bash
# Installs the NomadRight kiosk as a user service that starts with the desktop
# session and restarts after a crash. No root needed. Run again after editing
# the unit files here. Two one-time steps need sudo and are printed at the end.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
mkdir -p ~/.config/systemd/user ~/.local/bin
install -m 755 "$HERE/nomadright-wait-speech" "$HERE/nomadright-health-check" "$HERE/nomadright-session" "$HERE/nomadright-splash" ~/.local/bin/
install -m 644 "$HERE/splash.jpg" ~/.local/bin/nomadright-splash.jpg
install -m 644 "$HERE/nomadright-kiosk.service" "$HERE/nomadright-health.service" "$HERE/nomadright-health.timer" ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable nomadright-kiosk.service nomadright-health.timer
systemctl --user start nomadright-health.timer
if systemctl --user is-active --quiet nomadright-kiosk.service; then
  echo "kiosk service already running (restart it to load new code: systemctl --user restart nomadright-kiosk)"
elif ss -ltn | grep -q ':8765 '; then
  echo "a kiosk started by hand is using port 8765 - stop it, then: systemctl --user start nomadright-kiosk.service"
else
  systemctl --user start nomadright-kiosk.service
fi
systemctl --user --no-pager status nomadright-kiosk.service | head -5
cat <<'MSG'

Installed. The kiosk now starts with the desktop session and restarts itself after a crash.
Two one-time steps need sudo (run them yourself):
  1. Log in automatically after power-on (otherwise the login screen waits for a password):
       sudo sed -i 's/^#  AutomaticLoginEnable = true/AutomaticLoginEnable=true/; s/^#  AutomaticLogin = user1/AutomaticLogin=stark-x/' /etc/gdm3/custom.conf
  2. Keep logs across a power cut (optional but useful after a crash):
       sudo mkdir -p /var/log/journal && sudo systemctl restart systemd-journald
Everyday commands:
  systemctl --user status nomadright-kiosk      # is it running
  journalctl --user-unit=nomadright-kiosk.service -f   # its log (was the terminal before)
  systemctl --user restart nomadright-kiosk     # restart it
  systemctl --user stop nomadright-kiosk        # stop it (e.g. before running pocketinfer-retro by hand)
MSG
