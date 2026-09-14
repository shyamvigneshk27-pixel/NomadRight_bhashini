#!/bin/bash
# sudo bash system-services.sh disable|restore|status
# Turns off the system services and timers this device does not need for the
# NomadRight kiosk, and can put them back exactly as they were.
#   disable : records each unit's enabled state in /var/lib/nomadright/services.before the FIRST
#             time it sees the unit (running it again never overwrites that record),
#             then disables and stops the units listed below (never masks anything, never
#             touches tailscale, ssh, NetworkManager, GDM, the speech service or Jetson services)
#   restore : re-enables and starts every unit that was enabled before 'disable'
#   status  : shows the current state of every unit in the list
set -u
STATE_DIR=/var/lib/nomadright; BEFORE=$STATE_DIR/services.before

# Tier 1 - nothing on this device uses them (measured on 14 September 2026):
#   container runtime (docker+containerd, ~85 MB; no containers or images exist), the Ollama
#   daemon (the kiosk starts its own llama-server from Ollama's files; the service is not
#   involved), snaps (only Firefox; NOTE this also disables boards/hdmi.py's last-resort
#   `firefox -kiosk` fallback, used only if pywebview cannot be imported), mDNS (loses the
#   pocket-infer-0d29.local name; the tailscale name is unaffected), modems (none attached),
#   crash reporting, NFS RPC (no exports or NFS mounts), the systemd-networkd helper
#   (NetworkManager manages the Wi-Fi), software/firmware update daemons and every update,
#   news and reporting timer. The device IS online (Wi-Fi + tailscale): without this cut it
#   runs apt-get update 15 min after every boot, fwupdmgr refresh hourly, MOTD news twice a
#   day and snap refreshes - on a production kiosk.
UNITS=(
  # each socket before its service, so nothing re-activates a service between the two
  docker.socket docker.service containerd.service
  ollama.service
  # snapd.seeded.service Requires=snapd.socket and would pull snapd back in at every boot
  snapd.seeded.service snapd.socket snapd.service snapd.apparmor.service snapd.snap-repair.timer
  avahi-daemon.socket avahi-daemon.service
  ModemManager.service
  kerneloops.service apport.service apport-autoreport.timer apport-autoreport.path
  rpcbind.socket rpcbind.service
  networkd-dispatcher.service
  packagekit.service fwupd.service fwupd-refresh.timer
  apt-daily.timer apt-daily-upgrade.timer motd-news.timer ua-timer.timer
  update-notifier-download.timer update-notifier-motd.timer man-db.timer dpkg-db-backup.timer
)
# Tier 2 - harmless but not needed; left alone unless you add them to UNITS yourself:
#   jtop.service (the jetson_stats monitor you may use), bluetooth.service (allowed
#   connectivity), colord / switcheroo-control / power-profiles-daemon / upower (desktop
#   helpers, dbus-activated, idle in the kiosk session).
# Never: tailscaled, ssh, NetworkManager, wpa_supplicant, gdm3, accounts-daemon (GDM needs it),
#   polkit, udisks2, systemd-*, rsyslog, haveged, bhashini_models, nv* / nvidia* (Jetson).

[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
mkdir -p "$STATE_DIR"
exists() { systemctl cat "$1" >/dev/null 2>&1; }
case "${1:-status}" in
  status)
    for u in "${UNITS[@]}"; do exists "$u" && printf "%-34s enabled=%-10s active=%s\n" "$u" "$(systemctl is-enabled "$u" 2>/dev/null)" "$(systemctl is-active "$u" 2>/dev/null)"; done ;;
  disable)
    touch "$BEFORE"
    for u in "${UNITS[@]}"; do
      exists "$u" || continue
      grep -q "^$u " "$BEFORE" || printf "%s %s\n" "$u" "$(systemctl is-enabled "$u" 2>/dev/null)" >> "$BEFORE"
      systemctl disable --now "$u" >/dev/null 2>&1 && echo "off: $u" || echo "could not change: $u"
    done
    echo "previous states saved in $BEFORE; put back with: sudo bash $0 restore" ;;
  restore)
    [ -f "$BEFORE" ] || { echo "nothing recorded in $BEFORE"; exit 1; }
    while read -r u was; do
      case "$was" in enabled|enabled-runtime) systemctl enable --now "$u" >/dev/null 2>&1 && echo "on: $u" || echo "could not restore: $u" ;; esac
    done < "$BEFORE" ;;
  *) echo "usage: sudo bash $0 disable|restore|status"; exit 1 ;;
esac
