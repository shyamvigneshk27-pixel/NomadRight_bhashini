# NomadRight kiosk as an appliance

`install-kiosk-service.sh` installs three **user** systemd units (no root):

| Unit | Does |
|---|---|
| `nomadright-kiosk.service` | Runs `pocketinfer-service` with `NOMADRIGHT_QUERY_FALLBACK=sorry` as part of the desktop session; `Restart=always` brings it back 3 s after any exit, forever. Waits up to 4 min for the speech service first. |
| `nomadright-health.timer` / `.service` | Every minute, once the kiosk has been up 3 min: if the kiosk's web server fails three checks in a row, restart the kiosk (catches a hang, which `Restart=` cannot see). |

Boot chain: power → GDM auto-login (`/etc/gdm3/custom.conf`, needs sudo once) → GNOME session → `graphical-session.target` → the kiosk → Language Selection. `bhashini_models.service` (speech) is a system service enabled at boot already. The old system unit `/etc/systemd/system/pocketinfer.service` must stay **disabled**: it lacks the fallback setting and would start before the display exists.

Verified on the device on 14 September 2026: `kill -9` of the kiosk process → back on port 8765 in 17 s, one window, fallback setting intact; an off-topic question under the service gets the apology and no model server starts. Log: `journalctl --user-unit=nomadright-kiosk.service -f` (volatile until the persistent-journal step is done). Hang: a process frozen with SIGSTOP was restarted after three failed one-minute checks (the first version of the check never fired: awk's %d overflowed 2^31 microseconds of uptime, fixed with bash arithmetic).

First real power-on with automatic login (14 September, 12:40): the service started and the backend came up, but the window did not, because the unit hard-coded `DISPLAY=:1` while an auto-login session is `:0` (`:1` only exists when someone logs in by hand after the greeter). The unit now takes DISPLAY and XAUTHORITY from the session and only falls back to `POCKETINFER_DISPLAY=:0`; fixed at 12:45, window confirmed on `:0`.

## Kiosk session (no desktop) and service cut

| File | Does |
|---|---|
| `nomadright-session` (→ `~/.local/bin`) | The X session GDM starts for the auto-login user: screen never blanks, dark root, hands DISPLAY/XAUTHORITY to `systemd --user`, shows the splash, starts the kiosk unit and the health timer, then stays alive for ever. `NOMADRIGHT_SESSION_DRYRUN=1` runs it on a test display without touching the real session. |
| `nomadright-splash` (→ `~/.local/bin`) | GTK splash sized to the screen (no window manager needed): artwork + "Starting" in three languages; exits 1.5 s after a window called "PocketInfer" exists; after five minutes without one it says "still starting". |
| `nomadright.desktop` (→ `/usr/share/xsessions`, sudo) | The session entry. |
| `install-kiosk-session.sh` (sudo) | Installs the entry and selects it for the user in AccountsService (`Session=nomadright`, `XSession=nomadright`). |
| `restore-desktop.sh` (sudo) | Selects the Ubuntu session again; then `sudo systemctl restart gdm3` or reboot. |
| `system-services.sh disable\|restore\|status` (sudo) | Disables and stops the units in its list, recording their previous state in `/var/lib/nomadright/services.before`; `restore` puts them back. Never masks; never touches tailscale, ssh, NetworkManager, GDM, the speech service or Jetson services. |

`boards/webview_launcher.py` now passes the monitor geometry to pywebview as well as asking for fullscreen, so the kiosk window fills the screen with no window manager (verified on a nested X server: 1024x600+0+0). Tested on 14 September on Xephyr: splash first, kiosk window on top, splash gone 4 s later.
