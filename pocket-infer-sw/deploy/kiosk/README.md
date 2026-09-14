# NomadRight kiosk as an appliance

`install-kiosk-service.sh` installs three **user** systemd units (no root):

| Unit | Does |
|---|---|
| `nomadright-kiosk.service` | Runs `pocketinfer-service` with `NOMADRIGHT_QUERY_FALLBACK=sorry` as part of the desktop session; `Restart=always` brings it back 3 s after any exit, forever. Waits up to 4 min for the speech service first. |
| `nomadright-health.timer` / `.service` | Every minute, once the kiosk has been up 3 min: if the kiosk's web server fails three checks in a row, restart the kiosk (catches a hang, which `Restart=` cannot see). |

Boot chain: power → GDM auto-login (`/etc/gdm3/custom.conf`, needs sudo once) → GNOME session → `graphical-session.target` → the kiosk → Language Selection. `bhashini_models.service` (speech) is a system service enabled at boot already. The old system unit `/etc/systemd/system/pocketinfer.service` must stay **disabled**: it lacks the fallback setting and would start before the display exists.

Verified on the device on 14 September 2026: `kill -9` of the kiosk process → back on port 8765 in 17 s, one window, fallback setting intact; an off-topic question under the service gets the apology and no model server starts. Log: `journalctl --user-unit=nomadright-kiosk.service -f` (volatile until the persistent-journal step is done). Hang: a process frozen with SIGSTOP was restarted after three failed one-minute checks (the first version of the check never fired: awk's %d overflowed 2^31 microseconds of uptime, fixed with bash arithmetic).
