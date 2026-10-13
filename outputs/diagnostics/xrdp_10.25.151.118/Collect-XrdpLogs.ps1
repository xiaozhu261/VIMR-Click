$ErrorActionPreference = 'Continue'
$taskLogPath = Join-Path $PSScriptRoot 'xrdp-asus.log'
$taskKnownHosts = Join-Path $env:TEMP 'codex-xrdp-10.25.151.118-known_hosts'
$taskRemoteCommands = @'
printf '\n=== Identity and system ===\n'
id
date -Is
cat /etc/os-release
printf '\n=== Installed desktop and xrdp packages ===\n'
dpkg-query -W -f='${Package} ${Version}\n' xrdp xorgxrdp xserver-xorg-core gnome-session ubuntu-session xfce4-session dbus-x11 2>/dev/null
printf '\n=== Service state ===\n'
systemctl is-active xrdp xrdp-sesman
printf '\n=== Display and desktop processes ===\n'
ps -C Xorg,xrdp,xrdp-sesman,gnome-session-binary,gnome-shell,xfce4-session -o user,pid,ppid,stat,args
printf '\n=== Sessions ===\n'
loginctl list-sessions --no-pager
printf '\n=== Filesystem capacity ===\n'
df -h / /tmp "$HOME"
printf '\n=== Xorg log files ===\n'
ls -lt "$HOME"/.xorgxrdp.*.log 2>/dev/null
for taskFile in "$HOME"/.xorgxrdp.11.log "$HOME"/.xorgxrdp.10.log "$HOME"/.xsession-errors; do
  if [ -f "$taskFile" ]; then
    printf '\n=== %s ===\n' "$taskFile"
    tail -n 180 "$taskFile"
  fi
done
printf '\n=== Session startup configuration ===\n'
for taskFile in /etc/xrdp/startwm.sh /etc/X11/xrdp/xorg.conf "$HOME"/.xsession "$HOME"/startwm.sh; do
  if [ -r "$taskFile" ]; then
    printf '\n=== %s ===\n' "$taskFile"
    sed -n '1,180p' "$taskFile"
  fi
done
printf '\n=== Server logs (sudo may ask for your Linux password) ===\n'
sudo tail -n 180 /var/log/xrdp-sesman.log /var/log/xrdp.log
'@

Write-Host 'Connecting as asus to 10.25.151.118. Enter your password in this terminal when prompted.'
Write-Host 'This command reads diagnostic information; it does not change server settings or restart services.'
& ssh -t -o ConnectTimeout=10 -o ConnectionAttempts=1 -o ClearAllForwardings=yes -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$taskKnownHosts" asus@10.25.151.118 $taskRemoteCommands 2>&1 | Tee-Object -FilePath $taskLogPath
Write-Host "Diagnostic output saved to: $taskLogPath"
