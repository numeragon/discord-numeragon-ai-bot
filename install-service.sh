#!/usr/bin/env bash
# Reinstall + start the discord-vibe-bot systemd service.
# Needed after VM replacement, which wipes /etc (unit + proxy env file).
set -euo pipefail
cd "$(dirname "$0")"
sudo sh -c 'cat > /etc/discord-vibe-bot.env && chmod 600 /etc/discord-vibe-bot.env' <<EOF
http_proxy=$(printenv http_proxy)
https_proxy=$(printenv https_proxy)
HTTP_PROXY=$(printenv HTTP_PROXY)
HTTPS_PROXY=$(printenv HTTPS_PROXY)
no_proxy=$(printenv no_proxy)
NO_PROXY=$(printenv NO_PROXY)
EOF
sudo cp discord-vibe-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now discord-vibe-bot
sleep 6
sudo systemctl is-active discord-vibe-bot
sudo journalctl -u discord-vibe-bot --no-pager -n 3 | tail -3
