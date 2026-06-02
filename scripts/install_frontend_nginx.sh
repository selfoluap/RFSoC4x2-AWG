#!/usr/bin/env bash
set -euo pipefail

SITE_NAME="rfsoc-awg"
REPO_ROOT="/home/xilinx/RFSoC4x2-AWG"
NGINX_SOURCE="$REPO_ROOT/deploy/nginx/$SITE_NAME.conf"
NGINX_AVAILABLE="/etc/nginx/sites-available/$SITE_NAME"
NGINX_ENABLED="/etc/nginx/sites-enabled/$SITE_NAME"

if [[ "$EUID" -ne 0 ]]; then
    printf 'Run this script with sudo.\n' >&2
    exit 1
fi

if [[ ! -f "$NGINX_SOURCE" ]]; then
    printf 'nginx config not found: %s\n' "$NGINX_SOURCE" >&2
    exit 1
fi

install -d -m 0755 /var/www/rfsoc-awg
install -d -m 0755 /etc/nginx/sites-available /etc/nginx/sites-enabled
install -m 0644 "$NGINX_SOURCE" "$NGINX_AVAILABLE"
ln -sfn "$NGINX_AVAILABLE" "$NGINX_ENABLED"
nginx -t
systemctl reload-or-restart nginx
printf 'RFSoC AWG frontend is configured for http://<rfsoc-board-ip>:8080/\n'
