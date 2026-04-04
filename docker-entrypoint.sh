#!/bin/bash
# Runs as root, fixes volume permissions, then drops to soma user.
set -e

for dir in /data/media /data/media/thumbnails /data/avatars /data/db; do
    mkdir -p "$dir"
    chown soma:soma "$dir"
done

# Ensure model files are readable by the soma user (host may have restrictive permissions)
chmod -R o+r /data/models 2>/dev/null || true

exec gosu soma "$@"
