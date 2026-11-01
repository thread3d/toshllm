#!/bin/zsh
# Builds the app's chat map (Assets/rich-content/maplibre/src/toshmap.ts) into Assets/rich-content/maplibre: one
# self-contained script with MapLibre's worker inside it, and MapLibre's stylesheet. It takes the libraries the web
# chat is built with, so run `npm ci` in vendor/llama.cpp/tools/ui first, or give another tools/ui folder.
#
#   ./scripts/build-map.sh [tools/ui folder]
set -euo pipefail
ROOT="${0:a:h:h}"
UI="${1:-$ROOT/vendor/llama.cpp/tools/ui}"
OUT="$ROOT/Assets/rich-content/maplibre"
ESBUILD="$UI/node_modules/.bin/esbuild"
[ -x "$ESBUILD" ] || { echo "no esbuild in $UI/node_modules: run npm ci there first"; exit 1; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
"$ESBUILD" "$UI/node_modules/maplibre-gl/dist/maplibre-gl-worker.mjs" --bundle --format=iife --minify \
    --outfile="$work/worker.js" --log-level=warning
node -e 'const fs = require("fs"); fs.writeFileSync(process.argv[2], "window.TOSH_MAP_WORKER=" + JSON.stringify(fs.readFileSync(process.argv[1], "utf8")) + ";\n")' \
    "$work/worker.js" "$work/worker-text.js"
# the sources import the libraries by name; they resolve from the web chat's node_modules
NODE_PATH="$UI/node_modules" "$ESBUILD" "$OUT/src/toshmap.ts" --bundle --format=iife --minify --target=safari15 \
    --outfile="$work/map.js" --log-level=warning --resolve-extensions=.ts,.js,.mjs \
    --alias:maplibre-gl="$UI/node_modules/maplibre-gl" --alias:pmtiles="$UI/node_modules/pmtiles" \
    --alias:@protomaps/basemaps="$UI/node_modules/@protomaps/basemaps"
cat "$work/worker-text.js" "$work/map.js" > "$OUT/toshmap.js"
cp "$UI/node_modules/maplibre-gl/dist/maplibre-gl.css" "$OUT/maplibre-gl.css"
echo "built $OUT/toshmap.js ($(du -h "$OUT/toshmap.js" | cut -f1))"
