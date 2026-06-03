#!/bin/sh
set -e

# NeMo Guardrails does NOT expand ${VAR} in YAML; render through envsubst
# at startup so api_base / api_key actually point at LiteLLM with real key.
SRC=/app/config
DST=/app/config-rendered

mkdir -p "$DST"
# Render top-level YAML/colang
for f in "$SRC"/*; do
  [ -f "$f" ] || continue
  base=$(basename "$f")
  case "$base" in
    *.yml|*.yaml|*.co)
      envsubst < "$f" > "$DST/$base"
      ;;
    *)
      cp "$f" "$DST/$base"
      ;;
  esac
done
# Recurse into rails/ (or any subdir)
for d in "$SRC"/*/; do
  [ -d "$d" ] || continue
  subdir=$(basename "$d")
  mkdir -p "$DST/$subdir"
  for f in "$d"/*; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    case "$base" in
      *.yml|*.yaml|*.co)
        envsubst < "$f" > "$DST/$subdir/$base"
        ;;
      *)
        cp "$f" "$DST/$subdir/$base"
        ;;
    esac
  done
done

echo "[entrypoint] rendered config:"
ls -la "$DST"
echo "[entrypoint] launching nemoguardrails server"
exec nemoguardrails server --config "$DST" --port 8090
