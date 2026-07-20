#!/usr/bin/env bash
# Launches the synthetic hand-tracking data generator (C++ orchestrator +
# headless Blender workers).
#
# Why this script exists: data_generator.cpp reads its configuration from
# environment variables (GEN_*) whose compiled-in defaults point at the
# original author's machine (e.g. GEN_MODELS_MANIFEST defaults to
# /4/clones/free_hand_meshes/model_manifest.json, which doesn't exist here).
# Nothing in the repo previously set these, so a bare `./data_generator`
# would abort immediately. This script resolves them relative to the
# repo's actual on-disk layout instead of hardcoding another machine-
# specific path.
#
# Usage:
#   ./run_data_generator.sh                  # use all the defaults below
#   GEN_SUPERROOT=/some/other/dir ./run_data_generator.sh
#   GEN_NUM_BLENDER_INSTANCES=2 ./run_data_generator.sh   # lighter run
#
# Smoke test (recommended before a full run): confirms the manifest parses,
# each Blender worker spawns and connects back over the socket, and poses
# get generated -- without spending time on actual rendering.
#   GEN_DONT_RENDER=1 GEN_NUM_BLENDER_INSTANCES=1 ./run_data_generator.sh
# Stop it once you see "Got a request!" / a seq folder with CSVs appear
# under GEN_SUPERROOT; Ctrl-C, or type anything + Enter on its stdin.
#
# Any GEN_* variable can be overridden by exporting it (or prefixing the
# call, as above) before running -- this script only sets a default if the
# variable isn't already set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # .../mercury_train
THESIS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"              # .../thesis

BINARY="$SCRIPT_DIR/build/cpp/data_generator/data_generator"

# ---- Sanity checks -----------------------------------------------------

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: data_generator binary not found (or not executable) at:" >&2
    echo "  $BINARY" >&2
    echo "Build it first (from $SCRIPT_DIR): cmake --build build --target data_generator" >&2
    exit 1
fi

if ! command -v blender >/dev/null 2>&1; then
    echo "ERROR: 'blender' is not on PATH. The orchestrator spawns it as a" >&2
    echo "subprocess (subprocess::Popen({\"blender\", ...})) -- it will not" >&2
    echo "find a Blender install by any other means." >&2
    exit 1
fi

# ---- Required config, with portable defaults ---------------------------

: "${GEN_MODELS_MANIFEST:=$THESIS_ROOT/hand-tracking-playground/hand_scans/model_manifest.json}"
export GEN_MODELS_MANIFEST

if [[ ! -f "$GEN_MODELS_MANIFEST" ]]; then
    echo "ERROR: GEN_MODELS_MANIFEST not found: $GEN_MODELS_MANIFEST" >&2
    echo "Set GEN_MODELS_MANIFEST to point at model_manifest.json (see" >&2
    echo "hand-tracking-playground/hand_scans/ at the thesis root)." >&2
    exit 1
fi

# NOTE: data_generator wipes GEN_SUPERROOT's contents on every startup
# (main() does fs::remove_all() on every entry in it before listening).
# Defaulting to a fresh output_new/ dir instead of the existing output/ --
# which already holds your 15 rendered sequences -- is deliberate so this
# script can't silently destroy that data. Point GEN_SUPERROOT at
# $THESIS_ROOT/output yourself, explicitly, once you're ready to replace it.
: "${GEN_SUPERROOT:=$THESIS_ROOT/output_new}"
export GEN_SUPERROOT

if [[ "$(cd "$(dirname "$GEN_SUPERROOT")" 2>/dev/null && pwd)/$(basename "$GEN_SUPERROOT")" == "$THESIS_ROOT/output" ]]; then
    echo "WARNING: GEN_SUPERROOT is $THESIS_ROOT/output -- its contents will" >&2
    echo "be wiped in 5 seconds. Ctrl-C now to abort." >&2
    sleep 5
fi

: "${GEN_HDRIS_DIR:=$THESIS_ROOT/hdris}"
export GEN_HDRIS_DIR

if [[ ! -d "$GEN_HDRIS_DIR" ]]; then
    echo "ERROR: GEN_HDRIS_DIR not found: $GEN_HDRIS_DIR" >&2
    exit 1
fi

# Conservative default -- 9 parallel Blender instances (the compiled-in
# default) is a lot of RAM/CPU for a dev machine. Bump it up on a beefier
# box: GEN_NUM_BLENDER_INSTANCES=9 ./run_data_generator.sh
: "${GEN_NUM_BLENDER_INSTANCES:=4}"
export GEN_NUM_BLENDER_INSTANCES

# ---- Go -----------------------------------------------------------------

echo "== data_generator =="
echo "  manifest:        $GEN_MODELS_MANIFEST"
echo "  output (wiped!):  $GEN_SUPERROOT"
echo "  hdris:            $GEN_HDRIS_DIR"
echo "  blender instances: $GEN_NUM_BLENDER_INSTANCES"
echo

exec "$BINARY"
