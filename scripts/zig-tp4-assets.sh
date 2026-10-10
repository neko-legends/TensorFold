#!/usr/bin/env bash
# Per-rank runtime assets of the Zig engine at TP=4 (docs/ZIG-SERVE.md "Generate runtime assets", docs/ZIG-TP4-PLAN.md).
# No GPU anywhere; nothing here touches a running server.
#
#   scripts/zig-tp4-assets.sh aot RANK OUTDIR     Triton AOT programs of one rank (OUTDIR/aot). Any host: the cubins are
#                                                 sm_121 code, compiled offline. At TP=4 ranks 0/1 and 2/3 need
#                                                 different sets (their expert and head widths), so run it per rank.
#   scripts/zig-tp4-assets.sh host OUTDIR PACK    rope.bin and engram-host.bin. On a SPARK, in the Python image's torch
#                                                 (their bits come from that host's CPU cos / sin and normalizer), e.g.
#                                                 inside `docker run --rm --name zigtp4-assets ... dsv41-tensorfold:tp4rot`.
#                                                 The same files serve every rank; copy them into each rank's ASSETS_DIR.
#
# Env: CONFIG (config/tp4-zig.env: the served settings aot-needs reads), PY_SOURCE (the patched Python tree's src/:
# vendor/TensorFold + patches/*.patch), M1 (tf-dsv41-m1 of the image's build, run on this host), PYTHON (Triton 3.7.1 +
# torch for `aot`; the image's python for `host`), PTXAS (ptxas 13.3.73; PyPI nvidia-cuda-nvcc==13.3.73 ships it),
# ROPE_ROWS (default: TF_DSV41_CONTEXT + 4096).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
root=$PWD
step=${1:?usage: zig-tp4-assets.sh aot RANK OUTDIR | host OUTDIR PACK}
config=${CONFIG:-config/tp4-zig.env}
[[ -f "$config" ]] || { echo "missing $config" >&2; exit 2; }
py=${PYTHON:-python3}
src=${PY_SOURCE:?set PY_SOURCE to the patched Python tree src/}
# the served settings, placeholders blanked (aot-needs reads only the engine knobs)
load_config() { set -a; source <(sed -e 's#<[^>]*>#x#g' "$config"); set +a; }
case "$step" in
aot)
    rank=${2:?rank}; out=$(realpath -m "${3:?outdir}")
    m1=${M1:?set M1 to tf-dsv41-m1}
    ptxas=${PTXAS:?set PTXAS to ptxas 13.3.73}
    mkdir -p "$out"
    load_config
    [[ "$rank" =~ ^[0-9]+$ && "$rank" -lt "${TP_WORLD:-4}" ]] || { echo "rank must be 0..$((${TP_WORLD:-4} - 1))" >&2; exit 2; }
    (cd engine/zig && "$py" tools/zig/triton_fill.py sigs --py "$src" --out "$out/sigs.json")
    TF_TP_WORLD=${TP_WORLD:-4} TF_TP_RANK=$rank TF_DSV41_ASSETS=$out "$m1" aot-needs "$out/sigs.json" "$out/needs.json"
    (cd engine/zig && TRITON_PTXAS_PATH=$ptxas TRITON_PTXAS_BLACKWELL_PATH=$ptxas CUDA_VISIBLE_DEVICES="" \
        "$py" tools/zig/triton_fill.py compile --py "$src" --needs "$out/needs.json" \
        --options tools/zig/dsv41_triton_options.json --arch 121 --out "$out/aot")
    echo "rank $rank: $out/aot"
    ;;
host)
    out=$(realpath -m "${2:?outdir}"); pack=${3:?pack directory}
    mkdir -p "$out"
    load_config
    rows=${ROPE_ROWS:-$((TF_DSV41_CONTEXT + 4096))}
    PYTHONPATH="$src" "$py" engine/zig/tools/zig/dsv41_rope_tables.py --config "$pack/config.json" --rows "$rows" --out "$out/rope.bin"
    "$py" engine/zig/tools/zig/dsv41_engram_host.py --py-src "$src" --config "$pack/config.json" \
        --tokenizer "$pack/tokenizer.json" --out "$out/engram-host.bin"
    echo "rope.bin ($rows rows) and engram-host.bin in $out (the same for every rank)"
    ;;
*) sed -n '2,17p' "$0"; exit 2 ;;
esac
