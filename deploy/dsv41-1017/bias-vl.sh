#!/usr/bin/env bash
# The image-routing bias for the 1017 weights: the 43 `*.ffn.gate.bias_vl` tensors (40 layers F32[384] + 3 MTP F32[128])
# read straight from the 1017 EXL3 pack (it ships them) into the file the engine loads at TF_DSV41_BIAS_VL
# (/cache/dsv41-bias-vl/bias_vl.safetensors, i.e. CACHE_DIR/dsv41-bias-vl/ on the host), then copied to every node with
# config/pfdense-table.json. stdlib only, seconds. Run on rank 0.
#
# The file is byte-identical to the one served (sha256 6fe07ce482cdfbb35e081b0737d2ddf08e2bdaf58765533c0f340e614ac21f91).
# 34 of the 43 tensors differ from the earlier V4.1's: an earlier bias file would mis-route image tokens.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SPARKS_ENV:-$here/sparks.env}"
SSH="ssh -o BatchMode=yes"
out="$CACHE_DIR/dsv41-bias-vl"
mkdir -p "$out"
python3 "$here/bias_vl_from_pack.py" "$MODEL_DIR" "LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw@afc7aa0a (gate.bias_vl copied from deepseek-ai/DeepSeek-V4.1-Flash-1017)" \
    "$out/bias_vl.safetensors"
cp "$REPO_DIR/config/pfdense-table.json" "$CACHE_DIR/"
sum=$(cd "$CACHE_DIR" && sha256sum dsv41-bias-vl/bias_vl.safetensors pfdense-table.json)
echo "$sum"
for h in $WORKERS; do
    $SSH "$h" "mkdir -p '$out'"
    rsync -a -e "$SSH" "$CACHE_DIR/dsv41-bias-vl" "$CACHE_DIR/pfdense-table.json" "$h:$CACHE_DIR/"
    echo "$sum" | $SSH "$h" "cd '$CACHE_DIR' && sha256sum -c"
done
