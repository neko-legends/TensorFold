#!/usr/bin/env bash
# Download DeepSeek-V4.1-Flash-1017 on rank 0, pinned revisions, then check every file against the Hugging Face tree.
#   deploy/dsv41-1017/download.sh exl3     the served 2.9 bpw EXL3 pack -> MODEL_DIR (66 files, 210,819,040,903 bytes)
#   deploy/dsv41-1017/download.sh engram   the Engram source from the official FP8 checkpoint -> FP8_DIR: shards 47/48
#                                          (the two Engram tables) + index + config + tokenizer, ~203 GB. The full FP8
#                                          checkpoint (~476 GiB) is not needed to serve.
# Re-running resumes; the check (hf_verify.py: sizes + sha256 of every LFS file) runs after each download.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SPARKS_ENV:-$here/sparks.env}"
EXL3_REPO=LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw
EXL3_REV=afc7aa0aab98ee735c0f54e0a122019af1ac08f1
FP8_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
FP8_REV=1bae0e1088993b9b82de08f59de5be5a34e42170
case "${1:-}" in
exl3)
    mkdir -p "$MODEL_DIR"
    "$HF" download "$EXL3_REPO" --revision "$EXL3_REV" --local-dir "$MODEL_DIR" --max-workers 8
    python3 "$here/hf_verify.py" "$EXL3_REPO" "$EXL3_REV" "$MODEL_DIR" ;;
engram)
    files=(model-00047-of-00048.safetensors model-00048-of-00048.safetensors model.safetensors.index.json
           config.json tokenizer.json tokenizer_config.json)
    mkdir -p "$FP8_DIR"
    "$HF" download "$FP8_REPO" "${files[@]}" --revision "$FP8_REV" --local-dir "$FP8_DIR" --max-workers 8
    python3 "$here/hf_verify.py" "$FP8_REPO" "$FP8_REV" "$FP8_DIR" "${files[@]}" ;;
*) echo "usage: $0 exl3|engram" >&2; exit 2 ;;
esac
