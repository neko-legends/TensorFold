# Nemotron TP-4 swap window — agenda (needs Jun's explicit go)

Everything below was dry-run on the truncated model (TF_NEMOTRON_LAYERS) beside the live engine; the full model's
per-rank load was checked alone with TF_NEMOTRON_TP_SHARD (ranks 0 and 3: 8.5 s warm, no unused tensors).
Binaries: `~/nemotron-tf/bin/{tensorfold,tensorfold-native}` on all four Sparks (built from this branch on spark4);
NCCL: `~/nemotron-tf/lib/libnccl.so.2` (copied out of the DeepSeek image). Rank order = DeepSeek's (spark1 0 .. spark4 3).

1. `tools/nemotron/swap/nemotron-window.sh up` — maintenance flag, `docker stop dsv41-tf4-r0..r3` (exact names),
   Nemotron serving on `http://spark1:8100/v1` (context 65536, --parallel 1; `PARALLEL=4` for qbench's 4 workers).
   Re-run `nemotron-window.sh keep` every <= 30 min (the keeper's stand-down lasts 60 min after a touch).
2. Smoke: three chat prompts (coherent English, thinking on and off), one image (`a4_table_chart.png`), one 20K-token
   prompt. Note the `accepted=a/b` MTP ratio in rank 0's log (`~/nemotron-tf/runs/window-r0.log`).
3. `tools/nemotron/swap/window-bench.sh NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` — qbench all areas (speed, mmlu,
   math, coding, if, json, tools, longctx up to the 299K passkey: start the server with CONTEXT=307200 for that cell)
   and the vision read cells; results in `engine/zig-nemotron/results/` (NOT spark-bench).
4. Stop the server (`nemotron-window.sh down` does it, but run the gates first with the server stopped):
   - TP-4 per-block dump: tprun-style launch of `tensorfold prefill MODEL prompts.json p --dump DIR` on the four
     ranks (rank 0's DIR has NNN_h for all 88 blocks), then `tools/nemotron/layer_ref.py MODEL DIR --rows N --layers
     0,1,7,40,41,47,86,87` on CPU (the checkpoint's own HF blocks).
   - M1 parity: `tensorfold run MODEL --tokens ... --max-tokens 64 --no-drafts --report tp1.json` on ONE Spark (TP-1,
     70 GB, fits once DeepSeek is stopped) vs the same at TP-4: same first tokens, per-block cos >= 0.9999.
5. `nemotron-window.sh down` — Nemotron off, `serve4.sh start`, spark1:8000/health 200, flag removed.
6. After the window (DeepSeek live): `window-vision.py http://spark1:8000/v1 deepseek-v4.1-flash results/vision-dsv41.json`
   for the matching vision cells (qbench's DeepSeek numbers already exist: spark-bench/results/qbench-dsv41-zig-20261015.json).

Expected speed (from 24/48-layer TP-4 runs and the full single-rank share): 1-row window ~16 ms (~60 tok/s plain),
2/4/8 rows ~19/23/31 ms, a draft level ~0.74 ms -> ~100-110 tok/s on code-like text if the MTP head lands at the
config's prior (0.8, 0.72, 0.68 ...); the measured acceptance decides.

## Serving notes (2026-10-16)

- **Exactness depends on the prompt cache's state.** With `--prompt-cache-gib > 0` a prompt pass is cut at keep marks
  and resume points, so under TP greedy text drifts at rounding level from a cache-off pass (prose-type prompts flip,
  short code/essay prompts usually not), and a cache hit can differ again. It is deterministic per cache state: the same
  request sequence on a fresh server gives the same text. **A/B speed or exactness tests must pin the cache state**:
  run both sides with `CACHE_GIB=0`, or the same request sequence on freshly started servers.
- **Install a new binary by copy to a new name, then rename** (`cp new bin/.tensorfold-native.new && mv -f
  bin/.tensorfold-native.new bin/tensorfold-native` on each Spark); never scp over a running binary. Keep the previous
  one beside it (`bin/tensorfold-native.<what>-<sha>`) for rollback, on all four Sparks.
- Env switches: `TF_NEMOTRON_DISK` (prompt-cache disk tier folder; `DISK=` empty: none) with `TF_NEMOTRON_DISK_GIB`
  (cap, 128), `TF_NEMOTRON_DISK_FLOOR_GIB` (free space kept, 64), `TF_NEMOTRON_DISK_MIN` (smallest state written, 4096
  tokens); `TF_NEMOTRON_LONE_DEPTH=0` (lone-stream MTP chains back to the fixed acceptance-rate depth);
  `TF_NEMOTRON_DRAFT_TRACE=1` (rank 0 logs each lone round's drafts, confidences and landings).
- The disk tier's folder is `<TF_NEMOTRON_DISK>/<identity>` per rank; the identity hashes the checkpoint (config.json
  and each shard's name, size, header and last 64 KiB), the layout, the rank and the world, so another checkpoint never
  reads these states. One server holds a folder (`<identity>/lock`); a second one of the same identity runs without a
  disk tier. Rank 0 removes files its index does not name at start and the other ranks follow with the first call.
  States are written when evicted and at a clean stop (an idle keep-alive connection no longer forces an unclean exit).
