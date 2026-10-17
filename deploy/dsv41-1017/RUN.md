# Run DeepSeek-V4.1-Flash-1017 on four DGX Sparks (TensorFold Zig engine, TP=4)

DeepSeek's official `deepseek-ai/DeepSeek-V4.1-Flash-1017`, served as the third-party 2.9 bpw EXL3 quant
[`LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw),
on the same engine, image and serving settings as the earlier V4.1 pack. It is a **config-only drop-in**: no code change and
no new image. Three inputs derived from the weights must be rebuilt for 1017, because 1017 changed them:

| input | from | why it cannot be reused |
| --- | --- | --- |
| Engram tables, `engram-l{1,14}-r<rank>of4.bin` | the official checkpoint's shards 47/48 | the 1017 tables differ from the first row |
| image-routing bias, `bias_vl.safetensors` | the 43 `gate.bias_vl` tensors the 1017 pack ships | 34 of 43 differ from the earlier V4.1 |
| prepared rank folders | written by the engine on the first start | they are images of the weights |

The steps below are what the live service ran on 2026-10-17, with hostnames and paths turned into settings. Everything
runs on rank 0's node unless a step says otherwise. Results of this setup: [README](../../README.md#deepseek-v41-flash-1017-a-config-only-drop-in)
and [spark-bench](https://github.com/neko-legends/spark-bench#dsv41-1017).

## 0. Before you start

- Four DGX Sparks on one switch, Docker with the NVIDIA runtime on each, passwordless ssh from rank 0 to the other three.
- The Zig TP=4 engine already set up once: the image built and loaded on all four, `config/tp4-zig.env` filled in on
  every node, and the RoCE check passed. That is the runbook in [docs/ZIG-TP4-PLAN.md](../../docs/ZIG-TP4-PLAN.md#runbook-build-no-gpu-then-the-window)
  (image, assets) and [docs/ZIG-SERVE.md](../../docs/ZIG-SERVE.md). The image that serves 1017 is built from zig commit
  `3c5b773b` (native images on TP=4); nothing newer is required.
- Disk on each node: ~211 GB for the EXL3 pack, ~48 GB for its Engram shards, ~80 GB for the prepared folders, plus the
  session tier (up to 128 GiB). Rank 0 also holds the Engram source (~203 GB, deletable after step 4).
- This repository checked out at the same path on all four nodes.

## 1. Settings

```bash
cd tensorfold-4x-sparks
cp deploy/dsv41-1017/sparks.env.example   deploy/dsv41-1017/sparks.env      # WORKERS, REPO_DIR, the paths
cp deploy/dsv41-1017/zig-1017.env.example deploy/dsv41-1017/zig-1017.env    # IMAGE (your engine image tag)
# fill in every <placeholder>, then copy both files to the same path on the three workers
```

`sparks.env` names the three workers (ssh targets, rank order), the repo path and every directory. `zig-1017.env` is the
engine config each rank reads: your `config/tp4-zig.env` plus the 1017 weights and their own directories (rank `r`
reads `$ASSETS_ROOT/rank-r`). The knobs are the four-Spark serving profile unchanged: RoCE, `TF_DSV41_PREFILL_PIPE=1024`,
native images, 4 slots of 420,096 tokens. On the live cluster the effective engine environment of this template is
identical to the running rank 0's, apart from the link names and addresses.

Keep the 1017 directories apart from an earlier pack's. Switching between the two is then one stop and one start
with the other config.

## 2. Download and check (HF ids, pinned revisions)

```bash
deploy/dsv41-1017/download.sh exl3      # LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw @ afc7aa0a -> MODEL_DIR
deploy/dsv41-1017/download.sh engram    # deepseek-ai/DeepSeek-V4.1-Flash-1017 @ 1bae0e10, shards 47/48 + index/config/tokenizer -> FP8_DIR
```

After each download, `hf_verify.py` checks every file against the Hugging Face tree of that revision: size, and the
sha256 of every LFS file against its LFS oid (read with O_DIRECT, so the page cache is not flushed). Expected:

```text
LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw@afc7aa0a: files=66 bytes=210819040903 lfs-checked=46 bad=0
deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e10: files=6 bytes=203086919239 lfs-checked=2 bad=0
```

The two Engram shards' sha256 are `76a05106…` (47) and `9dc51433…` (48), as in the quant's README. The full FP8
checkpoint (~476 GiB) is not needed to serve.

## 3. Distribute the weights

```bash
deploy/dsv41-1017/fanout.sh
```

It writes a sha256 manifest of the checked pack on rank 0, rsyncs the pack to each worker in parallel, and runs
`sha256sum -c` on each worker. Expected: `verify <worker>: mismatches=0` three times. On the live cluster the copy
took 453 s over the CX7 link.

## 4. Pack the Engram tables (per rank)

```bash
deploy/dsv41-1017/engram-pack.sh        # PY_IMAGE= any image with python3 + numpy, or empty for the host python3
```

This runs [`scripts/pack_engram.py`](../../scripts/pack_engram.py) `--world 4` for each rank. It reads the source
shards and the EXL3 pack's `config.json` and writes `engram-l{1,14}-r<rank>of4.bin` (~48 GB a node). Each rank is then
checked with `--check` (headers, size, sampled rows); expect `ok (header, size, 258 rows)` for each file. Ranks 1-3 are
copied to their node and sha256-checked there. CPU and disk only: about 15 minutes for all four ranks.

## 5. Image-routing bias

```bash
deploy/dsv41-1017/bias-vl.sh
```

[`bias_vl_from_pack.py`](bias_vl_from_pack.py) reads the 43 `*.ffn.gate.bias_vl` tensors (F32[384] in 40 layers, F32[128]
in the 3 MTP layers) from the EXL3 pack. It writes them in the engine's format to `CACHE_DIR/dsv41-bias-vl/bias_vl.safetensors`
(`TF_DSV41_BIAS_VL=/cache/dsv41-bias-vl`), next to `config/pfdense-table.json`, on all four nodes. The file is
byte-identical to the served one: sha256 `6fe07ce482cdfbb35e081b0737d2ddf08e2bdaf58765533c0f340e614ac21f91`.

## 6. Runtime assets

Each rank's `ASSETS_DIR` holds `rope.bin`, `engram-host.bin` and `aot/`. They are built from `config.json`,
`tokenizer.json` and the serving knobs, and those are the same for 1017 as for the earlier V4.1 pack. If you already
serve the earlier pack with these knobs, copy its asset folders:

```bash
for r in 0 1 2 3; do cp -a <earlier-assets>/rank-$r "$ASSETS_ROOT/rank-$r"; done   # on each rank's node, its own rank
```

Otherwise generate them with [`scripts/zig-tp4-assets.sh`](../../scripts/zig-tp4-assets.sh) (`host` once against
`MODEL_DIR`, `aot` per rank). Export `TF_DSV41_IMAGES=native` and `TF_DSV41_PREFILL_PIPE=1024` for the `aot` step;
see the runbook in docs/ZIG-TP4-PLAN.md.

## 7. Start

```bash
deploy/dsv41-1017/serve-1017.sh start      # status | stop
```

It waits for the GPUs to be free, then starts ranks 3, 2 and 1 on their nodes (`CONFIG=zig-1017.env
scripts/run-zig.sh <rank>`) and rank 0 here, and waits for `/health`. **The first start writes the prepared rank
folders: 405 s** on the live cluster (`loaded in 405.3s`). Later starts load them in about 35 s. Set `ALIAS` in
sparks.env to serve a second model id (the live service lists `deepseek-v4.1-flash` too). Boot logs are in
`$LOGDIR/boot-r<rank>.log`.

## 8. Verify

```bash
python3 deploy/dsv41-1017/verify.py http://localhost:8000/v1 DeepSeek-V4.1-Flash-1017-TF
```

Seven gates, temperature 0, stdlib only. They cover: thinking on by default (answer plus a reasoning field), thinking
off, SSE streaming, an image ([`shapes.png`](shapes.png)), `json_schema` structured output, and a tool call with
thinking on and off. Expected (the live service, 2026-10-17):

```text
PASS chat_thinking_default: ('Paris', 69)
PASS chat_thinking_off: ('Paris', 2)
PASS stream: ('1, 2, 3, 4, 5', 5)
PASS image: ('Red square, blue circle.', 219)
PASS json_schema: ['city', 'landmarks', 'population_millions']
PASS tools_thinking: ([('get_weather', '{"city":"Tokyo","unit":"c"}')], 'tool_calls')
PASS tools_nothink: ([('get_weather', '{"city":"Tokyo","unit":"c"}')], 'tool_calls')
7/7 passed
```

## 9. Benchmark (optional)

The numbers in the README come from [spark-bench](https://github.com/neko-legends/spark-bench#dsv41-1017):
`bench/qbench.py` (quality and speed), `bench/prefill-ladder.py` (cold prompts at 32K / 131K / 200K tokens),
`bench/window-vision.py` (vision) and `bench/harness/run.py` (agent harnesses), all pointed at
`http://<rank0>:8000/v1` and model `DeepSeek-V4.1-Flash-1017-TF`.

| 2026-10-17, four Sparks | 1017 |
| --- | ---: |
| MMLU-200 | 89.5 % |
| one stream, code / prose | 151.8 / 67.3 tok/s |
| four streams, aggregate | 162.4 tok/s |
| cold prompt 32K / 131K / 200K tokens | 6.2 / 22.8 / 35.7 s |
| verify.py | 7/7 |

## Notes

- **Prompt format.** The engine's renderer matches 1017's `encoding/encoding.py` on every request shape the bench sends,
  and its effort map is already 1017's (low 50 / high 75 / max 100, default 75). Two corner shapes are not covered yet:
  tool namespaces (`namespace::name`), and replaying a tool history with thinking on but without `tools`.
- **The drafter is 6 bpw** in the 1017 pack (4 in the earlier one); the loader reads the width from the weights. Decode
  measured 2-7 % slower than the earlier pack (single runs); the drafter is the likely cause, not isolated.
- **Sampling default.** A request without `temperature` uses the engine's default sampling (like temperature 1). Agent
  harnesses that omit it are not greedy.
- **Long thinking.** 1017 can think for a very long time on agent turns; cap `max_tokens` or the effort for agent loops.
