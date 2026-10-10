# DeepSeek-V4.1-Flash on TensorFold, 4x NVIDIA DGX Spark

Serve DeepSeek-V4.1-Flash (the 2.9 bpw EXL3 pack) across **four** NVIDIA DGX Sparks, tensor-parallel (TP=4) over a
switched CX7 RoCE fabric, behind an OpenAI-compatible API: exact DSpark speculative decoding, Engram rows from local
NVMe, 4 request slots of up to 420K tokens, image input, NVMe sessions, structured output and DSML tool calls.

This is the four-Spark fork of **[jayleaton/deepseek-v41-tensorfold-spark](https://github.com/jayleaton/deepseek-v41-tensorfold-spark)**.
The engine, the DeepSeek-V4.1-Flash family and nearly everything in this repository are Jay Leaton's work on
[TensorFold](https://github.com/ashhart/TensorFold); this fork adds what four Sparks need (`patches/0003`), a
four-node launcher, and the measurements. Since 2026-10-07 it runs Jay's G19 engine. **Two Sparks? Use Jay's
repository**: it is the maintained two-Spark recipe.

> Measured on one cluster (four GB10s, one switch), one boot a configuration. Read [What is not solved](#what-is-not-solved)
> before relying on it.

Jay's [CUDA Zig serving path](docs/ZIG-SERVE.md) (TensorFold 1.0.2 plus the DeepSeek CUDA port, source in
[`engine/zig`](engine/zig)) is merged here as he ships it: **two Sparks (TP=2) and his q28-v2 2.8 bpw pack**. It
does not yet run on four Sparks; [ZIG-TP4-PLAN](docs/ZIG-TP4-PLAN.md) is the port plan. The four-Spark service on this
page is the Python engine.

## Results (four Sparks)

Weights: [`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw).
Configuration: [`config/prod.env.example`](config/prod.env.example) + [`config/tp4.env.example`](config/tp4.env.example)
(expert pruning on, `TF_DSV41_EXPERT_TOPP=0.85`), thinking off, temperature 0, 512-token replies, isolated runs.

### At a glance (live, 2026-10-07)

| | four Sparks (this fork) | two Sparks (Jay's) |
| --- | ---: | ---: |
| one user, short code prompt (`m2bench`, tok/s) | **122-124** | ~80 |
| one user, short prose prompt (`m2bench`, tok/s) | **68-69** | ~41-45 |
| four users at once, steady (`m2bench` C4, tok/s) | **~200** | |
| code / prose, 1k-160k prompts (geometric mean, tok/s) | **104 / 67** | |
| cold 160k-token prompt | **36 s** | |

Live build: Jay's G19 engine with `0003` (four Sparks), `0004` (rotated expert split, `TF_DSV41_EXPERT_ROTATE=1`) and
the 4-bit draft head (`TF_DSV41_DRAFT_HEAD=q4`), on RoCE; every reply exact (drafted == serial). Two-Spark figures are
Jay's published `m2bench` C1 numbers for his build.

### G19 on four Sparks (2026-10-07)

Jay's G19 engine (engine `7bd2d67`) with this fork's `0003`, live since 2026-10-07, against the G13 build it replaced,
one boot each in the same window:

| | G13 build | **G19 build** |
| --- | ---: | ---: |
| decode, prose 1k / 20k / 160k (cold, tok/s) | 53.1 / 50.1 / 47.5 | 54.5 / 50.3 / 48.6 |
| decode, code 1k / 20k / 160k (cold, tok/s) | 88.9 / 80.8 / 84.9 | 90.7 / 82.3 / 86.2 |
| cold first token, 20k / 160k | 5.9-6.7 s / 38.9-39.1 s | **5.4-6.0 s / 35.9-36.2 s** |
| context a request slot | 300K | **420K** (KV pool 1,201,152 tokens, `TF_DSV41_POOL_TOKENS`) |
| image input | no | **yes** (`TF_DSV41_IMAGES=native`) |

Checks on the G19 build: code word at 30 / 60 / 85% of 20k / 80k / 158k-token prompts 3 / 3, and at 50% of a
~405k-token prompt 1 / 1; short gates 7 / 7; six wide sampling cases (`top_k` 40,000 and 32,000, nucleus, JSON
nucleus) 6 / 6; an image question answered; the greedy reply byte for byte the G13 build's. Same tok/s within noise:
the gain is reading prompts (~7% at 160k: G19's speed switches and fused dense prefill), the 420K context and images.

Both columns of this table ran on the NCCL fallback, not RoCE (see the note under Decode below), which is most of
why they sit below the decode sweep. The harness also differs (1-stream, 512-token replies, 2 trials). Both columns
were measured the same way, so the comparison between them holds.

### Decode (2026-10-07, G19 on RoCE)

| prompt | 1k | 20k | 40k | 80k | 160k | geometric mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prose, tok/s | 66.0 | 66.7 | 68.7 | 69.6 | 62.0 | **66.5** |
| code, tok/s | 107.2 | 103.5 | 105.0 | 101.7 | 103.4 | **104.1** |
| cold first token, prose / code | 0.9 / 0.7 s | 6.0 / 5.3 s | 9.3 / 9.4 s | 17.9 / 17.8 s | 35.9 / 35.8 s | |
| *2026-10-04 build (G13, `0003` alone), prose / code* | *63.6 / 104.9* | *65.7 / 97.0* | *64.0 / 96.4* | *62.8 / 103.1* | *61.0 / 99.8* | *63.4 / 100.2* |

Median of 3 a cell (the published depth sweep: 512 tokens, greedy, thinking off, trial 0 cold, isolated). Four streams
at once: **122.2 tok/s** aggregate (`dsbench`, single-stream median 63.8; 2026-10-04: 119.2 / 62.6).

**Where single-user time goes at four ranks** ([spark-bench report](https://github.com/neko-legends/spark-bench/blob/main/artifacts/tensorfold-v41-decode-levers-20261007/REPORT.md)). None of the exact exchange and L2-prefetch switches moved decode beyond boot-to-boot noise. The 4-bit draft head (`TF_DSV41_DRAFT_HEAD=q4`, now in `config/tp4.env.example`) gave +2.5% code and +1.4% prose; the sweep above ran before it. A RoCE trace puts ~2.5 ms of a decode window in ranks waiting for the slowest one. About 0.9 ms of that comes from the uneven expert split (ranks 0 and 1 hold 640 of 2,304 columns, against 512 for ranks 2 and 3). `patches/0004` (`TF_DSV41_EXPERT_ROTATE=1`, live since 2026-10-07) rotates those wide slices round the ranks by expert id: one stream unchanged, four streams +1-1.5% (C4 aggregate 137.3 -> 138.7, steady 196.9 -> 199.8 tok/s, two boots each), 16-48-row verify windows 2-8% faster.

**Check the transport after every start.** A run-time RoCE failure writes `/cache/roce-failed` in the cache volume, and
while that file exists every start serves on NCCL; the round-plan link falls back to TCP too. A crash test left one
behind on 2026-10-05, and our server ran on NCCL for two days without an error. On the same build, NCCL vs RoCE
(`m2bench`, 1 stream): code 107.2 vs **119.6** tok/s, prose 55.8 vs **66.4**, 1-row verify window 18.7 vs **16.3** ms.
Rank 0's log should say `all-gathers of up to ... over RoCE` and `plan link: rdma`. If it doesn't, look for the file,
fix the cause and move the file aside.

### Reading a new prompt (2026-10-05)

| cold time to first token | 20k prompt | 160k prompt |
| --- | ---: | ---: |
| `0003` alone (2026-10-04) | 12.9-14.2 s | 97.4-99.8 s |
| + split selections and overlapped exchanges | 9.4-9.5 s | 74.6-75.2 s |
| **+ pipelined prompt reading (the shipped config)** | **5.8-6.8 s** | **38.8-39.6 s** |

Each build started with an empty session cache (every prompt read cold). With the pipeline a cold 160k-token prompt
reads at ~4,100 tokens/s; before it, TP=4 read prompts ~15-25% slower than two Sparks do (this repository's two-Spark
row: 1,953 tokens/s at 128K), because every rank recomputes the parts of a layer that tensor parallelism does not
split and waits at 43 exchanges a chunk. How the pipeline works: [docs/PREFILL_SPEED.md](docs/PREFILL_SPEED.md).
Warm repeats reuse the prefix: 0.26-0.64 s at every depth.

### Checks

- On the running server: arithmetic, forced tool call (`tool_choice=required`), tool continuation, strict JSON
  schema at T = 0 / 0.7 / 1.0, reasoning, and `scripts/canary.py` (chat, thinking, tool, json, tokenize): all pass.
- A code word hidden at 30 / 60 / 85% of 20k / 80k / 158k-token prompts: found 3 / 3 (pipelined and not).
- On the CPU (four ranks as threads, exact numerics): four ranks == one rank, the ranks agree bit for bit, a
  pipelined prompt leaves the same state as the tensor-parallel one, greedy / top-k / nucleus decoding on four ranks
  == one rank token for token (`tests/test_dsv41_tp4.py`, `test_dsv41_pipe.py`, `test_dsv41_four_sparks.py`).

Raw files: [`results/four-sparks-20261004/`](results/four-sparks-20261004/README.md),
[`results/prefill-speed-20261005/`](results/prefill-speed-20261005/README.md),
[`results/four-spark-fixes-20261005/`](results/four-spark-fixes-20261005/README.md). The same cluster's earlier deployments
(SGLang on the FP8 checkpoint, vLLM) and a dated history of every change: [spark-bench](https://github.com/neko-legends/spark-bench).

## What four Sparks change

| part | two Sparks (a rank) | four Sparks (a rank) |
| --- | --- | --- |
| query heads | 32 | 16 |
| routed + shared experts (intermediate 2,304) | 1,152 | 640 / 640 / 512 / 512 |
| vocabulary (head columns, embedding rows) | 64,640 | 32,384 / 32,384 / 32,256 / 32,256 |
| Engram hash heads | 12 | 6 (`engram-l{1,14}-r{rank}of4.bin`) |
| resident weights | ~99 GiB | ~56 GiB (+~27 GiB for the prompt pipeline's full-width layers) |

Every split stays on whole 128-wide Hadamard blocks (EXL3 rotations act per block), so the widths that do not divide
by four are uneven; every exchange sends the same size from every rank. Details: [docs/FOUR_SPARKS.md](docs/FOUR_SPARKS.md).

## Quick start

Requirements: four DGX Sparks on one switch, one IPv4 subnet per CX7 port function, Docker with the NVIDIA runtime on
each, passwordless ssh from the head (rank 0) to the three workers over the link; on each node's local NVMe ~200 GB
for the weights, ~47 GB for its Engram shards, ~50 GB for the prepared rank folder, and room for the session tier.
Nothing else on the GPUs.

```bash
git clone --recurse-submodules https://github.com/neko-legends/deepseek-v41-tensorfold-spark.git
cd deepseek-v41-tensorfold-spark
cp config/prod.env.example config/prod.env        # the engine knobs
cp config/tp4.env.example config/tp4.env          # every <placeholder>: workers, link addresses, paths

# 1. weights: the same EXL3 pack on all four nodes, same path
hf download dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw --local-dir <MODEL>

# 2. Engram shards, rank r on node r (from DeepSeek's original checkpoint: docs/FOUR_SPARKS.md)
python3 scripts/pack_engram.py --src <deepseek-ai/DeepSeek-V4.1-Flash dir> --config <MODEL>/config.json \
    --out <ENGRAM> --world 4 --rank <r>

# 3. the image (TensorFold v0.6.0 + patches/*), shipped to the workers, CUDA extensions prebuilt on all four
docker build -f docker/Dockerfile -t dsv41-tensorfold:tp4 .
bash scripts/serve4.sh ship
bash scripts/serve4.sh prebuild

# 4. serve (the first start writes the prepared rank folders; later starts load in ~35 s)
bash scripts/serve4.sh start
bash scripts/serve4.sh status | logs [R] | stop
```

Then:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "DeepSeek-V4.1-Flash-TF",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}], "reasoning_effort": "low"}'
```

Optional: `scripts/keeper4.sh` from cron restarts the four ranks after a reboot or three failed health checks.
Setup notes and lessons from the first boots: [docs/FOUR_SPARKS.md](docs/FOUR_SPARKS.md). Knobs and memory gates:
[docs/OPERATIONS.md](docs/OPERATIONS.md).

## API

OpenAI-compatible on `HOST:PORT` (`127.0.0.1:8000` by default: put your own proxy and authentication in front):
`/v1/chat/completions` (streaming, tool calls, `response_format`), `/v1/completions` (text or token ids),
`/tokenize`, `/v1/models`, `/health`, `/metrics` (TensorFold's own series and, from `0003`, the vLLM-named ones
fleet dashboards read), `/v1/model_info` (`max_num_seqs`: the request slots). Thinking follows DeepSeek-V4.1's encoding and is on by default
(`TF_DSV41_THINKING=0` turns it off); `reasoning_effort` `none` / `low` / `medium` / `high` / `max` or 1-100. The
full table: [docs/TWO_SPARKS.md#api](docs/TWO_SPARKS.md#api) (unchanged at four Sparks).

## The engine

`vendor/TensorFold` is upstream TensorFold v0.6.0, unmodified; the Dockerfile applies `patches/` in order:

| patch | from | what |
| --- | --- | --- |
| [`0001-spark-stack-060.patch`](patches/0001-spark-stack-060.patch) | jayleaton | the GLM-5.3-Flash Spark engine stack rebased onto 0.6.0: communicator, RoCE all-gather, server pieces |
| [`0002-deepseek-v41-family.patch`](patches/0002-deepseek-v41-family.patch) | jayleaton | the DeepSeek-V4.1-Flash family (engine G19, `7bd2d67`) |
| [`0003-four-sparks.patch`](patches/0003-four-sparks.patch) | this fork | four ranks (TP=4): whole-block uneven splits, `--tp 4 --rank 0..3`, N-way rank agreement, plan link (TCP and RDMA), fail-fast and memory floor; the fixes from Jay's review; pipelined prompt reading, split selections and overlapped exchanges (opt-in, on in `tp4.env.example`); `/health` draft counters; sync-free expert counts; vLLM-named `/metrics` series and `/v1/model_info` |
| [`0004-expert-rotate.patch`](patches/0004-expert-rotate.patch) | this fork | `TF_DSV41_EXPERT_ROTATE=1`: each expert's wide 128-column slices rotate round the four ranks by expert id instead of always landing on ranks 0 and 1 (`x3ld` reads an expert's own width); exact, +1-1.5% at four streams |
| [`0005-drafter-fidelity.patch`](patches/0005-drafter-fidelity.patch) | this fork | drafter-training tools: the capture logs each pass's candidates and noise flag; the trainer's drafter reads the engine's q4 head and bf16 mHC weights (port agreement 0.72 -> 0.90); `m2bench` / `draftcap --world` |

Until 2026-10-07 the fork ran on G13 with these as patches 0003-0006 (branch `four-sparks-g13`). Jay's PR #6 asks for
a smaller `0003` (four ranks and the review's fixes only, two-Spark output unchanged): that version is the `four-sparks`
branch, verified byte-identical at TP=2 against his main.

### Fixed 2026-10-05 (then patch 0005, now in 0003)

Jay reviewed the TP=4 port ([PR #6](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/pull/6)) and found
bugs that only uneven slices or more than two ranks expose. All fixed, with tests in `tests/test_dsv41_four_sparks.py`
(each review bug's test fails without its fix), and checked on the four Sparks (2026-10-05; serving since 11:30):

- the four benchmark replies (20k and 160k, prose and code) are byte-identical to the 0004 build's; cold first token
  5.8-6.8 s (20k) and 38.8-39.0 s (160k), as before; decode the same as 0004 in an A/B in one window;
- the hidden code word found 3 / 3, the short gates 7 / 7;
- sampled requests past the narrow slices pass: nucleus rows, nucleus with a JSON schema (full-vocabulary
  candidates), `top_k` 32,000 and 40,000, including `top_k` 40,000 as the first request after a boot.

The fixes:

- **Candidates past the narrow vocabulary slices**: every rank took `min(count, its own width)` candidates, so a
  count above 32,256 (`top_k` above that, or masked nucleus rows, which ask for the whole vocabulary) all-gathered
  unequal sizes. On the four Sparks, the 0004 build answered a `top_k` 40,000 request with an illegal memory access
  on ranks 2 and 3 (the narrow slices) and the server stopped. Every rank now sends the widest slice's count,
  narrower slices padded with entries that sort after every real token (`pick.rank_top`).
- **Trimmed draft head** (`TF_DSV41_DRAFT_HEAD=trim`): the per-rank id lists assumed equal slices; rank 2 refused to
  boot. They now follow the 128-block split.
- **TCP plan link** (`TF_DSV41_PLAN_LINK=tcp`): followers now say their rank and world size; rank 0 refuses strays,
  duplicates and another world size, orders the connections by rank, and names a rank that never connects.
- **`--tp 4`** is refused for families that do not declare it (`CUDA_TP`), instead of starting an engine that takes
  neither path.
- **`TF_DSV41_PREFILL_ATTN_BMQ=32`** (the two-Spark production value) on a 16-head rank now runs the fused prefill
  attention with 16 heads a program instead of silently falling back to the chunk kernels.
- **NVMe session tier**: the world size is part of the directory's identity, so an entry written by two Sparks is
  never resumed by four.
- **The first very wide window** (found while testing the above on the four Sparks): with the gather fixed, a
  sampled request with `top_k` 32,000-40,000 as the server's first wide request still timed out every follower's
  Engram gate at layer 14 (4 of 4 runs); runs whose first wide request was a JSON nucleus request passed. Most
  likely cause: the window needed the large-`k` top-k and large-row sort kernels for the first time, and CUDA's lazy
  module loading loaded them while the forward still waited on host threads (the Engram gate, the RoCE exchanges).
  The candidate kernels of every width class now run once at boot on every rank (`cand_warm`), and the candidates'
  pinned buffer grows before a window's forward is queued (`cand_reserve`): since then the same requests pass in
  every run (2 of 2, one with `top_k` 40,000 as the first request after a boot). Not fully explained: the 0004 build
  once answered a first `top_k` 32,000 request without the warm-up.
- The gate scorer and DSpark delta shards (both off in production) now handle any rank count and uneven splits.

## How it was tested (2026-10-05)

*This section describes the 2026-10-05 G13 build (patches 0001-0006, branch `four-sparks-g13`). The G19 build's checks are
in [G19 on four Sparks](#g19-on-four-sparks-2026-10-07). Its patches rebuild the same way. Its CPU suites pass except the
three tests that also fail on Jay's main (they need his unpublished draft vocabulary or a network this sandbox lacks)
and two that the fork's extra prompt-reading code trips (below, What is not solved).*

Every number and claim on this page was checked these ways before `main` moved to it.

**1. The patches rebuild the engine exactly.** TensorFold v0.6.0 (`vendor/TensorFold`) with `patches/0001`-`0006`
applied in order gives the same tree, file for file, as the engine branch the work was done on:

```bash
cp -r vendor/TensorFold /tmp/tf && cd /tmp/tf && rm -rf .git
for p in "$OLDPWD"/patches/*.patch; do git apply --whitespace=nowarn "$p"; done
```

**2. The CPU test suites pass on that tree.** No GPU needed: the tests run the engine's CPU twin, with four ranks as
threads where it matters (`pip install torch numpy safetensors pytest`; run each file in its own process, as some keep
large fixtures):

```bash
cd /tmp/tf
for f in tests/test_dsv41_*.py tests/test_cuda_cli.py; do python -m pytest -q "$f"; done
```

On the final tree, 23 files (every DeepSeek V4.1 suite that the TP=4 patches touch, and the CLI suites): **327
passed, 2 failed**. Both failures are older than this fork's patches and unrelated to four Sparks:
`test_cuda_cli.py::test_serve_parses_the_kv_cache_flag` (an upstream test that still expects `--kv-dtype fp8` to be
refused; `patches/0002` adds it) and `test_dsv41_draft_head.py::test_shipped_ranking` (needs the unpublished draft
vocabulary file). The four-Spark tests:

| file | what it checks | tests |
| --- | --- | ---: |
| `test_dsv41_tp4.py` (0003) | uneven 128-block splits; four ranks == one rank in exact numerics; ranks agree bit for bit; row invariance; greedy over uneven vocabulary slices; Engram from `of4` shards | 8 |
| `test_dsv41_overlap.py`, `test_dsv41_pipe.py` (0004) | split selections and overlapped exchanges == one segment; a pipelined prompt leaves the tensor-parallel state, at 2 and 4 ranks, through a mid-prompt snapshot, over point-to-point | 12 |
| `test_dsv41_four_sparks.py` (0005) | every fix in the list above; run without the fixes, each review bug's test fails | 20 |

The thread communicator in these tests refuses all-gathers whose ranks send different sizes, as NCCL requires, so a
mismatch fails a test instead of hanging a Spark.

**3. The image builds from a fresh clone.** `git clone --recurse-submodules` of this repository, then
`docker build -f docker/Dockerfile .` exactly as in [Quick start](#quick-start), shipped to the three workers with
`scripts/serve4.sh ship`: that image is the one serving on our four Sparks.

**4. On the four Sparks, before it went live.** Each candidate build ran in a test window (the live server stopped,
the build started with an empty session cache, the checks, the live server restored), and it was promoted only if
every check passed:

- the depth bench's 20k and 160k prompts, prose and code, read cold: same first-token times and byte-identical
  replies to the previous build;
- a code word hidden at 30 / 60 / 85% of 20k / 80k / 158k-token prompts: 3 / 3;
- the short gates (arithmetic, forced tool call, tool continuation, strict JSON at three temperatures, reasoning):
  7 / 7;
- sampled requests past the narrow slices: nucleus, nucleus with min_p, `top_k` 40,000 and 32,000, `top_k` 20, a
  JSON schema with nucleus sampling: 6 / 6, and `top_k` 40,000 as the first request after a fresh boot;
- decode against the previous build, four boots alternating in one window: the same within noise.

Three candidate builds failed the wide-request check before the shipped one passed; they never served. The previous
build was also tested on the same wide request: it crashed ranks 2 and 3. All of it, run by run:
[`results/four-spark-fixes-20261005/`](results/four-spark-fixes-20261005/README.md).

## What is not solved

- Two CPU tests fail on the G19 build's full `0003` (not on PR #6's smaller one): `test_dsv41_calib_knobs`
  (the prompt-reading switches are not yet in the calibration key or its exempt list) and `test_dsv41_pdl`'s
  forward-hook count (the overlapped-exchange refactor of the layer loop). Neither changes what the server computes.
- The boot memory budget (`memory.load_check`) is anchored on two-Spark measurements: conservative on ranks 0 and 1,
  skipped on ranks 2 and 3. The live floor is MemAvailable ~26 GB a node with the pipeline on.
- A pipelined prompt's bits are one rank's arithmetic, not the four-rank sum's (its own session tag): as different
  as the engine's own fast vs exact prefill kernels. Exact numerics give the same state either way.
- Short prompts gain less from the pipeline (it fills in four steps): cold 20k is 5.4-6.0 s.
- One cluster, one boot a configuration; expert pruning is lossy (delete its three lines for the unpruned model).

## Documentation

| | |
| --- | --- |
| [FOUR_SPARKS](docs/FOUR_SPARKS.md) | the TP=4 port: what changes, setup, first-boot lessons, results |
| [PREFILL_SPEED](docs/PREFILL_SPEED.md) | the prompt pipeline, split selections, overlapped exchanges, what did not work |
| [ZIG-SERVE](docs/ZIG-SERVE.md) / [ZIG-RESULTS](docs/ZIG-RESULTS.md) | Jay's Zig engine: build, TP=2 launch, exact settings, q28-v2 comparisons (two Sparks only) |
| [ZIG-TP4-PLAN](docs/ZIG-TP4-PLAN.md) | what the Zig engine needs for four Sparks, and the like-for-like bench recipe |
| [INSTALL](docs/INSTALL.md) | requirements, weights, Engram shards, build, start, watchdog |
| [API](docs/API.md) / [IMAGES](docs/IMAGES.md) | endpoints, thinking and effort, image input |
| [OPERATIONS](docs/OPERATIONS.md) | settings, memory gates, turning each lever off |
| [TWO_SPARKS](docs/TWO_SPARKS.md) | Jay's two-Spark README at the fork point: method, quality tables, strict mode, credits |
| [ENGINE](docs/ENGINE.md) / [ARCHITECTURE](docs/ARCHITECTURE.md) / [DECODE](docs/DECODE.md) | the patches, the model split, the decode roofline |
| [campaign/](docs/campaign/README.md) | Jay's development log, windows G1-G13 |

## Licensing

Apache License 2.0 for this project's own code, patches, scripts, benchmarks and docs ([`LICENSE`](LICENSE),
[`NOTICE`](NOTICE)): keep the copyright lines and the NOTICE attributions and state your changes. This fork's changes
are listed in [`NOTICE`](NOTICE). TensorFold (`vendor/TensorFold`, unmodified) is Apache-2.0 from 0.6.0; the code the
patches modify stays under its license. Files the patches add keep their own SPDX notice (the DeepSeek-V4.1-Flash
family: MIT, Copyright (c) 2026 Jay Leaton). Third-party code and the model weights: the full table in
[docs/TWO_SPARKS.md#licensing](docs/TWO_SPARKS.md#licensing). The uncensored weights have refusals removed; you are
responsible for how you use them.

## Credits

- **[Jay Leaton (jayleaton)](https://github.com/jayleaton)**: the DeepSeek-V4.1-Flash engine for TensorFold and the
  two-Spark recipe this fork extends, and the review that found the bugs 0005 fixes. Follow him on
  [X](https://x.com/jayleaton) / support him on [Buy Me a Coffee](https://buymeacoffee.com/jayleaton).
- [Ash Hart (ashhart) / TensorFold](https://github.com/ashhart/TensorFold): the engine, the EXL3 kernels, the drafting
  and the server.
- [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash): the model, its encoding, and the CED / CSA2 /
  mHC / DSpark / Engram designs.
- [dealignai](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw) (the uncensored pack),
  [Mia-AiLab](https://huggingface.co/Mia-AiLab) (the 2.9 bpw EXL3 pack), [turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3)
  (EXL3), and everyone credited in [docs/TWO_SPARKS.md#credits](docs/TWO_SPARKS.md#credits).
