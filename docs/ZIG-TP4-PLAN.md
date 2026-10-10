# The Zig engine on four Sparks (TP=4): port plan

Jay's Zig engine (`engine/zig`, TensorFold 1.0.2 + the DeepSeek-V4.1 CUDA port, merged from `jay/main` at
`199ea205`) serves DeepSeek-V4.1-Flash on **two** Sparks. This page lists every place it assumes two ranks, what each
of this fork's Python patches does and whether Zig has it, the order to port in, and the risks. Paths are relative
to `engine/zig/zig/src/` unless they start with `scripts/`, `config/` or `docker/`; `ds/` is
`families/deepseek_v41/`. Line numbers are at the merge (`199ea205`) unless a section says otherwise.

**Status of this branch** is in [the last section](#status-on-branch-zig-tp4). In short, the host-side port in step 1
is done and tested on the CPU. Nothing here has run on a GPU yet.

## Weights: what the Zig path expects and what we serve

| | Jay's Zig path | our four Sparks (live) |
| --- | --- | --- |
| pack | q28-v2 EXL3 2.8 bpw (`jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw`, publication pending) | `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw` (README, `config/tp4.env.example` `MODEL=`) |
| expert widths | per-expert bit widths (K2 4-10, "ragged" layout, `ds/exl3.zig:131-162`); every expert's columns are 2304 | one bit width a projection ("stack" layout) |
| Engram shards | `engram-l{1,14}-r{rank}of{world}.bin` (`ds/engram_rows.zig:290`) | `engram-l{1,14}-r{rank}of4.bin`, already packed (`scripts/pack_engram.py --world 4`) |
| prepared files | Zig's own (`TF_DSV41_PREPARED`, key has rank and world: `ds/prepared.zig:67`), not Python's | Python's prepared folders |
| assets | per-rank `rope.bin`, `engram-host.bin`, Triton AOT `aot/` (`docs/ZIG-SERVE.md`) | none (Python JITs) |

**Can the Zig path serve our 2.9 bpw pack?** By construction, yes. `exl3.Layout.of` reads each expert's width from
its trellis header and classifies a projection as `.stack` when all widths agree (`ds/exl3.zig:159-161`). A uniform
pack is the simpler case of the same loader, and `x3gm` has a uniform-stack form (`kernels/cuda/deepseek_v41/x3gm.cu:539`).
Jay's own README says the reverse direction is the restriction: the Python patches *predate* q28-v2's ragged loader,
so Python keeps the 2.9 bpw pack. The model config and tokenizer are the same model. Jay's measurements are all
q28-v2, so a Zig run on 2.9 bpw is not on any qualified path. The like-for-like comparison is **both engines on the
same 2.9 bpw pack**, which needs no conversion. Switching to q28-v2 would be a separate quality and speed question,
and the Python TP=4 engine cannot load it today.

Not verified on the Sparks: the live container's mounts, env and model directory. This session had no ssh or docker
access (both refused by the permission policy), so the "live" column above comes from this repository's
configuration and README, not from `docker inspect dsv41-tf4-r0`. Check before the window:
`docker inspect dsv41-tf4-r0 --format '{{json .Mounts}} {{json .Config.Env}}'` (read-only).

## Where the Zig engine assumes two ranks

### Transport (`fabric/tp/`)

The TP runtime is N-rank everywhere except the RoCE and mailbox fast path. `TF_TP_WORLD` accepts 1..64
(`fabric/tp/config.zig:69`). The bootstrap's all-gather, broadcast and agreement have a 3-rank unit test
(`fabric/tp/bootstrap.zig:219`). NCCL is generic (`fabric/tp/nccl.zig:45`). The plan link writes every frame to
`fds[1..world]` (`fabric/tp/planlink.zig:48`). Fail-fast is N-way (`fabric/tp/fate.zig:98,152,206`).

| file:line | assumption | TP=4 change |
| --- | --- | --- |
| `fabric/tp/roce.zig:154` | `if (b.world != 2) return error.Unsupported` | Remove once the parts below are N-way. Until then, TP=4 falls back to NCCL (`TF_TP_ROCE_FALLBACK=nccl`, the default; `fabric/tp/tp.zig:94,119-122`) |
| `fabric/tp/roce.zig:18,147-149` | `send [2][slot], recv [2][slot]` | receive area (world-1) x 2 slots |
| `fabric/tp/roce.zig:193-197,222-228,211,408-425` | one peer: `blobs: [2]`, `all[1 - r.me]`, `connect(peer)`, `hostGatherV` with `1 - r.me` | one queue pair per (peer, HCA); loop over peers |
| `fabric/tp/roce_verbs.zig:118-119,218-228` | one `peer_region` / `peer_rkey` | per peer |
| `fabric/tp/roce.zig:287-316` (`proxyLoop`) | posts each sequence number once to one wire; at most 2 pending | post to every peer; recheck the bound |
| `fabric/tp/mailbox.cu:447-590` (`tp_roce`) | one send slot, flags from one peer (`flags [2][kRoceHcas]`), copies out one inbox; gather at `out + (1-rank)*n` | N-1 inboxes and flags; place by sender rank |
| `fabric/tp/mailbox.cu:195-227` (`sum_bytes`) | `rank == 0 ? m : p`: a sum of exactly two operands | rank-ordered sum of N (`fabric/tp/collective.zig:43`'s contract) |
| `fabric/tp/roce.zig:52`, `mailbox.cu:420-423` | `lens: int32 [2]` | `[world]` |
| `fabric/tp/mailbox.zig:76,150,197`, `tp.zig:152` | the single-host mailbox | none: one host only, not used across Sparks |
| `fabric/tp/gpu_checks.zig:56,115,128,153`, `roce_host.zig:31-35`, `run_pair.sh`, `run_hosts.sh` | test harnesses with `1 - rank` | generalise when RoCE is |

### Weight, expert and vocabulary split (`ds/`)

The split is parameterized by world size, but only for **even** whole-128-block splits. At TP=4, two real widths
are uneven: the expert width 2304 (18 blocks, 640/640/512/512) and the vocabulary 129,280 (1,010 blocks,
32,384/32,384/32,256/32,256). The Python 0003 `block_bounds` (extra blocks on the low ranks) is the reference.

| file:line | assumption | TP=4 change |
| --- | --- | --- |
| `ds/plan.zig:266` | `c.vocab % (128 * o.world) != 0` → `BadWorld`: **boot refuses TP=4** | vocabulary in whole 128-blocks only |
| `ds/exl3.zig:82-87,90-95` | `Group.part`: `k % (world*128)`, `per = tiles / world`, `offset = rank * per` | block bounds (lo, len) per rank |
| `ds/block_moe.zig:154`, `ds/block_prefill.zig:1102`, `ds/block.zig:483` | `I = expert_width / world` | the rank's own expert width |
| `ds/block_moe.zig:329`, `ds/block_prefill.zig:454`, `ds/dspark_emit.zig:459` | head logits `V = vocab / world` | the rank's own head width |
| `ds/forward.zig:306-377` (greedy) | `V = vocab / W`, ids `rank * V` | the rank's (lo, len); the all-gathered (value, id) pairs stay one per row per rank |
| `ds/sampling_pipe.zig:163-316`, `ds/sampling_gpu.zig:51,224` | `V` per rank, `k = min(count, V)`, `wholeRow` gathers V floats with "global id = index" | k from the narrowest slice (the same on every rank); whole row padded to the widest slice and ids remapped |
| `ds/grammar_gpu.zig:92-93` | mask `width = vocab / W; lo = rank * width` | the rank's (lo, len) |
| `ds/dspark_gpu.zig:620-636`, `ds/dspark_slots.zig:156-171,1076-1086`, `ds/dspark_dev.zig:159,201-211`, `ds/m5.zig:152` | draft candidates over `V = vocab / W`, `id0 = rank * V` | the rank's (lo, len) |
| `ds/forward.zig:908,930`, `ds/batch.zig:704`, `ds/dspark_gpu.zig:583`, `ds/dspark_slots.zig:1040` | embedding rows `vocab / W`, offset `rank * V` | none: the embedding is a native tensor split evenly by rows (`ds/plan.zig:163-165`); 129,280 / 4 = 32,320 is even. It is independent of the head's split |
| `ds/block_prefill.zig:66,905,1000,1012` | `bmq = 32` and `H % bmq != 0` → `Unsupported`: at TP=4 a rank has 16 heads | `BMQ = min(32, H)`, as 0003 clamps it; the AOT fill then lists the BMQ-16 `_fused` variant |
| `ds/block_prefill.zig:127`, `ds/prod_knobs.zig:445` | `TF_DSV41_PF_OVERLAP` only at world 2 | leave unset at TP=4 (or port to N peers later) |
| `ds/aot_needs.zig:213` | `TF_TP_WORLD` defaults to 2 | needs the world **and the rank**: per-rank widths change the Triton variants |
| `ds/aot_needs.zig:236`, `ds/m5_emit.zig:118` | split pool `@divExact(pages, 2) + 2` | `/ world` |
| `ds/m2b.zig:96,276`, `ds/m2a.zig:79`, `ds/m1_main.zig:239`, `ds/forward.zig:1354` | gate drivers with `.world = 2` | only if those gates should run at TP=4 |

Heads (64/4 = 16), `o_groups` (8/4 = 2), `wq_b` (32,768/4), `wo_b` (8,192/4), the Engram head columns (24/4 = 6,
`ds/engram_host.zig:100-105`) and the paged KV pool (4,692 pages % 4 = 0) all divide evenly.

### KV pool, scheduler, sessions, calibration, server

Nothing here hard-codes two ranks. Page ownership is `(t >> psh) % world` (`ds/kv/split.zig:161`). The pool
rounds to `256 * world` (`ds/kv_state.zig:65`). Agreement is an N-way all-gather (`ds/graphs.zig:156-194`). The calibration table key includes the world
(`ds/draft/calib_env.zig:159,182`), so TP=4 measures its own table on first boot (`TF_DSV41_CALIB=zig`). The NVMe
session tier's compat id includes the world (`ds/sessions_gpu.zig:209-216`), so TP=2 session files are ignored,
not misread. Drain-first stop sends `op_stop` to every follower. Notes for TP=4:

- The round planner's memory floor reads **rank 0's** `/proc/meminfo` only (`ds/kv/sched.zig:166`,
  `ds/sched_host.zig:9-10`). The boot check runs on every rank (`ds/kv_state.zig:72-120`). Python 0003 checks the
  tightest follower. At TP=4, ranks 2 and 3 run with different headroom (different expert and head widths), so
  this matters more than at TP=2.
- `CONTEXT=420000` is not a multiple of the 256-token page. `aot-needs` does `@divExact(limit, 256)`
  (`ds/aot_needs.zig:233`) and would panic. Use **`TF_DSV41_CONTEXT=420096`** (1,641 pages), the smallest page
  multiple above 420,000. With `TF_DSV41_POOL_TOKENS=1201152` (4,692 pages, 1,173 a residue), two full 420K slots fit
  at once; four slots share the pool.
- Messages only: `server/spark.zig:255,355,380` "restart both ranks"; comments in `ds/kv/layout.zig:7`,
  `ds/kv/budget.zig:238,354` and others.

### Launcher and config

- `scripts/run-zig.sh:5-6,24`: rank 0 or 1 only, `TF_TP_WORLD=2` fixed. Needs ranks 0..WORLD-1 and the world
  from the config. Ranks above 0 already run `tf-dsv41-m1 follow`.
- `config/prod-zig.env.example` is the TP=2 profile: `TF_DSV41_CONTEXT=1048576`, `POOL_TOKENS=1400000`.

## Our Python patches and their Zig equivalents

| patch / feature | what it does | Zig today | Zig change |
| --- | --- | --- | --- |
| 0003 whole-block uneven splits | experts 640/640/512/512, vocab 32,384/32,384/32,256/32,256 | absent (`BadWorld`, `SplitNotOnBlocks`) | step 1 below |
| 0003 `--tp 4 --rank 0..3` | four ranks | `TF_TP_WORLD`/`TF_TP_RANK` (`fabric/tp/config.zig:69-70`); launcher 0/1 only | launcher |
| 0003 N-way rank agreement, plan link with rank handshake, fail-fast | | has equivalents (`fabric/tp/bootstrap.zig:80,174`, `fabric/tp/fate.zig`) | none |
| 0003 plan link over RDMA at 4 ranks | the round plan over the RoCE mailbox | `TF_DSV41_PLAN_LINK=rdma` is TCP plus spin-polling (`fabric/tp/config.zig:111-114`) | none needed first; measure |
| 0003 RoCE all-gather at 4 ranks | Python's RoCE pairs HCA h of every rank by subnet | 2-rank only, falls back to NCCL | step 3 below |
| 0003 memory floor on the tightest follower | | boot check per rank; round planner rank 0 only | gather every rank's MemAvailable each round (or every N rounds) |
| 0003 candidates past narrow vocab slices (`rank_top`) | wide top_k on uneven slices | not needed at TP=2; Zig caps device top-k at 1,024 and sends wider rows through `wholeRow` (`ds/sampling_pipe.zig:22-25,195-199`) | step 1: pad `wholeRow` to the widest slice |
| 0003 trimmed draft head (`TF_DSV41_DRAFT_HEAD=q4`) | the DSpark pass reads a 4-bit copy of the head (+2.5% C1 code) | absent: DSpark reuses the backbone head (`ds/dspark_emit.zig:459`) | later; a separate speed feature |
| 0003 BMQ 32 → 16 clamp | 16 heads a rank | absent (`Unsupported`) | step 1 |
| 0003 world size in the NVMe session identity | | has it (`ds/sessions_gpu.zig:212`) | none |
| 0003/0004 pipelined prompt reading (`TF_DSV41_PREFILL_PIPE`, `PIPE_P2P`) | long prompts through the four ranks as a pipeline (+~27 GB a node) | absent (NCCL send/recv exists, `fabric/tp/nccl.zig:117`) | large: new prefill scheduler; port after TP=4 parity |
| 0003/0004 split selections (`TF_DSV41_INDEX_SPLIT`) | each rank selects a quarter of the rows | absent | medium |
| 0003/0004 overlapped exchanges (`TF_DSV41_PREFILL_OVERLAP`) | exchanges behind compute | `TF_DSV41_PF_OVERLAP` exists but is TP=2 only (`ds/prod_knobs.zig:445`) | generalise its side-stream pieces to N peers |
| 0003 `/health` draft counters | | has them (`server/spark.zig:173,236-237`) | none |
| 0003 sync-free expert counts | removes a `torch.bincount` host sync | not applicable: Zig groups experts on the device (`ds/kernels_exl3.zig:579`) | none |
| metrics: vLLM-named `/metrics` series, `/v1/model_info` | dashboards read `vllm:*`; `max_num_seqs` | `tensorfold:*` mirrors only (`server/metrics.zig:6,138-146`); Spark wire `tensorfold_*`; no `/v1/model_info` | small: a `vllm:` alias block and one route |
| 0004 `TF_DSV41_EXPERT_ROTATE` | each expert's wide 128-column slices rotate round the ranks by expert id (+1-1.5% C4) | absent; conflicts with `exl3.Layout.of`'s one shape a projection (`ds/exl3.zig:155`), `stage.zig`'s one `npart`, the kernels' scalar `I` | medium-large: per-expert N tables in x3ld / x3gm, padded scratch |
| 0005 drafter-training capture | logs each pass's candidates and noise flag; trainer reads the q4 head | absent (`TF_DSV41_ACCEPT_LOG` logs policy events only) | only if drafter training moves to Zig; Python stays the capture tool |
| 0005 candidate warmup at boot | CUDA lazy loading mid-window caused wide top_k timeouts | not needed: Triton kernels are AOT and resolved at boot (`cuda/aot.zig:80-105`); a missing variant is `MissingTritonVariant`, not a JIT | none (verify `CUDA_MODULE_LOADING` in the window) |
| 0005/0003 pinned candidate buffer grown before the forward | | grown lazily after the forward is queued (`ds/sampling_gpu.zig:144,162-168`), bounded (`W*k <= 2048`) | low risk; grow at init |
| 0006 logprobs / top_logprobs (not live) | | refused explicitly (`server/fields.zig:276-286`) | medium: per-rank log-sum-exp parts in the candidates' gather, as 0006 |
| images (`TF_DSV41_IMAGES=native`, live) | DeepSeek's ViT on rank 0 | has it (`server/deepseek.zig:12-14`, `ds/vision_rows.zig:298-306`); jay's profile sets `placeholder` | set `native` and `TF_DSV41_BIAS_VL` on every rank |

## Order of work

1. **Host-side TP=4 (this branch).** Block-bounds partition, per-rank expert and head widths, the vocabulary
   shard in greedy, sampling, grammar and drafting, the BMQ clamp, the AOT world and rank, the launcher and a
   TP=4 config. Host tests at world 4. *Estimate: 1-2 days of work; this session's status is below.*
2. **First GPU window: correctness on NCCL.** Build `dsv41-tensorfold:zigtp4-<sha>` with fatbins. Run the AOT fill per
   rank with `TF_TP_WORLD=4 TF_TP_RANK=r`, then the rope and Engram host assets. Write the prepared files (first
   boot: several minutes a rank, about 50-100 GB of NVMe a node). Boot on NCCL. Then the gates: 17 x 23 = 391,
   greedy token parity against the live Python TP=4 on the depth bench's prompts, drafted equal to plain,
   concurrent equal to solo, and the code word at 30/60/85 % of 20k/80k/158k. *1-2 windows; most likely failures
   are a missing AOT variant, or a kernel shape check refusing 640/512 widths or 32,384/32,256 head slices.*
3. **N-way RoCE.** `mailbox.cu`'s `tp_roce` and `roce.zig` from one peer to N-1, with sums in rank order. Without it,
   TP=4 runs every exchange on NCCL. On the Sparks, Python's NCCL vs RoCE gap was large enough that the README warns
   about the `roce-failed` trap. *3-5 days plus a window.*
4. **Bench** (`bench/zig_vs_python.py`, below) against the live Python TP=4 on the same pack.
5. Features by measured value: overlapped exchanges at N peers, the prefill pipeline, `EXPERT_ROTATE`, the q4
   draft head, rank-wide memory floor, vLLM metrics and `/v1/model_info`, logprobs.

**Total estimate** to a measured, correct TP=4 Zig service on NCCL: about 1 week including two windows. With N-way
RoCE: 2 weeks. Matching every Python TP=4 feature (pipeline, rotation, q4 head, logprobs) is another 2-3 weeks.
Treat these as rough; the engine has no GPU qualification at TP>2 at all.

## Main risks

- **No TP>2 qualification anywhere.** Jay's gates (greedy parity before drafting, drafted equal to plain,
  concurrent equal to solo, resumed equal to fresh) all ran at TP=2. Kernels that size grids from a per-rank width
  may assume even halves where no host test reaches them.
- **Uneven slices are new to Zig's kernels.** Hand-written CUDA (`x3ld`, `x3gm`, the epilogue tickets) and the AOT
  Triton variants see 640 and 512 expert widths and 32,384 and 32,256 head widths on different ranks. Each rank needs
  its own AOT fill (assets are already per rank).
- **NCCL-only until step 3.** A TP=4 Zig run on NCCL vs Python TP=4 on RoCE is not like-for-like on transport.
  Label it, or compare against Python with `GLM53_TF_COMM_BACKEND=nccl` as well.
- **The pack is different from Jay's.** His numbers are q28-v2. Ours would be the first Zig run on the uniform
  2.9 bpw layout. Python's 2.9 bpw path is qualified; Zig's `.stack` path exists but is not.
- **Memory.** Zig's prepared cache is another weight-sized copy on NVMe a node (`docs/ZIG-SERVE.md`). The planner
  floor only sees rank 0. Ranks 0/1 hold 25 % more expert columns than ranks 2/3.
- **Feature regressions vs live:** no logprobs (0006 is not live either), no `vllm:` metrics names (dashboards),
  no `/v1/model_info`, no prompt pipeline (cold 160k prompts will be slower than live Python until ported).

## Status on branch `zig-tp4`

Filled in at the end of the session; see the commit list on the branch.
