# Pipelined prompt reading on the Zig TP=4 engine (`TF_DSV41_PREFILL_PIPE`)

Status: design and host-side implementation on `zig-tp4`, **off by default, not run on a GPU**. The hardware window
plan is in the last section. Paths are relative to `engine/zig/zig/src/families/deepseek_v41/` unless they start with
`docs/`, `config/`, `patches/` or `fabric/`.

## Why

At TP=4 the Zig engine beats live Python on every decode cell, but it reads long prompts 32-42 % slower:
~2,540-2,760 against ~4,050-4,350 prompt tok/s, and a 154K-token TTFT of ~61 s against ~35.5 s (window 4b,
2026-10-10). Python's number comes from its **pipelined** prefill (`TF_DSV41_PREFILL_PIPE=1024` in the live
`tp4.env`). Zig already beats pipe-less Python TP=4 (~1,900-2,150 tok/s). Zig's TP=4 prefill is no faster than its
TP=2 prefill: the replicated work (the indexer's 32 heads and its top-512 selection, the compressor and the single KV
head, the mHC streams) and one all-gather per sublayer do not shrink with more ranks.

## What Python's pipeline does (patches/0003 `pipe.py`, docs/PREFILL_SPEED.md)

- Each rank also loads a quarter of CED's encoder layers at **full width**: stages `[0,5) [5,10) [10,15) [15,21)`.
  The last stage ends at layer 20's attention site and compressor, as the encoder pass does. This costs about +27 GB a
  rank.
- A long prompt's segments (2,048 rows each) flow through the ranks like an assembly line. In round t, rank s runs
  segment t - s through its layers at world 1, with no collective inside the stage.
- Each round ends in one exchange:
  - point to point (NCCL p2p), to the next stage: the streams before the stage's last FFN post, that pending post (its
    gathered partial and mHC coefficients), and the latest index source's selection;
  - an all-gather to every rank: the compressed rows and index keys the round wrote for the kv sources, and the last
    stage's CED stash rows.
- Engram rows are gathered per round: each rank holds a quarter of the hash heads.
- The line stays open across prompt pieces. `finish_prompt` and snapshots drain it: W - 1 empty rounds, then every
  stage's last-window SWA rows and ratio-2 carries go to every rank. Reset and restore abandon it.
- Measured at 160K: stages take 366 / 383 / 401 / 341 ms, Engram rows ~33 ms and the exchange 20-45 ms per round. The
  whole prompt takes 38.8-39.6 s against 82.5-84.8 s for TP, about 2.1x. A 20K prompt goes from 10.5 to 5.8-6.8 s.
- Numerics: a stage computes at world 1, so a pipelined prompt carries one rank's bits rather than the TP=4 bits.
  Per-layer stream differences run from 0.4 % at layer 0 to 26 % at layer 19, the same curve as fast vs exact prefill.
  Sessions get their own tag (`tag_grid + 2`).

## What the Zig port does

Same protocol, so the behaviour and numerics questions are the ones Python already answered. It is mapped onto Zig's
emitter, runner and op stream, with five changes, each cheaper than Python's version.

### 1. Stages are emitted programs at world 1

`block_prefill.emitStage(o1, layers[lo..hi), n, start, entry, exit)` emits a stage the same way the encoder pass is
emitted, at `o1` = the forward's options with `world = 1, rank = 0`. That gives 64 heads, 8 `wo_a` groups and 2,304
expert columns. Entry and exit:

- **entry**: glue `pipe_in` loads the stage's input into its roles. That is the streams (stage 0: the embedded rows),
  and for later stages also the pending post's gathered partial (`L.recv.moe`, `[1, n, D]`) with its coefficient set,
  plus the incoming selection (`w.sel<src>`, `w.cnt<src>`) when the stage's first layers reuse an earlier source's. A
  stage after the first enters its first layer through a boundary site, which applies the pending post exactly as the
  next block applies it in one process. If that layer is Engram, the order is post-only, Engram, then site2, as in
  `Seg.run`.
- **exit**: a stage before the last stops after its last layer's MoE partial: glue `pipe_out` names the streams, the
  partial, the coefficient set and the latest selection. The last stage ends with `encoderEnd(20)`: layer 20's
  compressor, then `ced_stash`.

**Proof obligation, as a host test:** the stage programs, concatenated, equal the world-1 encoder program
`emitReplay(o1, layers 0..20, .encoder)` call for call. Only the `pipe_in` / `pipe_out` glue and the first stage's
embed are different. So the pipelined arithmetic is "one rank computing the encoder", Python's claim, checked
structurally, at the served boot's options and at sampled segments (not every row count and start): see "What the
tests prove" below.

### 2. Only the sharded tensors are loaded twice

Zig names weights without the world (`L5.attn.wq_b`), so the stage copy gets a `P.` prefix (`P.L5.attn.wq_b`).
**Only the TP-sharded tensors are copied:**

- `wq_b`, `wo_a.*`, `wo_b`, `attn.sink`;
- the routed experts `moe.w{1,2,3}`;
- the shared expert `moe.shared.0.w{1,2,3}`.

The stage reads every replicated tensor from its existing TP copy: `wq_a`, `wkv`, the compressor and indexer
projections, the norms, mHC, the router and the whole Engram projection (157M parameters).

The renaming is a pass over the stage program (`pipe.rename`). It covers:

- weight roles;
- the run-time forms of those weights (`s.<w>.lanes`, `s.<w>.T`);
- the expert tables (`s.L<i>.ex.*`);
- x3gm's pointer tables (`s.L<i>.gm.t{g,u,d}`).

A host test compares every weight and form view of the stage program with the same name in the TP=4 program of the
same layers. Every name whose shape differs must be renamed, and every renamed name must be a sharded tensor.

Layer 20's encoder part uses only replicated tensors, so the last stage loads no copy for it. Cost a rank (stage of
five encoder layers): ~4.6 GiB of routed experts plus ~0.1 GiB of attention and shared expert, a layer → **~23 GiB**.
That is a little below Python's ~27 GB, because Python reloads the replicated parts too.

### 3. The stage's scratch is its own namespace

The stage programs run with `ownCalls(.., "pp~")`, the mechanism that already gives TF_DSV41_PF_4K and PF_TBO their
own window, layer and scratch roles while sharing the slot's state: SWA rings, carries, the paged pool's comp rows and
index keys, and the CED stash. A stage therefore writes its layers' state straight into the slot's own stores, as
Python's `Stage.bind` does. Decode windows and TP prefills of other slots never touch the stage's roles, so the line
survives between rounds.

The namespace is sized at boot and priced in `kv_state.bootCheck` as resident.

### 4. Rounds, exchanges and drain (`pipe_run.zig`)

Every rank runs the same rounds from the same op stream. The leader sends the prompt pieces, and every follower runs
`prompt()` on the same ids.

- **Entry.** A segment enters the line in `forward_prefill.prompt` when all of these hold:
  - replay mode, KV not split, no image rows;
  - at least 17 rows;
  - either this slot's line is open, or the segment has at least `TF_DSV41_PREFILL_PIPE` rows and no line is open.

  At least 17 rows is the minimum in principle. The implementation asks for `min_seg` = 33, past the mHC split
  programs' 32. A stage then always enters at streams buffer 0 and coefficient set 0, exactly where the one-process
  encoder pass is at that layer.

  Its host state advances at entry: the position, the Engram tail, the stash ids and the session commit. Its pages are
  reserved on every rank. Device state completes at drain.
- **Round t:**
  1. The new segment's embedding: the TP glue's vocabulary-row gather, on every rank, landing as stage 0's input.
     This saves Python's full-width embedding copy (1.3 GB on rank 0).
  2. Each Engram layer's rows for the segment its owner runs this round: every rank's head columns, gathered, as in
     Python. The rows of round t + 1 are read ahead on the host during round t through the engine's Engram prefetch.
  3. Rank s runs its stage on segment t - s, if any, then copies its ratio-2 sources' carries (the stage-local commit).
  4. The exchange:
     - p2p from s to s + 1 of the stage output (packed: streams, partial, coefficients, selection), one grouped NCCL
       send/recv (new `Collective.sendRecv`);
     - one all-gather of the comp rows and index keys each stage's kv sources wrote. Python's rows-of rule applies:
       `max(0, start/r - 1) .. (start+n)/r`, one copy per page run through the slot's host page table.
- **Drain.** Rounds with no new segment until the line is empty. Then one all-gather of each stage's layers' SWA
  rings, the ratio-2 carries and the **CED stash ring**. The stash is sent once here and not every round as in Python;
  only the decoder replay reads it, and it holds the last 127 positions.
- **Drain points**, run on every rank at the same op:
  - the line slot's CED finish (`op_ced_finish`, `op_ced_multi`);
  - `op_sess_prompt` and `op_sess_save` snapshots, right after the op is sent;
  - a segment of the line slot under `min_seg` rows, or one with image rows.

  The line slot's windows only come after its CED finish (the verify tail), which drains it first.
  Release **abandons** the line. A restore needs an empty slot, so a release precedes it. A slot found at another
  position than its line's next is abandoned by the route. A long segment of another slot while a line is open runs
  tensor parallel; it does not open a second line, unlike Python, which drains and switches. That keeps every round on
  the active slot's views, and "long prefills 1 at a time" already holds that case off. Multi-slot piece runs
  (`pf.runnable`) leave the line slot out.
- **Fail-fast.** Any error from a segment's entry to the end of a drain goes through the tp session's fail-fast
  (`tp.Session.fatal`): a FAIL frame on the fate channel, every rank's communicator aborted by its own hook, exit 70 on
  every rank.
  - A local `comm.abort()` would not wake a peer already waiting in an NCCL kernel.
  - A drain inside a session op would otherwise be a follower's request-level error.

  No rank continues, and none waits, past a broken line.
- **The end of the line.** At boot the ranks agree, in one all-gather, on the end up to which every rank's stage emits a
  segment (`pipe.lastEnd`). The route keeps rows past it tensor parallel, so no rank meets a segment its own stage
  refuses. At the served config every stage emits up to the context's end, so the agreed end is the limit.

### 5. What is better than Python, and what is not

| | Python | Zig port |
| --- | --- | --- |
| extra weights a rank | whole layers (~27 GB) | sharded tensors only (~23 GiB) |
| stage 0 embedding | full-width copy (1.3 GB) | the TP gather each round |
| CED stash | all-gathered every round (~6.5 MB a rank) | once, at drain |
| Engram rows (~33 ms a round in Python) | read in the round | host reads issued a round ahead |
| stage compute | Python's world-1 kernels | the Zig kernels (PF_DENSE fused, x3gm v2, INDEX_BOUND, ...) at world 1 |

Not done here, and listed for later: `TF_DSV41_INDEX_SPLIT` and `TF_DSV41_PREFILL_OVERLAP` for short prompts (their
ceiling is ~10 % each, below the line's 2x), overlapping the exchange with the next round's compute on a comm stream,
and a total-length opening rule (it needs a leader hint in the op stream).

`TF_DSV41_PIPE_ROWS=4096` is a knob, with stages at 4K rows and x3gm one block. It halves the expert-weight traffic per
token, which is ~23 GB a stage-round at 2K rows. Python measured 4K as a tie at 160K and a loss at 20K. The window
should A/B it at 131K.

## Expected gain, and its basis

Python's round at 160K is ~470 ms: the slowest stage 401 ms, Engram ~33 ms and the exchange ~20-45 ms. The rounds of
a prompt are its segments + 3. Python's 38K and 154K cells fit that model (22 rounds in 9.3 s, 79 rounds in 35.5 s).

For Zig:
- a round = the slowest stage × r + Engram + exchange, where r is Zig's stage compute over Python's;
- Engram is ~3 ms, because the rows are read ahead;
- the exchange is ~10 ms: ~110 MB of p2p at ~20 GB/s plus ~3 MB of rows, with no stash.

The basis for r: Zig's prefill is ~1.3x Python's at TP=2, 2,500-2,686 against 1,833-2,068 tok/s, where collectives are
cheap. So r ≈ 0.8. A conservative estimate takes r = 1.

| cell | Python pipelined | Zig today | Zig piped, r = 1 | Zig piped, r = 0.8 |
| --- | ---: | ---: | ---: | ---: |
| long 32,768 (37,986 tokens, 19 segments) | 4,084 tok/s | 2,761 | ~4,300 | ~5,000 |
| long 131,072 (154,195 tokens, 76 segments) | 4,354 tok/s | 2,554 | ~4,500 | ~5,400 |

TTFT at 154K: ~34 s with r = 1, ~28.5 s with r = 0.8, against Python's 35.5 s.

**Decode is untouched.** Decode windows, graphs, row mode and the drafter use neither the stage weights nor the `pp~`
roles. The only decode-visible change is memory: ~23 GiB of weights plus the stage namespace (~1-2 GiB) come out of
MemAvailable. With 55 GiB of weights on ranks 0/1 and 45 GiB on ranks 2/3, the boot check and the graph floor have to
clear it. The window's first check is that boot and the decode cells' numbers.

## Output parity

- With the knob **off** (the default), nothing changes: no stage weights, no roles, no new op.
- With it **on**, a prompt's encoder rows carry world-1 bits, as in Python. The session tag gets the pipe bit
  (`ced.tag_pipe`), so pipelined and tensor-parallel prompt snapshots never resume each other. With the knob off, the
  session semantics are unchanged.
- A side effect worth measuring: Zig's world-1 kernels are the same ports that matched Python bit for bit at TP=2. A
  Zig pipelined prompt may therefore reproduce Python's pipelined long-cell hashes, which today are 0/6 equal.
- Short prompts, below the knob's rows, stay on the TP path. Decode bits are unchanged.

## Risks

- **Nothing here has run on a GPU.** New on hardware: the BMQ-32 `_fused` and other world-1 Triton variants on a TP=4
  rank (enumerated per rank by aot-needs, below), NCCL p2p connections (their buffers are made lazily at the first
  pipelined prompt), and the drain's state hand-off.
- **Memory**: +~25 GiB a rank. If ranks 0/1 do not fit, the bounds knob moves layers to ranks 2/3
  (`TF_DSV41_PREFILL_PIPE_BOUNDS=0,4,9,14,21`).
- **AOT**: the stage programs add world-1 variants. aot-needs enumerates the rank's own stage at every row count at
  each regime's first start, plus the boot's planned programs. A host test checks the planned programs and sampled
  segments against the regimes alone (see "What the tests prove"); any (rows, start) beyond those is covered by the
  regime argument, not by a test.

## What the tests prove, and what they sample

The equivalence (`pipe_emit_test.zig`): the four stages' programs minus their hand-off glue equal the one-rank
encoder pass minus its embedding, call for call. Every call's name, kind, grid, stream flags and every argument
(tensor role, dtype, shape, stride, offset) must match.

- **Options.** Proven at the served boot's options (`servedOptions`). These are every emitter switch
  `config/tp4-zig.env.example` puts on the forward through `prod_knobs.apply` / `branches.install`: R1, BRANCHES,
  MHC_DEFER with MHC_DEFER_AT=exchange, GM_V2, EXPERT_TOPP 0.85, INDEX_BUDGET 64 MiB, INDEX_BOUND, MHC_PFDEC=17,
  KV_NORM_STORE, ROUTER_GROUP_ROT, PF_DENSE=fused, and DRAFTS' taps, which `stageOptions` clears. The bare pipeline
  set (`bootOptions`) is checked too.
  - PF_DENSE (the fused dense GEMMs) is the only served switch that changes a stage program. The test asserts this
    (served ≠ bare at 2,048 rows on every stage). The bare set, which the first version of the test used, therefore
    never ran the served dense path.
  - INDEX_BOUND, MHC_PFDEC, KV_NORM_STORE, ROUTER_GROUP_ROT and MHC_DEFER_AT emit no stage call of their own at
    33-2,048 rows (probed 2026-10-11). They are decode-window switches.
  - PF_DENSE is checked with its default table. The served `/cache/pfdense-table.json` picks other tile
    configurations for the same calls, on both sides of the comparison.
- **Segments, 2,048-row line.**
  - Served options: rows {2048, 2047, 1500, 1000, 100, 64, 34, 33} × starts {0, 16, 4112, 154176, 300000, the line's
    last start limit − n}.
  - Bare set: rows {2048, 1000, 33} × starts {0, 300000}.
- **Segments, 4,096-row line** (`TF_DSV41_PIPE_ROWS=4096`). Call for call, both option sets, rows {4096, 2049, 33} ×
  starts {0, 8192, 300000, limit − n}.
- **Sampled, not swept.** Other row counts in 34..4095 and other starts are not compared. A full sweep is about 4,000 row
  counts × 205 starts of the line's rows × 5 programs, which is not a unit test. The samples
  cover the regime edges the emitter branches on: ≤ 64 rows (the decode-sized paths), ragged and odd counts, a
  16-grid start, the first page, and the last start.
- **Separately pinned.** The hand-off tensors (byte-equal, `HandoffLayout`). The renamed program's views against
  TP=4's, on every rank. The equivalence is on the pre-rename program; the two are not diffed end to end.

The AOT coverage (`pipe_aot_test.zig`): on every rank at CONTEXT=420096, every Triton launch of the boot's planned
programs and of sampled segments reduces to a variant in `aot_needs.enumerateRegimes`' list.

- **Not circular.** That list is the regime sweep without the planned loop. Neither leg is in it by construction:
  the planned 33-row program at 420,063 is not on the 16-row grid, and it is covered by the regimes.
  - The fill itself (`enumerateStage`) lists the planned programs too.
  - A mutation that drops the regimes' row sweep fails the test (`build/pipe-proofs-red2.log`: `_site` at 33 rows).
- **The plan is frozen.** `pipe.planned` at this config is frozen in the test, per rank: (2048, 0), (2048, 418048),
  (33, 0), (33, 420063), and on ranks 0-2 the selection's worst segments (1547, 5684) and (1366, 6825). A change to
  the plan or to `lastEnd` fails the test until the constant is updated deliberately.
- **Sampled.** Rows {2048, 2047, 1500, 1000, 100, 64, 34, 33} × starts {0, 16, 2048, 4112, 37984, 154176, 300000,
  417984, limit − n}. "Any segment at any start" is the regime argument: a start's variant set depends only on the
  regime the dedup sweep finds. It is not a test.
- **Not tested.**
  - A shorter context: a 420096 fill holds every start a shorter one has, by the sweep's design.
  - `lastEnd`'s assumption that a stage's emission limit is monotone in the rows. It probes only 2,048 and 33 rows,
    which hold at the served config: `lastEnd` = limit on every rank, tested.

The buffer plan (`pipe_emit_test.zig`, window 6). A stage program runs in its own namespace ("pp~"), so the boot's
buffer plan holds its window, layer and scratch roles only from the stage programs `pipe.planned` lists. The first
version listed a segment at its first and last starts, the forward's rule (`forward_prefill.plan`). That rule holds
for the forward only together with its other planned programs. On the stages the index selection's scratch is not
monotone in the start: past 2,730 keys a 2,048-row segment's materialised scores go to row blocks, each naming
"L.ix.pos<j>", and past 4,096 keys they stream. So the first pipelined prompt's third segment (rows 4,096-6,143 on
rank 0) named "L.pp~ix.pos0", which no planned program named: `the line failed (Unbound)` (Runner.tensorAddr). The
second segment had already overrun "L.pp~ix.scores" (16 MiB into an 8 MiB slot of the layer arena).

- **The fix.** `pipe.planned` adds the segments where the selection's start-dependent scratch is largest
  (`block_prefill.selectWorst`, exact over every (rows, start): each key count's last start, each key split's for the
  stream, through `selectScratch`, which takes select's own decisions). A stage that selects at the candidate source
  or with a top-k off the powers of two is refused at boot (not the model).
- **Tested.** Every stage-owned role ("pp~", "s.P.") of a stage at sampled segments is in the plan at least as large:
  2,048- and 4,096-row lines, every rank, rows {R, R − 1, 1547, 1366, 1365, 1024, 100, 64, 33} × starts every 512
  through 16,384, the regime edges, a key split's ends, 300,000 and the last start, plus 40 random segments a rank.
  It fails on e62f70a0 (`build/pipe-plan-red.log`). Sampled, not swept: the sweep is `selectWorst`'s argument.
- **The named lookups.** Every role `pipe_run.zig` looks up by name (the commit's "w.pp~L<i>.proj" and carry, the
  pool's families, the SWA rings, the stash) is in the stage plan, and no stage program names a bare
  "w.L<i>.proj" (the forward's own rows).

## Implementation map

| piece | where | host test |
| --- | --- | --- |
| knobs, stages, schedule, rows / hand-off / drain layouts, a call's route, page runs, stage options, the P. renaming | `pipe.zig`, `pipe_names.zig` | `pipe_test.zig` |
| a stage as a world-1 encoder program (`pipe_in` / `pipe_out`, the entered first block) | `block_prefill.emitStage` | `pipe_emit_test.zig`: the four stages are the one-rank encoder pass call for call, at the served options, 2,048- and 4,096-row lines, sampled segments |
| the AOT fill's stage variants | `aot_needs.enumerateStage` (`enumerateRegimes` + the planned programs) | `pipe_aot_test.zig`: at CONTEXT=420096 on every rank, the frozen planned programs and sampled segments ⊆ the regimes, through the served Triton signatures |
| the stage copies | `pipe_load.zig`, `prepared.zig` (key variant) | `host_test.zig` (a synthetic pack at world 1), `prepared.zig` |
| the shift | `Collective.sendRecv` (NCCL group) | `fabric/tp/host_test.zig` |
| rounds, exchange, drain, fail-fast | `pipe_run.zig` | device code; the route / layouts above are its decisions, and every role it binds or looks up is in the stage plan (`pipe_emit_test.zig`) |
| boot | `model.zig`; refusals in `pipe_run.check` | compile only |

`aot-needs` with `config/tp4-zig.env.example` and `TF_DSV41_PREFILL_PIPE=1024` on this host gives these counts. Every
rank's list is a superset of the window-4b fill (136 variants, none dropped), so the same assets also boot with the
pipeline off.

| rank | variants | new for the stages |
| ---: | ---: | --- |
| 0 | 163 | +27 (14 `_fused`, 6 `_gemm`, 6 `_site`, 1 `_kv_store`) |
| 1 | 155 | +19 |
| 2 | 159 | +23 |
| 3 | 155 | +19 |

## Hardware-window plan (for Depths; nothing here was run)

**Before the window (build host, no GPU)**

1. Build the image from `zig-tp4` HEAD.
2. Regenerate the per-rank AOT fill with the pipeline knob in the config that `scripts/zig-tp4-assets.sh aot <r>` reads.
   Add `TF_DSV41_PREFILL_PIPE=1024` to the copied `config/tp4-zig.env`, then run
   `for r in 0 1 2 3; do scripts/zig-tp4-assets.sh aot $r /srv/zigtp4-assets/rank-$r; done`.
   - Expect 163 / 155 / 159 / 155 variants in each rank's `needs.json`.
   - The new kernels are world-1 `_fused` (BMQ 32), `_gemm`, `_site` and `_kv_store` variants, compiled with Triton 3.7.1
     as before.
3. Leave `TF_DSV41_PIPE_ROWS` unset (2,048). 4,096 needs its own fill (stages at x3gm one block); see A/B 3 below.

**In the window.** Same image and assets for both boots. Use the RoCE transport if its check passes, else NCCL; pair it
against the Python baseline on the same transport.

1. **Control boot, pipeline off** (`TF_DSV41_PREFILL_PIPE` unset). Run `bench/zig_vs_python.py run` with a fresh salt
   tag. Python's baselines from window 4b stand: same pack, cells and salt rules.
2. **Pipeline boot**: `TF_DSV41_PREFILL_PIPE=1024` and `TF_DSV41_PIPE_TRACE=1` in every rank's `config/tp4-zig.env`.
   - The first boot writes the stage copies' prepared files: a new key folder a rank, ~23 GiB of NVMe, a few minutes.
   - Grep the boot logs for:
     - `pipeline: rank r's stage copies (layers ...): N GiB`, about 23 GiB each;
     - `pipeline: prompts of 1024+ rows through stages [0, 5, 10, 15, 21]`;
     - the boot check's MemAvailable lines on all four ranks;
     - no `MissingTritonVariant`, and no `refused` beyond today's x3gm variants.
3. **Gates on the pipeline boot**:
   - 17 × 23 = 391;
   - `bench/structured.py --suites schemas` (drafted == plain, concurrent == solo);
   - the code word at 30 / 60 / 85 % of 20k / 80k / 158k (Python's needle check: 3/3);
   - one long salted prompt re-sent after a short wait (a session hit of a pipelined prompt) must reply the same.
4. **Bench**: the same `zig_vs_python.py run` with a fresh salt tag. The cells that matter are `long 32768` and
   `long 131072` (prompt tok/s, TTFT), every decode cell, `mix` and `mixwin`. Keep every rank's log, including every
   `pipeline rank r: S segments, R rounds; …` trace line and the request log's prompt tokens. Together they are the
   "line ran" gate below.
5. **Optional A/B, time permitting**:
   1. `TF_DSV41_PREFILL_PIPE_BOUNDS`. If the trace shows one stage slower than the others by more than 10 %, move a layer
      away from it (Python's slowest stage held layer 14). If ranks 0/1 run short of memory, try `0,4,9,14,21`.
   2. `TF_DSV41_PREFILL_PIPE=8192`. Prompts under ~8K stay tensor parallel: a short line's fill and drain cost more than
      it gains.
   3. `TF_DSV41_PIPE_ROWS=4096`, with its own AOT fill: `aot-needs` under that knob.

**Pass criteria.** Two verdicts, judged separately. **NOT RUN**: any "the line ran" gate fails. The pipeline did not
execute, and the tok/s cells say nothing about it. **NO GAIN**: the line ran, but a prefill gate fails. Judge "the line
ran" first; a NOT RUN boot is not scored on gain.

- **The line ran** (any miss is NOT RUN). These are read from every rank's log, so they need `TF_DSV41_PIPE_TRACE=1`
  on the pipeline boot.
  - Control boot: no line containing `pipeline` on any rank. This proves the control really ran with the pipeline off.
  - Pipeline boot, at boot:
    - each rank r = 0..3: `pipeline: rank r's stage copies (layers a-b at world 1, sharded tensors only): X GiB`,
      X about 23 on every rank (5 layers each: layer 20 is capped out of rank 3's copies);
    - rank 0: `pipeline: prompts of 1024+ rows through stages … (2048 rows a segment)`;
    - the boot check's MemAvailable about 23 GiB a rank below the control boot's (the copies load before the check;
      the roles and buffers are priced in it).
  - Per cold long prompt. Cold means salted, with no session or lookup hit; a warm repeat may legitimately read under
    1,024 new rows and never open the line. Each rank logs, between the request's arrival and its first token:
    - first line: `pipeline: rank r: slot s's prompt from P through the stages … (line k)`. It is printed for lines
      1-4 and every 64th, so judge it on the first long prompts only;
    - then one or more trace lines `pipeline rank r: S segments, R rounds; a round: pre … ms, stage X ms, exchange …
      ms; drain … ms`.
  - Each trace line must have R = S + 3 and X > 0, and all four ranks must print the same S and R for it.
  - For a prompt of T tokens (the request log's prompt tokens), the line reads the T − 1 prefill rows. The expected
    segments are E = ⌊(T − 1) / 2048⌋, plus one if (T − 1) mod 2048 ≥ 33 (a shorter tail runs tensor parallel).
    - `long 32768` (T = 32,768): S = 16, R = 19.
    - `long 131072` (T = 131,072): S = 64, R = 67.
    - Recompute E from the logged T if the template or salt changes it.
    - A snapshot drain inside the prompt splits it into several trace lines. Their S must then sum to at least
      E − (lines − 1).
    - A trace line right after a `… line abandoned …` line also counts the abandoned segments; that is still R = S + 3.
  - Fail as NOT RUN if a cold long prompt has no trace line on some rank, or ΣS falls short. Fail it too if any
    `pipeline: rank r: segments end by …` line shows an agreed end below the context; then rows past it ran tensor
    parallel.
- Boot: all four ranks up, the boot check passes on every rank, no Triton variant missing, and the transport line names
  the intended backend.
- Prefill: `long 32768` ≥ 4,084 prompt tok/s and `long 131072` ≥ 4,354 (Python pipelined, RoCE; NCCL pair: 4,084 /
  4,334). TTFT at 154K ≤ 35.5 s.
- No decode regression against the control boot: every decode and aggregate cell at ≥ −2 % (code / prose, T0 / T0.7,
  x1 / x4, long decode, mix, mixwin 214.76 on RoCE). Expected: unchanged, because decode never runs the line's code. The
  memory taken does show up in the boot check, the graph floor and MemAvailable at the prefill → decode hand-off.
- Correctness: 391, structured PASS, needle 3/3, no fleet exits. Long-cell hashes against Python's pipelined ones are
  recorded but are not a gate (the parity section above).
- Rollback on any hang or regression: stop the ranks, boot again without the knob (same image and assets), then the
  usual live restore.
