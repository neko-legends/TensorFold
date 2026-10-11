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
structurally.

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
  - `op_sess_prompt` and `op_sess_save` snapshots;
  - a window or keep on the line slot (defensive);
  - a short segment (16 rows or fewer) or an image segment of the line slot.

  Release, reset, restore and stop **abandon** the line. A long segment of another slot while a line is open runs
  tensor parallel; it does not open a second line, unlike Python, which drains and switches. That keeps every round on
  the active slot's views, and "long prefills 1 at a time" already holds that case off. Multi-slot piece runs
  (`pf.runnable`) leave the line slot out.
- **Fail-fast.** Any error inside a round or a drain aborts this rank's communicator (`comm.abort()`) before it
  returns, so every other rank's pending collective fails instead of hanging. Followers exit as for any prefill error,
  and the fate channel takes rank 0 down. No rank continues past a broken line.

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
- **AOT**: the stage programs add world-1 variants. aot-needs enumerates the rank's own stage at every row count and
  regime, and a host test checks that the boot's stage programs are a subset of that list.

## Hardware-window plan

The window plan, exact config and pass criteria are written up in the window plan (see the report and
docs/ZIG-TP4-PLAN.md's status section once the implementation lands).
