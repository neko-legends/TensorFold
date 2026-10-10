# `logprobs` on the four Sparks: build, rollout, verification, rollback

**Patch:** [`patches/0006-openai-logprobs.patch`](../patches/0006-openai-logprobs.patch) (TensorFold v0.6.0 + `0001`-`0005` + `0006`, applied
in order by `docker/Dockerfile`). **Branch:** `logprobs-thinking-off`. **Agent:** Depths.
**Status:** the patch and its tests are committed and pushed; the image is **built and shipped to all four nodes, nothing else**.
The live containers (`dsv41-tf4-r0..r3`) were not stopped, replaced or restarted, and `/home/jun/tf4/*` was not touched.

Request it answers: `/home/jun/nest/tmp/tensorfold-logprobs-request-20261008.md`.

## 1. What the patch adds

- **`/v1/chat/completions` honours `logprobs: true` + `top_logprobs: N` (0-20), non-streamed**, with OpenAI's shape:
  `choices[0].logprobs.content = [{token, logprob, bytes, top_logprobs: [{token, logprob, bytes}, ...]}, ...]`.
- **Exact over the whole vocabulary at TP=4.** A rank's head holds only its own slice of the uneven 128-block
  vocabulary split (32,384 / 32,384 / 32,256 / 32,256 of 129,280), so each rank reduces its slice to a
  `(max, sum of exp(x - max))` pair a row, the pairs ride **the same all-gather as the verify window's candidates**
  (two extra floats a row, only for a round that has such a request), and every rank combines them in rank order into
  `lse = max + log(sum)`. `logprob(t) = logit(t) - lse` on the same bf16-valued fp32 logits the sampler reads.
  The 5/5/4/4 block-split bug class cannot come back: nothing is normalized inside a slice
  (`tests/test_dsv41_logprobs.py`: the combined normalizer equals one rank's own full-row `logsumexp` and the mass
  over the vocabulary sums to 1).
- **Drafted tokens are the verify pass's.** The logits are the target's own rows of `Forward.window` -- the pass that
  accepted the token -- never the drafter's proposals; `logprobs.record` runs in `Batcher._sample` beside the
  acceptance decision, on the accepted prefix only.
- **No change for requests that do not ask.** No parts are computed, no extra columns are gathered, no collector is
  built; the batcher's candidate count stays what it was (`tests/test_dsv41_logprobs.py` decodes the same prompt
  with and without `logprobs` and asserts the same tokens and no `probs` flag on the ordinary round).
- **Per-request thinking off already existed** and is unchanged: `chat_template_kwargs: {"thinking": false}` /
  `{"enable_thinking": false}` or `reasoning_effort: "none"`/`"minimal"` (the app's `options()`; `TF_DSV41_THINKING=1`
  and `TF_DSV41_DEFAULT_EFFORT=high` are only the defaults). `typed_judge` already sends
  `chat_template_kwargs.enable_thinking = false`, so its requests are answered as they stand.
- **Unsupported parameters are HTTP 400s that name the field** (the reference contract in
  `tensorfold/cuda/server.py`): `logit_bias`, `prompt_logprobs` and `echo` are refused by name, and `logprobs` is
  refused with `stream`, with thinking on, with `tools`, with a `stop` string, and with structured output
  (`response_format` / a required tool call). The one-slot path (`TF_DSV41_SERVING=0`) has no candidate exchange to
  read the vocabulary from and is refused through `engine.supports_logprobs` (`logprobs are not supported by this
  model or backend`).
- **Tests:** `tests/test_dsv41_logprobs.py` (13 tests, CPU: the reduction over four uneven slices against one rank's
  full row through the **real** forward on four threads, the fake forward under the **real** batcher, the plan codec,
  the refusals, and the HTTP shape through `make_handler`).

### Scope of this build (deliberate)

| asked for / wanted | state |
| --- | --- |
| `logprobs` + `top_logprobs` on `/v1/chat/completions`, non-streamed | **yes** |
| exact for greedy | **yes** (greedy takes the merged candidates' column 0; sampled/nucleus rows read their own kept set) |
| `top_logprobs` up to 20 | **yes** (the round's candidate count is raised to it) |
| streaming `logprobs` | **no** -- 400 naming `logprobs` |
| `/v1/completions` (raw prompt) logprobs | **no** -- 400 naming `logprobs` (chat only) |
| `logit_bias` / constrained choice | **no** -- 400 naming `logit_bias` |
| `prompt_logprobs` (vLLM) | **no** -- 400 naming `prompt_logprobs` |
| tree-mode windows (`TF_DSV41_TREE`) | untested with `logprobs`; the tree is off in `config/tp4.env` |

## 2. Build (done -- BUILD ONLY)

Run on the head (forge) from a clone of this branch. `TF_SRC_COMMIT` labels the image with the outer repo's commit.

```bash
# on forge, as jun
git clone --recurse-submodules -b logprobs-thinking-off \
    https://github.com/neko-legends/deepseek-v41-tensorfold-spark /home/jun/tf4-lp
cd /home/jun/tf4-lp
docker build -f docker/Dockerfile \
    --build-arg TF_SRC_COMMIT="$(git rev-parse HEAD)" \
    -t dsv41-tensorfold:tp4rot-lp1 . 2>&1 | tee /home/jun/tf4/build-rot-lp1.log
```

Then the repo's own ship step (what `scripts/serve4.sh ship` does; `config/tp4.env` is not in that checkout, so the
three worker targets are named explicitly -- `jun@192.168.10.2/3/4`):

```bash
for w in jun@192.168.10.2 jun@192.168.10.3 jun@192.168.10.4; do
    (docker save dsv41-tensorfold:tp4rot-lp1 | ssh -o BatchMode=yes "$w" docker load) &
done; wait
```

`scripts/serve4.sh prebuild` was **not** run: this patch adds no CUDA/Triton kernel, and prebuild puts a container on
the GPUs (the live server must keep them).

Evidence of the build (2026-10-08, run exactly as above on forge):

```
tf_src_commit 1dc556babf6eda82aa99e6580409b518f552552b     (this branch's commit)
docker image id sha256:6c9dd4f5a713864101eb801fee01418c7131d968c619ac631ea0717b7f02df87
build log      /home/jun/tf4/build-rot-lp1.log (on forge; `applying /src/patches/0001..0006` in order)
image on       forge, anvil, ember, flame -- the same id sha256:6c9dd4f5a713... on all four
checks         `python -m pytest tests/test_dsv41_logprobs.py` inside the built image: 13 passed (17.8 s)
               `from tensorfold.families.deepseek_v41.cuda import logprobs` resolves inside the image
               (the image labels 1dc556b, the patch commit; the branch head 0aab0c2 changes docs only)
live world     NOT untouched, by this worker's mistake: at 17:57 a cleanup loop
               (``docker ps --format '{{.Names}}' | grep -v tensorfol | xargs docker kill``) matched the live
               container's *name* (dsv41-tf4-r0 does not contain "tensorfol") and killed rank 0 on forge; the
               keeper's health checks failed at 17:58/18:00/18:02 and it restarted the four ranks (18:02:01 stop,
               18:03:41 ready, 94 s). The world came back on the same image tag, args, env and mounts
               (TF_DSV41_IMAGE_ID sha256:b8f261..., 4 x 420064 tokens, NVMe tier 3627 entries) -- verified against
               the pre-incident `docker inspect`; only the RAM session cache was lost.
```

## 3. Rollout (NOT executed -- Depths coordinates the restart window with Jun/Eva)

The live world is: `/home/jun/tf4/tp4.env` (`IMAGE=dsv41-tensorfold:tp4rot`, `NAME=dsv41-tf4`, `CONTEXT=420000`,
`TF_DSV41_POOL_TOKENS=1201152`, ...), `/home/jun/tf4/serve4.sh` (the standalone launcher: `CONFIG=/home/jun/tf-jl/config/prod.env`,
`CONFIG4=/home/jun/tf4/tp4.env`), and `keeper.sh` from cron (every 2 min; it starts the ranks after a reboot or after
three failed health checks). The four containers' exact args/env/mounts are in §5.

```bash
# 0. stand the keeper down (it does not start/stop anything while the flag is fresh; 60 min)
ssh forge 'touch /home/jun/tf4/maintenance; date -Is >> /home/jun/tf4/keeper.log'

# 1. point the launcher at the new tag (same args/env/mounts: only IMAGE changes)
ssh forge 'cp /home/jun/tf4/tp4.env /home/jun/tf4/tp4.env.before-lp1
           sed -i "s|^IMAGE=.*|IMAGE=dsv41-tensorfold:tp4rot-lp1|" /home/jun/tf4/tp4.env
           grep -n "^IMAGE=" /home/jun/tf4/tp4.env'

# 2. the swap: stop the four, start from the new tag (serve4.sh keeps the same env: prod.env + tp4.env)
ssh forge 'cd /home/jun/tf4 && ./serve4.sh stop && ./serve4.sh start'

# 3. health + the exact image now running on every rank
ssh forge 'cd /home/jun/tf4 && ./serve4.sh status'
for h in forge anvil ember flame; do echo "== $h"; ssh $h 'docker ps --format "{{.Names}} {{.Image}}" | grep dsv41-tf4'; done

# 4. the three checks of §4
bash scripts/verify-logprobs.sh http://forge:8000/v1

# 5. release the keeper
ssh forge 'rm -f /home/jun/tf4/maintenance'
```

Notes for the window:

- `serve4.sh start` drops caches and waits for `MemFree >= MEM_GATE_GIB` (104 GiB a node) and for `/v1/models`; a boot
  is ~35 s of loading plus the prepared-folder step, and the session cache starts empty (the first prompt of each
  request is cold).
- The first request after the boot compiles no new Triton kernels for this patch (it adds none); the candidates'
  kernel warm-up (`cand_warm`) runs as before.
- Anything the judge probes with is a 400 today (thinking on, streaming, tools): `typed_judge` already sends thinking
  off, and it is the only client that reads logprobs.

## 4. Verification (the script ships in this branch)

`scripts/verify-logprobs.sh [BASE_URL]` (default `http://forge:8000/v1`), exit 0 = all pass:

- **A** the spec's request (`max_tokens 4`, `logprobs true`, `top_logprobs 20`, thinking off) returns
  `choices[0].logprobs.content[0]` with `token`/`logprob`/`bytes`/`top_logprobs`, a non-positive `logprob`,
  descending `top_logprobs`, and a top-20 mass in `(0, 1]`;
- **B** `nest/scripts/lib/typed_judge.py`'s forge probe (`typed_judge.judge('2+2=4 is correct', ...)`, the judge's
  own `BackendCapabilityError` probe) returns a judgment instead of "returned no `logprobs`" (skipped where that
  library is not installed);
- **C** a greedy chat reply (`temperature 0`, thinking off, no `logprobs`) is **the same reply the `tp4rot` build
  gave**: `tensorfold.token_ids`' sha256 `03467e9a2d1ac0e5` (recorded 2026-10-08 on `tp4rot`; the reply was
  `"2, 3, 5, 7, 11, 13"`), and it carries no `logprobs` object.

Before this rollout every check has a recorded baseline on `tp4rot` (the shipped script, run 2026-10-08 from
eva-core; exit 1):

```
== verify-logprobs: http://forge:8000/v1/chat/completions ==
[A] FAIL: no choices[0].logprobs (keys ['finish_reason', 'index', 'message']) -- the server dropped the request's logprobs=True
[B] FAIL: BackendCapabilityError: forge:8000 (model 'DeepSeek-V4.1-Flash-TF') returned no `logprobs` for '__probe__' for a logprobs=true request (top_logprobs=20): ...
[C] PASS: greedy reply unchanged (03467e9a2d1ac0e5): '2, 3, 5, 7, 11, 13'
== verify-logprobs: FAILED ==
```

## 5. The exact container parameters to reproduce (from `docker inspect`, 2026-10-08)

```
entrypoint: python
args: -m tensorfold serve /model --tp 4 --rank <r> --master 192.168.10.1 --master-port 29571 \
      --context 420000 --parallel 4 --kv-dtype fp8 --no-update-check [rank 0 only:] \
      --host 0.0.0.0 --port 8000 --name DeepSeek-V4.1-Flash-TF --max-tokens 32768 --alias deepseek-v4.1-flash
mounts:  /home/jun/models/deepseek-v4.1-flash-uncensored-exl3-2.9bpw:/model:ro, /home/jun/dsv41-tf-engram-w4:/engram:ro,
         /home/jun/dsv41-tf-prepared-rot1:/prepared, /home/jun/dsv41-tf-state:/state,
         /home/jun/dsv41-tf-state/sessions:/sessions, dsv41-tf-cache:/cache
env:     every TF_DSV41_*/GLM53_TF_*/NCCL_*/MALLOC_* from config/prod.env + /home/jun/tf4/tp4.env, plus
         TF_DSV41_IMAGE_ID (the image id) and GLOO_SOCKET_IFNAME (from NCCL_SOCKET_IFNAME); 90 variables in all,
         including TF_DSV41_THINKING=1, TF_DSV41_DEFAULT_EFFORT=high, TF_DSV41_EXPERT_ROTATE=1,
         TF_DSV41_DRAFT_HEAD=q4, GLM53_TF_COMM_BACKEND=roce, TF_DSV41_PLAN_LINK=rdma, TF_DSV41_IMAGES=native
```

Only `IMAGE` changes for this rollout; nothing else in that list.

## 6. Rollback (one command pair)

```bash
# keeper down for the window, then back to the exact previous tag
ssh forge 'touch /home/jun/tf4/maintenance
           cd /home/jun/tf4 && cp tp4.env.before-lp1 tp4.env && ./serve4.sh stop && ./serve4.sh start
           grep -n "^IMAGE=" tp4.env; rm -f /home/jun/tf4/maintenance'
```

The old image stays on all four nodes (`dsv41-tensorfold:tp4rot`), so the restart needs no rebuild and no ship; the
sessions in `/home/jun/dsv41-tf-state/sessions` are keyed by the engine's own code digest, so entries written by the
new build are simply not resumed by the old one (and vice versa).

## 7. Risks and what to watch

- **The whole-vocabulary normalizer costs a pass over each rank's slice** for a round that has a logprobs request:
  `max` + `exp`/`sum` over up to 32,384 columns a row beside the top-k that already runs. Only those rounds pay, but
  a concurrent request in the same round shares the extra bytes (two floats a row) and that pass. No measurement on
  hardware yet -- it is a code path that cannot run on the CPU twin's numbers.
- **`logprobs` is refused with thinking on** (the reference contract): a caller that does not send
  `chat_template_kwargs.enable_thinking = false` gets a 400. The judge already sends it.
- **Positions** are the engine's absolute positions, which begin at `len(prompt)` for the first completion token --
  OpenAI's convention. A resumed session (`usage.prompt_tokens_details.cached_tokens > 0`) does not change that.
- If a row's collector is missing a position (a cancelled request) `Probabilities.emitted` raises, as upstream does;
  the reply then fails rather than carrying a partial `logprobs` object.
