# Nemotron 3.5 Super VL: vision serving under load (2026-10-16)

The live server (four DGX Sparks, TP-4, build c3ede919, drafting on, 16 slots, 307,200-token window, 40 GiB prompt
cache + disk tier), read-only, one lock hold per run. Chart: [`../charts-20261016/nemotron-vision-serving-light.png`](../charts-20261016/nemotron-vision-serving-light.png).

## Load: 1 / 4 / 8 / 16 image requests at once

[`tools/nemotron/vision-serve.py`](../../tools/nemotron/vision-serve.py): 16 requests a level through N closed-loop
workers, 8 code-rendered bench images (a1_poster, a2_dashboard, a4_table_chart, k_a_spritesheet, k_h1_lowcontrast,
k_h2_busy, k_h3_wide, a3_shapes) cycled, one fixed prompt ("Transcribe every piece of text in this image exactly as
written, one item per line, in reading order. Then describe the image in two sentences."), greedy, thinking off, up
to 600 tokens, streamed. Text exact = ground-truth strings found verbatim (whitespace and case folded) in the reply.

**Cold** ([`load.json`](load.json)): every request's image differs from every other in one pixel's lowest red bit,
so neither the image cache nor the prompt cache can serve it (`cached_tokens` ≤ 7 on every request: the chat
template's first tokens). This is what a new image takes.

| concurrent | images/min | prefill (server, median) | TTFT median / worst | decode tok/s per stream | aggregate tok/s | MTP accepted | text exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 18.5 | 0.69 s | 1.9 / 2.7 s | 144.0 | 52 | 78% | 304/314 |
| 4 | 22.9 | 0.81 s | 3.8 / 7.8 s | 32.8 | 64 | 83% | 304/314 |
| 8 | 24.1 | 0.79 s | 6.4 / 13.7 s | 13.3 | 66 | 90% | 304/314 |
| 16 | 22.5 | 0.85 s | 25.3 / 31.1 s | 11.7 | 62 | 88% | 304/314 |

**Warm** ([`load-warm.json`](load-warm.json), `VISION_SERVE_WARM=1`): the same images unchanged, each sent once
before the levels, so every request is a follow-up on an image the server has already read (prompt cache hits).

| concurrent | images/min | prefill (server, median) | TTFT median / worst | decode tok/s per stream | aggregate tok/s | MTP accepted | text exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 45.1 | 0.05 s | 0.1 / 0.3 s | 137.2 | 124 | 76% | 304/314 |
| 4 | 67.5 | 0.11 s | 0.2 / 0.5 s | 54.2 | 185 | 82% | 304/314 |
| 8 | 68.9 | 0.10 s | 0.4 / 1.0 s | 25.9 | 189 | 89% | 304/314 |
| 16 | 69.8 | 0.10 s | 1.3 / 2.1 s | 19.1 | 192 | 86% | 304/314 |

What it shows:

- **New images are taken one at a time.** Cold throughput stays at ~23 images a minute (about 2.6 s an image)
  from 4 requests on, so at 16 at once the median wait for the first token is 25 s. On one stream the client waits
  1.9 s for the first token while the server's own prefill timer reads 0.69 s, so about 1.2 s an image is spent
  outside that timer (reading the picture in and encoding it). The flat throughput says this per-image work does
  not overlap across requests; where exactly it serializes is not measured here.
- **Images the server has seen are cheap:** ~70 a minute at 8-16 at once, 1.3 s median first token with all 16
  slots busy, ~190 tok/s aggregate (text-only serving measured 214-230 tok/s aggregate at 16 streams).
- **Accuracy does not move with load:** 304 of 314 strings at every level, cold and warm.
- MTP acceptance is higher under concurrency (78% → 88-90%; not investigated); per-stream speed falls as the
  streams share the GPUs.

## One stream: image prompts vs text

Same file, `single`: each of the 8 images once (cold) vs draft-speed.py's code and prose prompts (3 reps), 320-token
cap, greedy, thinking off.

| prompt | decode tok/s (median) | MTP drafts accepted |
| --- | ---: | ---: |
| image: transcribe + describe | **140** (93-165 by image) | 78% |
| prose: a 250-word description | 95 | 65% |
| code: a Python function | 185 | 86% |

Image replies decode **1.47× faster than prose**: a transcription is predictable text that the MTP head drafts
well (78% accepted vs 65%). Code is faster still. The slowest image runs are the shortest replies (31 and 48
tokens), where the fixed per-reply overhead dominates.

## Accuracy under concurrency: 16 structured reads at once

[`accuracy/`](accuracy/): window-vision.py's tiled read (the same harness and score.py as the published vision
cells) of a1_poster, a3_shapes, a4_table_chart and k_f_counts, with spark-vision pinned to one request per read
(`wv1.py`). One stream ([`single-tiles.json`](accuracy/single-tiles.json)) vs 16 reads at once, 4 pixel-distinct
copies of each image ([`make-copies.py`](accuracy/make-copies.py): low red bit flipped on a sparse grid, so no copy
reuses another's cached crops). The zoomed re-read (refine) is off on both sides: with it on a read is ~90 requests
and 16 cold reads at once would take about an hour at the ingest rate above. The refine-on single-stream read of
the same images today is in [`single-refine.json`](accuracy/single-refine.json) (text 70/70, a3 shapes 17/20).

| | one stream | 16 at once (4 copies each) |
| --- | ---: | ---: |
| text read exactly (a1 + a4) | 69/70 (98.6%) | 277/280 (98.9%): a1 27, 28, 27, 27 of 28; a4 42/42 ×4 |
| a3 shapes matched | 19/20 | 74/80: 19, 19, 19 and 17 of 20 |
| k_f_counts objects matched | 1/23 | 12, 12, 12 and 0 of 23 |
| hallucinated text | 0 | 0 |
| wall time | 4 reads in 139 s | 16 reads in 334 s |

Text accuracy holds under load. The a3 shapes cell moves by 2 on one copy of four (as it does between runs: 17/20
and 19/20 are both on record for this build); the object-count cell is unreliable at any concurrency (0-12 of 23)
and is not a load effect.
