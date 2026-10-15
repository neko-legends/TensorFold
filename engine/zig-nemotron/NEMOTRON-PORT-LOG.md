# Nemotron 3.5 Super VL 120B-A12B (NVFP4) on TensorFold — port log (branch nemotron-supervl-120b-20261015)

Base: upstream TensorFold v1.0.11 vendored here (VENDORED.md). Target: `NemotronH_Omni_Reasoning_V3` = 88 blocks
(40 Mamba-2 + 40 latent-MoE + 8 attention), 512 routed experts top-22, hidden 4096, MTP head, C-RADIO vision.

## Done (M1 numerics, verified on flame 2026-10-15, no swap window needed)
- Config: wrapper `nemotron_h_omni` (llm_config; quantization from the wrapper; `modelopt_mixed`; `{"__float__": "Infinity"}`;
  8.6 MB config.json; eos from generation_config.json); `latent` (moe_latent_size), `checkCudaShapes` (Lightning or Super VL),
  max_layers 96. `TF_NEMOTRON_LAYERS=N` truncates the backbone for numerics tests.
- Checkpoint names: `openModelStrip` reads `language_model.*` as a plain model (vision tensors are left for the vision path).
- Latent MoE (cuda_forward.zig `latentExperts`): router on hidden -> fc1 (bf16 lane) -> NVFP4 experts at the latent width
  (K=1024/N=2688, 512 experts) -> `tf_latent_sum` -> fc2 -> + shared expert (two NVFP4 projections, `tf_relu2_bf16`, hidden
  width) -> delta. `slots()` is top_k for latent configs. MTP head loads through the same path (`mtpLatent`, FP8 draft head).
- Router kernel: 512 experts, top-22 (MAXC 16, MAXK 32); K slices 8 for the 4096-wide row (smem under 48 KiB).
- FP8 q/k/v with separate weight scales are stacked as bf16 at load (exact e4m3 x scale -> bf16).
- Numerics gate `tools/nemotron/layer_ref.py`: the checkpoint's own HF blocks (transformers 5.8.1, ModelOpt dequant) vs
  tensorfold dumps: prompt path and one-row decode windows, blocks 0-7 (mamba, latent MoE, attention).
  Result: mamba cos >= 0.9997, latent MoE >= 0.9999 (identical router picks), attention 0.9994.

## Done (r3, Claude lane, 2026-10-15 afternoon; all checks on the Sparks beside the live engine, no swap)
- TP across the Sparks (`families/nemotron/cuda_tp.zig`, `cuda_tp_load.zig`; `zig/src/fabric/tp` copied from the
  DeepSeek tree): Megatron split (Mamba heads + groups, attention q heads with the one KV head they read, latent MoE
  expert width split 672 -> padded 704, fc1 gather / latent all-reduce / fc2 + shared expert one all-reduce, lm_head and
  the draft head by vocabulary rows -> gather). `Config.shard` gives the rank's widths. RoCE backend
  (`TF_COMM_BACKEND=roce`, `TF_TP_ROCE_HCA=roceP2p1s0f1`; flame's 192.168.2.x port is unreachable from the others).
  Gate (8 layers): per-block residual cos vs TP-1 >= 0.99997 (TP-2), >= 0.99992 (TP-4), logits 0.99997; `widths`
  and `shared-widths` (4 streams) pass at TP-4.
- Serving under TP (`cuda_mirror.zig`): rank 0 runs `tensorfold-native serve`; ranks 1-3 replay its lane-backend calls
  (plan link frames) and its image embeds. No prompt cache, no lone driver, prompt passes run to their end, cost
  timing without cache/retimes. Checked: greedy/sampled chat, images, 20K-token prompt, 4 concurrent streams.
- Decode speedups: warp-per-expert router for windows, one-pass top-k (same bits), routing + shared expert on the side
  stream. 24 layers TP-4: 5.8 -> 5.05 ms/token; 48 layers: 9.1 ms 1-row window -> full model ~16 ms (~60 tok/s plain).
- `TF_NEMOTRON_TP_SHARD=RANK/WORLD`: one rank's full share alone (ranks 0 and 3 load all 88 blocks + MTP, 8.5 s warm).
- M3 image front end vs the checkpoint's HF processor (`tools/nemotron/vision_e2e_ref.py`): same grids, same served
  prompt length, history ids = reference prefix, pixels >= 99.96% bf16-exact.
- Swap-window runbook + agenda: `tools/nemotron/swap/` (nemotron-window.sh, WINDOW.md, window-bench.sh, window-vision.py).

## Not done (needs the swap window: Jun's go)
Full-model TP-4 serving and TP-1 text parity, the per-layer HF gate on late layers, MTP acceptance, qbench + vision
benches and the comparison table.
