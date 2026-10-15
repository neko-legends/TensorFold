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

## Not done
Full-model text parity (needs the 70 GB on one Spark = the DeepSeek swap window), TP-4, vision, serving, benchmarks.
