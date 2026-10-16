#!/bin/bash
# TP-4 tensorfold command, rank r on HOSTS[r], RoCE like the server. usage: tp4run.sh TAG -- args (RANK -> rank)
# env: TFBIN (bin/tensorfold), EXTRA (more env for every rank, e.g. TF_NEMOTRON_LAYERS=8 NCCL_MAX_NCHANNELS=2)
tag=$1; shift 2; args="$*"
. "$(cd "$(dirname "$0")" && pwd)/sparks.sh"; : "${MASTER:?set TF_TP_MASTER to rank 0 RoCE link address}"
for r in 0 1 2 3; do h=${HOSTS[$r]}
  ssh $h "cd ~/nemotron-tf && env TF_TP_WORLD=4 TF_TP_RANK=$r TF_TP_MASTER=$MASTER TF_TP_PORT=29651 \
    TF_COMM_BACKEND=${BACKEND:-roce} TF_TP_ROCE_HCA=roceP2p1s0f1 NCCL_IB_HCA=roceP2p1s0f1 GLM53_TF_ROCE_FAST=1 TF_TP_ROCE_FALLBACK=nccl \
    TF_NCCL_LIB=$HOME/nemotron-tf/lib/libnccl.so.2 NCCL_SOCKET_IFNAME=enP2p1s0f1np1 ${EXTRA:-} \
    timeout ${TO:-900} ${TFBIN:-bin/tensorfold} ${args//RANK/$r} > runs/$tag.r$r.log 2>&1; echo \"$h rank $r exit \$?\"; tail -${TAIL:-3} runs/$tag.r$r.log" &
done; wait
