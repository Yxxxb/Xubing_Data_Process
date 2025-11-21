#!/usr/bin/env bash
set +x
set -e
export ENV_NAME=mmq-bensenliu-h800-latest
source /opt/superpod_utils/setup-mamba-env-once

env | grep NCCL_
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_IB_QPS_PER_CONNECTION=4

micromamba activate nv-megatron-new-tccl-cuda121

export http_proxy=http://hk-mmhttpproxy.woa.com:11113
export https_proxy=http://hk-mmhttpproxy.woa.com:11113
export no_proxy=11.152.212.103
pip install --upgrade jsonargparse==4.24.1
pip install timm

cd $1
pip install -e .

torchrun \
    --nnodes="${WORLD_SIZE}" \
    --nproc_per_node=8 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --node_rank="${RANK}" \
    $2

sleep 10000000000
