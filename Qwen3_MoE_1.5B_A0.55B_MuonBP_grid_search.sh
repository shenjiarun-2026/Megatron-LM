#!/usr/bin/env bash
set -euo pipefail

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MEGATRON_LOGGING_LEVEL=${MEGATRON_LOGGING_LEVEL:-20}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

# export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-19}
# export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
# export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
# export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}
# export NCCL_AVOID_RECORD_STREAMS=${NCCL_AVOID_RECORD_STREAMS:-1}

# =========================
# User-configurable paths
# =========================
TOKENIZER_ARG=${TOKENIZER_ARG:-/225045001/Megatron-LM/Qwen3-tokenizer}
TRAIN_DATA_ARG=${TRAIN_DATA_ARG:-/225045001/Megatron-LM/c4_Qwen3_merged_train/c4_full}
VALIDATION_DATA_ARG=${VALIDATION_DATA_ARG:-/225045001/Megatron-LM/c4_Qwen3_merged_valid/c4_full}
PRETRAIN_SCRIPT_PATH=${PRETRAIN_SCRIPT_PATH:-pretrain_gpt.py}

# =========================
# Distributed training setup
# =========================
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6015}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE * $NUM_NODES))

# =========================
# Qwen3-like 1.5B MoE setup
# =========================
TP_SIZE=${TP_SIZE:-8}
CP_SIZE=${CP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-32}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}

NUM_LAYERS=${NUM_LAYERS:-20}
HIDDEN_SIZE=${HIDDEN_SIZE:-1024}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-3072}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-16}
NUM_QUERY_GROUPS=${NUM_QUERY_GROUPS:-4}
MOE_FFN_HIDDEN_SIZE=${MOE_FFN_HIDDEN_SIZE:-384}
NUM_EXPERTS=${NUM_EXPERTS:-48}
TOPK=${TOPK:-8}

DTYPE=${DTYPE:-bf16}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-32768}

OPTIMIZER_TYPE="muon"
MUON_MODE=${MUON_MODE:-blockwise}      # blockwise / duplicated / distributed
MUON_USE_NESTEROV=${MUON_USE_NESTEROV:-true}
MOE_MODEL=${MOE_MODEL:-true}
USE_MEGATRON_FSDP=${USE_MEGATRON_FSDP:-false}

# =========================
# Search controls
# =========================
# SEARCH_STAGE=1 : coarse search over LR x WD
# SEARCH_STAGE=2 : refine only muon-extra-scale-factor around chosen LR/WD
SEARCH_STAGE=${SEARCH_STAGE:-1}

# Proxy-run length for grid search
GRID_STEPS=${GRID_STEPS:-2000}
GRID_WARMUP=${GRID_WARMUP:-200}
EVAL_INTERVAL=${EVAL_INTERVAL:-200}
EVAL_ITERS=${EVAL_ITERS:-32}

# Fixed Muon defaults for stage 1
MUON_MOMENTUM=${MUON_MOMENTUM:-0.95}
MUON_NUM_NS_STEPS=${MUON_NUM_NS_STEPS:-5}
MUON_SCALE_MODE=${MUON_SCALE_MODE:-unit_rms_norm}

# Default coarse grid: tune LR, WD, and MuonBP full-update interval first
LR_GRID_STAGE1=(${LR_GRID_STAGE1:-1.5e-2})
WD_GRID_STAGE1=(${WD_GRID_STAGE1:-0.05})
MUON_EXTRA_SCALE_GRID_STAGE1=(${MUON_EXTRA_SCALE_GRID_STAGE1:-1.0})
MUONBP_FULL_UPDATE_INTERVAL_GRID_STAGE1=(${MUONBP_FULL_UPDATE_INTERVAL_GRID_STAGE1:-2 3 4 6})

# Stage 2 refinement around chosen best LR/WD/interval
BEST_LR=${BEST_LR:-8e-4}
BEST_WD=${BEST_WD:-0.10}
BEST_MUONBP_FULL_UPDATE_INTERVAL=${BEST_MUONBP_FULL_UPDATE_INTERVAL:-100}
MUON_EXTRA_SCALE_GRID_STAGE2=(${MUON_EXTRA_SCALE_GRID_STAGE2:-1.0})
MUONBP_FULL_UPDATE_INTERVAL_GRID_STAGE2=(${MUONBP_FULL_UPDATE_INTERVAL_GRID_STAGE2:-${BEST_MUONBP_FULL_UPDATE_INTERVAL}})

# Optional behavior
SKIP_EXISTING=${SKIP_EXISTING:-true}

# Output locations
BASE_CHECKPOINT_DIR=${BASE_CHECKPOINT_DIR:-checkpoints/Qwen3_MoE_1p5B_muon_grid}
DATA_CACHE_PATH=${DATA_CACHE_PATH:-${PWD}/cache_Qwen3_MoE_1p5B_bf16}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$DATA_CACHE_PATH" "$LOG_DIR" "$BASE_CHECKPOINT_DIR"

if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Please run this script from the root of the Megatron-LM repository."
    exit 1
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --num-attention-heads "$NUM_ATTENTION_HEADS"
    --num-query-groups "$NUM_QUERY_GROUPS"

    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
    --position-embedding-type rope
    --rotary-base 1000000

    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --normalization RMSNorm
    --norm-epsilon 1e-06
    --qk-layernorm
    --init-method-std 0.02

    --attention-backend fused
    --apply-layernorm-1p
    --untie-embeddings-and-output-weights
    --disable-bias-linear

    --optimizer "${OPTIMIZER_TYPE}"
    --logging-level 20
)

if [[ "$USE_MEGATRON_FSDP" == "true" ]]; then
    MODEL_ARGS+=(
        --use-megatron-fsdp
        --ckpt-format fsdp_dtensor
    )
fi

MOE_ARGS=()
if [[ "$MOE_MODEL" == "true" ]]; then
    MOE_ARGS+=(
        --num-experts "$NUM_EXPERTS"
        --moe-router-topk "$TOPK"
        --moe-ffn-hidden-size "$MOE_FFN_HIDDEN_SIZE"
        --moe-layer-freq 1
        --moe-router-dtype fp32
        --moe-router-load-balancing-type aux_loss
        --moe-aux-loss-coeff 1e-3
        --moe-token-dispatcher-type alltoall
        --expert-model-parallel-size 4
        --expert-tensor-parallel-size 1
        --overlap-moe-expert-parallel-comm
        # --moe-grouped-gemm
        # --moe-permute-fusion
        # --moe-router-fusion
    )
fi

DTYPE_ARGS=()
if [[ "$DTYPE" == "fp8" ]]; then
    DTYPE_ARGS+=(
        --fp8-format hybrid
        --fp8-amax-history-len 1024
        --fp8-amax-compute-algo max
        --fp8-param-gather
    )
fi

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP_SIZE"
    --context-parallel-size "$CP_SIZE"
    --pipeline-model-parallel-size "$PP_SIZE"
    --sequence-parallel
)

DATA_ARGS_LIST=(
    --train-data-path "$TRAIN_DATA_ARG"
    --valid-data-path "$VALIDATION_DATA_ARG"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_ARG"
    --data-cache-path "${DATA_CACHE_PATH}"
    --no-create-attention-mask-in-dataloader
    --no-mmap-bin-files
    --num-workers 8
    --vocab-size 151936
)

run_one() {
    local current_lr="$1"
    local current_wd="$2"
    local current_extra_scale="$3"
    local current_muonbp_full_update_interval="$4"

    local current_min_lr
    current_min_lr=$(awk "BEGIN {printf \"%e\", $current_lr * 1e-2}")

    local log_name
    log_name="Qwen3MoE1p5B_${OPTIMIZER_TYPE}_muonbp_lr${current_lr}_wd${current_wd}_scale${current_extra_scale}_fullint${current_muonbp_full_update_interval}_TP_SIZE_${TP_SIZE}"

    local ckpt_dir="${BASE_CHECKPOINT_DIR}/${log_name}"
    local tb_dir="${DATA_CACHE_PATH}/${log_name}"
    local log_file="${LOG_DIR}/${log_name}.log"

    if [[ "$SKIP_EXISTING" == "true" && -f "$log_file" ]]; then
        echo "${log_name} already has a log file: ${log_file}... Reruning"
    fi

    mkdir -p "$ckpt_dir" "$tb_dir"

    echo "============================================================"
    echo "Starting grid run:"
    echo "  STAGE                = ${SEARCH_STAGE}"
    echo "  LR                            = ${current_lr}"
    echo "  WEIGHT_DECAY                  = ${current_wd}"
    echo "  MUON_EXTRA_SCALE              = ${current_extra_scale}"
    echo "  MUONBP_FULL_UPDATE_INTERVAL   = ${current_muonbp_full_update_interval}"
    echo "  GRID_STEPS                    = ${GRID_STEPS}"
    echo "  GRID_WARMUP                   = ${GRID_WARMUP}"
    echo "  LOG_NAME             = ${log_name}"
    echo "============================================================"

    torchrun \
        "${DISTRIBUTED_ARGS[@]}" \
        "$PRETRAIN_SCRIPT_PATH" \
        "${MODEL_ARGS[@]}" \
        "${MOE_ARGS[@]}" \
        --seed 42 \
        --use-flash-attn \
        --micro-batch-size "$MICRO_BATCH_SIZE" \
        --global-batch-size "$GLOBAL_BATCH_SIZE" \
        --train-iters "$GRID_STEPS" \
        --lr "$current_lr" \
        --min-lr "$current_min_lr" \
        --lr-decay-style cosine \
        --lr-decay-iters "$GRID_STEPS" \
        --lr-warmup-iters "$GRID_WARMUP" \
        --clip-grad 1.0 \
        --weight-decay "$current_wd" \
        --adam-beta1 0.9 \
        --adam-beta2 0.98 \
        --bf16 \
        --manual-gc \
        --empty-unused-memory-level 1 \
        --cross-entropy-loss-fusion \
        --no-gradient-accumulation-fusion \
        --use-checkpoint-opt_param-scheduler \
        --muon-config-mode muonbp \
        --muon-scale-mode "$MUON_SCALE_MODE" \
        --muon-tp-mode "$MUON_MODE" \
        --muon-momentum "$MUON_MOMENTUM" \
        --muon-num-ns-steps "$MUON_NUM_NS_STEPS" \
        --muon-extra-scale-factor "$current_extra_scale" \
        --muonbp-full-update-interval "$current_muonbp_full_update_interval" \
        --muon-snecv-z-low 1.0 \
        --muon-snecv-z-high 1.2 \
        $( [[ "$MUON_USE_NESTEROV" == "true" ]] && echo "--muon-use-nesterov" ) \
        "${DTYPE_ARGS[@]}" \
        "${MODEL_PARALLEL_ARGS[@]}" \
        "${DATA_ARGS_LIST[@]}" \
        --log-interval 1 \
        --eval-iters "$EVAL_ITERS" \
        --eval-interval "$EVAL_INTERVAL" \
        --save-interval 2000 \
        --save "$ckpt_dir" \
        --log-throughput \
        --distributed-timeout-minutes 3600 \
        --tensorboard-dir "$tb_dir" \
        | tee "$log_file"
}

if [[ "$SEARCH_STAGE" == "1" ]]; then
    echo "[Info] Running Stage 1 search: LR x weight_decay"
    for CURRENT_LR in "${LR_GRID_STAGE1[@]}"; do
        for CURRENT_WD in "${WD_GRID_STAGE1[@]}"; do
            for CURRENT_EXTRA_SCALE in "${MUON_EXTRA_SCALE_GRID_STAGE1[@]}"; do
                run_one "$CURRENT_LR" "$CURRENT_WD" "$CURRENT_EXTRA_SCALE"
            done
        done
    done
elif [[ "$SEARCH_STAGE" == "2" ]]; then
    echo "[Info] Running Stage 2 refinement: muon-extra-scale-factor around BEST_LR/BEST_WD"
    echo "[Info] BEST_LR=${BEST_LR}, BEST_WD=${BEST_WD}"
    for CURRENT_EXTRA_SCALE in "${MUON_EXTRA_SCALE_GRID_STAGE2[@]}"; do
        run_one "$BEST_LR" "$BEST_WD" "$CURRENT_EXTRA_SCALE"
    done
else
    echo "Error: SEARCH_STAGE must be 1 or 2, got '${SEARCH_STAGE}'"
    exit 1
fi

set +x