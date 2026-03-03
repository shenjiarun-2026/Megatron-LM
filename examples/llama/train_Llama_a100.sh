#!/usr/bin/env bash

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# export LOG_LEVEL=${LOG_LEVEL:-INFO}
# export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-19}
# export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
# export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
# export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}
# export NCCL_AVOID_RECORD_STREAMS=${NCCL_AVOID_RECORD_STREAMS:-1}

# CHECKPOINT_PATH=${1:-"checkpoints/llama3_8b_fp8"}
# TENSORBOARD_LOGS_PATH=${2:-"tensorboard_logs/llama3_8b_fp8"}
TOKENIZER_ARG=./Megatron-LM-dev/Llama-3-8B/ # Path to tokenizer model, or "MOCK"
TRAIN_DATA_ARG=./Megatron-LM-dev/c4_merged_train/c4_full # Data prefix, or "MOCK"
VALIDATION_DATA_ARG=./Megatron-LM-dev/c4_merged_valid/c4_full # Data prefix, or "MOCK"

# Distributed training setup
GPUS_PER_NODE=8
NUM_NODES=1
MASTER_ADDR=localhost
MASTER_PORT=6000
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Path to the pretrain_gpt.py script, assuming this script is run from the root of the Megatron-LM repository
PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

# Fixed model and training parameters
TP_SIZE=2
CP_SIZE=1     
PP_SIZE=1     
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=256
NUM_LAYERS=32
DTYPE="bf16"
SEQ_LENGTH=4096
MAX_POSITION_EMBEDDINGS=4096
OPTIMIZER_TYPE="muon"
MUON_MODE="distributed"  # Options: blockwise, duplicated, distributed
USE_MEGATRON_FSDP="false"
MUON_USE_NESTEROV="true"
MOE_MODLE="false"
STEPS=28992

CHECKPOINT_PATH=checkpoints/Llama3_190M_${OPTIMIZER_TYPE}_bf16

# Create directories if they don't exist
# mkdir -p "$(dirname "$CHECKPOINT_PATH")"
# mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

# Data cache path (useful for both mock and real data)
DATA_CACHE_PATH="${PWD}/cache_Llama2_130M_bf16"
mkdir -p "$DATA_CACHE_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers $NUM_LAYERS
    --hidden-size 512
    --ffn-hidden-size 2048
    --num-attention-heads 8
    --num-query-groups 2
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 500000
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --normalization RMSNorm
    --norm-epsilon 1e-05
    --init-method-std 0.02
    --attention-backend fused
    --apply-layernorm-1p 
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --optimizer ${OPTIMIZER_TYPE}
)

MOE_ARGS=()
if [[ "$MOE_MODLE" == "true" ]]; then
    MOE_ARGS+=(
        --num-experts 8
        --moe-router-topk 2
        --moe-router-load-balancing-type aux_loss
        --moe-aux-loss-coeff 1e-2
        --moe-grouped-gemm
        --moe-token-dispatcher-type alltoall
        --expert-model-parallel-size 4
    ) 
fi


if [[ "$USE_MEGATRON_FSDP" == "true" ]]; then
    MODEL_ARGS+=(
        --use-megatron-fsdp
        --ckpt-format fsdp_dtensor
    )
fi

if [[ "$OPTIMIZER_TYPE" == "muon" ]]; then
    TRAINING_ARGS=(
        --seed 42
        --use-flash-attn
        --micro-batch-size $MICRO_BATCH_SIZE
        --global-batch-size $GLOBAL_BATCH_SIZE
        --train-iters $STEPS
        --lr 2e-3
        --min-lr 2e-5
        --lr-decay-style cosine
        --lr-warmup-iters 87
        --clip-grad 1.0
        --weight-decay 3.2e-4
        --adam-beta1 0.9
        --adam-beta2 0.98
        --bf16
        --manual-gc
        --empty-unused-memory-level 1
        --exit-duration-in-mins 3600
        --cross-entropy-loss-fusion
        --no-gradient-accumulation-fusion
        --use-checkpoint-opt_param-scheduler
        --muon-scale-mode unit_rms_norm
        --muon-tp-mode $MUON_MODE
        --muon-lr-multiplier 1.6
        --muon-momentum 0.95
    )
else
    TRAINING_ARGS=(
        --seed 42
        --use-flash-attn
        --micro-batch-size $MICRO_BATCH_SIZE
        --global-batch-size $GLOBAL_BATCH_SIZE
        --train-iters $STEPS # 28992 for 8x Chinchilla
        --lr 4e-3
        --min-lr 4e-5 # 1% of lr
        --lr-decay-style cosine
        --lr-decay-iters $STEPS
        --lr-warmup-iters 5798
        --clip-grad 1.0
        --weight-decay 2e-4
        --adam-beta1 0.9
        --adam-beta2 0.98
        --lr-warmup-init 0.0
        --bf16
        --cross-entropy-loss-fusion
        --no-gradient-accumulation-fusion
        --manual-gc 
        --use-checkpoint-opt_param-scheduler
        --empty-unused-memory-level 1 
        --exit-duration-in-mins 3600
    )
fi

if [[ "$OPTIMIZER_TYPE" == "muon" ]]; then
    if [[ "$MUON_USE_NESTEROV" == "true" ]]; then
        TRAINING_ARGS+=(
            --muon-use-nesterov
        )
    fi
fi


# Conditional arguments based on DTYPE (FP8)
DTYPE_ARGS=()
if [[ "$DTYPE" == "fp8" ]]; then
    DTYPE_ARGS+=(
        "--fp8-format hybrid"
        "--fp8-amax-history-len 1024"
        "--fp8-amax-compute-algo max"
        "--fp8-param-gather"
    )
fi

# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
    --pipeline-model-parallel-size $PP_SIZE # Not explicitly set in llama script options, assume 1 if not multi-node PP
    --sequence-parallel  # Always enable sequence parallelism with TP_SIZE=2
)

# Distributed Data Parallel (DDP) arguments
# From original script's ddp_args
DDP_ARGS=()
if [[ "$OPTIMIZER_TYPE" == "adam" ]]; then
    DDP_ARGS+=(
        --use-distributed-optimizer
        --overlap-grad-reduce
        --overlap-param-gather
    )
fi
TRAINING_ARGS+=("${DDP_ARGS[@]}")


# Data arguments (conditional for mock vs real data)
DATA_ARGS_LIST=()
# Settings for real data
DATA_ARGS_LIST+=(
    "--train-data-path $TRAIN_DATA_ARG"
    "--valid-data-path $VALIDATION_DATA_ARG"
    "--tokenizer-type HuggingFaceTokenizer" 
    "--tokenizer-model $TOKENIZER_ARG"
    "--data-cache-path ${DATA_CACHE_PATH}"
    # "--split '99,1,0'"
    "--no-create-attention-mask-in-dataloader"
    "--no-mmap-bin-files"
    "--num-workers 8"
    "--vocab-size 128256"
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 32
    --eval-interval 400
    --save-interval 2000
    --load ${CHECKPOINT_PATH}
    --save ${CHECKPOINT_PATH}
    --log-throughput
    --distributed-timeout-minutes 3600
)

# Ensure pretrain_gpt.py is found
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Please ensure you are running this script from the root of the Megatron-LM repository, and pretrain_gpt.py is present."
    exit 1
fi

if [[ "$OPTIMIZER_TYPE" == "muon" ]]; then
  LOG_NAME="train_GBS${GLOBAL_BATCH_SIZE}_muon_${MUON_MODE}_STEP${STEPS}_8xchinchilla"
else
  LOG_NAME="train_GBS${GLOBAL_BATCH_SIZE}_${OPTIMIZER_TYPE}_STEP${STEPS}_8xchinchilla"
fi

# Run the training command
torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    | tee logs/"${LOG_NAME}.log"

set +x