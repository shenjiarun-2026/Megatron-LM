set -euo pipefail

C4_DIR=/sds_lix/hf_home/hub/datasets--allenai--c4/snapshots/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en # contains c4-train.*.json.gz and c4-validation.*.json.gz
OUT_DIR=/225045001/Megatron-LM/Qwen3_c4_mcore
TOKENIZER_MODEL=/225045001/Megatron-LM/Qwen3-tokenizer/ # or your HF tokenizer path if you use HuggingFaceTokenizer
WORKERS=16

export HF_DATASETS_OFFLINE=1
export HF_HOME="/sds_lix/hf_home"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT_DIR/train" "$OUT_DIR/valid"

# Train (1024 shards)
for f in "$C4_DIR"/c4-train.*.json.gz; do
  base=$(basename "$f" .json.gz)

  # ==== 新增：断点续传逻辑 ====
  # 检查是否已经存在对应的 .idx 索引文件
  shopt -s nullglob
  existing_idx=("$OUT_DIR/train/${base}"*.idx)
  shopt -u nullglob

  if [ ${#existing_idx[@]} -gt 0 ]; then
    echo "⏩ 跳过已处理的训练集: $base"
    continue
  fi
  # ============================

  echo "▶️ 正在处理训练集: $base"
  tmp=$(mktemp --suffix=.jsonl)
  gzip -dc "$f" > "$tmp"   # use `gzip -dc` if pigz not available

  python tools/preprocess_data.py \
    --input "$tmp" \
    --output-prefix "$OUT_DIR/train/$base" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --workers "$WORKERS" \
    --append-eod

  rm -f "$tmp"
done

# Validation (8 shards)
for f in "$C4_DIR"/c4-validation.*.json.gz; do
  base=$(basename "$f" .json.gz)

  # ==== 新增：验证集断点续传逻辑 ====
  shopt -s nullglob
  existing_idx=("$OUT_DIR/valid/${base}"*.idx)
  shopt -u nullglob

  if [ ${#existing_idx[@]} -gt 0 ]; then
    echo "⏩ 跳过已处理的验证集: $base"
    continue
  fi
  # ==================================

  echo "▶️ 正在处理验证集: $base"
  tmp=$(mktemp --suffix=.jsonl)
  gzip -dc "$f" > "$tmp"

  python tools/preprocess_data.py \
    --input "$tmp" \
    --output-prefix "$OUT_DIR/valid/$base" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --workers "$WORKERS" \
    --append-eod

  rm -f "$tmp"
done

echo "✅ 所有数据处理完毕！"
