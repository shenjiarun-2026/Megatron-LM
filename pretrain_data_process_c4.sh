set -euo pipefail

C4_DIR=./hf_home/hub/datasets--allenai--c4/snapshots/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en/    # contains c4-train.*.json.gz and c4-validation.*.json.gz
OUT_DIR=./shenjiarun/Megatron-LM-dev/c4_mcore_gpt2
TOKENIZER_MODEL=./shenjiarun/Megatron-LM-dev/gpt2/   # or your HF tokenizer path if you use HuggingFaceTokenizer
WORKERS=16

export HF_DATASETS_OFFLINE=1
export HF_HOME="./hf_home"

mkdir -p "$OUT_DIR/train" "$OUT_DIR/valid"

# Train (1024 shards)
# for f in "$C4_DIR"/c4-train.*.json.gz; do
#   base=$(basename "$f" .json.gz)
#   tmp=$(mktemp --suffix=.jsonl)
#   gzip -dc "$f" > "$tmp"   # use `gzip -dc` if pigz not available

#   python tools/preprocess_data.py \
#     --input "$tmp" \
#     --output-prefix "$OUT_DIR/train/$base" \
#     --tokenizer-type HuggingFaceTokenizer \
#     --tokenizer-model "$TOKENIZER_MODEL" \
#     --workers "$WORKERS" \
#     --append-eod

#   rm -f "$tmp"
# done

# Validation (8 shards)
for f in "$C4_DIR"/c4-validation.*.json.gz; do
  base=$(basename "$f" .json.gz)
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