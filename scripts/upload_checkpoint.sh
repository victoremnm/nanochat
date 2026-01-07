#!/bin/bash
# Upload NanoChat checkpoints to HuggingFace
# Usage: ./upload_checkpoint.sh [step] [repo]
#   step: specific step number (default: latest)
#   repo: HuggingFace repo (default: victoremnm/nanochat-d34-sft)

set -e

REPO="${2:-victoremnm/nanochat-d34-sft}"
CHECKPOINT_DIR=~/.cache/nanochat/chatsft_checkpoints/d34
MID_CHECKPOINT_DIR=~/.cache/nanochat/mid_checkpoints/d34

# Determine which step to upload
if [ -n "$1" ]; then
    STEP=$1
    MODEL_FILE="$CHECKPOINT_DIR/model_$(printf '%06d' $STEP).pt"
    META_FILE="$CHECKPOINT_DIR/meta_$(printf '%06d' $STEP).json"
else
    # Find latest model file
    MODEL_FILE=$(ls -t $CHECKPOINT_DIR/model_*.pt 2>/dev/null | head -1)
    if [ -z "$MODEL_FILE" ]; then
        echo "Error: No model files found in $CHECKPOINT_DIR"
        exit 1
    fi
    STEP=$(basename $MODEL_FILE | sed 's/model_0*\([0-9]*\)\.pt/\1/')
    META_FILE="$CHECKPOINT_DIR/meta_$(printf '%06d' $STEP).json"
fi

# Validate files exist
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found: $MODEL_FILE"
    exit 1
fi

if [ ! -f "$META_FILE" ]; then
    echo "Warning: Meta file not found: $META_FILE"
fi

echo "========================================"
echo "Uploading to: $REPO"
echo "Step: $STEP"
echo "Model: $MODEL_FILE"
echo "Meta: $META_FILE"
echo "========================================"

# Upload model
echo "Uploading model file..."
hf upload $REPO $MODEL_FILE model_$(printf '%06d' $STEP).pt --repo-type model

# Upload meta if exists
if [ -f "$META_FILE" ]; then
    echo "Uploading meta file..."
    hf upload $REPO $META_FILE meta_$(printf '%06d' $STEP).json --repo-type model
fi

echo "========================================"
echo "Upload complete!"
echo "View at: https://huggingface.co/$REPO"
echo "========================================"

# List all available checkpoints
echo ""
echo "Available checkpoints:"
ls -la $CHECKPOINT_DIR/*.pt 2>/dev/null | awk '{print "  " $NF}' || echo "  (none)"
