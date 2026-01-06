# NanoChat Setup Guide

Quick reference for setting up and training NanoChat models.

## Known Issues & Fixes

### 1. Package Not Installable

**Problem:** `uv sync` doesn't install the nanochat package, causing `ModuleNotFoundError: No module named 'nanochat'`

**Fix:** Ensure `pyproject.toml` has:
```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["nanochat*", "scripts*", "tasks*"]
exclude = ["dev*"]
```

Then run `uv sync` again.

---

### 2. HuggingFace Model Placement

**Problem:** HuggingFace instructions say to put pretrained model in `chatsft_checkpoints/`, but SFT expects it in `base_checkpoints/`.

**Correct directory structure:**
```
~/.cache/nanochat/
├── tokenizer/
│   ├── token_bytes.pt
│   └── tokenizer.pkl
├── base_checkpoints/d34/        # Pretrained model goes HERE
│   ├── model_169150.pt
│   └── meta_169150.json
├── mid_checkpoints/d34/         # Output of mid-training
├── chatsft_checkpoints/d34/     # Output of SFT
└── chatrl_checkpoints/d34/      # Output of RL
```

---

### 3. Memory Requirements

| Model | Params | Training Memory | Inference Memory | MacBook MPS |
|-------|--------|-----------------|------------------|-------------|
| d12   | ~125M  | ~8 GB           | ~2 GB            | Works       |
| d20   | ~350M  | ~16 GB          | ~4 GB            | Tight       |
| d34   | 2.2B   | ~40-70 GB       | ~16 GB           | Too big     |

**For d34:** Use cloud GPU (H100 80GB recommended, A100 40GB minimum).

---

### 5. Mid-Training OOM on Single GPU

**Problem:** Mid-training crashes with OOM on single H100 using default `device_batch_size=32`.

**Fix:** Reduce batch size to match base training:
```bash
uv run python -m scripts.mid_train \
  --model_tag=d34 \
  --model_step=169150 \
  --device_batch_size=4 \
  --run=d34-mid
```

Memory usage: ~56GB with batch_size=4 vs 79GB+ with batch_size=32.

---

### 4. Loading Specific Checkpoints

Always specify `--step` to load the correct model:
```bash
# Load pretrained base model
--source=base --model_tag=d34 --model_step=169150

# Load your SFT-trained model
--source=sft --model_tag=d34 --model_step=700
```

Without `--step`, it auto-selects the last checkpoint in the directory.

---

## Training Pipeline

```
Pretrained Model (from HuggingFace)
        |
   Mid-Training (--source=base)      ← RECOMMENDED, not optional!
        |
   Chat SFT (--source=mid)
        |
   [Optional] RL (--source=sft)
        |
   Chat Model (ready to deploy)
```

**Important:** Skipping mid-training produces a poor chat model. The base model only learned language patterns, not conversational ability. Mid-training teaches reasoning, math, and conversation structure.

---

## Quick Start: SFT on Cloud GPU

### 1. Setup (H100/A100)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

git clone https://github.com/karpathy/nanochat.git
cd nanochat
uv sync
uv pip install huggingface_hub
```

### 2. Download Model
```bash
uv run huggingface-cli download karpathy/nanochat-d34 --local-dir ~/nanochat-d34
```

### 3. Setup Directories
```bash
mkdir -p ~/.cache/nanochat/tokenizer
mkdir -p ~/.cache/nanochat/base_checkpoints/d34

cp ~/nanochat-d34/token_bytes.pt ~/.cache/nanochat/tokenizer/
cp ~/nanochat-d34/tokenizer.pkl ~/.cache/nanochat/tokenizer/
cp ~/nanochat-d34/model_169150.pt ~/.cache/nanochat/base_checkpoints/d34/
cp ~/nanochat-d34/meta_169150.json ~/.cache/nanochat/base_checkpoints/d34/

curl -L -o ~/.cache/nanochat/identity_conversations.jsonl \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
```

### 4. Run SFT
```bash
uv run python -m scripts.chat_sft \
  --source=base \
  --model_tag=d34 \
  --model_step=169150 \
  --run=d34-sft
```

### 5. Test
```bash
uv run python -m scripts.chat_web \
  --source=sft \
  --model_tag=d34 \
  --step=<final_step> \
  --temperature=0.6
```

### 6. Download Before Terminating
```bash
scp ubuntu@<ip>:~/.cache/nanochat/chatsft_checkpoints/d34/model_*.pt ~/Downloads/
scp ubuntu@<ip>:~/.cache/nanochat/chatsft_checkpoints/d34/meta_*.json ~/Downloads/
```

---

## Firewall / Port Access

If `http://<server-ip>:8000` doesn't work, use SSH tunneling:
```bash
ssh -L 8000:localhost:8000 ubuntu@<server-ip>
```
Then open `http://localhost:8000`

---

## Cost Estimates

| Task | GPU | Time | Cost |
|------|-----|------|------|
| Mid-training (d34) | 1x H100 80GB | ~5.5 hours | ~$14 |
| SFT (d34, 1 epoch) | 1x H100 80GB | ~30-60 min | ~$1-2 |
| Inference testing | 1x H100 80GB | ~10 min | ~$0.50 |
| **Total** | | **~6.5 hours** | **~$16** |

---

## Complete Training Workflow (Tested)

This is the exact workflow that produced a working chat model:

### 1. Setup Lambda H100
```bash
ssh ubuntu@<lambda-ip>

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# Clone and setup
git clone https://github.com/karpathy/nanochat.git
cd nanochat
uv sync
uv pip install huggingface_hub
```

### 2. Download Pretrained Model
```bash
uv run huggingface-cli download karpathy/nanochat-d34 --local-dir ~/nanochat-d34
```

### 3. Setup Directory Structure (CRITICAL)
```bash
mkdir -p ~/.cache/nanochat/tokenizer
mkdir -p ~/.cache/nanochat/base_checkpoints/d34

# Tokenizer
cp ~/nanochat-d34/token_bytes.pt ~/.cache/nanochat/tokenizer/
cp ~/nanochat-d34/tokenizer.pkl ~/.cache/nanochat/tokenizer/

# Model -> base_checkpoints (NOT chatsft_checkpoints!)
cp ~/nanochat-d34/model_169150.pt ~/.cache/nanochat/base_checkpoints/d34/
cp ~/nanochat-d34/meta_169150.json ~/.cache/nanochat/base_checkpoints/d34/

# Identity conversations
curl -L -o ~/.cache/nanochat/identity_conversations.jsonl \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
```

### 4. Run Mid-Training (~5.5 hours)
```bash
# Use device_batch_size=4 to avoid OOM!
nohup uv run python -m scripts.mid_train \
  --model_tag=d34 \
  --model_step=169150 \
  --device_batch_size=4 \
  --run=d34-mid > mid_train.log 2>&1 &

# Monitor
tail -f mid_train.log
nvidia-smi  # Should show ~56GB usage
```

**Expected output when complete:**
```
Step 00813 (100.00%) | loss: 1.068
Minimum validation bpb: 0.3282
```

### 5. Run SFT (~30-60 min)
```bash
uv run python -m scripts.chat_sft \
  --source=mid \
  --model_tag=d34 \
  --run=d34-sft
```

**Expected output when complete:**
```
Step 00700 | mmlu_acc: 0.425781, arc_easy_acc: 0.719727
✅ Saved model checkpoint to /home/ubuntu/.cache/nanochat/chatsft_checkpoints/d34
```

### 6. Test
```bash
uv run python -m scripts.chat_web \
  --source=sft \
  --model-tag=d34 \
  --step=700 \
  --temperature=0.6
```

From local machine:
```bash
ssh -L 8000:localhost:8000 ubuntu@<lambda-ip>
# Open http://localhost:8000
```

### 7. Download Models Before Terminating
```bash
# On server - create tarball
cd ~/.cache/nanochat
tar -czvf ~/nanochat-d34-trained.tar.gz \
  tokenizer/ \
  mid_checkpoints/d34/ \
  chatsft_checkpoints/d34/

# On local machine - download
scp ubuntu@<lambda-ip>:~/nanochat-d34-trained.tar.gz ~/Downloads/

# Extract and setup locally
cd ~/Downloads
tar -xzvf nanochat-d34-trained.tar.gz
cp -r tokenizer ~/.cache/nanochat/
cp -r mid_checkpoints ~/.cache/nanochat/
cp -r chatsft_checkpoints ~/.cache/nanochat/
```

---

## Expected Model Quality

| Metric | Without Mid-Training | With Mid-Training |
|--------|---------------------|-------------------|
| MMLU Accuracy | 25% (random) | 42.6% |
| ARC-Easy Accuracy | 30% | 72% |
| Chat Quality | Gibberish | Coherent conversations |
| Math | Broken | Basic arithmetic works |
| Code Generation | Broken | Working code |

---

## Best Practices

1. **Never skip mid-training** - It's the difference between gibberish and a working model

2. **Always use `--device_batch_size=4`** for d34 on single GPU - Default of 32 causes OOM

3. **Always specify `--step`** when loading models - Auto-selection can load wrong checkpoint

4. **Put pretrained model in `base_checkpoints/`** - NOT `chatsft_checkpoints/`

5. **Use SSH tunneling** for web UI - Cloud providers block port 8000

6. **Download models before terminating** - You'll lose 6+ hours of work otherwise

7. **Monitor with WandB** - Logs may be buffered, WandB shows real-time progress
