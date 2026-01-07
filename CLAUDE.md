# NanoChat Project Guide

## Project Overview
NanoChat is a minimal full-stack ChatGPT clone by Andrej Karpathy. This fork includes fixes for package installation and documentation for training with the pretrained d34 model.

## Quick Reference

### Training Pipeline (DO NOT SKIP STEPS)
```
Base Model (HuggingFace d34)
    → Mid-Training (REQUIRED for good results)
    → SFT (Chat fine-tuning)
    → Working Chat Model
```

### Key Commands
```bash
# Mid-training (use device_batch_size=4 for d34 on single GPU)
uv run python -m scripts.mid_train --model_tag=d34 --model_step=169150 --device_batch_size=4 --run=d34-mid

# SFT after mid-training
uv run python -m scripts.chat_sft --source=mid --model_tag=d34 --run=d34-sft

# Run chat web server
uv run python -m scripts.chat_web --source=sft --model-tag=d34 --step=<step> --temperature=0.6
```

## Critical Learnings

### 1. Package Installation
The repo requires `[build-system]` in pyproject.toml to be installable:
```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["nanochat*", "scripts*", "tasks*"]
exclude = ["dev*"]
```

### 2. Model File Placement
HuggingFace instructions are WRONG. Place pretrained model in `base_checkpoints/`, NOT `chatsft_checkpoints/`:
```
~/.cache/nanochat/
├── tokenizer/
│   ├── token_bytes.pt
│   └── tokenizer.pkl
├── base_checkpoints/d34/      ← Pretrained model HERE
│   ├── model_169150.pt
│   └── meta_169150.json
├── mid_checkpoints/d34/       ← Output of mid-training
└── chatsft_checkpoints/d34/   ← Output of SFT
```

### 3. Memory Requirements (d34 model)
- **Mid-training default batch_size=32**: OOM on single H100 (uses 79GB+)
- **Mid-training with batch_size=4**: Works (~56GB)
- **SFT**: Works with default settings (~40GB)
- **Inference**: ~16GB

### 4. Always Specify --step
Without `--step`, scripts auto-select the last checkpoint which may not be what you want:
```bash
# Correct
--source=sft --model_tag=d34 --step=700

# Dangerous - might load wrong checkpoint
--source=sft --model_tag=d34
```

### 5. Mid-Training is NOT Optional
Skipping mid-training produces a broken chat model that outputs:
- Random forum/Wikipedia text
- Repetitive gibberish
- Wrong conversation format

Mid-training teaches the model conversation structure, reasoning, and domain knowledge.

## Cloud GPU Setup (Lambda Labs H100)

### SSH Access
```bash
ssh ubuntu@<lambda-ip>
```

### SSH Tunneling for Web Interface
Lambda/cloud providers block direct port access. Use SSH tunneling:

```bash
# Terminal 1 (local) - Create tunnel
ssh -i ~/.ssh/lambda.pem -L 8080:localhost:8080 ubuntu@209-20-159-9

# Terminal 2 (on Lambda via the tunnel) - Start server
cd ~/nanochat
uv run python -m scripts.chat_web --source=sft --model-tag=d34 --step=719 --port=8080

# Then open http://localhost:8080 in browser
```

Alternative: tunnel-only mode (no shell):
```bash
ssh -N -L 8080:localhost:8080 ubuntu@209-20-159-9
```

### Quick Model Test via curl (no tunnel needed)
Test directly on Lambda without tunneling:
```bash
# On Lambda - start server in background
cd ~/nanochat
uv run python -m scripts.chat_web --source=sft --model-tag=d34 --step=719 --port=8080 &

# Wait for model to load (~30s), then test
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Write a Clickhouse query for top 10 traders by volume"}]}'
```

### Background Training
```bash
nohup uv run python -m scripts.mid_train \
  --model_tag=d34 \
  --model_step=169150 \
  --device_batch_size=4 \
  --run=d34-mid > mid_train.log 2>&1 &

# Monitor
tail -f mid_train.log
nvidia-smi
```

### Download Model Before Terminating
```bash
scp ubuntu@<ip>:~/.cache/nanochat/chatsft_checkpoints/d34/model_*.pt ~/Downloads/
scp ubuntu@<ip>:~/.cache/nanochat/chatsft_checkpoints/d34/meta_*.json ~/Downloads/
```

## Development Workflow

### Git Branches
- Never push directly to main/master
- Create feature branches: `git checkout -b feature/description`
- Fork is at: https://github.com/victoremnm/nanochat

### Virtual Environment
- Use uv for dependency management
- Deactivate conda if active (conflicts with uv)
- Run commands with `uv run python -m ...`

## File Locations

| File | Purpose |
|------|---------|
| `scripts/mid_train.py` | Mid-training script |
| `scripts/chat_sft.py` | SFT training script |
| `scripts/chat_web.py` | Web chat server |
| `scripts/chat_cli.py` | CLI chat interface |
| `nanochat/checkpoint_manager.py` | Model loading logic |
| `nanochat/gpt.py` | Model architecture |
| `tasks/customjson.py` | Custom training data loader |

## Adding Custom Training Data

Create JSONL file:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Add to training mixture in `scripts/chat_sft.py`:
```python
from tasks.customjson import CustomJSON

train_ds = TaskMixture([
    # ... existing tasks ...
    CustomJSON(filepath="/path/to/my_data.jsonl"),
])
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'nanochat'"
Run `uv sync` - the build-system config should fix this.

### OOM during mid-training
Add `--device_batch_size=4` to the command.

### Model outputs gibberish
- Check you're loading the correct checkpoint (`--step=X`)
- Ensure you didn't skip mid-training
- Try lower temperature (`--temperature=0.3`)

### Logs not appearing
Python output buffering. Check WandB dashboard or use `PYTHONUNBUFFERED=1`.

### Port 8000 not accessible
Use SSH tunneling: `ssh -L 8000:localhost:8000 ubuntu@<ip>`

## Cost Estimates (Lambda Labs H100 @ $2.49/hr)

| Task | Time | Cost |
|------|------|------|
| Mid-training (d34) | ~5.5 hours | ~$14 |
| SFT (d34) | ~30-60 min | ~$1-2 |
| Testing | ~10 min | ~$0.50 |
| **Total** | ~6.5 hours | **~$16** |

## Uploading Models to HuggingFace

Upload directly from the cloud GPU (much faster than downloading locally first):

```bash
# Login to HuggingFace
huggingface-cli login

# Create repo
huggingface-cli repo create nanochat-d34-sft --type model

# Upload from ~/.cache/nanochat/
cd ~/.cache/nanochat

hf upload <username>/nanochat-d34-sft tokenizer/ tokenizer/ --repo-type model
hf upload <username>/nanochat-d34-sft chatsft_checkpoints/d34/model_000700.pt model_000700.pt --repo-type model
hf upload <username>/nanochat-d34-sft chatsft_checkpoints/d34/meta_000700.json meta_000700.json --repo-type model

# Optional: upload mid-training checkpoint
hf upload <username>/nanochat-d34-sft mid_checkpoints/d34/model_000813.pt mid_model_000813.pt --repo-type model
hf upload <username>/nanochat-d34-sft mid_checkpoints/d34/meta_000813.json mid_meta_000813.json --repo-type model
```

## Downloading from HuggingFace

```bash
huggingface-cli download victoremnm/nanochat-d34-sft --local-dir ~/nanochat-d34-sft

# Setup directories
mkdir -p ~/.cache/nanochat/{tokenizer,chatsft_checkpoints/d34}
cp ~/nanochat-d34-sft/tokenizer/* ~/.cache/nanochat/tokenizer/
cp ~/nanochat-d34-sft/model_000700.pt ~/.cache/nanochat/chatsft_checkpoints/d34/
cp ~/nanochat-d34-sft/meta_000700.json ~/.cache/nanochat/chatsft_checkpoints/d34/
```

## Links

- Original repo: https://github.com/karpathy/nanochat
- Fork: https://github.com/victoremnm/nanochat
- PR with fixes: https://github.com/victoremnm/nanochat/pull/1
- Pretrained d34: https://huggingface.co/karpathy/nanochat-d34
- Trained SFT model: https://huggingface.co/victoremnm/nanochat-d34-sft
  - step 700: Base SFT (MMLU 42.6%, ARC-Easy 72%)
  - step 719: Custom SQL/analytics SFT (MMLU 41.99%, ARC-Easy 73.93%)
- WandB dashboard: https://wandb.ai/victoremnm-victor-em-llc/nanochat-sft
