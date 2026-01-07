# NanoChat Training Progress Log

A living document tracking learnings, decisions, and progress for our NanoChat deployment.

---

## Project Overview

**Goal:** Train and deploy a custom NanoChat model fine-tuned on our SQL/analytics data for internal use.

**Model:** nanochat-d34 (2.2B parameters)

**HuggingFace:** https://huggingface.co/victoremnm/nanochat-d34-sft

---

## Training History

### Session 1: Initial Training (Jan 2025)

**Infrastructure:** Lambda Labs H100 80GB @ $2.49/hr

| Stage | Duration | Cost | Output |
|-------|----------|------|--------|
| Mid-training | ~5.5 hours | ~$14 | model_000813.pt |
| SFT (base) | ~30 min | ~$1-2 | model_000700.pt |
| **Total** | ~6.5 hours | ~$16 | |

**Results:**
- MMLU: 42.6%
- ARC-Easy: 72%
- Chat quality: Coherent conversations, basic math, working code

### Session 2: Custom SQL/Analytics Fine-tuning (Jan 6-7, 2026)

**Attempt 1 - Diluted Training (629 examples)**
- Custom data mixed with 23K base examples (~2.7% custom)
- Only 19 training steps on custom data
- **Result:** Model still hallucinated table/column names

| Metric | Score |
|--------|-------|
| MMLU | 41.99% |
| ARC-Easy | 73.93% |
| SQL Quality | Poor - wrong table names |

**Attempt 2 - Custom-Only Training (1,092 examples)**
- Added `--custom_only` flag to train ONLY on custom data
- Expanded dataset from 629 → 1,092 examples
- 34 training steps (1 epoch)

**Custom Data Sources (1,092 total):**
- Clickhouse queries (various batches)
- BigQuery table definitions
- BonkBot analytics
- Pum3 data
- Geyser-geezer metrics
- Web-terminal documentation

| Metric | Score |
|--------|-------|
| MMLU | 41.80% |
| ARC-Easy | 71.19% |
| SQL Quality | Better - correct table names, wrong SQL dialect |

**Key Finding:** Model now uses correct table names (swap_events) but outputs SQL Server syntax (`SELECT TOP 10`) instead of Clickhouse (`LIMIT 10`). Needs more epochs or dialect-specific training.

### Session 3: Environment Crisis & Recovery (Jan 7, 2026)

**Problem:** Deleted uv.lock causing cascading dependency failures
- PyTorch 2.9+ requires cuDNN 9
- Lambda had cuDNN 8 in system, cuDNN 9 only in tensorflow package
- Multiple failed attempts with sed patching, system Python, new venvs

**Solution:**
1. `git checkout -- uv.lock` to restore
2. `rm uv.lock` and `uv lock` to regenerate fresh
3. `uv sync --extra gpu` - this time pulled `nvidia-cudnn-cu12==9.10.2.21`
4. Environment works with PyTorch 2.9.1+cu128

---

## Key Learnings

### Training Pipeline
1. **Never skip mid-training** - Base model outputs gibberish without it
2. **Use `--device_batch_size=4`** for d34 on single H100 (default 32 causes OOM)
3. **Always specify `--step`** when loading models
4. **Put pretrained model in `base_checkpoints/`** not `chatsft_checkpoints/`

### Infrastructure
1. **Upload to HuggingFace from server** - Much faster than downloading locally (100MB/s vs 130KB/s)
2. **Use SSH tunneling** for web UI - Cloud providers block port 8000
3. **Use tmux/nohup for long training runs** - Prevents losing output on disconnect
4. **Use `hf upload` not `huggingface-cli`** - Different command on newer versions

### Dependency Management (CRITICAL)
1. **NEVER delete uv.lock casually** - Can cause cascading dependency hell
2. **If uv.lock corrupted:** `rm uv.lock && uv lock && uv sync --extra gpu`
3. **PyTorch 2.9+ needs cuDNN 9** - uv should pull `nvidia-cudnn-cu12` automatically
4. **If cuDNN missing:** Check if uv pulled the nvidia-cudnn-cu12 package

### Custom Training
1. **Diluted training doesn't work** - 629 examples in 23K = only 2.7%, not enough
2. **Use `--custom_only` flag** - Trains ONLY on your data, no base mixture
3. **More epochs needed** - 1 epoch (34 steps) learns table names but not SQL dialect
4. **SQL dialect matters** - Need Clickhouse-specific examples, not generic SQL

### Data Preparation
1. **NanoChat JSONL format:** `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`
2. **NOT:** `{"messages": [...]}`  - Needs conversion
3. **Validation:** Roles must alternate user/assistant starting with user

### Cost Optimization
- Mid-training is the expensive part (~$14 for d34)
- SFT is cheap (~$1-2 per run)
- Can iterate on SFT quickly without re-running mid-training

---

## Issues Encountered & Solutions

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'nanochat'` | Add `[build-system]` to pyproject.toml |
| OOM during mid-training | Use `--device_batch_size=4` |
| Model outputs gibberish | Ensure mid-training was done, check `--step` |
| Port 8000 blocked | SSH tunnel: `ssh -L 8000:localhost:8000` |
| Slow local download | Upload directly from server to HuggingFace |
| Wrong JSONL format | Convert `{"messages": [...]}` to `[...]` |
| `uv.lock` parse error (matplotlib-inline) | Delete uv.lock and regenerate: `rm uv.lock && uv lock` |
| `libcudnn.so.9 not found` | Ensure `uv sync --extra gpu` pulls nvidia-cudnn-cu12 |
| Custom data diluted in base training | Use `--custom_only` flag |
| Model uses wrong SQL dialect | Need more epochs + dialect-specific training data |
| `huggingface-cli` not found | Use `hf` command instead (newer API) |
| Git push fails on Lambda | Need to set up SSH keys or use token auth |

---

## Data Sources

### Current Training Data
- **Clickhouse:** Production analyst SQL queries
- **BigQuery:** Table definitions and common patterns
- **API Documentation:** Web terminal endpoints, WebSocket topics
- **Dataform SQL:** Analytics transformation logic

### Future Data (Planned)
- [ ] Additional query logs
- [ ] Internal documentation
- [ ] Code repositories
- [ ] Support tickets / Q&A

---

## Model Versions

All models available at: https://huggingface.co/victoremnm/nanochat-d34-sft

| File | Step | Training Data | MMLU | ARC-Easy | Notes |
|------|------|---------------|------|----------|-------|
| model_000700.pt | 700 | Base SFT (23K) | 42.6% | 72% | Initial chat model |
| model_000719.pt | 719 | Base + Custom (23K + 629) | 41.99% | 73.93% | Diluted - doesn't help |
| custom_only_model_000033.pt | 33 | Custom only (1,092) | 41.80% | 71.19% | Better domain knowledge |
| mid_model_000813.pt | 813 | Mid-training | - | - | Required base for SFT |

**Training Data:** `training_data/combined_training_nanochat.jsonl` (1,092 examples)

---

## Deployment Plan

### Phase 1: Internal Testing
- [ ] Deploy on cloud GPU for team testing
- [ ] Gather feedback on SQL query quality
- [ ] Identify gaps in training data

### Phase 2: Organization Deployment
- [ ] Set up persistent hosting (see DEPLOYMENT.md)
- [ ] Add authentication
- [ ] Create usage documentation
- [ ] Monitor usage and costs

### Phase 3: Iteration
- [ ] Collect user queries for future training
- [ ] Fine-tune on feedback
- [ ] Expand to more data sources

---

## Next Steps

### Immediate
- [x] ~~Complete custom SQL SFT training~~ (Done - custom_only_model_000033.pt)
- [x] ~~Upload new checkpoint to HuggingFace~~ (Done)
- [ ] Run more epochs for better SQL dialect learning
- [ ] Add more Clickhouse-specific examples (LIMIT vs TOP, etc.)

### Short-term
- [ ] Push analyst-ai to GitHub private repo
- [ ] Test model with real analyst queries
- [ ] Gather feedback on SQL quality
- [ ] Decide on Lambda termination (currently running @ $2.49/hr)

### Long-term
- [ ] Set up persistent deployment infrastructure
- [ ] Create internal usage documentation
- [ ] Build query collection pipeline for continuous training

---

## Lambda Status

**Current Instance:** 209.20.159.9 (H100 80GB)
**Cost:** $2.49/hr (~$60/day)
**Status:** Running with working environment (PyTorch 2.9.1+cu128)

**To resume work:**
```bash
ssh ubuntu@209.20.159.9
cd ~/nanochat
export PATH=$HOME/.local/bin:$PATH
uv run python -m scripts.chat_sft --help
```

---

## Resources

- **Fork:** https://github.com/victoremnm/nanochat
- **HuggingFace:** https://huggingface.co/victoremnm/nanochat-d34-sft
- **Original repo:** https://github.com/karpathy/nanochat
- **Setup guide:** [docs/SETUP_GUIDE.md](./SETUP_GUIDE.md)
- **Deployment guide:** [docs/DEPLOYMENT.md](./DEPLOYMENT.md)

---

*Last updated: January 7, 2026*
