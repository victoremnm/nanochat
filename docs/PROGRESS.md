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

### Session 2: Custom SQL/Analytics Fine-tuning (Jan 2025)

**Custom Data:** 629 examples from:
- Clickhouse query logs (374 examples)
- Query log training batch 2 (88 examples)
- Documentation training (55 examples)
- Test docs (45 examples)
- Clickhouse training (35 examples)
- BigQuery training (32 examples)

**Training:** Continuing from step 700 with custom data mixed in

| Stage | Duration | Cost | Output |
|-------|----------|------|--------|
| SFT (custom) | TBD | TBD | TBD |

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
3. **Use tmux for long training runs** - Prevents losing output on disconnect

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

| Version | Step | Training Data | Notes |
|---------|------|---------------|-------|
| v1 | 700 | Base SFT (23K examples) | Initial chat model |
| v2 | TBD | Base + Custom SQL (23K + 629) | SQL/analytics focused |

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

- [ ] Complete custom SQL SFT training
- [ ] Test SQL query generation quality
- [ ] Upload new checkpoint to HuggingFace
- [ ] Set up deployment infrastructure
- [ ] Create internal documentation for users

---

## Resources

- **Fork:** https://github.com/victoremnm/nanochat
- **HuggingFace:** https://huggingface.co/victoremnm/nanochat-d34-sft
- **Original repo:** https://github.com/karpathy/nanochat
- **Setup guide:** [docs/SETUP_GUIDE.md](./SETUP_GUIDE.md)
- **Deployment guide:** [docs/DEPLOYMENT.md](./DEPLOYMENT.md)

---

*Last updated: January 2025*
