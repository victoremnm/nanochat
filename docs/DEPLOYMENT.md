# NanoChat Deployment Guide

Options for hosting NanoChat for your organization.

---

## Quick Comparison

| Option | Cost | Setup Time | Best For |
|--------|------|------------|----------|
| [Modal](#1-modal-serverless) | ~$0.001/query | 30 min | Low-medium traffic, pay-per-use |
| [RunPod](#2-runpod-dedicated) | ~$0.50-2/hr | 1 hour | Consistent traffic, full control |
| [HuggingFace Inference](#3-huggingface-inference-endpoints) | ~$1-4/hr | 15 min | Easiest setup, managed |
| [Self-hosted](#4-self-hosted-vps) | $200-800/mo | 2-4 hours | Full control, predictable costs |
| [Replicate](#5-replicate) | ~$0.002/query | 1 hour | Simple API, auto-scaling |

---

## 1. Modal (Serverless)

**Best for:** Variable traffic, pay only when used

### Setup

```bash
pip install modal
modal setup  # Login with GitHub/Google
```

Create `modal_app.py`:
```python
import modal

app = modal.App("nanochat")

# Create volume for model weights
volume = modal.Volume.from_name("nanochat-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "huggingface_hub", "uvicorn", "fastapi", "jinja2")
    .run_commands("pip install git+https://github.com/victoremnm/nanochat.git@feature/setup-guide-and-fixes")
)

@app.function(
    image=image,
    gpu="A100",  # or "T4" for cheaper
    volumes={"/root/.cache/nanochat": volume},
    timeout=300,
)
@modal.web_endpoint(method="POST")
def chat(request: dict):
    from nanochat.checkpoint_manager import load_model
    from nanochat.engine import Engine

    model, tokenizer, meta = load_model("sft", "cuda", model_tag="d34", step=700)
    engine = Engine(model, tokenizer)

    messages = request.get("messages", [])
    response = engine.generate(messages, temperature=0.6)
    return {"response": response}

@app.function(image=image, gpu="A100", volumes={"/root/.cache/nanochat": volume})
def download_model():
    """Run once to download model to volume"""
    import subprocess
    subprocess.run([
        "huggingface-cli", "download",
        "victoremnm/nanochat-d34-sft",
        "--local-dir", "/root/.cache/nanochat/chatsft_checkpoints/d34"
    ])
```

Deploy:
```bash
# First, download model to volume
modal run modal_app.py::download_model

# Deploy endpoint
modal deploy modal_app.py
```

### Cost
- A100: ~$0.001-0.002 per request (cold start ~30s)
- T4: ~$0.0005 per request (slower inference)

---

## 2. RunPod (Dedicated GPU)

**Best for:** Consistent traffic, need full control

### Setup

1. Create RunPod account at https://runpod.io
2. Deploy a GPU pod:
   - Template: PyTorch 2.0
   - GPU: RTX 4090 ($0.44/hr) or A100 ($1.99/hr)
   - Disk: 100GB

3. SSH in and setup:
```bash
# Install dependencies
pip install huggingface_hub uvicorn fastapi

# Clone repo
git clone https://github.com/victoremnm/nanochat.git
cd nanochat
pip install -e .

# Download model
huggingface-cli download victoremnm/nanochat-d34-sft --local-dir ~/model
mkdir -p ~/.cache/nanochat/chatsft_checkpoints/d34
cp ~/model/model_*.pt ~/.cache/nanochat/chatsft_checkpoints/d34/
cp ~/model/meta_*.json ~/.cache/nanochat/chatsft_checkpoints/d34/
mkdir -p ~/.cache/nanochat/tokenizer
cp ~/model/tokenizer/* ~/.cache/nanochat/tokenizer/

# Run server
python -m scripts.chat_web --source=sft --model-tag=d34 --step=700 --port=8000
```

4. Access via RunPod proxy URL or set up your own domain

### Cost
- RTX 4090: ~$0.44/hr (~$320/mo always-on)
- A100 40GB: ~$1.99/hr (~$1,400/mo always-on)

---

## 3. HuggingFace Inference Endpoints

**Best for:** Easiest setup, fully managed

### Setup

1. Go to https://huggingface.co/victoremnm/nanochat-d34-sft
2. Click "Deploy" → "Inference Endpoints"
3. Select:
   - Region: Closest to your users
   - Instance: GPU (A10G or A100)
   - Min replicas: 0 (scale to zero when idle)

4. Wait for deployment (~5-10 min)

5. Use the API:
```python
import requests

API_URL = "https://your-endpoint.huggingface.cloud"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

response = requests.post(API_URL, headers=headers, json={
    "inputs": "How do I query the top traders?",
})
print(response.json())
```

### Cost
- A10G: ~$1.30/hr
- A100: ~$4/hr
- Scale-to-zero: Only pay when processing requests

---

## 4. Self-Hosted (VPS with GPU)

**Best for:** Full control, predictable costs, data privacy

### Providers
- **Lambda Labs**: H100 $2.49/hr, A100 $1.29/hr
- **Vast.ai**: Variable pricing, spot instances
- **CoreWeave**: Enterprise, dedicated
- **Your own servers**: NVIDIA GPU required

### Setup Script

```bash
#!/bin/bash
# setup_server.sh - Run on a fresh GPU server

# Install system deps
sudo apt update && sudo apt install -y python3-pip nginx certbot

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# Clone and install
git clone https://github.com/victoremnm/nanochat.git
cd nanochat
uv sync

# Download model
uv pip install huggingface_hub
huggingface-cli download victoremnm/nanochat-d34-sft --local-dir ~/model

# Setup directories
mkdir -p ~/.cache/nanochat/{tokenizer,chatsft_checkpoints/d34}
cp ~/model/tokenizer/* ~/.cache/nanochat/tokenizer/
cp ~/model/model_*.pt ~/.cache/nanochat/chatsft_checkpoints/d34/
cp ~/model/meta_*.json ~/.cache/nanochat/chatsft_checkpoints/d34/

# Create systemd service
sudo tee /etc/systemd/system/nanochat.service << EOF
[Unit]
Description=NanoChat Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/nanochat
ExecStart=$HOME/.local/bin/uv run python -m scripts.chat_web --source=sft --model-tag=d34 --step=700 --port=8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable nanochat
sudo systemctl start nanochat

echo "NanoChat running on port 8000"
```

### Add NGINX + SSL

```bash
# Install certbot
sudo apt install -y certbot python3-certbot-nginx

# Create NGINX config
sudo tee /etc/nginx/sites-available/nanochat << EOF
server {
    listen 80;
    server_name chat.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/nanochat /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Get SSL cert
sudo certbot --nginx -d chat.yourdomain.com
```

### Add Basic Auth

```bash
# Create password file
sudo apt install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd admin

# Update NGINX config to add auth
# Add inside location block:
#     auth_basic "NanoChat";
#     auth_basic_user_file /etc/nginx/.htpasswd;
```

---

## 5. Replicate

**Best for:** Simple API, auto-scaling, no infrastructure

### Setup

1. Create account at https://replicate.com
2. Fork the model or create custom deployment
3. Use their API:

```python
import replicate

output = replicate.run(
    "victoremnm/nanochat-d34-sft:latest",
    input={"prompt": "How do I query the top traders?"}
)
print(output)
```

### Cost
- ~$0.002 per prediction
- No minimum, pay per use

---

## API Wrapper (Optional)

For any deployment, wrap with a simple FastAPI server for better control:

```python
# api_server.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os

app = FastAPI(title="NanoChat API")
security = HTTPBearer()

# Simple API key auth
API_KEYS = set(os.getenv("API_KEYS", "").split(","))

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

class ChatRequest(BaseModel):
    messages: list[dict]
    temperature: float = 0.6
    max_tokens: int = 1024

class ChatResponse(BaseModel):
    response: str
    model: str = "nanochat-d34-sft"

# Load model once at startup
engine = None

@app.on_event("startup")
async def load_model():
    global engine
    from nanochat.checkpoint_manager import load_model
    from nanochat.engine import Engine
    model, tokenizer, meta = load_model("sft", "cuda", model_tag="d34", step=700)
    engine = Engine(model, tokenizer)

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, token: str = Depends(verify_token)):
    response = engine.generate(request.messages, temperature=request.temperature)
    return ChatResponse(response=response)

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": engine is not None}
```

Run:
```bash
API_KEYS=key1,key2,key3 uvicorn api_server:app --host 0.0.0.0 --port=8000
```

---

## Recommendations

| Use Case | Recommended Option |
|----------|-------------------|
| Testing/demos | Modal (serverless) |
| Small team (<10 users) | RunPod or self-hosted |
| Medium team (10-50 users) | Self-hosted with NGINX |
| Enterprise | Self-hosted or HuggingFace Endpoints |
| Variable/unpredictable traffic | Modal or Replicate |

---

## Security Checklist

- [ ] API authentication (API keys or OAuth)
- [ ] HTTPS/SSL certificate
- [ ] Rate limiting
- [ ] Input validation/sanitization
- [ ] Logging and monitoring
- [ ] Regular model updates
- [ ] Backup strategy

---

*Last updated: January 2025*
