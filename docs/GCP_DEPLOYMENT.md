# NanoChat GCP Deployment Guide

Deploy NanoChat on Google Cloud with Google Workspace authentication - your team can access immediately with their work accounts.

---

## Architecture

```
Users (Google Workspace accounts)
         ↓
   Cloud Load Balancer (HTTPS)
         ↓
   Identity-Aware Proxy (IAP) ← Google Workspace Auth
         ↓
   Compute Engine (GPU VM)
         ↓
   NanoChat Server
```

**Benefits:**
- No separate auth system - uses existing Google accounts
- Single URL for your team
- Managed SSL certificate
- Access control via Google Groups

---

## Quick Start (~30 min)

### 1. Create GPU VM

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Create VM with NVIDIA T4 (cheaper) or A100 (faster)
gcloud compute instances create nanochat-server \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --maintenance-policy=TERMINATE \
  --tags=nanochat-server

# Or for A100 (faster, more expensive):
# --machine-type=a2-highgpu-1g \
# --accelerator=type=nvidia-tesla-a100,count=1
```

### 2. SSH and Setup

```bash
gcloud compute ssh nanochat-server --zone=us-central1-a

# On the VM:
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# Clone repo
git clone https://github.com/victoremnm/nanochat.git
cd nanochat
git checkout feature/setup-guide-and-fixes
uv sync

# Download model from HuggingFace
uv pip install huggingface_hub
huggingface-cli download victoremnm/nanochat-d34-sft --local-dir ~/model

# Setup directories
mkdir -p ~/.cache/nanochat/{tokenizer,chatsft_checkpoints/d34}
cp ~/model/tokenizer/* ~/.cache/nanochat/tokenizer/
cp ~/model/model_*.pt ~/.cache/nanochat/chatsft_checkpoints/d34/
cp ~/model/meta_*.json ~/.cache/nanochat/chatsft_checkpoints/d34/

# Test it works
uv run python -m scripts.chat_web --source=sft --model-tag=d34 --step=700 --port=8080 &
curl localhost:8080  # Should return HTML
```

### 3. Create Systemd Service

```bash
sudo tee /etc/systemd/system/nanochat.service << 'EOF'
[Unit]
Description=NanoChat Server
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/nanochat
Environment="PATH=/home/YOUR_USERNAME/.local/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/YOUR_USERNAME/.local/bin/uv run python -m scripts.chat_web --source=sft --model-tag=d34 --step=700 --port=8080
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Replace YOUR_USERNAME
sudo sed -i "s/YOUR_USERNAME/$USER/g" /etc/systemd/system/nanochat.service

sudo systemctl daemon-reload
sudo systemctl enable nanochat
sudo systemctl start nanochat
sudo systemctl status nanochat
```

### 4. Setup Identity-Aware Proxy (IAP)

This lets your team log in with Google Workspace accounts.

```bash
# Exit SSH, run these locally

# Enable APIs
gcloud services enable compute.googleapis.com
gcloud services enable iap.googleapis.com

# Create firewall rule for IAP
gcloud compute firewall-rules create allow-iap-to-nanochat \
  --direction=INGRESS \
  --action=allow \
  --rules=tcp:8080 \
  --source-ranges=35.235.240.0/20 \
  --target-tags=nanochat-server

# Create health check
gcloud compute health-checks create http nanochat-health \
  --port=8080 \
  --request-path=/

# Create backend service
gcloud compute backend-services create nanochat-backend \
  --protocol=HTTP \
  --port-name=http \
  --health-checks=nanochat-health \
  --global

# Create instance group
gcloud compute instance-groups unmanaged create nanochat-group \
  --zone=us-central1-a

gcloud compute instance-groups unmanaged add-instances nanochat-group \
  --zone=us-central1-a \
  --instances=nanochat-server

gcloud compute instance-groups unmanaged set-named-ports nanochat-group \
  --zone=us-central1-a \
  --named-ports=http:8080

# Add instance group to backend
gcloud compute backend-services add-backend nanochat-backend \
  --instance-group=nanochat-group \
  --instance-group-zone=us-central1-a \
  --global

# Create URL map
gcloud compute url-maps create nanochat-urlmap \
  --default-service=nanochat-backend

# Create managed SSL cert (replace with your domain)
gcloud compute ssl-certificates create nanochat-cert \
  --domains=chat.yourdomain.com \
  --global

# Create HTTPS proxy
gcloud compute target-https-proxies create nanochat-https-proxy \
  --url-map=nanochat-urlmap \
  --ssl-certificates=nanochat-cert

# Create forwarding rule
gcloud compute forwarding-rules create nanochat-https-rule \
  --global \
  --target-https-proxy=nanochat-https-proxy \
  --ports=443

# Get the IP address
gcloud compute forwarding-rules describe nanochat-https-rule --global --format="get(IPAddress)"
```

### 5. Configure DNS

Point your domain to the IP address from the previous step:
```
chat.yourdomain.com  A  <IP_ADDRESS>
```

### 6. Enable IAP Authentication

```bash
# Enable IAP on backend
gcloud compute backend-services update nanochat-backend \
  --iap=enabled \
  --global

# Grant access to your team (Google Group or individual emails)
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="group:team@yourdomain.com" \
  --role="roles/iap.httpsResourceAccessor"

# Or individual user:
# --member="user:alice@yourdomain.com"
```

### 7. Configure OAuth Consent Screen

1. Go to https://console.cloud.google.com/apis/credentials/consent
2. Select "Internal" (for Workspace users only)
3. Fill in app name: "NanoChat"
4. Add your domain
5. Save

---

## Cost Estimate

| Component | Monthly Cost |
|-----------|-------------|
| T4 GPU VM (n1-standard-8) | ~$250-400 |
| A100 GPU VM (a2-highgpu-1g) | ~$1,200-1,800 |
| Load Balancer | ~$20 |
| Egress (depends on usage) | ~$10-50 |
| **Total (T4)** | **~$300-500/mo** |
| **Total (A100)** | **~$1,300-2,000/mo** |

**Cost optimization:**
- Use preemptible/spot VMs (60-80% cheaper, but can be stopped)
- Schedule VM to stop outside work hours
- Use T4 for testing, A100 for production

---

## Auto-Shutdown Script (Save Money)

Stop VM outside work hours:

```bash
# Create Cloud Scheduler job to stop VM at 8 PM
gcloud scheduler jobs create http stop-nanochat \
  --schedule="0 20 * * 1-5" \
  --uri="https://compute.googleapis.com/compute/v1/projects/YOUR_PROJECT/zones/us-central1-a/instances/nanochat-server/stop" \
  --http-method=POST \
  --oauth-service-account-email=YOUR_SERVICE_ACCOUNT@YOUR_PROJECT.iam.gserviceaccount.com \
  --time-zone="America/Los_Angeles"

# Start VM at 8 AM
gcloud scheduler jobs create http start-nanochat \
  --schedule="0 8 * * 1-5" \
  --uri="https://compute.googleapis.com/compute/v1/projects/YOUR_PROJECT/zones/us-central1-a/instances/nanochat-server/start" \
  --http-method=POST \
  --oauth-service-account-email=YOUR_SERVICE_ACCOUNT@YOUR_PROJECT.iam.gserviceaccount.com \
  --time-zone="America/Los_Angeles"
```

---

## Quick Alternative: Cloud Run (Simpler, No GPU)

If you have a smaller model or want simpler deployment:

```bash
# Build container
gcloud builds submit --tag gcr.io/YOUR_PROJECT/nanochat

# Deploy to Cloud Run
gcloud run deploy nanochat \
  --image=gcr.io/YOUR_PROJECT/nanochat \
  --platform=managed \
  --region=us-central1 \
  --memory=16Gi \
  --cpu=4 \
  --allow-unauthenticated=false

# Note: Cloud Run GPU support is in preview (L4 GPUs)
# For d34 model, you need the GPU option
```

---

## Terraform (Infrastructure as Code)

For reproducible deployments, see `terraform/gcp/` in the repo.

```hcl
# main.tf
resource "google_compute_instance" "nanochat" {
  name         = "nanochat-server"
  machine_type = "n1-standard-8"
  zone         = "us-central1-a"

  boot_disk {
    initialize_params {
      image = "deeplearning-platform-release/pytorch-latest-gpu"
      size  = 200
    }
  }

  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
  }

  network_interface {
    network = "default"
    access_config {}
  }

  tags = ["nanochat-server"]
}
```

---

## Troubleshooting

### VM won't start with GPU
```bash
# Check quota
gcloud compute regions describe us-central1 --format="get(quotas)"
# Look for NVIDIA_T4_GPUS or NVIDIA_A100_GPUS
# Request quota increase if needed
```

### IAP not working
```bash
# Check IAP is enabled
gcloud compute backend-services describe nanochat-backend --global

# Check user has access
gcloud projects get-iam-policy YOUR_PROJECT --filter="bindings.role:roles/iap.httpsResourceAccessor"
```

### Model loading slow
```bash
# Pre-load model on VM startup
# Add to /etc/rc.local or systemd ExecStartPre
curl localhost:8080/health  # Triggers model load
```

---

## Access URL

Once deployed, your team accesses:
```
https://chat.yourdomain.com
```

They'll see Google login → then the NanoChat interface.

---

*Last updated: January 2025*
