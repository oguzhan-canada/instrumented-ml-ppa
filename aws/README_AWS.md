# AWS Cloud Deployment Guide

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Local Machine                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  1. terraform init/apply  →  VPC + S3 + IAM             │   │
│  │  2. aws s3 sync           →  Upload code + scripts      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
          ┌─────────────────┐  ┌─────────────────┐
          │  c6i.8xlarge    │  │  g4dn.xlarge    │
          │  (Spot, CPU)    │  │  (Spot, T4 GPU) │
          │  DLAMI Ubuntu   │  │  DLAMI Ubuntu   │
          │  + OpenROAD     │  │  + OpenROAD     │
          │                 │  │                 │
          │  Stage 1: Data  │  │  Stage 3: Train │
          │  Stage 2: Feat  │  │  Stage 4: Optim │
          │                 │  │  Stage 5: Eval  │
          └────────┬────────┘  └────────┬────────┘
                   │                    │
                   └────────┬───────────┘
                            ▼
                   ┌─────────────────┐
                   │   S3 Bucket     │
                   │                 │
                   │  code/          │
                   │  features/      │
                   │  data/          │
                   │  models/        │
                   │  results/       │
                   └─────────────────┘
```

**No Docker required.** Both instances use the AWS Deep Learning AMI (Ubuntu 22.04)
which comes with Python 3.10, PyTorch, and CUDA pre-installed. OpenROAD is built
from source during bootstrap (~15 min). This eliminates the 6.5 GB Docker image
build/push overhead.

## Cost Estimate

| Phase | Instance | Spot Rate | Hours | Cost |
|-------|----------|-----------|-------|------|
| Stage 1-2 | c6i.8xlarge | ~$0.45/hr | 8 | ~$3.60 |
| Stage 3-5 | g4dn.xlarge | ~$0.22/hr | 12 | ~$2.64 |
| S3 + transfer | — | — | — | ~$0.75 |
| **Total** | | | | **~$7** |

---

## Step-by-Step Deployment

### Prerequisites
- AWS CLI configured (`aws configure`)
- Terraform >= 1.5.0 installed
- An EC2 key pair created in your target region

### Step 0: Set Budget Alerts (before any terraform apply)

Create a **monthly project budget** ($25 ceiling, alerts at $10 and $20):
```bash
aws budgets create-budget \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget '{"BudgetName":"ppa-project","BudgetLimit":{"Amount":"25","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST"}' \
  --notifications-with-subscribers '[
    {"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":10},"Subscribers":[{"SubscriptionType":"EMAIL","Address":"oguzhantekin@gmail.com"}]},
    {"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":20},"Subscribers":[{"SubscriptionType":"EMAIL","Address":"oguzhantekin@gmail.com"}]}
  ]'
```

PowerShell equivalent:
```powershell
$acct = aws sts get-caller-identity --query Account --output text
aws budgets create-budget --account-id $acct `
  --budget '{"BudgetName":"ppa-project","BudgetLimit":{"Amount":"25","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST"}' `
  --notifications-with-subscribers '[{"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":10},"Subscribers":[{"SubscriptionType":"EMAIL","Address":"oguzhantekin@gmail.com"}]},{"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":20},"Subscribers":[{"SubscriptionType":"EMAIL","Address":"oguzhantekin@gmail.com"}]}]'
```

Create a **daily anomaly alert** ($5/day ceiling, alerts at 80% = $4):
```bash
aws budgets create-budget \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget '{"BudgetName":"ppa-anomaly","BudgetLimit":{"Amount":"5","Unit":"USD"},"TimeUnit":"DAILY","BudgetType":"COST"}' \
  --notifications-with-subscribers '[
    {"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":80},"Subscribers":[{"SubscriptionType":"EMAIL","Address":"oguzhantekin@gmail.com"}]}
  ]'
```

PowerShell equivalent:
```powershell
aws budgets create-budget --account-id $acct `
  --budget '{"BudgetName":"ppa-anomaly","BudgetLimit":{"Amount":"5","Unit":"USD"},"TimeUnit":"DAILY","BudgetType":"COST"}' `
  --notifications-with-subscribers '[{"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":80},"Subscribers":[{"SubscriptionType":"EMAIL","Address":"oguzhantekin@gmail.com"}]}]'
```

> **Why these thresholds?** Expected total spend is ~$7. The $10 alert catches a forgotten instance within hours. The $20 alert is the "stop everything" signal. The daily $4 alert catches an overnight instance before it reaches $5/day. The main risk is the GPU Spot instance during RL training — if reclaimed and relaunched without spot_watcher.sh catching it, a second instance could run.

### Step 1: Provision Infrastructure

```bash
cd aws/terraform

# Copy and edit the example config
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars — set key_pair_name and allowed_ssh_cidr

terraform init
terraform apply
```

This creates: VPC, S3 bucket, IAM role, launch templates.
No instances are launched yet (`launch_cpu = false`, `launch_gpu = false`).

### Step 2: Upload Code to S3

```bash
# From the project root — upload code + orchestration scripts
bash aws/scripts/upload_code.sh
```

Or manually with PowerShell:
```powershell
$bucket = (cd aws/terraform; terraform output -raw s3_bucket)
aws s3 sync . "s3://$bucket/code/" --exclude ".git/*" --exclude "aws/*" --exclude "data/raw/*"
aws s3 sync aws/scripts "s3://$bucket/scripts/"
```

### Step 3: Launch CPU Instance (Stages 1-2)

```bash
cd aws/terraform
terraform apply -var="launch_cpu=true"
```

Wait ~15 minutes for bootstrap (OpenROAD build from source), then SSH in:
```bash
CPU_IP=$(terraform output -raw cpu_instance_public_ip)
ssh -i ~/.ssh/ppa-key.pem ubuntu@$CPU_IP
```

On the instance:
```bash
# Check bootstrap completed
tail -20 /var/log/bootstrap.log

# Run Stages 1-2
source /opt/ppa/env.sh
tmux new -s ppa    # Use tmux in case SSH disconnects
bash /opt/ppa/scripts/run_stages_1_2.sh
```

**Expected time:** ~8 hours (OpenABC-D download + 55 OpenROAD runs).

After completion, data is synced to S3 automatically. Terminate the CPU instance:
```bash
cd aws/terraform
terraform apply -var="launch_cpu=false"
```

### Step 4: Launch GPU Instance (Stages 3-5)

```bash
cd aws/terraform
terraform apply -var="launch_gpu=true"
```

Wait ~15 minutes for bootstrap, then SSH in:
```bash
GPU_IP=$(terraform output -raw gpu_instance_public_ip)
ssh -i ~/.ssh/ppa-key.pem ubuntu@$GPU_IP
```

On the instance:
```bash
tail -20 /var/log/bootstrap.log

# Verify GPU
nvidia-smi
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run Stages 3-5
source /opt/ppa/env.sh
tmux new -s ppa
bash /opt/ppa/scripts/run_stages_3_5.sh
```

**Expected time:** ~12 hours (training + BO + RL + eval).

After completion:
```bash
cd aws/terraform
terraform apply -var="launch_gpu=false"
```

### Step 5: Download Results

```bash
S3_BUCKET=$(cd aws/terraform && terraform output -raw s3_bucket)

# Download everything
aws s3 sync s3://$S3_BUCKET/ppa-run/results/ results/
aws s3 sync s3://$S3_BUCKET/ppa-run/models/ models/
```

### Step 6: Cleanup (when done)

```bash
cd aws/terraform
terraform destroy    # Removes ALL resources including S3 bucket
```

---

## Spot Interruption Recovery

The deployment includes automatic Spot interruption handling:

1. **spot_watcher.sh** runs in background, polls IMDSv2 every 5 seconds
2. On 2-minute warning: emergency syncs models/results/features to S3
3. **run_openroad.py** has built-in resume — skips completed runs
4. **train_gat.py** saves per-fold checkpoints
5. S3 sync happens after every major step (not just at end)

**To resume after interruption:**
```bash
# Re-launch the same instance type
terraform apply -var="launch_cpu=true"   # or launch_gpu=true

# SSH in and re-run the same script
bash /opt/ppa/scripts/run_stages_1_2.sh --resume
# or
bash /opt/ppa/scripts/run_stages_3_5.sh
```

Scripts will sync from S3 and skip already-completed work.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Spot capacity unavailable | Change `instance_type` in terraform.tfvars to next fallback |
| Bootstrap log shows errors | `cat /var/log/bootstrap.log` — check OpenROAD build or pip install |
| OpenROAD build fails | Ensure 200GB EBS for CPU instance (build needs ~10GB tmp space) |
| S3 sync fails | Verify IAM role has `s3:PutObject` and bucket policy allows it |
| `nvidia-smi` not found | AMI may not have GPU drivers — verify using Deep Learning AMI |
| Python package missing | `pip install <package>` — DLAMI has pip pre-configured |

---

## File Structure

```
aws/
├── terraform/
│   ├── main.tf                 # VPC, SG, IAM, S3, Launch Templates, Spot
│   ├── variables.tf            # All configurable parameters
│   ├── outputs.tf              # IPs, bucket name, AMI ID
│   └── terraform.tfvars.example
├── docker/                     # Retained for reference (not used in deployment)
│   ├── Dockerfile.cpu          # CPU image definition
│   └── Dockerfile.gpu          # GPU image definition
├── scripts/
│   ├── upload_code.sh          # Upload project code to S3
│   ├── bootstrap_cpu.sh        # EC2 userdata: OpenROAD + pip install
│   ├── bootstrap_gpu.sh        # EC2 userdata: verify GPU + pip install
│   ├── run_stages_1_2.sh       # Stage 1-2 orchestration (direct Python)
│   ├── run_stages_3_5.sh       # Stage 3-5 orchestration (direct Python)
│   ├── spot_watcher.sh         # Spot interruption monitor
│   ├── sync_to_s3.sh           # Upload artifacts to S3
│   └── sync_from_s3.sh         # Download artifacts from S3
└── README_AWS.md               # This file
```
