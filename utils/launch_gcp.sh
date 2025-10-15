#!/bin/bash
set -euo pipefail

# ----------------- Config -----------------
PROJECT="testml-467904"
ZONE="us-central1-f"
REGION="us-central1"
INSTANCE_NAME="instance-$(date +%Y%m%d-%H%M%S)"
MACHINE="n1-standard-1"
ACCEL="nvidia-tesla-t4"
DISK_SIZE_GB=100
IMAGE="projects/ml-images/global/images/c1-deeplearning-tf-2-15-cu122-v20240922-debian-11-py310"
SUBNET="default"

# Option A: lock SSH to your IP (recommended)
YOUR_IP_CIDR="${YOUR_IP_CIDR:-}"

# Option B: use IAP instead of opening port 22 (no firewall rule needed)
USE_IAP="${USE_IAP:-false}"

# The tag our firewall rule will target
FW_TAG="allow-ssh-from-ip"

# ----------------- Preflight: Firewall/IAP -----------------
if [[ "$USE_IAP" != "true" ]]; then
  if [[ -z "${YOUR_IP_CIDR}" ]]; then
    echo "Detecting your public IP..."
    PUBIP=$(curl -fsS https://ifconfig.me || curl -fsS https://ipinfo.io/ip)
    YOUR_IP_CIDR="${PUBIP}/32"
  fi
  echo "Using SSH source range: ${YOUR_IP_CIDR}"

  if ! gcloud compute firewall-rules describe allow-ssh-from-ip --project="$PROJECT" >/dev/null 2>&1; then
    echo "Creating firewall rule allow-ssh-from-ip..."
    gcloud compute firewall-rules create allow-ssh-from-ip \
      --project="$PROJECT" \
      --allow=tcp:22 \
      --direction=INGRESS \
      --source-ranges="$YOUR_IP_CIDR" \
      --target-tags="$FW_TAG"
  else
    echo "Firewall rule allow-ssh-from-ip already exists."
  fi
fi

# ----------------- Create the instance -----------------
echo "Creating $INSTANCE_NAME in $ZONE..."

TAGS_ARG=""
if [[ "$USE_IAP" != "true" ]]; then
  TAGS_ARG="--tags=$FW_TAG"
fi

gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet="$SUBNET" \
  --maintenance-policy=TERMINATE \
  --provisioning-model=STANDARD \
  --service-account=830368009513-compute@developer.gserviceaccount.com \
  --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
  --accelerator=count=1,type="$ACCEL" \
  --create-disk=auto-delete=yes,boot=yes,device-name="$INSTANCE_NAME",disk-resource-policy=projects/"$PROJECT"/regions/"$REGION"/resourcePolicies/default-schedule-1,image="$IMAGE",mode=rw,size="$DISK_SIZE_GB",type=pd-balanced \
  --no-shielded-secure-boot \
  --shielded-vtpm \
  --shielded-integrity-monitoring \
  --labels=goog-ops-agent-policy=v2-x86-template-1-4-0,goog-ec-src=vm_add-gcloud \
  --metadata=enable-osconfig=TRUE,enable-oslogin=FALSE \
  --reservation-affinity=any \
  $TAGS_ARG

# ----------------- ( #1 ) Wait for sshd: poll port 22 (non-IAP w/ external IP) -----------------
if [[ "$USE_IAP" != "true" ]]; then
  IP="$(gcloud compute instances describe "$INSTANCE_NAME" --project "$PROJECT" --zone "$ZONE" \
        --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"
  if [[ -n "${IP}" ]]; then
    echo "Public IP is ${IP}. Waiting for sshd on port 22..."
    # Clear stale host keys (harmless if not present)
    ssh-keygen -R "$IP" 2>/dev/null || true
    ssh-keygen -R "$INSTANCE_NAME" 2>/dev/null || true

    for i in {1..60}; do
      if nc -z "$IP" 22 2>/dev/null; then
        echo "Port 22 is open."
        break
      fi
      echo "Waiting for sshd… (${i}/60)"
      sleep 5
    done
  else
    echo "No external IP detected; skipping port-22 readiness check."
  fi
else
  echo "USE_IAP=true; skipping port-22 readiness check against a public IP."
fi

# ----------------- Single-shot SSH (no retry) -----------------
SSH_BASE=(gcloud --verbosity=debug compute ssh "$INSTANCE_NAME" --project="$PROJECT" --zone="$ZONE")
# If you use IAP, uncomment:
# SSH_BASE+=(--tunnel-through-iap)
# Suggested SSH flags for scripts (optional):
# SSH_BASE+=(-- -T -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o ConnectionAttempts=1 -o IdentitiesOnly=yes)

echo "Sleeping briefly before troubleshoot..."

# 1) Preflight once with --troubleshoot (auto-accept fixes); ignore its exit
yes | "${SSH_BASE[@]/%/}" --troubleshoot --command 'exit 0' || true

# 2) Give sshd a moment to come up after first-boot init
sleep 10

# 3) Real SSH (single attempt)
"${SSH_BASE[@]}" <<'REMOTE_CMDS'
set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA drivers already installed."
  nvidia-smi || true
else
  echo "Installing NVIDIA drivers..."
  if [ -x /opt/deeplearning/install-driver.sh ]; then
    yes | sudo /opt/deeplearning/install-driver.sh
  elif [ -x /opt/deeplearning/install-gpu-driver.sh ]; then
    yes | sudo /opt/deeplearning/install-gpu-driver.sh
  else
    export DEBIAN_FRONTEND=noninteractive
    sudo apt-get update
    sudo apt-get -y install nvidia-driver
  fi
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
fi

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get -y install build-essential pkg-config cmake curl git

if [ ! -d "$HOME/can-t" ]; then
  git clone https://github.com/TuckerBMorgan/can-t "$HOME/can-t"
fi
cd "$HOME/can-t"

# Rust install: fully non-interactive (equivalent to choosing '1')
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile default --default-toolchain stable --no-modify-path
. "$HOME/.cargo/env"

rustc --version && cargo --version
cargo build
REMOTE_CMDS
