#!/usr/bin/env bash
# Ubuntu 22.04/24.04 or WSL2 Ubuntu. Creates ~/pos/venv with the tested stack.
set -euo pipefail
sudo apt-get update
sudo apt-get install -y ffmpeg git git-annex python3-venv
# Python 3.11 (uv fetches it if the system python is different). Safe to re-run.
if [ ! -x ~/pos/venv/bin/python ]; then
  if command -v python3.11 >/dev/null; then
    python3.11 -m venv ~/pos/venv
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv python install 3.11
    uv venv --seed -p 3.11 ~/pos/venv
  fi
fi
source ~/pos/venv/bin/activate
python -m pip --version >/dev/null 2>&1 || python -m ensurepip --upgrade
python -m pip install --upgrade pip
# CUDA 12.1 wheels for the RTX 4070 (driver >= 530)
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r "$(dirname "$0")/../requirements.txt"
git config --global user.name  >/dev/null || git config --global user.name  "pos-user"
git config --global user.email >/dev/null || git config --global user.email "pos-user@example.com"
python - <<'PY'
import torch, transformers, inspect, himalaya
from transformers.models.vjepa2 import modeling_vjepa2 as m
assert torch.cuda.is_available(), "CUDA not visible to torch"
src = inspect.getsource(m.VJEPA2Predictor.forward)
assert "apply_masks(encoder_hidden_states, context_mask)" in src and "self.proj(hidden_states)" in src
assert "max_patch_num = target_mask[0].max() + 1" in inspect.getsource(m.VJEPA2PredictorEmbeddings.forward)
print("OK torch", torch.__version__, "| transformers", transformers.__version__, "| himalaya", himalaya.__version__,
      "| GPU", torch.cuda.get_device_name(0))
PY
