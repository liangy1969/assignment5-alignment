#!/bin/bash
# CUDA + GPU setup for WSL2 Ubuntu 24.04 - Official NVIDIA Installer

set -e

echo "🔧 Starting CUDA + GPU setup for WSL2..."

# Step 1: Update packages
echo "📦 Updating package manager..."
sudo apt update
sudo apt install -y wget

# Step 2: Install NVIDIA CUDA Toolkit from official repo
echo "🎯 Installing NVIDIA CUDA Toolkit (official)..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit

# Step 3: Verify CUDA installation
echo "✅ Verifying CUDA installation..."
nvidia-smi || echo "⚠️  nvidia-smi not available yet (may need WSL GPU driver on Windows side)"

# Step 4: Set up CUDA environment variables
echo "⚙️  Setting up CUDA environment variables..."
mkdir -p ~/.config/uv

cat > ~/.config/uv/uv.toml << 'EOF'
# UV Configuration for CUDA + GPU Support

[pip]
index-strategy = "unsafe-best-match"
prefer-prebuilt-wheels = true

[build]
allow-build-isolation = false

[env]
CUDA_HOME = "/usr/local/cuda"
CUDACXX = "/usr/local/cuda/bin/nvcc"
PATH = "/usr/local/cuda/bin:$PATH"
LD_LIBRARY_PATH = "/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
VLLM_USE_CUDA = "1"
FLASH_ATTENTION_2_ENABLED = "1"
TORCH_CUDA_ARCH_LIST = "8.0;8.6;8.9;9.0"
EOF

echo "✅ Created ~/.config/uv/uv.toml"

# Step 5: Update shell profile
echo "⚙️  Adding CUDA to shell profile..."
if ! grep -q "CUDA_HOME" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# CUDA Environment Variables
export CUDA_HOME="/usr/local/cuda"
export CUDACXX="/usr/local/cuda/bin/nvcc"
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
EOF
    source ~/.bashrc
fi

# Step 6: Install Python dependencies
echo "📚 Installing Python dependencies (this may take 10-15 minutes)..."
cd /mnt/e/projects/assignment5-alignment
uv sync

# Step 7: Verify PyTorch CUDA support
echo "🧪 Verifying CUDA support in PyTorch..."
python_check='import torch; cuda_avail = torch.cuda.is_available(); print(f"✅ CUDA available: {cuda_avail}"); print(f"Device: {torch.cuda.get_device_name() if cuda_avail else \"CPU\"}"); print(f"PyTorch version: {torch.__version__}")'
uv run python -c "$python_check"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "⚠️  IMPORTANT: If you see 'CUDA available: False', you need to:"
echo "   1. Check that your Windows host has NVIDIA drivers installed"
echo "   2. Ensure WSL GPU support is enabled in Windows (requires driver >= 560.35 for NVIDIA)"
echo "   3. Restart WSL and try again"
echo ""
echo "To run the benchmark:"
echo "  cd /mnt/e/projects/assignment5-alignment"
echo "  uv run python -m cs336_alignment.gsm_benchmark_script"
