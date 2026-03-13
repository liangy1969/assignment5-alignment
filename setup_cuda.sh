#!/bin/bash
# CUDA + GPU setup script for WSL2 Ubuntu 24.04

set -e  # Exit on any error

echo "🔧 Starting CUDA + GPU setup for WSL2..."

# Step 1: Update package manager
echo "📦 Updating package manager..."
sudo apt update
sudo apt upgrade -y

# Step 2: Install NVIDIA CUDA Toolkit & Utils
echo "🎯 Installing NVIDIA CUDA Toolkit..."
sudo apt install -y nvidia-cuda-toolkit nvidia-utils

# Step 3: Verify installation
echo "✅ Verifying CUDA installation..."
nvidia-smi

# Step 4: Set up CUDA environment variables
echo "⚙️  Setting up CUDA environment variables..."
mkdir -p ~/.config/uv

# Already created by the assistant, but just in case
if [ ! -f ~/.config/uv/uv.toml ]; then
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
    echo "Created ~/.config/uv/uv.toml"
fi

# Step 5: Install Python dependencies
echo "📚 Installing Python dependencies (this may take a few minutes)..."
cd /mnt/e/projects/assignment5-alignment
uv sync --no-install-package flash-attn
uv sync

# Step 6: Verify PyTorch CUDA support
echo "🧪 Verifying CUDA support in PyTorch..."
uv run python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "🎉 Setup complete! CUDA is ready to use with vllm."
echo ""
echo "To run the benchmark:"
echo "  cd /mnt/e/projects/assignment5-alignment"
echo "  uv run <your-benchmark-script>"
