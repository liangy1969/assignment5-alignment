#!/bin/bash
# Configure passwordless sudo for current user

USER=$(whoami)
echo "Configuring passwordless sudo for: $USER"

# Add sudoers entry
echo "$USER ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/$USER-nopass > /dev/null

# Verify
if sudo -n true 2>/dev/null; then
    echo "✅ Passwordless sudo configured successfully!"
else
    echo "❌ Configuration failed"
    exit 1
fi
