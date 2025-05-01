#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
MODEL_REPO='https://github.com/g-levine/pupper_v3_description'
PUPPER_MJX_REPO='https://github.com/Nate711/pupperv3-mjx.git'
PUPPER_MJX_COMMIT='0e6a40386b5e6bb8decffbb582c5c07211a5ce24'
PUPPER_DESC_COMMIT='218f24acf67ee2b2f8a80fa50ae6e4e3a3247468'
PUPPER_MJX_DIR='pupperv3_mjx'
PUPPER_DESC_DIR='pupper_v3_description'

# --- Installation Steps ---

echo ">>> Installing Python dependencies..."
pip install -q mujoco
pip install -q mujoco-mjx
pip install -q brax
echo ">>> Python dependencies installed."

echo ">>> Setting up pupperv3-mjx repository..."
# Remove the directory if it exists to ensure a clean clone
if [ -d "$PUPPER_MJX_DIR" ]; then
  echo "Removing existing $PUPPER_MJX_DIR directory..."
  rm -rf "$PUPPER_MJX_DIR"
fi
# Clone the repository
git clone "$PUPPER_MJX_REPO" "$PUPPER_MJX_DIR"
# Change into the directory, checkout the specific commit, and install the package
cd "$PUPPER_MJX_DIR"
echo "Checking out commit $PUPPER_MJX_COMMIT..."
git checkout "$PUPPER_MJX_COMMIT"
echo "Installing pupperv3_mjx package..."
pip install -q .
# Go back to the parent directory
cd ..
echo ">>> pupperv3-mjx setup complete."

echo ">>> Setting up pupper_v3_description repository..."
# Remove the directory if it exists
if [ -d "$PUPPER_DESC_DIR" ]; then
  echo "Removing existing $PUPPER_DESC_DIR directory..."
  rm -rf "$PUPPER_DESC_DIR"
fi
# Clone the repository
git clone "$MODEL_REPO" "$PUPPER_DESC_DIR"
# Change into the directory and checkout the specific commit
cd "$PUPPER_DESC_DIR"
echo "Checking out commit $PUPPER_DESC_COMMIT..."
git checkout "$PUPPER_DESC_COMMIT"
# Go back to the parent directory
cd ..
echo ">>> pupper_v3_description setup complete."

echo ">>> All setup steps finished successfully!"

