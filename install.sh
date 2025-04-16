# Get path to the conda environment
CONDA_ENV_PATH=$(conda info --base)/envs/$(basename "$CONDA_PREFIX")

# Find CUDA toolkit path (adjust version if needed)
CUDA_HOME=$(find "$CONDA_ENV_PATH" -type d -path "*cudatoolkit*" | grep -E "cuda|cudatoolkit" | head -n 1)

# Fallback if above fails
if [ -z "$CUDA_HOME" ]; then
    CUDA_HOME="$CONDA_ENV_PATH"
fi

export CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "CUDA_HOME set to $CUDA_HOME"
echo "LD_LIBRARY_PATH set to $LD_LIBRARY_PATH"

pip3 install -r requirements.txt

pip3 install submodules/simple-knn
pip3 install submodules/fused-ssim
pip3 install submodules/diff-gaussian-rasterization
pip3 install -e ./submodules/slang-gaussian-rasterization
