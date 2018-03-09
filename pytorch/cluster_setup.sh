# Link CUDA
export CONDA_HOME=/deep/group/packages/miniconda3
export CUDA_HOME=/usr/local/cuda-8.0
export PATH=${CUDA_HOME}/bin:${PATH}
export PATH=${CONDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

source activate imaging
