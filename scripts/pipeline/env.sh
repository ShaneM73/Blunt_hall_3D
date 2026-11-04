# source this from anywhere:  DATASET=2026-11-04-142939 . scripts/pipeline/env.sh
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export TVGO_RAW="$(ls -d $R/raw/${DATASET}*)" BH_WORK="$R/work/${DATASET}"
export PATH=$HOME/venvs/blunt3d/bin:/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 PYTHONWARNINGS=ignore
mkdir -p $BH_WORK/logs
