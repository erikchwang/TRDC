ROOT=$(dirname $(realpath ${0}))
SCALE=$(nvidia-smi --list-gpus | wc -l)
source ${ROOT}/anaconda/bin/activate ${ROOT}/anaconda

python ${ROOT}/preprocess.py 0
python ${ROOT}/preprocess.py 1

python ${ROOT}/construct.py

python -m torch.distributed.launch --use_env --nproc_per_node=${SCALE} ${ROOT}/optimize.py

python ${ROOT}/execute.py
