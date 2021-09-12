ROOT=$(dirname $(realpath ${0}))
SCALE=$(nvidia-smi --list-gpus | wc -l)
source ${ROOT}/anaconda/bin/activate ${ROOT}/anaconda

# preprocess the raw data to obtain the statistics
python ${ROOT}/preprocess.py 0

# preprocess the raw data to generate and split the dataset
python ${ROOT}/preprocess.py 1

# construct the model
python ${ROOT}/construct.py

# optimize the model on multiple GPUs in parallel
python -m torch.distributed.launch --use_env --nproc_per_node=${SCALE} ${ROOT}/optimize.py

# execute the trained model on the test set
python ${ROOT}/execute.py
