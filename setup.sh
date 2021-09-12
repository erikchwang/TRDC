ROOT=$(dirname $(realpath ${0}))
mkdir ${ROOT}/anaconda ${ROOT}/transformers ${ROOT}/task ${ROOT}/dataset ${ROOT}/checkpoint

wget -P ${ROOT} https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
wget -P ${ROOT} https://hrcdn.net/s3_pub/istreet-assets/DiqN0KqXi6XE2BADA_xrxg/TRDCdata.zip

sh ${ROOT}/Miniconda3-latest-Linux-x86_64.sh -b -f -p ${ROOT}/anaconda
source ${ROOT}/anaconda/bin/activate ${ROOT}/anaconda
conda install -y python=3.6
pip install -f https://download.pytorch.org/whl/torch_stable.html torch==1.6.0+cu101
pip install conda-pack==0.5.0 transformers==3.1.0

python -c "
import transformers;
transformers.AutoConfig.from_pretrained('bert-base-uncased').save_pretrained('${ROOT}/transformers');
transformers.AutoTokenizer.from_pretrained('bert-base-uncased').save_pretrained('${ROOT}/transformers');
transformers.AutoModel.from_pretrained('bert-base-uncased').save_pretrained('${ROOT}/transformers')
"

unzip -j ${ROOT}/TRDCdata.zip -d ${ROOT}/task

conda pack -o ${ROOT}/anaconda.tar.gz
conda deactivate
rm -rf ${ROOT}/anaconda
mkdir ${ROOT}/anaconda
tar -xzvf ${ROOT}/anaconda.tar.gz -C ${ROOT}/anaconda
rm ${ROOT}/anaconda.tar.gz

rm ${ROOT}/Miniconda3-latest-Linux-x86_64.sh
rm ${ROOT}/TRDCdata.zip
