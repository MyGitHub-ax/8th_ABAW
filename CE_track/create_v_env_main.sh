#!/usr/bin/env bash

local="/data/liuran"
env="abaw-7"


cdir=$(pwd)

mkdir $local/venvs


echo "Create virtual env $env"
if test -d $local/venvs/$env; then
  echo "Deleting" $local/venvs/$env
  rm -r $local/venvs/$env
fi

python3.10 -m venv $local/venvs/$env
source $local/venvs/$env/bin/activate



python --version

pip install texttable more-itertools -i https://mirrors.aliyun.com/pypi/simple/

echo "Installing..."

pip install pyparsing attrs certifi click requests jinja2 markupsafe pyyaml typing-extensions -i https://mirrors.aliyun.com/pypi/simple/
pip install munch -i https://mirrors.aliyun.com/pypi/simple/
pip install pynvml -i https://mirrors.aliyun.com/pypi/simple/

pip install numpy==1.23 -i https://mirrors.aliyun.com/pypi/simple/
# these versions are required by facenet-pytorch 2.6.0
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
#pip install torch-1.9.0+cu111-cp39-cp39-linux_x86_64.whl
#pip install torchvision-0.10.0+cu111-cp39-cp39-linux_x86_64.whl
pip install tqdm matplotlib scipy pandas  scikit-learn ipython -i https://mirrors.aliyun.com/pypi/simple/

pip install facenet-pytorch==2.6.0 -i https://mirrors.aliyun.com/pypi/simple/

pip install ffmpeg-python -i https://mirrors.aliyun.com/pypi/simple/
pip install face-alignment -i https://mirrors.aliyun.com/pypi/simple/

pip install opencv-python tqdm opensmile resampy vosk -i https://mirrors.aliyun.com/pypi/simple/
pip install --upgrade nltk -i https://mirrors.aliyun.com/pypi/simple/

pip install deepmultilingualpunctuation -i https://mirrors.aliyun.com/pypi/simple/
pip install ffmpeg-python -i https://mirrors.aliyun.com/pypi/simple/
python -c "import nltk;nltk.set_proxy('http://mirrors.ustc.edu.cn/pypi/simple'); nltk.download('punkt')"


#  face.evoLVe: https://github.com/ZhaoJ9014/face.evoLVe
# 722ecfd769006c9c9de1cf81203807e02ddac7e5

if [ ! -d "face_evoLVe" ]; then
  git clone https://github.com/ZhaoJ9014/face.evoLVe.git
  cd face.evoLVe
  git checkout 722ecfd769006c9c9de1cf81203807e02ddac7e5
  rm -rf .git
  cd ..
  mv "face.evoLVe" "face_evoLVe"
fi


cd $cdir

deactivate

echo "Done creating and installing virt.env: $env."
