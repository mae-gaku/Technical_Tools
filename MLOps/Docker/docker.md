# build

- docker build -t docker .

- docker run -t docker 

# image
docker images

docker rmi imageID

# container
docker rm containerID

docker ps
docker ps -a


history | grep build


docker images

container ID取得


docker run --rm -it container id  /bin/bash


# dockerマウント

**(マウントした場合、dockerfile内のCOPYはコメントアウトする)**

- docker run -it -v /home:/docker (image name) /bin/bash

**マウントしたコンテナに入る方法**

- docker exec -it (container id) /bin/bash


# イメージをpushする手順

**https://gray-code.com/blog/container-image-push-for-dockerhub/**

docker tag docker avintondocker/docker:latest

docker tag app avintondocker/app:latest

docker images

docker login

avintondocker/docker:latest

avintondocker/app:latest

docker push avintondocker/docker:latest

docker push avintondocker/app:latest



# gpu使用時のDocker

### イメージをpull

- docker pull nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 

docker上でNVIDIAドライバが認識されていればOK


### コンテナ作成

- docker run -it -d --gpus all --name (container name) nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 bash

- docker run -it -d --gpus all --name G-test -v /home:/home --shm-size=10g  nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 bash


- docker run -it -d --gpus all --name my-test -v /home:/home bash

### コンテナに入る

- docker exec -it コンテナ名 bash

- docker exec -it develop bash

# gpuを使用しているか確認する

- nvidia-smi

# PythonでGPUが利用できるか確認する
```
import torch
print(torch.cuda.is_available())
```

Falseの場合、CUDAのバージョン10.1にあるPytorchのバージョンをインストールする必要がある

- pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

- pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio===0.7.2

- pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

- pip3 install torch==1.7.0 torchvision==0.8.1 -f https://download.pytorch.org/whl/cu101/torch_stable.html


- pip3 install torch==1.10.1+cpu torchmetrics==0.11.4 torchvision==0.11.2+cpu -f https://download.pytorch.org/whl/cu101/torch_stable.html


**参考資料：**　https://varhowto.com/install-pytorch-cuda-10-1/



#### GPU1を使用する
- export CUDA_VISIBLE_DEVICES="1"


### まとめ
CUDA = Pytorch = yolov5の全てのバージョンが合うようにする必要がある。
完了すれば、コンテナ上でGPUが使用できYolov5のtrain.pyをGPU有りで走らせることができる。


# Docker上でGPUを使用し、且つYolov5を使用して学習＆推論

export CUDA_VISIBLE_DEVICES="0","1"

```
# 必須
apt-get update

# pipをインストール
sudo apt install python3-pip

# ModuleNotFoundError: No module named 'yaml' 
python3 -m pip install pyyaml

# ModuleNotFoundError: No module named 'tqdm'
pip3 install tqdm


# ModuleNotFoundError: No module named 'cv2'
pip3 install opencv-python


# ModuleNotFoundError: No module named 'skbuild'
pip3 install -U pip

# ImportError: libGL.so.1: cannot open shared object file: No such file or directory
apt update
sudo apt-get install libgl1-mesa-dev -y

# ModuleNotFoundError: No module named 'requests'
sudo python3 -m pip install requests

# ModuleNotFoundError: No module named 'torchvision'
pip3 install pytorch torchvision

# ModuleNotFoundError: No module named 'seaborn'
python3 -m pip install seaborn

# ModuleNotFoundError: No module named 'tensorboard'
pip3 install tensorboard

```

### バックグラウンドで学習させる
nohup python3 train.py --data ../datasets/OCR_MTP/fold_1/fold_1.yaml --img 640 --epochs 200 --weights weights/yolov5s.pt --batch-size 2 --project runs/train/OCR_MTP_s640_v2 > /dev/null 2>&1 &


## pytorchのインストール注意点

- pip3 install torch==1.7.0 torchvision==0.8.1 -f https://download.pytorch.org/whl/cu101/torch_stable.html

- pip3 install torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html


- pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

- pip install torch==1.10.2+cpu torchvision==0.11.3+cpu -f https://download.pytorch.org/whl/torch_stable.html
torch==1.10.2+cpu

- pip3 install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

- pip3 install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

- pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html



# docker detail
### docker build cashを削除

- docker builder prune

### docker imageからファイルをローカルにコピーする

- docker cp <コンテナID>:<コンテナ内のファイルパス> <ローカルディレクトリパス>

- docker cp <コンテナID>:/mnt/src/utils/pipeline.py /home










