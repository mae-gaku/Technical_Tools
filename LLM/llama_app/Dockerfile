FROM python@sha256:9b0d7419e2811710aacee87c40a2c94693e2b6810c3e7e466b8c7fc5bde4cd66 as builder

WORKDIR /mnt

COPY ./requirements.txt ./

RUN apt-get update && \
    apt-get install -y build-essential && \
    pip3 install --no-cache-dir \
    --upgrade pip \
    -r requirements.txt \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    && rm -rf ~/.cache/pip