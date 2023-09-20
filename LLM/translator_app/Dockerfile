FROM python@sha256:9b0d7419e2811710aacee87c40a2c94693e2b6810c3e7e466b8c7fc5bde4cd66 as builder

WORKDIR /mnt

# Install linux packages
RUN apt-get update && apt-get install -y build-essential

COPY ./ ./

# Install python packages
RUN pip3 install --no-cache-dir \
    --upgrade pip \
    -r requirements.txt \
    -f https://download.pytorch.org/whl/torch_stable.html \
    && rm -rf ~/.cache/pip \
    && apt-get install -y git \
    && pip3 install git+https://github.com/huggingface/transformers.git

FROM python@sha256:9b0d7419e2811710aacee87c40a2c94693e2b6810c3e7e466b8c7fc5bde4cd66


WORKDIR /mnt

# Install necessary libraries for the application
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/local/src/*

# Copy files from builder stage
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /usr/local/bin/uwsgi /usr/local/bin/uwsgi
COPY --from=builder /mnt /mnt

CMD ["/bin/bash", "-c", "uwsgi uwsgi.ini"]
