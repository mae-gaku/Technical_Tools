## How to Uses
```
wget https://huggingface.co/localmodels/Llama-2-7B-Chat-ggml/resolve/main/llama-2-7b-chat.ggmlv3.q2_K.bin
```

```
git clone https://github.com/ggerganov/llama.cpp.git
```
```
cd llama.cpp
```

### Convert
```
python3 convert-llama-ggml-to-gguf.py --input /home/gaku/llama_app/models/llama-2-7b-chat.ggmlv3.q2_K.bin --output llama-model.gguf
```

### Docker
```
docker build -t llama
```

```
docker run -it -v /home/gaku/llama_app:/mnt "image" /bin/bash

```

```
streamlit run app.py
```


