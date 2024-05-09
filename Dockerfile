FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

WORKDIR /app

# install python
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y curl git software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.10 python3.10-venv python3.10-dev
RUN ls -la /usr/bin/python3
RUN rm /usr/bin/python3
RUN ln -s python3.10 /usr/bin/python3
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN apt-get install -y python3-pip python-is-python3
RUN rm -rf /var/lib/apt/lists/*

# install unsloth requirements
RUN pip install packaging
RUN pip install --upgrade --no-cache-dir torch==2.2.0 triton --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-deps networkx ninja einops flash-attn xformers trl peft accelerate bitsandbytes
RUN pip install "unsloth[cu121-ampere-torch220] @ git+https://github.com/unslothai/unsloth.git"

# install llama-cpp for GGUF output
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=ON"
RUN pip install llama-cpp-python

# install local_rag_llm for synthetic datasets
RUN pip install -r https://raw.githubusercontent.com/dhopp1/local_rag_llm/main/requirements.txt
RUN pip install local-rag-llm