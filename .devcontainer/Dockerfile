FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel as compile_server
WORKDIR /workspace
ENV CUDA_HOME /usr/local/cuda
RUN apt update -y &&  apt install -y  --no-install-recommends \
    git \
    wget \
    vim \
    gcc \
    g++ \
    openssh-server && \
rm -rf /var/lib/apt/lists/* && \
pip install --upgrade pip && \
pip install ninja pyproject numpy cpufeature cmake && \
pip install flash-attn && \
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/

ENV PATH=/opt/conda/bin:$PATH
ENV CMAKE_COMMAND=/opt/conda/bin/cmake
ENV CMAKE_ROOT=/opt/conda/lib/python*/site-packages/cmake/data

CMD ["/bin/bash"]
