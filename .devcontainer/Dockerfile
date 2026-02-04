FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel as compile_server
WORKDIR /workspace
ENV CUDA_HOME /usr/local/cuda
RUN apt update -y &&  apt install -y  --no-install-recommends \
    git \
    wget \
    vim \
    gcc \
    g++ \
    cmake \
    openssh-server && \
rm -rf /var/lib/apt/lists/* && \
pip install --upgrade pip && \
pip install ninja pyproject numpy cpufeature && \
pip install flash-attn && \
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/

# 创建 SSH 目录并设置权限
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# 将本地主机的公钥复制到容器中（需要通过 --build-arg 或直接 COPY 公钥文件）
# 1. 使用 --build-arg 将公钥作为构建参数传递给 Dockerfile（此方法可以避免在 Dockerfile 中暴露敏感信息）
# 2. 也可以直接从本地复制 authorized_keys 文件（如果有多个公钥需要添加）

# 假设你将主机的公钥保存在本地，直接复制到容器中的 authorized_keys 文件
COPY .ssh/id_ed25519.pub /root/.ssh/authorized_keys
COPY .ssh/authorized_keys /root/.ssh/authorized_keys

# 设置权限
RUN chmod 600 /root/.ssh/authorized_keys && \
    chown -R root:root /root/.ssh

# 配置 SSH 服务
RUN mkdir /var/run/sshd && \
    # echo 'root:password' | chpasswd && \
    sed -i 's/^#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    # sed -i 's/^PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    # sed -i 's/^#UsePAM yes/UsePAM no/' /etc/ssh/sshd_config && \
    sed -i 's/^#Port 22/Port 2222/' /etc/ssh/sshd_config && \
    sed -i 's/^#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config
RUN service ssh restart
# Set the default shell to bash
CMD ["/bin/bash"]
