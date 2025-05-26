FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    python3-pip wget git tree \
    && apt-get clean

WORKDIR /workspace

# pip 업그레이드
RUN python3 -m pip install --upgrade pip

# requirements.txt 복사 후 설치
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# 진입점은 shell
CMD ["/bin/bash"]
