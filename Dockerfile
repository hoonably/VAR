FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y git wget unzip tree vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ARG LOCAL_USER_ID=1002
ARG LOCAL_GROUP_ID=1002

RUN groupadd -g $LOCAL_GROUP_ID user && \
    useradd -m -d /workspace -u $LOCAL_USER_ID -g $LOCAL_GROUP_ID -s /bin/bash user

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install jupyterlab notebook

ENV PATH="/opt/conda/bin:$PATH"
ENV PATH="$HOME/.local/bin:$PATH"

USER user
WORKDIR /workspace

CMD ["/bin/bash"]

