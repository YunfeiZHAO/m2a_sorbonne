FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

# prepare python dependencies for pyenv
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -yq \
	&& apt-get install -y make build-essential libssl-dev zlib1g-dev \
						libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
						libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
						libffi-dev liblzma-dev git \
	&& rm -rf /var/lib/apt/lists/*
	
# install and config pyenv
ARG PY_VER=3.7.12
ENV PYENV_ROOT=${HOME}/.pyenv
ENV PATH=${PYENV_ROOT}/bin:${PATH}
RUN git clone https://github.com/pyenv/pyenv.git ${PYENV_ROOT} \
	&& pyenv install ${PY_VER} \
	&& pyenv global ${PY_VER}

# install project environment
ENV PATH=${PYENV_ROOT}/shims:${PATH}
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt \
	&& rm -rf /tmp/requirements.txt
	
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /root
