FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/work/paper_replication/.cache/matplotlib
ENV NUMBA_CACHE_DIR=/work/paper_replication/.cache/numba
ENV XDG_CACHE_HOME=/work/paper_replication/.cache/xdg

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work/paper_replication
COPY environment/requirements_docker_original.txt /tmp/requirements_docker_original.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/requirements_docker_original.txt

CMD ["bash"]

