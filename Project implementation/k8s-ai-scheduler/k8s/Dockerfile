ARG PYTHON_BASE_IMAGE=python:3.11.13-slim-bookworm
FROM ${PYTHON_BASE_IMAGE}

ARG APP_VERSION=0.3.0
LABEL org.opencontainers.image.title="ml-sim-job" \
      org.opencontainers.image.description="Synthetic ML training workload for scheduler experiments" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.source="local-workspace"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    HOME=/tmp \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY constraints.txt /tmp/constraints.txt
COPY requirements-trainer.txt /tmp/requirements-trainer.txt
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
      -r /tmp/requirements-trainer.txt \
    && rm /tmp/requirements-trainer.txt /tmp/constraints.txt
COPY --chown=10001:10001 k8s /app/k8s

USER 10001:10001
ENTRYPOINT ["python", "-m", "k8s.train"]
