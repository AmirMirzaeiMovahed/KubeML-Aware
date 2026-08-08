ARG PYTHON_BASE_IMAGE=python:3.11.13-slim-bookworm
FROM ${PYTHON_BASE_IMAGE}

ARG APP_VERSION=0.2.0
LABEL org.opencontainers.image.title="ml-aware-scheduler" \
      org.opencontainers.image.description="ML-aware Kubernetes scheduling controller" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.source="local-workspace"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    HOME=/tmp

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /results \
    && chown -R 10001:10001 /results

WORKDIR /app
COPY constraints.txt /tmp/constraints.txt
COPY requirements-scheduler.txt /tmp/requirements-scheduler.txt
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
      -r /tmp/requirements-scheduler.txt \
    && rm /tmp/requirements-scheduler.txt /tmp/constraints.txt
COPY --chown=10001:10001 scheduler /app/scheduler

USER 10001:10001
EXPOSE 8080
ENTRYPOINT ["python", "-m", "scheduler.gate_controller"]
