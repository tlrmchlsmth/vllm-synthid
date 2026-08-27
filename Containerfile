ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.15@sha256:a5727064a0de127bdb7c9d3c1383f3a9ac307d9f2d8a391edc7896c54289ced0
ARG VLLM_IMAGE

FROM ${UV_IMAGE} AS uv
FROM ${VLLM_IMAGE}

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /opt/vllm-synthid
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv pip install --system --no-deps .
