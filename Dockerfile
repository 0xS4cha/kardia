FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libsm6 \
        libegl1 \
        && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY config ./config
COPY data ./data

RUN uv sync --frozen --no-dev

ENV QT_API=pyside6
ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "kardia"]
