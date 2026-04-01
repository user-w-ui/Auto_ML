FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY kb /app/kb
COPY house_prices_train.csv /app/house_prices_train.csv

RUN pip install --upgrade pip \
    && pip install -e .

CMD ["python", "-m", "src.cli", "run", "--foreground", "--config", "configs/container.yaml"]
