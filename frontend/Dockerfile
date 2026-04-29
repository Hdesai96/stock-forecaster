FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY frontend/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "import cmdstanpy; cmdstanpy.install_cmdstan(version='2.34.1', cores=2, progress=False, overwrite=False)"

COPY frontend/ .

RUN mkdir -p /tmp/stock_cache

CMD streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true
