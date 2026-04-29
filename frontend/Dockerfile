FROM python:3.11-slim

# System packages needed to compile Prophet
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /tmp/stock_cache

CMD streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true
