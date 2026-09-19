FROM python:3.12-slim

WORKDIR /app

# gcc and libpq-dev build psycopg2. libgomp1 is the OpenMP runtime LightGBM
# links against: the slim image lacks it, and without it the API fails to
# import lightgbm and never starts (T-19).
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
