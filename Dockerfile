FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY src ./src
COPY sdss_sample.csv .
COPY README.md .
COPY outputs/.gitkeep ./outputs/.gitkeep

RUN mkdir -p /app/outputs/metrics /app/outputs/plots

CMD ["python", "main.py"]

