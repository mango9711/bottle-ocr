FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ocr_server.py bottle_ocr.py index.html ./

ENV OCR_ENGINE=rapid
ENV OCR_MAX_SIDE=1280
ENV PORT=7860

EXPOSE 7860

CMD ["sh", "-c", "uvicorn ocr_server:app --host 0.0.0.0 --port ${PORT:-7860}"]
