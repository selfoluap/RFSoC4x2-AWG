FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=60
ENV PIP_RETRIES=5

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN python -m pip install --no-cache-dir --timeout 60 --retries 5 -r backend/requirements.txt

COPY backend/ backend/
COPY firmware/ firmware/

CMD ["uvicorn", "backend.backend:app", "--host", "0.0.0.0", "--port", "8001"]
