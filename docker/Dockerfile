# Placeholder OpenEnv server image. Replace PYTHON_VERSION / org as needed.
FROM python:3.11-slim

WORKDIR /app

COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY . .

ENV PYTHONPATH=/app
EXPOSE 8000

# Health: rely on OpenEnv / FastAPI routes after start
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
