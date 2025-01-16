FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir fastapi uvicorn pydantic

EXPOSE 8080

CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
