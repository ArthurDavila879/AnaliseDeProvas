FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependências primeiro: a camada fica em cache enquanto requirements.txt não mudar.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY tests/ tests/
COPY exemplos/ exemplos/

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["simular"]
