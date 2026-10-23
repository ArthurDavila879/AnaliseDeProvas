FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependências primeiro: a camada fica em cache enquanto requirements.txt não mudar.
COPY requirements.txt requirements-dev.txt pyproject.toml ./
RUN pip install -r requirements-dev.txt

COPY src/ src/
COPY tests/ tests/
COPY exemplos/ exemplos/
COPY app.py .
COPY .streamlit/ .streamlit/

EXPOSE 8501

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["simular"]
