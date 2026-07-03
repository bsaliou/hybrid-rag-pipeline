FROM python:3.11-slim

WORKDIR /app

# Dépendances système nécessaires à pypdf / sentence-transformers / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ingère automatiquement les documents d'exemple au premier démarrage si la
# base vectorielle est vide, puis lance l'interface Streamlit.
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
