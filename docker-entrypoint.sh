#!/bin/sh
# Ingère automatiquement data/sample_docs si la base vectorielle est vide,
# pour que `docker compose up` donne une app immédiatement utilisable sans
# étape manuelle supplémentaire.
set -e

if [ -z "$(ls -A /app/chroma_db 2>/dev/null)" ]; then
  echo "→ Base vectorielle vide, ingestion des documents d'exemple..."
  python scripts/ingest.py data/sample_docs || echo "⚠️  Ingestion échouée (clé API manquante ?), l'app démarre quand même."
fi

exec "$@"
