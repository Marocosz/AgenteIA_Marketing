from pathlib import Path

# --- Configurações de Caminho ---
# Constrói o caminho para a pasta 'docs'
# O __file__ aqui se refere a este arquivo (config.py), então a lógica de 'parent' muda um pouco
# .parent -> /app/
# .parent -> / (pasta raiz do projeto)
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_PATH = PROJECT_ROOT / "docs"


# --- Configurações do LLM ---
MODEL_ID = "llama3-70b-8192"
TEMPERATURE = 0.7


# --- Configurações do RAG ---
EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# --- Configurações do Retriever ---
SEARCH_TYPE = "mmr"
SEARCH_K = 3
FETCH_K = 4