from pathlib import Path

# --- Configurações de Caminho
# Constrói o caminho para a pasta 'docs'
# O __file__ aqui se refere a este arquivo (config.py), então a lógica de 'parent' muda um pouco
# .parent -> /app/
# .parent -> / (pasta raiz do projeto)
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_PATH = PROJECT_ROOT / "docs"


# --- Configurações do LLM
MODEL_ID = "llama3-70b-8192"
TEMPERATURE = 0.7


# --- Configurações do RAG
EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# --- Configurações do Retriever
SEARCH_TYPE = "mmr"
SEARCH_K = 3
FETCH_K = 4


# --- Configurações do Analisador de Currículos
CV_MODEL_ID = "llama3-70b-8192" 
CV_TEMPERATURE = 0.5 
CV_JSON_FILE = "curriculos.json"
CV_JOB_CSV_FILE = "vagas.csv"

CV_REQUIRED_FIELDS  = [
    "name",
    "area",
    "summary",
    "skills",
    "education",
    "interview_questions",
    "strengths",
    "areas_for_development",
    "important_considerations",
    "final_recommendations",
    "score"
]

JOB_DETAILS = {}
JOB_DETAILS['title'] = "Desenvolvedor(a) Full Stack"
JOB_DETAILS['description'] = "Estamos em busca de um(a) Desenvolvedor(a) Full Stack para integrar o time de tecnologia da nossa empresa, atuando em projetos estratégicos com foco em soluções escaláveis e orientadas a dados. O(a) profissional será responsável por desenvolver, manter e evoluir aplicações web robustas, além de colaborar com times multidisciplinares para entregar valor contínuo ao negócio."
JOB_DETAILS['details'] = """
Atividades:
- Desenvolver e manter aplicações web em ambientes modernos, utilizando tecnologias back-end e front-end.
- Trabalhar com equipes de produto, UX e dados para entender demandas e propor soluções.
- Criar APIs, integrações e dashboards interativos.
- Garantir boas práticas de versionamento, testes e documentação.
- Participar de revisões de código, deploys e melhorias contínuas na arquitetura das aplicações.

Pré-requisitos:
- Sólidos conhecimentos em Python, JavaScript e SQL.
- Experiência prática com frameworks como React, Node.js e Django.
- Familiaridade com versionamento de código usando Git.
- Experiência com serviços de nuvem, como AWS e Google Cloud Platform.
- Capacidade de trabalhar em equipe, com boa comunicação e perfil colaborativo.

Diferenciais:
- Conhecimento em Power BI ou outras ferramentas de visualização de dados.
- Experiência anterior em ambientes ágeis (Scrum, Kanban).
- Projetos próprios, contribuições open source ou portfólio técnico disponível.
- Certificações em nuvem ou áreas relacionadas à engenharia de software.
"""