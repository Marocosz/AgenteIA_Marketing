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
JOB_DETAILS['title'] = "Analista de Dados"
JOB_DETAILS['description'] = "Estamos em busca de um(a) Analista de Dados para integrar o time de tecnologia da nossa empresa, atuando em projetos estratégicos com foco em soluções escaláveis e orientadas a dados. O(a) profissional será responsável por analisar grandes volumes de dados, gerar insights e construir dashboards para apoiar a tomada de decisão, além de colaborar com times multidisciplinares para entregar valor contínuo ao negócio."
JOB_DETAILS['details'] = """
Atividades:
- Desenvolver e manter pipelines de dados e processos de ETL.
- Trabalhar com equipes de produto, marketing e operações para entender demandas e propor análises.
- Criar relatórios, dashboards interativos e visualizações de dados.
- Garantir boas práticas de qualidade, governança e documentação dos dados.
- Participar de análises exploratórias, testes de hipóteses e melhorias contínuas na cultura de dados da empresa.

Pré-requisitos:
- Sólidos conhecimentos em SQL, Python (com bibliotecas como Pandas e NumPy) e Excel avançado.
- Experiência prática com ferramentas de Business Intelligence como Power BI, Tableau ou Looker.
- Familiaridade com bancos de dados relacionais e não relacionais.
- Experiência com análise estatística e modelagem de dados.
- Capacidade de traduzir dados complexos em insights acionáveis, com boa comunicação e perfil colaborativo.

Diferenciais:
- Conhecimento em serviços de nuvem, como AWS (S3, Redshift) ou Google Cloud Platform (BigQuery).
- Experiência anterior em ambientes ágeis (Scrum, Kanban).
- Conhecimento em ferramentas de versionamento como Git.
- Certificações em análise de dados ou áreas relacionadas.
"""