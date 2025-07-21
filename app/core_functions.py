import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import json
import pandas as pd
import csv
import os
from docling.document_converter import DocumentConverter

# Imports dos nossos novos arquivos
from config import *
from prompts import *


# --- FUNÇÕES GERAIS ---

@st.cache_resource  # Decorator de ativar o cache do resultado da func
def load_llm():
    """Carrega o modelo de linguagem uma única vez e o guarda em cache."""
    llm = ChatGroq(
        model=MODEL_ID,
        temperature=TEMPERATURE,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    return llm


# --- FUNÇÕES DO GERADOR DE CONTEÚDO ---

@st.cache_resource  # Decorator de ativar o cache do resultado da func
# OBS: O '_' no parâmetro é uma convenção para o sistema de cache do Streamlit
def get_content_generation_chain(_llm):
    """
    Cria a cadeia de prompt e a armazena em cache para reutilização.
    """
    # É uma função do LangChain que constrói um template a partir de uma lista de mensagens
    template = ChatPromptTemplate.from_messages([
        # ("system", ...): Define a persona e a instrução principal da IA para esta tarefa.
        ("system", CONTENT_GENERATOR_SYSTEM_PROMPT),
        ("human", "{prompt}"),  # Aqui seria oq o usuário escrever ou selecionar
    ])
    # Corrente de acontecimentos do langchain
    # O parser é pra fazer a resposta da llm virar usável para gente aqui no código
    chain = template | _llm | StrOutputParser()
    return chain


# --- FUNÇÕES DO ATENDIMENTO SAFE BANK (RAG) ---

@st.cache_resource  # Decorator de ativar o cache do resultado da func (para rodar 1 vez só visto q ela é pesada)
def config_retriever(_folder_path):

    docs_path = Path(_folder_path)  # Carrega o local do documento
    pdf_files = [f for f in docs_path.glob("*.pdf")]  # Pegam todos arquivos .pdf que estão na pasta

    if not pdf_files:
        st.error(f"Nenhum arquivo PDF encontrado no caminho: {_folder_path}")
        st.stop()  # Interrompe o script

    # Aqui extrai todo conteúdo dos pdfs e guarda numa lista de listas das páginas dos pdfs
    docs_list = [PyMuPDFLoader(str(pdf)).load() for pdf in pdf_files]

    # Aqui nós pegamos e deixamos o conteudo todo em uma lista só
    docs = [item for sublist in docs_list for item in sublist]

    # Aqui nós criamos a função de separação de texto de acordo com a qt de caracteres queremos e quanto cada separação compartilhará (chunk_overlap)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    chunks = text_splitter.split_documents(docs)  # Guardamos em uma lista os "chunks" dos textos dos pdfs

    # Aqui acontece o embedding que é transformar cada chunk de texto em uma lista de números (um vetor)
    # que representa o seu significado, de acordo com o modelo definido
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Aqui é onde de fato a função de Embeddings acontece e é salvo num "banco de dados de vetores" (FAISS)
    # para permitir buscas por similaridade de forma instantanea
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    # Aqui nós recuperamos (retriever) as informações dos documentos que estão em vetor no FAISS de acordo com oq precisamos de fato
    # E o "fetch_k" define quantos documentos o sistema busca inicialmente, ANTES de aplicar o algoritmo de diversidade (mmr) para
    # selecionar os 'k' melhores. Um 'fetch_k' maior dá mais opções para o algoritmo escolher, podendo resultar em respostas melhores.
    retriever = vectorstore.as_retriever(
        search_type=SEARCH_TYPE, search_kwargs={'k': SEARCH_K, 'fetch_k': FETCH_K})
    return retriever


@st.cache_resource   # Decorator de ativar o cache do resultado da func (para rodar 1 vez só visto q ela é pesada)
def config_rag_chain(_llm, _retriever):

    # É uma função do LangChain que constrói um template a partir de uma lista de mensagens
    # Esse é o prompt que pegará o historico, a pipeline
    # Esse é o prompt que vai refazer a pergunta do usuário, para melhorar a pergunta dando os contextos
    context_q_prompt = ChatPromptTemplate.from_messages(
        [("system", CONTEXT_Q_SYSTEM_PROMPT),
         # Aqui é permitido receber uma lista de textos, no caso o historico do chat
         MessagesPlaceholder("chat_history"),
         ("human", "{input}")])

    # Aqui é a subchain que vai ser "um pesquisador ciente do histórico".
    # Seu output é os trechos dos pdfs uteis
    history_aware_retriever = create_history_aware_retriever(
        llm=_llm, retriever=_retriever, prompt=context_q_prompt)

    # Aqui é onde será o prompt final que a IA responderá de fato
    qa_prompt = ChatPromptTemplate.from_messages([("system", QA_SYSTEM_PROMPT),
                                                 MessagesPlaceholder(
                                                     "chat_history"),
                                                 # Pergunta original do usuário
                                                 ("human", "{input}")])

    # Aqui é a segunda subchain que junta o resultado do retriever e coloca o context dentro do qa_prompt
    qa_chain = create_stuff_documents_chain(_llm, qa_prompt)

    # E aqui é a junção das duas chain que dará de fato a resposta final da IA
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return rag_chain


# --- FUNÇÕES DO ANALISADOR DE CURRÍCULOS ---

@st.cache_resource
def get_cv_analysis_chain(_llm):
    """Cria e armazena em cache a chain para análise de currículos."""
    chain = CV_PROMPT_TEMPLATE | _llm | StrOutputParser() # Adicionado o StrOutputParser
    return chain

def parse_doc(file_path):
    """Converte um documento (PDF, DOCX) para Markdown."""
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()

def parse_res_llm(response_text: str, required_fields: list) -> dict:
    """Extrai um objeto JSON da resposta de texto do LLM."""
    try:
        if "</think>" in response_text:
            response_text = response_text.split("</think>")[-1].strip()
        
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            st.error("Nenhum JSON válido encontrado na resposta do modelo.")
            return None # Retorna None em caso de erro

        json_str = response_text[start_idx:end_idx]
        info_cv = json.loads(json_str)

        for field in required_fields:
            if field not in info_cv:
                info_cv[field] = "N/A" # Preenche campos ausentes
        return info_cv
    except json.JSONDecodeError:
        st.error("Erro ao decodificar o JSON da resposta do modelo.")
        return None # Retorna None em caso de erro

def save_json_cv(new_data, path_json, key_name="name"):
    """Salva os dados de um novo currículo em um arquivo JSON, evitando duplicatas."""
    if os.path.exists(path_json):
        with open(path_json, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = [] # Se o arquivo estiver corrompido/vazio, começa uma nova lista
    else:
        data = []

    if not isinstance(data, list):
        data = [] # Garante que estamos trabalhando com uma lista

    candidates = [entry.get(key_name) for entry in data]
    if new_data.get(key_name) in candidates:
        st.warning(f"Currículo '{new_data.get(key_name)}' já registrado. Ignorando.")
        return

    data.append(new_data)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def show_cv_result(result: dict):
    """Formata os dados extraídos do CV em uma string Markdown para exibição."""
    md = f"### 📄 Análise e Resumo do Currículo\n"
    
    # Usar .get() é uma prática mais segura para evitar erros se uma chave não existir
    if result.get("name"): 
        md += f"- **Nome:** {result.get('name', 'N/A')}\n"
    if result.get("area"): 
        md += f"- **Área de Atuação:** {result.get('area', 'N/A')}\n"
    if result.get("summary"): 
        md += f"- **Resumo do Perfil:** {result.get('summary', 'N/A')}\n"
    if result.get("skills"): 
        md += f"- **Competências:** {', '.join(result.get('skills', []))}\n"
    
    # --- AQUI ESTÃO OS CAMPOS QUE FALTAVAM ---
    if result.get("interview_questions"):
        md += f"- **Perguntas sugeridas:**\n"
        md += "\n".join([f"  - {q}" for q in result.get("interview_questions", [])]) + "\n"
    if result.get("strengths"):
        md += f"- **Pontos fortes (ou Alinhamentos):**\n"
        md += "\n".join([f"  - {s}" for s in result.get("strengths", [])]) + "\n"
    if result.get("areas_for_development"):
        md += f"- **Pontos a desenvolver (ou Desalinhamentos):**\n"
        md += "\n".join([f"  - {a}" for a in result.get("areas_for_development", [])]) + "\n"
    if result.get("important_considerations"):
        md += f"- **Pontos de atenção:**\n"
        md += "\n".join([f"  - {i}" for i in result.get("important_considerations", [])]) + "\n"
    if result.get("final_recommendations"):
        md += f"- **Conclusão e recomendações:** {result.get('final_recommendations', 'N/A')}\n"
        
    return md

def save_job_to_csv(data, filename):
    """Salva a descrição da vaga em um CSV."""
    headers = ['title', 'description', 'details']
    file_exists = os.path.exists(filename)
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter=';')
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def load_job(csv_path):
    """Carrega a última vaga salva do CSV."""
    try:
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
        if df.empty:
            return "Nenhuma vaga registrada."
        job = df.iloc[-1]
        prompt_text = f"**Vaga para {job['title']}**\n\n**Descrição:**\n{job['description']}\n\n**Detalhes:**\n{job['details']}"
        return prompt_text.strip()
    except FileNotFoundError:
        return "Arquivo de vagas não encontrado."

def display_json_table(path_json):
    """Carrega o JSON e o converte em um DataFrame do Pandas para exibição."""
    try:
        with open(path_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except (FileNotFoundError, json.JSONDecodeError):
        return pd.DataFrame() # Retorna um DataFrame vazio em caso de erro