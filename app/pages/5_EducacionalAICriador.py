# -*- coding: utf-8 -*-
# ---
# NOME DO ARQUIVO: educacao_llm.py
# DESCRIÇÃO: Script convertido do notebook "LLMs para empresas e negócios - Educação.ipynb"
# ---

# ==============================================================================
# SEÇÃO 0: INSTALAÇÃO DE DEPENDÊNCIAS
# ==============================================================================
# Antes de executar este script, instale as bibliotecas necessárias no seu terminal:
# pip install langchain-groq langchain-community langchain-core langchain-huggingface langchain-qdrant qdrant_client langchain-docling pypandoc streamlit python-dotenv langgraph tavily numexpr

# Em alguns sistemas Linux, você também pode precisar do pandoc:
# sudo apt-get install pandoc

import os
import getpass
from datetime import datetime
from pathlib import Path
import math
import numexpr
import qdrant_client

# LangChain e bibliotecas relacionadas
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from langchain_docling import DoclingLoader
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool

# LangGraph para Agentes
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from typing import Annotated

# NOTA: O código original usava ipywidgets e IPython.display para criar
# formulários interativos em um notebook. Essa funcionalidade não é
# transferível diretamente para um script .py. A seção "Interface com Streamlit"
# (mais abaixo) é a abordagem correta para criar uma UI para este script.

# ==============================================================================
# SEÇÃO 1: CONFIGURAÇÃO INICIAL E MODELO
# ==============================================================================

# --- Configuração da API Key ---
# Para segurança, é recomendado usar variáveis de ambiente.
# Descomente a linha abaixo e substitua pela sua chave ou configure um arquivo .env
# os.environ["GROQ_API_KEY"] = "SUA_CHAVE_API_AQUI"
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Digite sua GROQ API Key: ")

# --- Carregamento do LLM ---
def load_llm(id_model="llama3-8b-8192", temperature=0.7):
    """Carrega e retorna uma instância do modelo de linguagem."""
    return ChatGroq(
        model=id_model,
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

llm = load_llm()
print("Modelo de Linguagem carregado.")

# --- Funções de Formatação ---
def format_res(res, return_thinking=False):
    """Formata a resposta do modelo, removendo ou exibindo a cadeia de pensamento."""
    res = res.strip()
    if return_thinking:
        res = res.replace("<think>", "[pensando...] ")
        res = res.replace("</think>", "\n---\n")
    else:
        if "</think>" in res:
            res = res.split("</think>")[-1].strip()
    return res

# ==============================================================================
# SEÇÃO 2: GERAÇÃO DE EXERCÍCIOS SIMPLES (SEM RAG)
# ==============================================================================

def build_prompt_simple(topic, quantity, level, interests):
    """Constrói o prompt para a geração de exercícios simples."""
    prompt = f"""
    Você é um tutor especialista em {topic}. Gere {quantity} exercícios para um aluno de nível {level}.
    {f"- Apenas caso faça sentido no contexto, adapte de forma natural e sutil os enunciados dos exercícios para refletir a afinidade do aluno com o tema '{interests}'." if interests else ""}
    - Formato dos exercícios: Múltipla escolha com 4 opções.
    - Incluir explicação passo a passo e o raciocínio usado para chegar à resposta.
    - Não use LaTeX e nenhuma sequência iniciada por barra invertida (como \\frac, \\sqrt, ou similares). Use apenas linguagem natural e símbolos comuns do teclado.

    Exemplo de estrutura:
    1. [Enunciado]
       a) Opção 1
       b) Opção 2
       c) Opção 3
       d) Opção 4
       Resposta: [Letra correta]
       Explicação: [Passo a passo detalhado]
    """
    return prompt

def generate_exercises_simple(topic, quantity, level, interests):
    """Gera exercícios com base nos parâmetros fornecidos."""
    print("\nGerando exercícios simples...")
    prompt = build_prompt_simple(topic, quantity, level, interests)
    res = llm.invoke(prompt)
    formatted_content = format_res(res.content)
    print(formatted_content)
    return formatted_content

# ==============================================================================
# SEÇÃO 3: RAG E INTEGRAÇÃO COM QDRANT
# ==============================================================================

# --- Configuração das API Keys do Qdrant ---
# Use variáveis de ambiente para segurança
if 'QDRANT_HOST' not in os.environ or 'QDRANT_API_KEY' not in os.environ:
    # os.environ['QDRANT_HOST'] = "URL_DO_SEU_QDRANT_AQUI"
    # os.environ['QDRANT_API_KEY'] = "SUA_CHAVE_QDRANT_AQUI"
    print("Variáveis de ambiente QDRANT não configuradas. Funções de RAG podem falhar.")

# --- Conexão com Qdrant ---
def get_qdrant_client():
    """Retorna um cliente Qdrant conectado."""
    try:
        return QdrantClient(
            url=os.environ['QDRANT_HOST'],
            api_key=os.environ['QDRANT_API_KEY']
        )
    except Exception as e:
        print(f"Não foi possível conectar ao Qdrant: {e}")
        return None

client = get_qdrant_client()

# --- Modelo de Embedding ---
def get_embedding_model(model_name="BAAI/bge-m3"):
    """Carrega e retorna o modelo de embedding."""
    return HuggingFaceEmbeddings(model_name=model_name)

embeddings = get_embedding_model()
print("Modelo de Embedding carregado.")

# --- Funções de Processamento de Documentos ---
def load_documents(input_path):
    """Carrega documentos de um arquivo ou diretório."""
    input_path = Path(input_path)
    documents = []
    if input_path.is_dir():
        pdf_files = list(input_path.glob("*.pdf"))
    elif input_path.is_file() and input_path.suffix == '.pdf':
        pdf_files = [input_path]
    else:
        raise ValueError("Caminho inválido. Forneça um diretório ou um arquivo .pdf")

    for file in pdf_files:
        loader = DoclingLoader(str(file))
        documents.extend(loader.load())
    print(f"Documentos carregados: {len(documents)}")
    return documents

def split_chunks(docs, chunk_size=1000, chunk_overlap=200):
    """Divide os documentos em chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    print(f"Chunks gerados: {len(chunks)}")
    return chunks

# --- Configuração do Retriever ---
def get_retriever(qdrant_collection, embeddings_model):
    """Cria ou carrega um retriever a partir de uma coleção Qdrant existente."""
    if not client:
        print("Retriever não pode ser criado: cliente Qdrant indisponível.")
        return None
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings_model,
        url=os.environ['QDRANT_HOST'],
        api_key=os.environ['QDRANT_API_KEY'],
        collection_name=qdrant_collection,
    )
    return vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 6, 'fetch_k': 10})

def get_context(retriever, topic):
    """Obtém o contexto relevante usando o retriever."""
    if not retriever:
        return ""
    retrieved_docs = retriever.invoke(topic)
    return "\n\n".join([doc.page_content for doc in retrieved_docs])

# --- Geração com RAG ---
def build_prompt_rag(quantity, level, interests):
    """Constrói o prompt para geração de exercícios com RAG."""
    return f"""
    Gere {quantity} exercícios em português sobre o conteúdo fornecido como contexto abaixo. Nível de dificuldade: {level}.
    Cada exercício deve ser de múltipla escolha com 4 alternativas, incluir a resposta correta e uma explicação passo a passo.
    Não invente dados externos nem saia do escopo do material apresentado, utilize exclusivamente o conteúdo fornecido.
    Para a explicação da resposta, não justifique mencionando que foi obtido com o contexto fornecido abaixo. Você deve justificá-la com base no conhecimento que você tem.
    {f"- Apenas caso faça sentido no contexto, adapte de forma natural e sutil os enunciados dos exercícios para refletir a afinidade do aluno com o tema '{interests}'." if interests else ""}
    """

def generate_exercises_rag(retriever, topic, quantity, level, interests):
    """Gera exercícios usando a abordagem RAG."""
    print("\nGerando exercícios com RAG...")
    context = get_context(retriever, topic)
    if not context:
        print("Não foi possível obter contexto do retriever. Abortando geração RAG.")
        return "Erro: contexto não encontrado."

    prompt_rag = build_prompt_rag(quantity, level, interests)
    prompt_template_rag = PromptTemplate(
        input_variables=["context", "input"],
        template="""
        {input}
        ---
        Contexto: {context}
        """
    )
    prompt_llm = prompt_template_rag.format(input=prompt_rag, context=context)
    res = llm.invoke(prompt_llm)
    formatted_content = format_res(res.content)
    print(formatted_content)
    return formatted_content


# ==============================================================================
# SEÇÃO 4: CRIAÇÃO DE AGENTES COM LANGGRAPH
# ==============================================================================

# --- Definindo o State do Agente ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

# --- Ferramentas (Tools) do Agente ---
@tool
def calculator_tool(expression: str) -> str:
    """Use esta ferramenta quando for explicitamente solicitado a resolver um cálculo matemático."""
    try:
        local_dict = {"pi": math.pi, "e": math.e}
        return str(numexpr.evaluate(expression.strip(), global_dict={}, local_dict=local_dict))
    except Exception as e:
        return f"Erro ao avaliar a expressão: {e}"

def create_retriever_tool_rag(retriever):
    """Cria uma ferramenta de retriever dinamicamente."""
    return create_retriever_tool(
        retriever,
        "retriever_docs_tool",
        "Use esta ferramenta para buscar informações e documentos sobre tópicos específicos no banco de dados."
    )

# --- Configuração e Execução do Agente ---
def create_agent_graph(tools):
    """Cria e compila o grafo do agente com as ferramentas fornecidas."""
    llm_with_tools = llm.bind_tools(tools)
    
    def agent_node(state: State):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

def run_agent_chatbot(graph):
    """Inicia um loop de chatbot interativo com o agente."""
    config = {"configurable": {"thread_id": "1"}}
    print("\n--- Tutor Digital Ativado ---")
    print("Olá! Sou seu tutor digital. Você pode me fazer perguntas, pedir cálculos ou buscar em meus documentos. Digite 'sair' para terminar.")

    while True:
        user_input = input("\nUsuário: ")
        if user_input.lower() in ["q", "sair"]:
            print("Até mais!")
            break
        
        events = graph.stream({"messages": [("user", user_input)]}, config)
        for event in events:
            for value in event.values():
                if isinstance(value['messages'][-1].content, str) and value['messages'][-1].content:
                    print(f"\nTutor: {value['messages'][-1].content}")

# ==============================================================================
# SEÇÃO 5: FUNÇÃO PRINCIPAL (MAIN) E EXECUÇÃO
# ==============================================================================
def main():
    """Função principal para executar as funcionalidades do script."""
    
    # --- Demonstração da Geração Simples ---
    print("="*50)
    print("PARTE 1: DEMONSTRAÇÃO DE GERAÇÃO SIMPLES")
    print("="*50)
    generate_exercises_simple(
        topic="Física Quântica",
        quantity=2,
        level="Iniciante",
        interests="Filmes de Ficção Científica"
    )

    # --- Demonstração da Geração com RAG ---
    print("\n" + "="*50)
    print("PARTE 2: DEMONSTRAÇÃO DE GERAÇÃO COM RAG")
    print("="*50)
    qdrant_collection_name = "proj_edu"
    # NOTA: Para o RAG funcionar, você precisa ter uma coleção no Qdrant
    # e documentos carregados nela. O código para carregar documentos
    # (`load_documents`, `split_chunks`) precisa ser executado previamente.
    # Exemplo (descomente para usar):
    #
    # docs_path = "/caminho/para/seus/artigos/"
    # if os.path.exists(docs_path) and client:
    #     documents = load_documents(docs_path)
    #     chunks = split_chunks(documents)
    #     # A primeira vez, use from_documents para criar e popular
    #     vectorstore = QdrantVectorStore.from_documents(
    #         chunks, embeddings, url=os.environ['QDRANT_HOST'], api_key=os.environ['QDRANT_API_KEY'], collection_name=qdrant_collection_name
    #     )
    #     print(f"Coleção '{qdrant_collection_name}' criada/populada.")
    # else:
    #      print("Diretório de documentos não encontrado ou cliente Qdrant indisponível.")

    rag_retriever = get_retriever(qdrant_collection_name, embeddings)
    if rag_retriever:
        generate_exercises_rag(
            retriever=rag_retriever,
            topic="astronomia",
            quantity=2,
            level="Intermediário",
            interests="Jogos"
        )
    else:
        print("Não foi possível executar a demonstração com RAG.")

    # --- Demonstração do Agente (Chatbot) ---
    print("\n" + "="*50)
    print("PARTE 3: DEMONSTRAÇÃO DO AGENTE (TUTOR DIGITAL)")
    print("="*50)
    if rag_retriever:
        rag_tool = create_retriever_tool_rag(rag_retriever)
        agent_tools = [calculator_tool, rag_tool]
        agent_graph = create_agent_graph(agent_tools)
        run_agent_chatbot(agent_graph)
    else:
        print("Não foi possível iniciar o agente com a ferramenta RAG. Iniciando com ferramentas básicas.")
        agent_graph = create_agent_graph([calculator_tool])
        run_agent_chatbot(agent_graph)


if __name__ == "__main__":
    main()

# ==============================================================================
# SEÇÃO 6: CÓDIGO PARA APLICAÇÃO STREAMLIT (app.py)
# ==============================================================================
# O código abaixo estava comentado no notebook original. Ele define uma aplicação
# web simples usando Streamlit. Para executá-lo:
# 1. Salve este bloco de código em um arquivo separado (ex: `app.py`).
# 2. Certifique-se de que as dependências estão instaladas (`pip install streamlit`).
# 3. Execute no terminal: `streamlit run app.py`
# ==============================================================================

# %%writefile app.py
# import streamlit as st
# from datetime import datetime
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# load_dotenv()
#
# # Configurações iniciais
# st.set_page_config(page_title="Gerador de Exercícios", layout="centered", page_icon="📖")
# st.title("Gerador de Exercícios 📖")
#
# # Função para carregar o modelo
# @st.cache_resource
# def load_llm(id_model, temperature):
#     return ChatGroq(
#         model=id_model,
#         temperature=temperature,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2,
#     )
#
# # Função para formatar a resposta da LLM
# def format_res(res, return_thinking=False):
#     res = res.strip()
#     if return_thinking:
#         res = res.replace("<think>", "[pensando...] ")
#         res = res.replace("</think>", "\n---\n")
#     else:
#         if "</think>" in res:
#             res = res.split("</think>")[-1].strip()
#     return res
#
# # Função para criar o prompt
# def build_prompt(topic, quantity, level, interests):
#     prompt = f"""
# Você é um tutor especialista em {topic}. Gere {quantity} exercícios para um aluno de nível {level}.
# {f"- Apenas caso faça sentido no contexto, adapte de forma natural e sutil os enunciados dos exercícios para refletir a afinidade do aluno com o tema '{interests}'." if interests else ""}
# - Formato dos exercícios: Múltipla escolha com 4 opções.
# - Incluir explicação passo a passo e o raciocínio usado para chegar à resposta.
# - Não use LaTeX e nenhuma sequência iniciada por barra invertida (como \\frac, \\sqrt, ou similares). Use apenas linguagem natural e símbolos comuns do teclado.
#
# Exemplo de estrutura:
# 1. [Enunciado]
#    a) Opção 1
#    b) Opção 2
#    c) Opção 3
#    d) Opção 4
#    Resposta: [Letra correta]
#    Explicação: [Passo a passo detalhado]
# """
#     return prompt
#
# st.sidebar.header("Configurações do modelo")
# id_model = st.sidebar.text_input("ID do modelo", value="llama3-8b-8192")
# temperature = st.sidebar.slider("Temperatura", 0.1, 1.5, 0.7, 0.1)
#
# with st.form("formulario"):
#     level = st.selectbox("Nível", ['Iniciante', 'Intermediário', 'Avançado'], index=1)
#     topic = st.text_input("Tema", placeholder="Matemática, Inglês, Física, etc.")
#     quantity = st.slider("Quantidade de Exercícios", 1, 10, 5)
#     interests = st.text_input("Interesses ou Preferências", placeholder="Ex: Filmes, Música, etc.")
#     gerar = st.form_submit_button("Gerar Exercícios")
#
# if gerar:
#     if not topic:
#         st.error("Por favor, insira um tema para gerar os exercícios.")
#     else:
#         with st.spinner("Gerando exercícios..."):
#             llm = load_llm(id_model, temperature)
#             prompt = build_prompt(topic, quantity, level, interests)
#             res = llm.invoke(prompt)
#             res_formatado = format_res(res.content, return_thinking=True)
#             st.markdown(res_formatado)