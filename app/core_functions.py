# app/core_functions.py

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

# Imports dos nossos novos arquivos
from config import *
from prompts import *


# --- FUNÇÕES GERAIS ---

@st.cache_resource
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


@st.cache_resource
def get_content_generation_chain(_llm):
    """
    Cria a cadeia de prompt e a armazena em cache para reutilização.
    """
    template = ChatPromptTemplate.from_messages([
        ("system", CONTENT_GENERATOR_SYSTEM_PROMPT),
        ("human", "{prompt}"),
    ])
    # Corrente de acontecimentos do langchain
    chain = template | _llm | StrOutputParser()
    return chain


# --- FUNÇÕES DO ATENDIMENTO SAFE BANK (RAG) ---

@st.cache_resource
def config_retriever(_folder_path):
    # (Sua função config_retriever completa aqui, sem alterações)
    docs_path = Path(_folder_path)
    # ... resto da função ...
    pdf_files = [f for f in docs_path.glob("*.pdf")]
    if not pdf_files:
        st.error(f"Nenhum arquivo PDF encontrado no caminho: {_folder_path}")
        st.stop()
    docs_list = [PyMuPDFLoader(str(pdf)).load() for pdf in pdf_files]
    docs = [item for sublist in docs_list for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type=SEARCH_TYPE, search_kwargs={
                                         'k': SEARCH_K, 'fetch_k': FETCH_K})
    return retriever


@st.cache_resource
def config_rag_chain(_llm, _retriever):
    # (Sua função config_rag_chain completa aqui, sem alterações)
    context_q_prompt = ChatPromptTemplate.from_messages(
        [("system", CONTEXT_Q_SYSTEM_PROMPT), MessagesPlaceholder("chat_history"), ("human", "{input}")])
    history_aware_retriever = create_history_aware_retriever(
        llm=_llm, retriever=_retriever, prompt=context_q_prompt)
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", QA_SYSTEM_PROMPT), MessagesPlaceholder("chat_history"), ("human", "{input}")])
    qa_chain = create_stuff_documents_chain(_llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return rag_chain
