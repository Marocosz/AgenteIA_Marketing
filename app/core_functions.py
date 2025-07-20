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


# --- FUNÇÕES DO GERADOR DE CONTEÚDO

@st.cache_resource # Decorator de ativar o cache do resultado da func
                   # OBS: variáveis com "_" no começo indica que são variáveis cache tmb
def get_content_generation_chain(_llm):
    """
    Cria a cadeia de prompt e a armazena em cache para reutilização.
    """
    # É uma função do LangChain que constrói um template a partir de uma lista de mensagens
    template = ChatPromptTemplate.from_messages([
        ("system", CONTENT_GENERATOR_SYSTEM_PROMPT),  # System é a msg mais importnte
        ("human", "{prompt}"),  # Aqui seria oq o usuário escrever ou selecionar
    ])
    # Corrente de acontecimentos do langchain
    # O parser é pra fazer a resposta da llm virar usável para gente aqui no código
    chain = template | _llm | StrOutputParser()
    return chain


# --- FUNÇÕES DO ATENDIMENTO SAFE BANK (RAG)

@st.cache_resource  # Decorator de ativar o cache do resultado da func (para rodar 1 vez só visto q ela é pesada)
def config_retriever(_folder_path):

    docs_path = Path(_folder_path)  # Carrega o local do documento
    pdf_files = [f for f in docs_path.glob("*.pdf")]  # Pegam todos arquivos .pdf que estão na pasta
    
    if not pdf_files:
        st.error(f"Nenhum arquivo PDF encontrado no caminho: {_folder_path}")
        st.stop()  # Interrompe o script
        
    docs_list = [PyMuPDFLoader(str(pdf)).load() for pdf in pdf_files]  # Aqui extrai todo conteúdo dos pdfs e guarda numa lista de listas das páginas dos pdfs
    
    docs = [item for sublist in docs_list for item in sublist]  # Aqui nós pegamos e deixamos o conteudo todo em uma lista só
    
    # Aqui nós criamos a função de separação de texto de acordo com a qt de caracteres queremos e quanto cada separação compartilhará (chunk_overlap)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    chunks = text_splitter.split_documents(docs)  # Guardamos em uma lista os "chunks" dos textos dos pdfs
    
    # Aqui acontece o embedding que é transformar cada chunk de texto em uma lista de números (um vetor) 
    # que representa o seu significado, de acordo com o modelo definido
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL) 
    
    # Aqui é onde de fato a função de Embeddings acontece e é salvo num "banco de dados de vetores" (FAISS) 
    # para permitir buscas por similaridade de forma instantanea
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    
    # Aqui nós recuperamos (retriever) as informações dos documentos que estão em vetor no FAISS de acordo com oq precisamos de fato
    # Sendo que o "K" é: Me entregue apenas os k melhores e mais relevantes documentos.
    # E o "fetch_k" é para iniciar a pesquisa a aprtir de fetch_k números de documentos, quanto maior mais poder processual e mais especifico e melhor fica
    retriever = vectorstore.as_retriever(search_type=SEARCH_TYPE, search_kwargs={'k': SEARCH_K, 'fetch_k': FETCH_K})
    return retriever


@st.cache_resource   # Decorator de ativar o cache do resultado da func (para rodar 1 vez só visto q ela é pesada)
def config_rag_chain(_llm, _retriever):
    
    # É uma função do LangChain que constrói um template a partir de uma lista de mensagens
    # Esse é o prompt que pegará o historico, a pipeline
    # Esse é o prompt que vai refazer a pergunta do usuário, para melhorar a pergunta dando os contextos
    context_q_prompt = ChatPromptTemplate.from_messages(
        [("system", CONTEXT_Q_SYSTEM_PROMPT),
         MessagesPlaceholder("chat_history"), # Aqui é permitido receber uma lista de textos, no caso o historico do chat
         ("human", "{input}")])
    
    # Aqui é a subchain que vai ser "um pesquisador ciente do histórico".
    # Seu output é os trechos dos pdfs uteis
    history_aware_retriever = create_history_aware_retriever(llm=_llm, retriever=_retriever, prompt=context_q_prompt)
    
    # Aqui é onde será o prompt final que a IA responderá de fato
    qa_prompt = ChatPromptTemplate.from_messages([("system", QA_SYSTEM_PROMPT), 
                                                  MessagesPlaceholder("chat_history"), 
                                                  ("human", "{input}")])  # Pergunta original do usuário
    
    # Aqui é a segunda subchain que junta o resultado do retriever e coloca o context dentro do qa_prompt
    qa_chain = create_stuff_documents_chain(_llm, qa_prompt)
    
    # E aqui é a junção das duas chain que dará de fato a resposta final da IA
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return rag_chain
