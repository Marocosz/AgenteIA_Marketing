# app/pages/2_aisup.py

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

# --- NOVOS IMPORTS ---
# Importa as configurações e a lógica dos outros arquivos
from config import DOCS_PATH
from core_functions import load_llm, config_retriever, config_rag_chain

load_dotenv()

# --- CONFIGURAÇÃO INICIAL DA PÁGINA ---
st.set_page_config(page_title="Atendimento SafeBank 🤖", page_icon="🤖")
st.title("Atendimento SafeBank")

# --- LÓGICA PRINCIPAL DA APLICAÇÃO ---

# 1. Carrega os recursos pesados usando o cache.
# O spinner mostra uma mensagem amigável enquanto a função é executada pela PRIMEIRA VEZ.
with st.spinner("Carregando modelo de linguagem..."):
    llm = load_llm()
with st.spinner("Analisando e indexando documentos PDF... (isso pode levar um tempo na primeira vez)"):
    retriever = config_retriever(DOCS_PATH)

# 2. Cria a RAG chain, também com cache.
rag_chain = config_rag_chain(llm, retriever)

# 3. Inicializa o histórico do chat na sessão se ele não existir.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(
            content="Olá! Sou o assistente virtual do SafeBank. Como posso te ajudar hoje?"),
    ]

# 4. Mostra o histórico do chat na tela. 
for message in st.session_state.chat_history:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        st.write(message.content)

# 5. Captura e processa a entrada do usuário.
if user_input := st.chat_input("Digite sua mensagem aqui..."):
    st.chat_message("Human").write(user_input)
    with st.chat_message("AI"):
        with st.spinner("Pensando..."):
            # Adiciona a nova pergunta ao histórico antes de invocar a chain
            st.session_state.chat_history.append(
                HumanMessage(content=user_input))

            response = rag_chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })
            answer = response["answer"]

            # Adiciona a resposta da IA ao histórico e à tela
            st.session_state.chat_history.append(AIMessage(content=answer))
            st.write(answer)