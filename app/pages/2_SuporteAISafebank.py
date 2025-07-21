# app/pages/2_aisup.py

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from config import DOCS_PATH
from core_functions import load_llm, config_retriever, config_rag_chain

load_dotenv()

# --- CONFIGURAÇÃO INICIAL DA PÁGINA
st.set_page_config(page_title="Atendimento SafeBank 🤖", page_icon="🤖")
st.title("Atendimento SafeBank")


# Carrega os recursos pesados usando o cache.
# O spinner mostra uma mensagem enquanto a função é executada pela PRIMEIRA VEZ.
with st.spinner("Carregando modelo de linguagem..."):
    llm = load_llm()
    
with st.spinner("Analisando e indexando documentos PDF... (isso pode levar um tempo na primeira vez)"):
    retriever = config_retriever(DOCS_PATH)

# Cria a RAG chain, também com cache.
rag_chain = config_rag_chain(llm, retriever)

# Inicializa o histórico do chat na sessão se ele não existir.
# Cria o "chat_history" dentro da sessão do streamlit e dá a mensagem inicial
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        # AIMessage é a classe para dizer que essa mensagem foi declarada pelo bot
        AIMessage(
            content="Olá! Sou o assistente virtual do SafeBank. Como posso te ajudar hoje?"),
    ]

# Mostra o histórico do chat na tela. 
for message in st.session_state.chat_history:
    # Define "AI" se a mensagem ta sendo instanciada pela classe AIMessage, se não Role = "Human"
    role = "AI" if isinstance(message, AIMessage) else "Human"
    
    # Aqui o streamlit define os balões de mensagem de acordo com a role
    with st.chat_message(role):
        # Aqui é conteúdo que vai aparecer
        st.write(message.content)

# Captura e processa a entrada do usuário.
# Definimos a caixa de mensagem do streamlit com o placeholder "digite sua mensagem aqui..."
# e a mensagem do usuário é atribuido a essa variavel "user_input"
if user_input := st.chat_input("Digite sua mensagem aqui..."):
    # Cria o balão de usuário no estilo "Human" com oque o usuário escreveu
    st.chat_message("Human").write(user_input)
    
    with st.chat_message("AI"):
        with st.spinner("Pensando..."):
            # Adiciona a nova pergunta ao histórico antes de invocar a chain
            st.session_state.chat_history.append(
                HumanMessage(content=user_input))

            # Executa a rag_chain 
            response = rag_chain.invoke({
                "input": user_input,  # A pergunta atual do usuário
                "chat_history": st.session_state.chat_history  # O histórico de conversa
            })
            
            # A response retornada pela chain é um dicionário que pode conter várias informações 
            # Aqui pegamos o valor da chave "answer" que é texto final da resposta gerada pela IA
            answer = response["answer"]

            # Adiciona a resposta da IA ao histórico e à tela
            st.session_state.chat_history.append(AIMessage(content=answer))
            st.write(answer)