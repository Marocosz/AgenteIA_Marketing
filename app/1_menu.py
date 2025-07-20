# 1_🏠_Pagina_Inicial.py

import streamlit as st

st.set_page_config(
    page_title="Menu das IAs",
    page_icon="🌟",
    layout="wide"
)

st.title("🌟 Bem-vindo ao Menu das IAs")

st.markdown("---")

st.header("Sobre este projeto")
st.write(
    "Este é um espaço central para demonstrar diversas aplicações que desenvolvi usando Streamlit e outras tecnologias de Inteligência Artificial."
)
st.write(
    "Cada aplicação é uma página independente. Use o menu na barra lateral à esquerda para navegar entre elas."
)

st.sidebar.success("Selecione uma aplicação acima para começar.")

st.info("O código de todas as aplicações está disponível no meu GitHub!") # Exemplo