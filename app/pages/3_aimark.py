# app/pages/3_aimark.py

import streamlit as st
from dotenv import load_dotenv

# Importa as funções de lógica do nosso arquivo central
from core_functions import load_llm, get_content_generation_chain

load_dotenv()

# --- CONFIGURAÇÃO DA PÁGINA E CARREGAMENTO DA LÓGICA ---
st.set_page_config(page_title="Gerador de Conteúdo 🤖", page_icon="🤖", layout="wide")
st.title("🤖 Gerador de Conteúdo com IA")
st.markdown("Use o painel à esquerda para configurar os detalhes e gerar conteúdo otimizado para suas redes.")

# Carregando o llm e a chain uma única vez
llm = load_llm()
content_chain = get_content_generation_chain(llm)

# --- LAYOUT DA PÁGINA ---
left_column, right_column = st.columns([2, 3])

with left_column:
    st.header("Configurações")

    # Definição das variáveis da página
    topic = st.text_input("Tema:", placeholder="Ex: saúde mental, IA no marketing")
    platform = st.selectbox("Plataforma:", ['Instagram', 'Facebook', 'LinkedIn', 'Blog', 'E-mail'])
    tone = st.selectbox("Tom:", ['Normal', 'Informativo', 'Inspirador', 'Urgente', 'Informal'])
    length = st.selectbox("Tamanho:", ['Curto', 'Médio', 'Longo'])
    audience = st.selectbox("Público-alvo:", ['Geral', 'Jovens adultos', 'Famílias', 'Idosos', 'Adolescentes'])
    keywords = st.text_area("Palavras-chave (SEO):", placeholder="Ex: bem-estar, medicina preventiva...")
    
    st.subheader("Opções Adicionais")
    cta = st.checkbox("Incluir uma Chamada para Ação (CTA)")
    hashtags = st.checkbox("Incluir Hashtags relevantes")
    
    generate_button = st.button("Gerar Conteúdo", type="primary", use_container_width=True)

with right_column:
    with st.container(border=True):
        if generate_button:
            if not topic:
                # Warning para colocar o conteudo no tópico
                st.warning("Por favor, insira um tema para gerar o conteúdo.")
            else:
                # Criação do prompt
                prompt = f"""
                Escreva um texto com SEO otimizado sobre o tema '{topic}'.
                Sua resposta deve ser apenas o texto final, sem frases introdutórias.
                ### Regras:
                - Plataforma: {platform}.
                - Tom: {tone}.
                - Público-alvo: {audience}.
                - Comprimento: {length}.
                - {"Inclua um CTA." if cta else "Não inclua CTA."}
                - {"Inclua hashtags." if hashtags else "Não inclua hashtags."}
                {f"- Use estas palavras-chave: {keywords}" if keywords else ""}
                """
                
                st.header("Conteúdo Gerado:")
                with st.spinner("Criando..."):
                    # Usamos a 'content_chain' que já está pronta e em cache.
                    response_stream = content_chain.stream({"prompt": prompt}) 
                    st.write_stream(response_stream)
        else:
            # Mensagem inicial para preencher o container
            st.info("Aguardando os detalhes para gerar o conteúdo.")