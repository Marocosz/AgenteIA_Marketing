import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Seus comentários originais...

load_dotenv()


@st.cache_resource
def get_llm():
    """
    Cria e retorna uma instância do modelo de linguagem da Groq.
    """
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL_ID", "llama3-70b-8192"),
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    return llm


def llm_generate_stream(llm, prompt):
    """
    Cria a cadeia de prompt e retorna o gerador de streaming da resposta.
    """
    template = ChatPromptTemplate.from_messages([
        ("system", "Você é um especialista em marketing digital com foco em SEO e escrita persuasiva. Sempre gere conteúdo criativo e de alta qualidade."),
        ("human", "{prompt}"),
    ])
    chain = template | llm | StrOutputParser()
    return chain.stream({"prompt": prompt})


st.set_page_config(page_title="Gerador de Conteúdo 🤖",
                   page_icon="🤖", layout="wide")

st.title("🤖 Gerador de Conteúdo com IA")
st.markdown(
    "Use o painel à esquerda para configurar os detalhes e gerar conteúdo otimizado para suas redes.")

left_column, right_column = st.columns([2, 3])

with left_column:
    st.header("Configurações")

    topic = st.text_input(
        "Tema:", placeholder="Ex: saúde mental, IA no marketing")
    platform = st.selectbox(
        "Plataforma:", ['Instagram', 'Facebook', 'LinkedIn', 'Blog', 'E-mail'])


    tone = st.selectbox(
        "Tom:", ['Normal', 'Informativo', 'Inspirador', 'Urgente', 'Informal'])
    length = st.selectbox("Tamanho:", ['Curto', 'Médio', 'Longo'])

    audience = st.selectbox(
        "Público-alvo:", ['Geral', 'Jovens adultos', 'Famílias', 'Idosos', 'Adolescentes'])
    keywords = st.text_area("Palavras-chave (SEO):",
                            placeholder="Ex: bem-estar, medicina preventiva...")

    st.subheader("Opções Adicionais")
    cta = st.checkbox("Incluir uma Chamada para Ação (CTA)")
    hashtags = st.checkbox("Incluir Hashtags relevantes")

    generate_button = st.button(
        "Gerar Conteúdo", type="primary", use_container_width=True)

with right_column:
    # MUDANÇA: Usando st.container para que ele pegue a cor 'secondaryBackgroundColor' do nosso tema
    with st.container(border=True):
        if generate_button:
            if not topic:
                st.warning("Por favor, insira um tema para gerar o conteúdo.")
            else:
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
                    llm = get_llm()
                    response_stream = llm_generate_stream(llm, prompt)
                    st.write_stream(response_stream)
        else:
            # Mensagem inicial para preencher o container
            st.info("Aguardando os detalhes para gerar o conteúdo.")
