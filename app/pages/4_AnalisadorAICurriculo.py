import streamlit as st
import os
import uuid
from dotenv import load_dotenv


from config import (
    CV_JSON_FILE, CV_JOB_CSV_FILE, JOB_DETAILS, CV_REQUIRED_FIELDS
)
from prompts import CV_SCHEMA, CV_PROMPT_SCORE

from core_functions import (
    load_llm, get_cv_analysis_chain, parse_doc, parse_res_llm,
    save_json_cv, show_cv_result, display_json_table,
    format_job_details # Importa a nova função de formatação
)

load_dotenv()

# --- FUNÇÃO DE CALLBACK PARA O BOTÃO ---
def set_selected_cv(cv_data):
    """Esta função é chamada quando um botão 'Ver detalhes' é clicado para definir o estado."""
    st.session_state.selected_cv = cv_data

def clear_selected_cv():
    """Esta função é chamada pelo botão 'Fechar Detalhes' para limpar a seleção."""
    st.session_state.selected_cv = None

# --- CONFIGURAÇÃO DA PÁGINA E CARREGAMENTO DA LÓGICA ---
st.set_page_config(page_title="Triagem e Análise de Currículos", page_icon="📄", layout="wide")

# Carrega o LLM e a Chain uma única vez usando o cache
llm = load_llm()
cv_chain = get_cv_analysis_chain(llm)

# --- LÓGICA DA PÁGINA
# Inicializa o estado da sessão para resetar o uploader e guardar a seleção
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())
    
if "selected_cv" not in st.session_state:
    st.session_state.selected_cv = None

# Formata a descrição da vaga (importada do config.py) para usar no prompt
job_details_text = format_job_details(JOB_DETAILS)


# --- LAYOUT DA PÁGINA
st.header("Triagem e Análise de Currículos")
st.markdown(f"#### Vaga: {JOB_DETAILS['title']}")

# Seção de upload os curriculos
uploaded_files = st.file_uploader(
    "Envie um ou mais currículos em PDF ou DOCX",
    type=["pdf", "docx"],
    key=st.session_state.uploader_key,
    accept_multiple_files=True
)

# Bloco principal que é executado quando arquivos são enviados
if uploaded_files:
    # Para cada curriculo 
    for single_upload in uploaded_files:
        # Abre um warning
        with st.spinner(f"Analisando: {single_upload.name}..."):
            temp_path = os.path.join(".", single_upload.name) # Indica aonde o single_upload é pra ser salvo
            
            with open(temp_path, "wb") as f:  # Abre o local do cache
                f.write(single_upload.read())  # Le o conteudo do curriculo

            content = parse_doc(temp_path)  # Le o arquivo temporário que acabamos de criar e o transforma em markdown com a função
            
            # Aqui é onde chamamos a chain da IA com os devidos esquemas de prompt
            output = cv_chain.invoke({
                "schema": CV_SCHEMA, "cv": content, "job": job_details_text, "prompt_score": CV_PROMPT_SCORE
            })
            
            # Aqui pegamos a resposta da IA e extraimos o JSON limpo e formatado
            structured_data = parse_res_llm(output, CV_REQUIRED_FIELDS)

            if structured_data:
                save_json_cv(structured_data, path_json=CV_JSON_FILE, key_name="name")  # Salvamos o json para um arquivo (curriculos.json)
                
                st.success(f"'{structured_data.get('name', single_upload.name)}' analisado com sucesso!")
                st.session_state.selected_cv = structured_data
            else:
                st.error(f"Não foi possível extrair dados do currículo '{single_upload.name}'.")
            
            os.remove(temp_path)

    # Reseta o uploader e força o reinício APENAS UMA VEZ, após o loop
    st.session_state.uploader_key = str(uuid.uuid4()) # Gera um nove id aleatorio pra seção de upload de arquivos e ai, reseta


# --- SEÇÃO DE EXIBIÇÃO DOS CURRÍCULOS
if os.path.exists(CV_JSON_FILE):  # Se o json existir
    st.subheader("Lista de currículos analisados", divider="gray")
    
    # Cria a tabela de acordo com json file
    df = display_json_table(CV_JSON_FILE)

    if not df.empty:
        # Lógica para exibir cada currículo com um botão de detalhes
        for i, row in df.iterrows():
            cols = st.columns([1, 3, 1, 5])
            with cols[0]:
                st.button(
                    "Ver detalhes",
                    key=f"btn_{i}",
                    on_click=set_selected_cv,    # Chama a função de callback
                    args=(row.to_dict(),)       # Passa os dados do CV para a função
                )
            with cols[1]:
                st.write(f"**Nome:** {row.get('name', '-')}")
            with cols[2]:
                st.write(f"**Score:** {row.get('score', '-')}")
            with cols[3]:
                st.write(f"**Resumo:** {row.get('summary', '-')}")

        # Se um CV foi selecionado, mostra os detalhes completos abaixo
        if st.session_state.selected_cv:
            st.markdown("---")
            st.subheader(f"Detalhes de: {st.session_state.selected_cv.get('name', 'N/A')}")
            st.write(show_cv_result(st.session_state.selected_cv))
            
            with st.expander("Ver dados estruturados (JSON) do candidato selecionado"):
                st.json(st.session_state.selected_cv)
            
            st.button("Fechar Detalhes", key="close_details", on_click=clear_selected_cv)

        # Adiciona a tabela de visão geral e o botão de download
        st.subheader("Visão Geral em Tabela", divider="gray")
        st.dataframe(df) # Dataframe com altura definida

        with open(CV_JSON_FILE, "r", encoding="utf-8") as f:
            json_data = f.read()
            
        st.download_button(
            label="📥 Baixar arquivo .json com todos os candidatos",
            data=json_data,
            file_name=CV_JSON_FILE,
            mime="application/json"
        )
    else:
        st.info("Nenhum currículo foi analisado ainda.")