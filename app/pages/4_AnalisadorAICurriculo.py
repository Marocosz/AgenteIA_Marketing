import streamlit as st
import os
import uuid
from dotenv import load_dotenv

# --- IMPORTS DOS NOSSOS MÓDULOS ---
# Importa as configurações, prompts e a lógica dos outros arquivos
from config import (
    CV_JSON_FILE, CV_JOB_CSV_FILE, JOB_DETAILS, CV_REQUIRED_FIELDS
)
from prompts import CV_SCHEMA, CV_PROMPT_SCORE
from core_functions import (
    load_llm, get_cv_analysis_chain, parse_doc, parse_res_llm,
    save_json_cv, save_job_to_csv, load_job, show_cv_result,
    display_json_table
)

load_dotenv()

# --- CONFIGURAÇÃO DA PÁGINA E CARREGAMENTO DA LÓGICA ---
st.set_page_config(page_title="Triagem e Análise de Currículos", page_icon="📄", layout="wide")

# Carrega o LLM e a Chain uma única vez usando o cache do core_functions.py
llm = load_llm()
cv_chain = get_cv_analysis_chain(llm)

# --- LÓGICA DA PÁGINA ---
# Inicializa o estado da sessão para resetar o uploader e guardar a seleção
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())
if "selected_cv" not in st.session_state:
    st.session_state.selected_cv = None

# Salva a descrição da vaga (importada do config.py) em um .csv e a carrega
save_job_to_csv(JOB_DETAILS, CV_JOB_CSV_FILE)
job_details_text = load_job(CV_JOB_CSV_FILE)

# --- LAYOUT DA PÁGINA ---
st.header("Triagem e Análise de Currículos")
st.markdown(f"#### Vaga: {JOB_DETAILS['title']}")

uploaded_file = st.file_uploader(
    "Envie um currículo em PDF ou DOCX",
    type=["pdf", "docx"],
    key=st.session_state.uploader_key
)

# Bloco principal que é executado quando um arquivo é enviado
if uploaded_file is not None:
    with st.spinner("Analisando o currículo... (Isso pode levar um momento)"):
        # Salva o arquivo temporariamente para processamento local
        temp_path = os.path.join(".", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # 1. Extrai o conteúdo do CV
        content = parse_doc(temp_path)

        # 2. Executa a chain de análise (que já está em cache)
        output = cv_chain.invoke({
            "schema": CV_SCHEMA,
            "cv": content,
            "job": job_details_text,
            "prompt_score": CV_PROMPT_SCORE
        })
        
        # 3. Extrai o JSON da resposta do modelo
        structured_data = parse_res_llm(output, CV_REQUIRED_FIELDS)

        # 4. Se a extração foi bem-sucedida, salva e mostra os resultados
        if structured_data:
            save_json_cv(structured_data, path_json=CV_JSON_FILE, key_name="name")
            st.success("Currículo analisado com sucesso!")
            st.write(show_cv_result(structured_data))
            with st.expander("Ver dados estruturados (JSON)"):
                st.json(structured_data)
        else:
            st.error("Não foi possível extrair os dados do currículo. A resposta do modelo pode estar mal formatada. Tente novamente ou com outro arquivo.")

        # 5. Limpa o arquivo temporário e reseta o uploader
        os.remove(temp_path)
        st.session_state.uploader_key = str(uuid.uuid4())
        st.rerun()


# --- SEÇÃO DE EXIBIÇÃO DOS CURRÍCULOS JÁ ANALISADOS ---
if os.path.exists(CV_JSON_FILE):
    st.subheader("Lista de currículos analisados", divider="gray")
    
    # Usa a função importada para carregar e mostrar a tabela
    df = display_json_table(CV_JSON_FILE)

    if not df.empty:
        # Lógica para exibir cada currículo com um botão de detalhes
        for i, row in df.iterrows():
            cols = st.columns([1, 3, 1, 5])
            with cols[0]:
                if st.button("Ver detalhes", key=f"btn_{i}"):
                    st.session_state.selected_cv = row.to_dict()
            with cols[1]:
                st.write(f"**Nome:** {row.get('name', '-')}")
            with cols[2]:
                st.write(f"**Score:** {row.get('score', '-')}")
            with cols[3]:
                st.write(f"**Resumo:** {row.get('summary', '-')}")

        # Se um CV foi selecionado, mostra os detalhes completos abaixo
        if st.session_state.selected_cv:
            st.markdown("---")
            st.write(show_cv_result(st.session_state.selected_cv))
            with st.expander("Ver dados estruturados (JSON) do candidato selecionado"):
                st.json(st.session_state.selected_cv)

        # Adiciona o botão de download
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