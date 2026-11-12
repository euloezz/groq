import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

# Layout wide para permitir controle fino das colunas
st.set_page_config(page_title="Chatbot Didático: Groq + RAG + CSV", page_icon="🤖", layout="wide")

# CSS para alinhar o st.chat_input exatamente na largura e posição da coluna central (60% do total)
st.markdown("""
<style>
    /* Trava o chat_input na largura da coluna do meio e centraliza */
    [data-testid="stChatInput"] {
        max-width: 60% !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Laboratório de IA: Chatbot, RAG & Análise de CSV com Groq")

# Inicialização do histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================================================================
# PROPORÇÃO DAS COLUNAS: LATERAIS (0.8) E CHAT EXPANDIDO (2.4)
# ==============================================================================
col_controles, col_chat, col_preview = st.columns([0.8, 2.4, 0.8], gap="medium")

# ==============================================================================
# 1. COLUNA DA ESQUERDA: API, PERSONA, UPLOAD E CONTROLES
# ==============================================================================
with col_controles:
    st.subheader("⚙️ Controles")

    # Bloco API
    st.markdown("##### 🔑 API & Modelo")
    user_api_key = st.text_input(
        "API Key (Groq):", 
        type="password", 
        placeholder="gsk_...",
        help="Insira a chave obtida em console.groq.com"
    )

    BASE_URL = "https://api.groq.com/openai/v1"
    modelo_selecionado = None
    client = None

    if user_api_key:
        try:
            client = OpenAI(base_url=BASE_URL, api_key=user_api_key.strip())
            
            # Filtra modelos de texto/chat disponíveis
            modelos_disponiveis = [
                m.id for m in client.models.list().data 
                if not any(termo in m.id.lower() for termo in ["whisper", "guard", "audio"])
            ]
            
            if modelos_disponiveis:
                padrao_idx = 0
                for idx, nome in enumerate(sorted(modelos_disponiveis)):
                    if "llama" in nome.lower() or "gpt-oss" in nome.lower():
                        padrao_idx = idx
                        break
                        
                modelo_selecionado = st.selectbox(
                    "Modelo Ativo:", 
                    sorted(modelos_disponiveis),
                    index=padrao_idx
                )
                st.success("Conectado!", icon="✅")
            else:
                st.error("Nenhum modelo de chat liberado.")
        except Exception as e:
            st.error(f"Erro: {e}")
            client = None
    else:
        st.info("Cole sua chave acima.")

    st.markdown("---")

    # Bloco Persona
    st.markdown("##### 🎭 Persona da IA")
    persona_padrao = "Você é um cientista de dados especialista em analisar datasets, extrair insights e explicar tendências de forma clara."
    persona_customizada = st.text_area(
        "Instrução da IA:",
        value=persona_padrao,
        height=100,
        help="Orienta o comportamento do modelo durante as respostas."
    )

    st.markdown("---")

    # Bloco Upload
    st.markdown("##### 📁 Documento / Dataset (RAG)")
    arquivo_enviado = st.file_uploader(
        "Upload (.txt, .csv, .md):",
        type=["txt", "csv", "md"]
    )
    
    conteudo_documento = ""
    df_dados = None
    
    if arquivo_enviado:
        try:
            nome_arquivo = arquivo_enviado.name.lower()
            if nome_arquivo.endswith(".csv"):
                df_dados = pd.read_csv(arquivo_enviado)
                conteudo_documento = df_dados.to_string(index=False)
                st.caption(f"Status: CSV Carregado ({df_dados.shape[0]} linhas, {df_dados.shape[1]} colunas)")
            else:
                conteudo_documento = arquivo_enviado.read().decode("utf-8", errors="replace")
                st.caption(f"Status: Texto OK ({len(conteudo_documento)} chars)")
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")

    st.markdown("---")

    # Bot
