import streamlit as st
import pandas as pd
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
                # Tenta ler o CSV de forma robusta
                df_dados = pd.read_csv(arquivo_enviado)
                conteudo_documento = df_dados.to_string(index=False)
                st.caption(f"Status: CSV Carregado ({df_dados.shape[0]} linhas, {df_dados.shape[1]} colunas)")
            else:
                conteudo_documento = arquivo_enviado.read().decode("utf-8", errors="replace")
                st.caption(f"Status: Texto OK ({len(conteudo_documento)} chars)")
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")

    st.markdown("---")

    # Botões de Controle
    if st.button("🧹 Limpar Histórico", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# 2. COLUNA DO CENTRO: CHAT (HISTÓRICO E DIÁLOGO)
# ==============================================================================
with col_chat:
    st.subheader("💬 Diálogo")
    
    # Caixa com rolagem vertical
    chat_container = st.container(height=580)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

# ==============================================================================
# 3. COLUNA DA DIREITA: PRÉ-VISUALIZAÇÃO, ESTATÍSTICAS E GRÁFICOS
# ==============================================================================
with col_preview:
    st.subheader("📊 Visualização & Gráficos")
    if arquivo_enviado:
        st.caption(f"Arquivo: **{arquivo_enviado.name}**")
        
        # Se for um DataFrame CSV, mostra abas para Tabela, Estatísticas e Gráficos
        if df_dados is not None:
            tab_tabela, tab_stats, tab_graficos = st.tabs(["Tabela", "Estatísticas", "Gráficos"])
            
            with tab_tabela:
                st.dataframe(df_dados.head(50), height=480)
            
            with tab_stats:
                st.write(df_dados.describe())
            
            with tab_graficos:
                st.markdown("##### Gerador Rápido de Gráficos")
                colunas_numericas = df_dados.select_dtypes(include=["number"]).columns.tolist()
                
                if colunas_numericas:
                    tipo_grafico = st.selectbox("Tipo de Gráfico:", ["Linha", "Barra"])
                    coluna_selecionada = st.selectbox("Métrica/Coluna:", colunas_numericas)
                    
                    if tipo_grafico == "Linha":
                        st.line_chart(df_dados[coluna_selecionada])
                    elif tipo_grafico == "Barra":
                        st.bar_chart(df_dados[coluna_selecionada])
                else:
                    st.info("O dataset não possui colunas numéricas para gráficos.")
        else:
            # Caso seja arquivo de texto comum (.txt / .md)
            st.text_area(
                label="Conteúdo em memória:",
                value=conteudo_documento,
                height=580,
                disabled=True
            )
    else:
        st.info("Nenhum arquivo carregado. Anexe um arquivo CSV ou TXT no painel esquerdo.")

# ==============================================================================
# 4. CAMPO DE ENTRADA DO CHAT (ALINHADO À COLUNA CENTRAL)
# ==============================================================================
if user_input := st.chat_input("Faça sua pergunta sobre os dados ou converse com o bot..."):
    if not client or not modelo_selecionado:
        st.warning("⚠️ Insira uma chave de API válida no painel de controle antes de enviar perguntas.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with chat_container:
        with st.chat_message("user"):
            st.write(user_input)

    # Constrói o System Prompt (Persona + Contexto RAG estruturado)
    mensagens_api = []
    prompt_sistema = f"### IDENTIDADE E REGRAS DA PERSONA:\n{persona_customizada.strip()}\n\n"

    if conteudo_documento:
        prompt_sistema += (
            "### REGRAS ESPECÍFICAS DE CONSULTA (RAG / DADOS):\n"
            "1. Responda à dúvida do usuário usando EXCLUSIVAMENTE os dados contidos no documento/dataset abaixo.\n"
            "2. Seja analítico, preciso e direto ao responder.\n"
            "3. Se a informação solicitada não constar no documento, responda estritamente: "
            "'A informação solicitada não consta no documento fornecido.'\n\n"
            f"--- INÍCIO DOS DADOS ---\n{conteudo_documento}\n--- FIM DOS DADOS ---"
        )
    else:
        prompt_sistema += "Responda de forma clara, objetiva e coerente com sua persona."

    mensagens_api.append({"role": "system", "content": prompt_sistema})

    for m in st.session_state.messages:
        mensagens_api.append({"role": m["role"], "content": m["content"]})

    # Chamada à API
    try:
        with chat_container:
            with st.spinner("Processando resposta..."):
                response = client.chat.completions.create(
                    model=modelo_selecionado,
                    messages=mensagens_api,
                    temperature=0.1 if conteudo_documento else 0.7,
                    max_tokens=500
                )
                resposta = response.choices[0].message.content

            st.session_state.messages.append({"role": "assistant", "content": resposta})
            with st.chat_message("assistant"):
                st.write(resposta)

    except Exception as e:
        st.error(f"Erro na geração da resposta: {e}")
