import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

# Configuração da página em modo wide
st.set_page_config(page_title="Chatbot Didático: Groq + RAG + CSV", page_icon="🤖", layout="wide")

st.title("🤖 Laboratório de IA: Chatbot, RAG & Análise de CSV com Groq")

# Inicialização do histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================================================================
# BARRA LATERAL (SIDEBAR): CONTROLOS, API, PERSONA E UPLOAD
# ==============================================================================
with st.sidebar:
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
        st.info("Cole sua chave acima na barra lateral.")

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

    # Botão de Limpar Histórico
    if st.button("🧹 Limpar Histórico", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# ÁREA PRINCIPAL: DIVIDIDA EM DUAS COLUNAS (CHAT E VISUALIZAÇÃO/GRÁFICOS)
# ==============================================================================
col_chat, col_preview = st.columns(spec=[1.2, 1.8], gap="medium")

# 1. Coluna do Chat
with col_chat:
    st.subheader("💬 Diálogo")
    chat_container = st.container(height=550)
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

# 2. Coluna de Visualização, Estatísticas e Gráficos Plotly
with col_preview:
    st.subheader("📊 Visualização & Gráficos")
    
    if arquivo_enviado:
        st.caption(f"Arquivo ativo: **{arquivo_enviado.name}**")
        
        if df_dados is not None:
            tab_tabela, tab_stats, tab_graficos = st.tabs(["Tabela", "Estatísticas", "Gráficos"])
            
            with tab_tabela:
                st.dataframe(df_dados, height=450, use_container_width=True)
            
            with tab_stats:
                st.write(df_dados.describe())
            
            with tab_graficos:
                st.markdown("##### 📈 Análise Visual Avançada (Plotly)")
                colunas_numericas = df_dados.select_dtypes(include=["number"]).columns.tolist()
                colunas_todas = df_dados.columns.tolist()
                
                if colunas_numericas and len(colunas_todas) >= 2:
                    tipo_grafico_avancado = st.selectbox(
                        "Selecione o tipo de gráfico:", 
                        ["Gráfico de Barras", "Gráfico de Dispersão (Scatter)", "Gráfico de Linha", "Histograma"]
                    )
                    
                    col_x = st.selectbox("Eixo X:", colunas_todas, index=0)
                    col_y = st.selectbox("Eixo Y (Métrica):", colunas_numericas, index=0)
                    
                    if tipo_grafico_avancado == "Gráfico de Barras":
                        fig = px.bar(df_dados, x=col_x, y=col_y, title=f"{col_y} por {col_x}")
                        st.plotly_chart(fig, use_container_width=True)
                    elif tipo_grafico_avancado == "Gráfico de Dispersão (Scatter)":
                        fig = px.scatter(df_dados, x=col_x, y=col_y, title=f"Dispersão: {col_y} vs {col_x}")
                        st.plotly_chart(fig, use_container_width=True)
                    elif tipo_grafico_avancado == "Gráfico de Linha":
                        fig = px.line(df_dados, x=col_x, y=col_y, title=f"Evolução de {col_y} por {col_x}", markers=True)
                        st.plotly_chart(fig, use_container_width=True)
                    elif tipo_grafico_avancado == "Histograma":
                        fig = px.histogram(df_dados, x=col_y, title=f"Distribuição de {col_y}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("O dataset precisa de pelo menos uma coluna numérica para gerar gráficos.")
        else:
            st.text_area(label="Conteúdo em memória:", value=conteudo_documento, height=480, disabled=True)
    else:
        st.info("👈 Faça o upload de um arquivo CSV ou TXT na barra lateral para começar a visualizar os dados e gráficos.")

# ==============================================================================
# CAMPO DE ENTRADA DO CHAT
# ==============================================================================
if user_input := st.chat_input("Faça sua pergunta sobre os dados ou converse com o bot..."):
    if not client or not modelo_selecionado:
        st.warning("⚠️ Insira uma chave de API válida na barra lateral antes de enviar perguntas.")
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

    # Chamada à API da Groq
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
