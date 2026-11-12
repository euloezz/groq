import plotly.express as px

# Dentro da aba de gráficos na coluna da direita (tab_graficos):
with tab_graficos:
    st.markdown("##### 📈 Análise Visual Avançada (Plotly)")
    
    colunas_numericas = df_dados.select_dtypes(include=["number"]).columns.tolist()
    colunas_todas = df_dados.columns.tolist()
    
    if colunas_numericas and len(colunas_todas) >= 2:
        tipo_grafico_avancado = st.selectbox(
            "Selecione o tipo de gráfico:", 
            ["Gráfico de Barras", "Gráfico de Dispersão (Scatter)", "Gráfico de Linha", "Histograma"]
        )
        
        # Seleção de Eixos baseada nas colunas do CSV
        col_x = st.selectbox("Eixo X:", colunas_todas, index=0)
        col_y = st.selectbox("Eixo Y (Métrica):", colunas_numericas, index=0)
        
        # Geração do gráfico interativo com Plotly
        if tipo_grafico_avancado == "Gráfico de Barras":
            fig = px.bar(df_dados, x=col_x, y=col_y, title=f"{col_y} por {col_x}", color=col_x if col_x in colunas_todas else None)
            st.plotly_chart(fig, use_container_width=True)
            
        elif tipo_grafico_avancado == "Gráfico de Dispersão (Scatter)":
            fig = px.scatter(df_dados, x=col_x, y=col_y, title=f"Dispersão: {col_y} vs {col_x}", size=colunas_numericas[0] if colunas_numericas else None)
            st.plotly_chart(fig, use_container_width=True)
            
        elif tipo_grafico_avancado == "Gráfico de Linha":
            fig = px.line(df_dados, x=col_x, y=col_y, title=f"Evolução de {col_y} por {col_x}", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            
        elif tipo_grafico_avancado == "Histograma":
            fig = px.histogram(df_dados, x=col_y, title=f"Distribuição de {col_y}")
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.info("O dataset precisa ter pelo menos uma coluna numérica para gerar gráficos analíticos.")
