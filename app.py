import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime
import numpy as np

# Configuração da página Streamlit
st.set_page_config(
    page_title="Assistente Vendas - Analytics",
    page_icon="🚗",
    layout="wide"
)

# Estilo CSS personalizado
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stTextInput > div > div > input { padding: 0.5rem; }
    .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .user-message { background-color: #e6f3ff; }
    .bot-message { background-color: #f0f2f6; }
    </style>
""", unsafe_allow_html=True)

class ClaudeAutopartsSystem:
    def __init__(self):
        self.base_config = {
            "anthropic_api_key": os.getenv('ANTHROPIC_API_KEY'),
            "temperature": 0.7,
            "model": "claude-3-opus-20240229"
        }
        
        self.master_agent = ChatAnthropic(**self.base_config)
        
        self.master_prompt = ChatPromptTemplate.from_template("""
        Você é um especialista em análise de dados de vendas de autopeças.
        IMPORTANTE: RESPONDA SEMPRE EM PORTUGUÊS DO BRASIL.

        Analise os dados fornecidos e responda à pergunta considerando:
        - Variações percentuais
        - Margens de lucro
        - Impactos tributários
        - Projeções futuras
        - Análise de comportamento de clientes

        Dados disponíveis:
        {available_data}

        Pergunta: {question}

        Forneça uma análise detalhada, incluindo números e percentuais relevantes.
        Se possível, sugira ações baseadas nos insights encontrados.
        """)

        self.customer_analysis_prompt = ChatPromptTemplate.from_template("""
        Você é um especialista em análise de churn e comportamento de clientes.
        IMPORTANTE: RESPONDA SEMPRE EM PORTUGUÊS DO BRASIL.

        Analise os dados fornecidos e identifique:
        1. Liste os clientes que compraram em 2023 mas não em 2024
        2. Para cada cliente identificado, forneça:
           - Valor total comprado em 2023
           - Última data de compra
           - Produtos mais comprados
           - Regional onde estava localizado
        3. Sugira possíveis razões para a não renovação com base nos padrões de compra

        Dados disponíveis:
        {available_data}

        Formatação desejada:
        - Liste os clientes em ordem alfabética
        - Apresente os valores monetários formatados em reais
        - Inclua porcentagens quando relevante

        Após a análise, forneça recomendações práticas para reativação desses clientes.
        """)

        self.customer_chain = self.customer_analysis_prompt | self.master_agent

    def process_query(self, query, df):
        """
        Processa todas as queries através do Claude para análises mais completas e contextualizadas.
        Cada tipo de análise tem seu prompt especializado para garantir respostas direcionadas.
        """
        query_lower = query.lower()
        
        try:
            # Define o prompt específico baseado no tipo de análise solicitada
            if "variação" in query_lower and "vendedor" in query_lower:
                analysis_prompt = ChatPromptTemplate.from_template("""
                Analise a variação percentual das vendas entre 2023 e 2024 por vendedor.
                
                Forneça:
                1. Variação percentual para cada vendedor
                2. Análise das possíveis razões para as variações encontradas
                3. Destaque os vendedores com melhor e pior desempenho
                4. Tendências observadas nas vendas
                5. Recomendações para melhorias
                
                Dados disponíveis:
                {available_data}
                """)
                
            elif "melhor margem" in query_lower and "regional" in query_lower:
                analysis_prompt = ChatPromptTemplate.from_template("""
                Analise as margens por regional e forneça um ranking detalhado.
                
                Inclua:
                1. Ranking completo das regionais por margem média
                2. Análise detalhada das 3 melhores regionais
                3. Fatores que contribuem para o sucesso dessas regionais
                4. Oportunidades de melhoria para as demais
                5. Análise de produtos com melhores margens por regional
                
                Dados disponíveis:
                {available_data}
                """)
                
            elif "impacto" in query_lower and "imposto" in query_lower:
                analysis_prompt = ChatPromptTemplate.from_template("""
                Analise o impacto do aumento de 3% nos impostos da regional Sudeste.
                
                Considere:
                1. Impacto nos preços atuais
                2. Efeito na competitividade dos produtos
                3. Possível impacto nas vendas
                4. Estratégias de mitigação
                5. Comparação com outras regionais
                6. Recomendações de ajustes de preço e margem
                
                Dados disponíveis:
                {available_data}
                """)
                
            elif "projete" in query_lower and "faturamento" in query_lower:
                analysis_prompt = ChatPromptTemplate.from_template("""
                Projete o faturamento para os próximos 3 anos considerando:
                - Dólar atual: BRL 5,00
                - Aumento projetado do dólar: 10% ao ano
                - Inflação projetada: 4,5% ao ano
                
                Forneça:
                1. Projeção detalhada ano a ano
                2. Impacto do dólar e inflação separadamente
                3. Análise de riscos e oportunidades
                4. Cenários otimista e pessimista
                5. Recomendações estratégicas
                
                Dados disponíveis:
                {available_data}
                """)
                
            elif "clientes" in query_lower and "2023" in query_lower and "2024" in query_lower:
                analysis_prompt = ChatPromptTemplate.from_template("""
                Analise os clientes que compraram em 2023 mas não em 2024.
                
                Para cada cliente identifique:
                1. Perfil completo de compras em 2023:
                   - Valor total comprado
                   - Frequência de compras
                   - Produtos mais comprados
                   - Margem média das vendas
                
                2. Análise do histórico:
                   - Última compra realizada
                   - Padrão de comportamento
                   - Regional e vendedor responsável
                
                3. Possíveis razões para não renovação:
                   - Análise de preços praticados
                   - Comparação com concorrência
                   - Mudanças no mercado
                
                4. Recomendações:
                   - Estratégias de reativação
                   - Ajustes necessários
                   - Priorização de ações
                
                Dados disponíveis:
                {available_data}
                """)
                
            else:
                # Prompt padrão para outras análises
                analysis_prompt = self.master_prompt
            
            # Cria uma nova chain com o prompt específico
            analysis_chain = analysis_prompt | self.master_agent
            
            # Processa a análise
            response = analysis_chain.invoke({
                "available_data": df.to_string(),
                "question": query
            })
            
            return response.content

        except Exception as e:
            return f"Erro ao processar a análise: {str(e)}"

def generate_visualizations(df):
    """Gera visualizações específicas para dados de autopeças"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Vendas por Regional
        fig_regional = px.pie(
            df.groupby('Regional de Vendas')['Valor da Venda'].sum().reset_index(),
            values='Valor da Venda',
            names='Regional de Vendas',
            title='Distribuição de Vendas por Regional'
        )
        st.plotly_chart(fig_regional, use_container_width=True)
        
    with col2:
        # Margem por Regional
        fig_margin = px.box(
            df,
            x='Regional de Vendas',
            y='Margem (%)',
            title='Distribuição de Margem por Regional'
        )
        st.plotly_chart(fig_margin, use_container_width=True)
    
    # Vendas ao longo do tempo
    df['Data Venda'] = pd.to_datetime(df['Data Venda'])
    vendas_tempo = df.groupby('Data Venda')['Valor da Venda'].sum().reset_index()
    fig_timeline = px.line(
        vendas_tempo,
        x='Data Venda',
        y='Valor da Venda',
        title='Evolução das Vendas ao Longo do Tempo'
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

def main():
    st.title("🚗 Assistente de Vendas Autopeças")
    
    if 'agent_system' not in st.session_state:
        st.session_state.agent_system = ClaudeAutopartsSystem()
    
    # Carrega dados
    df = pd.read_excel('data/dados_autoparts.xlsx')
    
    with st.sidebar:
        st.header("📊 Dados Gerais")
        st.write(f"Total de Vendas: {len(df)}")
        st.write(f"Regionais: {', '.join(df['Regional de Vendas'].unique())}")
        st.write(f"Período: {df['Data Venda'].min()} a {df['Data Venda'].max()}")
        
        st.header("🎯 Sugestões de Perguntas")
        st.write("- Qual a variação em % das vendas por vendedor?")
        st.write("- Qual regional possui a melhor margem?")
        st.write("- Qual o impacto no preço com aumento de impostos?")
        st.write("- Projete o faturamento para os próximos 3 anos")
        st.write("- Quais clientes não compraram em 2024?")
    
    tab1, tab2 = st.tabs(["💬 Chat com Agente IA", "📈 Visualizações"])

    with tab1:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Digite sua pergunta sobre as vendas..."):
            if prompt.strip():
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner('Analisando dados...'):
                        try:
                            response = st.session_state.agent_system.process_query(prompt, df)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response
                            })
                            st.markdown(response)
                        except Exception as e:
                            error_msg = f"Erro ao processar sua pergunta: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": error_msg
                            })
            else:
                st.warning("Por favor, digite uma pergunta válida.")
            
    with tab2:
        generate_visualizations(df)

if __name__ == "__main__":
    main()