import streamlit as st
import plotly.express as px
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Configuração da página Streamlit (mantido seu estilo original)
st.set_page_config(
    page_title="Assistente RH - Analytics",
    page_icon="👥",
    layout="wide"
)

# Estilo CSS personalizado (mantido do seu código)
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stTextInput > div > div > input { padding: 0.5rem; }
    .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .user-message { background-color: #e6f3ff; }
    .bot-message { background-color: #f0f2f6; }
    </style>
""", unsafe_allow_html=True)

class ClaudeHRSystem:
    
    def __init__(self):
        # Configuração base do Claude
        self.base_config = {
            "anthropic_api_key": os.getenv('ANTHROPIC_API_KEY'),
            "temperature": 0.7,
            "model": "claude-3-opus-20240229"  # Usando o modelo mais recente
        }
        
        # Inicialização dos agentes especializados
        self.master_agent = ChatAnthropic(**self.base_config)
        self.hr_agent = ChatAnthropic(**self.base_config)
        self.data_agent = ChatAnthropic(**self.base_config)
        self.career_agent = ChatAnthropic(**self.base_config)
        
        # Templates especializados em português
        self.master_prompt = ChatPromptTemplate.from_template("""
        Você é o agente mestre do sistema de RH, especializado em análise de dados.
        IMPORTANTE: RESPONDA SEMPRE EM PORTUGUÊS DO BRASIL.

        Analise os dados e coordene a resposta mais adequada:
        {available_data}

        Pergunta: {question}

        Forneça uma análise completa e profissional, sempre em português.
        """)
        
        self.hr_prompt = ChatPromptTemplate.from_template("""
        Como especialista em RH, analise os dados focando em gestão de pessoas.
        RESPONDA EM PORTUGUÊS DO BRASIL.

        Dados: {available_data}
        Pergunta: {question}
        """)
        
        self.data_prompt = ChatPromptTemplate.from_template("""
        Como analista de dados de RH, forneça insights quantitativos.
        RESPONDA EM PORTUGUÊS DO BRASIL.

        Dados: {available_data}
        Pergunta: {question}
        """)
        
        self.career_prompt = ChatPromptTemplate.from_template("""
        Como especialista em carreira e remuneração, analise os dados.
        RESPONDA EM PORTUGUÊS DO BRASIL.

        Dados: {available_data}
        Pergunta: {question}
        """)
        
        # Chains
        self.master_chain = self.master_prompt | self.master_agent
        self.hr_chain = self.hr_prompt | self.hr_agent
        self.data_chain = self.data_prompt | self.data_agent
        self.career_chain = self.career_prompt | self.career_agent

    def process_query(self, query, data):
        """Processa a query usando os agentes especializados"""
        if not query or not isinstance(query, str):
            return "Por favor, forneça uma pergunta válida."
            
        try:
            # Determina qual agente usar baseado em palavras-chave
            query = query.lower()  # Converte para minúsculo uma única vez
            
            if any(word in query for word in ['cultura', 'equipe', 'gestão', 'clima']):
                specialist_chain = self.hr_chain
            elif any(word in query for word in ['média', 'número', 'percentual', 'quantidade']):
                specialist_chain = self.data_chain
            elif any(word in query for word in ['salário', 'carreira', 'desenvolvimento', 'habilidades']):
                specialist_chain = self.career_chain
            else:
                specialist_chain = self.hr_chain

            # Obtém análise especializada
            response = specialist_chain.invoke({
                "available_data": data.to_string(),
                "question": query
            })
            
            return response
            
        except Exception as e:
            return f"Erro ao processar a análise: {str(e)}"
def generate_visualizations(df):
    """Gera visualizações com tratamento adequado dos dados"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de distribuição por departamento
        fig_dept = px.pie(
            df['departamento'].value_counts().reset_index(),
            values='count',
            names='departamento',
            title='Distribuição por Departamento'
        )
        st.plotly_chart(fig_dept, use_container_width=True)
        
    with col2:
        # Gráfico de salários por departamento
        fig_salary = px.box(
            df,
            x='departamento',
            y='salario',
            title='Distribuição Salarial por Departamento'
        )
        st.plotly_chart(fig_salary, use_container_width=True)
    
    # Tratamento correto para as avaliações sem warning
    try:
        # Primeiro, converte avaliações se necessário
        if isinstance(df['avaliacoes'].iloc[0], str):
            df['avaliacoes'] = df['avaliacoes'].apply(eval)
        
        # Calcula médias por departamento sem gerar warning
        dept_ratings = df.groupby('departamento', group_keys=False).agg({
            'avaliacoes': lambda x: sum(sum(y)/len(y) for y in x)/len(x)
        }).reset_index()
        
        # Cria o gráfico de barras
        fig_ratings = px.bar(
            dept_ratings,
            x='departamento',
            y='avaliacoes',
            title='Média de Avaliações por Departamento'
        )
        fig_ratings.update_layout(yaxis_title='Média de Avaliações')
        st.plotly_chart(fig_ratings, use_container_width=True)
        
    except Exception as e:
        st.warning("Não foi possível gerar o gráfico de avaliações.")
        st.error(f"Erro: {str(e)}")

def format_claude_response(response):
    """Formata a resposta do Claude para exibição"""
    # Extrai apenas o conteúdo da resposta
    if hasattr(response, 'content'):
        content = response.content
    elif isinstance(response, dict):
        content = response.get('content', str(response))
    else:
        content = str(response)
    
    # Remove metadados e formata o texto
    content = content.split('response_metadata')[0]  # Remove metadados
    content = content.replace('content=', '')  # Remove prefixo
    
    # Remove aspas extras se existirem
    content = content.strip("'\"")
    
    return content

def main():
    st.title("🤖 Assistente de RH Analytics - Powered by Claude")
    
    # Inicialização do sistema
    if 'agent_system' not in st.session_state:
        st.session_state.agent_system = ClaudeHRSystem()
    
    # Carrega dados
    df = pd.read_excel(os.path.join('E:\\Python\\LLM Local', 'dados_rh.xlsx'))
    
    # Sidebar com informações e filtros
    with st.sidebar:
        st.header("📊 Dados Gerais")
        st.write(f"Total de Funcionários: {len(df)}")
        st.write(f"Departamentos: {', '.join(df['departamento'].unique())}")
        st.write(f"Faixa Salarial: R${df['salario'].min():,.2f} - R${df['salario'].max():,.2f}")
        
        st.header("🎯 Sugestões de Perguntas")
        st.write("- Qual departamento tem maior média salarial?")
        st.write("- Quais são as habilidades mais comuns na TI?")
        st.write("- Como está o clima organizacional por departamento?")
        st.write("- Quais são as tendências de desenvolvimento de carreira?")
    
    # Área principal
    tab1, tab2 = st.tabs(["💬 Chat com Claude", "📈 Visualizações"])

    with tab1:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Exibe mensagens anteriores
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input do usuário e processamento
        if prompt := st.chat_input("Digite sua pergunta sobre os dados de RH..."):
            if prompt.strip():  # Verifica se não é string vazia
                # Adiciona mensagem do usuário
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Processa e exibe resposta
                with st.chat_message("assistant"):
                    with st.spinner('Analisando dados com Claude...'):
                        try:
                            response = st.session_state.agent_system.process_query(prompt, df)
                            formatted_response = format_claude_response(response)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": formatted_response
                            })
                            st.markdown(formatted_response)
                        except Exception as e:
                            error_msg = f"Ocorreu um erro ao processar sua pergunta: {str(e)}"
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