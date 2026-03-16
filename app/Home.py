"""
Dashboard Streamlit - Análise da Saúde Financeira do Banco Master

Projeto acadêmico: Extração e Preparação de Dados
Fonte de dados: BCB IF.data (dados públicos)
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Banco Master - Análise Financeira",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "preparado"


def main():
    st.title("Caso Banco Master: Análise da Saúde Financeira")
    st.markdown("---")

    st.markdown("""
    ### Projeto Acadêmico — Extração e Preparação de Dados

    Este dashboard analisa a **deterioração da saúde financeira do Banco Master**
    entre 2019 e 2025, comparando seus indicadores com bancos de porte similar.

    **Pipeline de dados:**
    1. **Origem** — BCB IF.data (API OData), BCB SGS, fontes de notícias
    2. **Download** — Crawlers automatizados em Python
    3. **Extração** — Parsing e normalização dos dados brutos
    4. **Preparação** — Cálculo de indicadores derivados e score de estresse
    5. **Exibição** — Dashboard interativo (esta página)
    6. **Predição** — Modelos de classificação de risco (Decision Tree + Logistic Regression)
    """)

    st.markdown("---")

    # Verificar se dados existem
    arquivos = {
        "indicadores_resumo.csv": "Indicadores Financeiros",
        "indicadores_capital.csv": "Indicadores de Capital (Basileia)",
        "score_estresse.csv": "Score de Estresse",
        "predicao_risco.csv": "Predição de Risco",
    }

    st.subheader("Status dos Dados")
    cols = st.columns(4)
    for i, (arquivo, nome) in enumerate(arquivos.items()):
        with cols[i]:
            existe = (DATA_DIR / arquivo).exists()
            if existe:
                st.success(f"✅ {nome}")
            else:
                st.warning(f"⏳ {nome}")

    if not (DATA_DIR / "indicadores_resumo.csv").exists():
        st.info("""
        **Para gerar os dados, execute na ordem:**
        ```bash
        python -X utf8 pipeline.py
        ```
        """)

    st.markdown("---")

    # Cards de contexto
    st.subheader("O Caso em Números")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ativo Total (Q4/2024)", "R$ 63 bi", "+74% em 1 ano")
    c2.metric("Índice de Basileia", "11,52%", "Mínimo = 11%", delta_color="inverse")
    c3.metric("Perda FGC Estimada", "R$ 41 bi", "Maior da história")
    c4.metric("Crescimento 2019-2025", "22x", "Ativo Total")

    st.markdown("---")

    st.subheader("Navegação")
    st.markdown("""
    Use o menu lateral para navegar entre as páginas:

    - **Documentação** — Detalhamento do projeto, metodologia, limitações e referências
    - **Saúde Financeira** — Gráficos comparativos de indicadores ao longo do tempo
    - **Sinais de Alerta** — Heatmap de red flags e score de estresse financeiro
    - **Notícias** — Timeline interativa dos eventos do caso
    - **Predição** — Dois modelos de classificação de risco, simulador interativo e matriz de confusão
    """)

    st.markdown("---")

    # Contexto do caso
    st.subheader("Contexto do Caso")
    st.markdown("""
    O **Banco Master**, sob a gestão de Daniel Vorcaro, cresceu de R\$ 3 bilhões para
    R\$ 68 bilhões em ativos entre 2019 e 2025, financiado majoritariamente por CDBs
    com taxas acima do mercado (até 130% do CDI).

    Em **novembro de 2025**, o Banco Central decretou sua liquidação extrajudicial após
    a Polícia Federal revelar uma fraude de R\$ 12 bilhões. O caso é considerado o maior
    escândalo financeiro do Brasil desde a Operação Lava Jato.

    **Crimes identificados:** fraude em precatórios, lavagem de dinheiro (incluindo para
    o PCC), manipulação de mercado, corrupção de diretores do Banco Central.
    """)

    st.markdown("---")

    # Bancos analisados
    st.subheader("Bancos Analisados")
    st.markdown("""
    | Banco | Porte | Papel na Análise |
    |-------|-------|------------------|
    | **Banco Master** | Médio | Alvo — liquidado em Nov/2025 |
    | Banco Inter | Médio-grande | Controle — crescimento acelerado mas saudável |
    | Banco Pine | Médio | Controle — tradicional, conservador |
    | Banco Original | Médio | Controle — porte similar |
    | Banco Daycoval | Médio-grande | Controle — referência de estabilidade |
    """)

    st.markdown("---")
    st.caption(
        "Dados: BCB IF.data (dados públicos) | "
        "Bancos comparados: Inter, Pine, Original, Daycoval | "
        "Período: Q1/2019 a Q3/2025"
    )


if __name__ == "__main__":
    main()
