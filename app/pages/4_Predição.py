"""
Página 4: Predição — Score de Estresse e Modelo de Risco.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Predição de Risco", layout="wide")

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "preparado"

CORES = {
    "Banco Master": "#e74c3c",
    "Banco Inter": "#3498db",
    "Banco Pine": "#2ecc71",
    "Banco Original": "#f39c12",
    "Banco Daycoval": "#9b59b6",
}


@st.cache_data
def carregar_score():
    caminho = DATA_DIR / "score_estresse.csv"
    if not caminho.exists():
        return pd.DataFrame()
    return pd.read_csv(caminho, encoding="utf-8-sig", parse_dates=["DataRef"])


@st.cache_data
def carregar_predicao():
    caminho = DATA_DIR / "predicao_risco.csv"
    if not caminho.exists():
        return pd.DataFrame()
    return pd.read_csv(caminho, encoding="utf-8-sig", parse_dates=["DataRef"])


def retreinar_modelo():
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from predicao.modelo_risco import executar_predicao
        return executar_predicao()
    except Exception as e:
        st.error(f"Erro ao retreinar: {e}")
        return {}


def main():
    st.title("Predição — Análise de Risco Bancário")

    df_score = carregar_score()
    df_pred = carregar_predicao()

    if df_score.empty and df_pred.empty:
        st.error("Dados não encontrados. Execute o pipeline primeiro.")
        return

    # =====================================================================
    # PARTE 1: SCORE DE ESTRESSE (evolução temporal)
    # =====================================================================
    st.header("1. Score de Estresse — Evolução do Risco ao Longo do Tempo")

    st.markdown("""
    O Score de Estresse é um indicador composto (0-100 pontos) que combina 4 dimensões
    de risco, calculado **trimestre a trimestre**. Diferente do modelo binário (risco/normal),
    ele captura a **intensificação progressiva** do risco.
    """)

    if not df_score.empty and "ScoreEstresse" in df_score.columns:
        # Gráfico principal: Score ao longo do tempo
        fig_score = go.Figure()

        for banco in ["Banco Inter", "Banco Daycoval", "Banco Pine", "Banco Original", "Banco Master"]:
            sub = df_score[df_score["NomeBanco"] == banco].sort_values("DataRef")
            if sub.empty:
                continue
            fig_score.add_trace(go.Scatter(
                x=sub["DataRef"],
                y=sub["ScoreEstresse"],
                mode="lines+markers",
                name=banco,
                line=dict(
                    color=CORES.get(banco, "#888"),
                    width=3 if banco == "Banco Master" else 1.5,
                ),
                marker=dict(size=5),
            ))

        fig_score.add_hrect(y0=60, y1=100, fillcolor="red", opacity=0.08,
                            annotation_text="Crítico", annotation_position="top left")
        fig_score.add_hrect(y0=40, y1=60, fillcolor="orange", opacity=0.08,
                            annotation_text="Alto", annotation_position="top left")
        fig_score.add_hrect(y0=20, y1=40, fillcolor="yellow", opacity=0.05,
                            annotation_text="Moderado", annotation_position="top left")
        fig_score.add_hrect(y0=0, y1=20, fillcolor="green", opacity=0.05,
                            annotation_text="Baixo", annotation_position="top left")

        fig_score.update_layout(
            title="Score de Estresse Financeiro ao Longo do Tempo",
            yaxis_title="Score (0-100)",
            yaxis_range=[0, 75],
            xaxis_title="",
            height=550,
            hovermode="x unified",
        )
        st.plotly_chart(fig_score, use_container_width=True)

        # Interpretação
        master_score = df_score[df_score["NomeBanco"] == "Banco Master"]
        if not master_score.empty:
            score_medio = master_score["ScoreEstresse"].mean()
            score_max = master_score["ScoreEstresse"].max()
            tri_max = master_score.loc[master_score["ScoreEstresse"].idxmax(), "AnoTri"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Score Médio (Master)", f"{score_medio:.0f} pts", "Faixa: Alto")
            col2.metric("Score Máximo", f"{score_max:.0f} pts", f"{tri_max}")
            col3.metric("Score Médio (Pares)", f"{df_score[df_score['NomeBanco'] != 'Banco Master']['ScoreEstresse'].mean():.0f} pts", "Faixa: Baixo-Moderado")

        st.markdown("""
        **Leitura do gráfico:**
        - O Banco Master opera **consistentemente na faixa Alto/Crítico** (40-67 pts)
          durante todo o período, enquanto os pares ficam entre 10-36 pts (Baixo/Moderado).
        - O **pico de 67 pts** ocorre em Q2/2020 (pandemia amplificou a alavancagem).
        - A partir de **2022**, o componente de **Crescimento** atinge o máximo (25 pts),
          refletindo a explosão do ativo de R\$ 20B para R\$ 69B.
        - O risco era **estrutural desde 2019** (Basileia + Alavancagem), mas **se intensificou
          progressivamente** com o crescimento anormal a partir de 2022.
        """)

        # Decomposição do score do Master
        st.subheader("Decomposição do Score — Banco Master")
        componentes = ["ScoreBasileia", "ScoreAlavancagem", "ScoreCrescimento", "ScoreCaptacoes"]
        comp_nomes = {"ScoreBasileia": "Basileia", "ScoreAlavancagem": "Alavancagem",
                      "ScoreCrescimento": "Crescimento", "ScoreCaptacoes": "Captações"}
        comp_cores = {"ScoreBasileia": "#e74c3c", "ScoreAlavancagem": "#f39c12",
                      "ScoreCrescimento": "#3498db", "ScoreCaptacoes": "#9b59b6"}
        comp_disp = [c for c in componentes if c in master_score.columns]

        if comp_disp:
            ms = master_score.sort_values("DataRef")
            fig_decomp = go.Figure()
            for comp in comp_disp:
                fig_decomp.add_trace(go.Bar(
                    x=ms["AnoTri"],
                    y=ms[comp],
                    name=comp_nomes.get(comp, comp),
                    marker_color=comp_cores.get(comp, "#888"),
                ))

            fig_decomp.update_layout(
                barmode="stack",
                title="Composição do Score de Estresse — Banco Master",
                yaxis_title="Pontos (máx. 25 cada)",
                height=400,
            )
            st.plotly_chart(fig_decomp, use_container_width=True)

            st.markdown("""
            **Evolução dos componentes:**
            - **2019-2021:** Risco dominado por **Basileia** (próxima do mínimo) e **Alavancagem** (16-21x)
            - **2022-2025:** **Crescimento** assume como componente principal (ativo crescendo 50-100% a.a.)
            - **Captações** contribuem de forma constante mas moderada (4-6 pts)
            """)

    # =====================================================================
    # PARTE 2: MODELO DE CLASSIFICAÇÃO (Decision Tree)
    # =====================================================================
    st.markdown("---")
    st.header("2. Modelo de Classificação — Decision Tree")

    st.markdown("""
    Complementando o Score de Estresse, treinamos uma **Decision Tree** para identificar
    quais indicadores mais distinguem o Banco Master dos pares saudáveis.
    """)

    resultado = retreinar_modelo()
    if resultado:
        metricas = resultado["metricas"]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Métricas**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Acurácia", f"{metricas['acuracia']:.1%}")
            m2.metric("F1 (CV)", f"{metricas['cv_f1_mean']:.1%}")
            m3.metric("Amostras", f"{metricas['n_amostras']}")

        with col_b:
            st.markdown("**Árvore de Decisão**")
            st.code(metricas["arvore_texto"], language="text")

        # Feature importance
        fi = pd.DataFrame([
            {"Feature": k, "Importância": v}
            for k, v in metricas["feature_importance"].items()
        ]).sort_values("Importância", ascending=True)

        fig_fi = px.bar(
            fi, x="Importância", y="Feature", orientation="h",
            title="Importância das Features",
            color="Importância",
            color_continuous_scale="Reds",
        )
        fig_fi.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    # =====================================================================
    # LIMITAÇÕES
    # =====================================================================
    st.markdown("---")
    st.header("3. Limitações")
    st.warning("""
    **Sobre o modelo Decision Tree:**
    - Com apenas **1 banco liquidado entre 5**, o modelo aprende a separar o Master
      dos pares — não detecta risco bancário em geral.
    - A árvore classifica Master como risco em **todos os trimestres** porque a feature
      principal (dependência de CDBs > 84%) era uma característica **estrutural** do banco,
      presente desde 2019.
    - Para um modelo que capture **quando** o risco se intensificou, o **Score de Estresse**
      (Parte 1) é mais adequado.

    **Sobre o Score de Estresse:**
    - O score é uma heurística baseada em desvios da mediana dos pares, não um modelo
      estatístico calibrado.
    - Os pesos dos componentes (25 pts cada) são arbitrários — em produção, seriam
      calibrados com dados históricos de múltiplos bancos.
    """)


if __name__ == "__main__":
    main()
