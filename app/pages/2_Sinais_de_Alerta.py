"""
Página 2: Sinais de Alerta — Heatmap e Score de Estresse.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Sinais de Alerta", layout="wide")

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "preparado"


@st.cache_data
def carregar_resumo():
    caminho = DATA_DIR / "indicadores_resumo.csv"
    if not caminho.exists():
        return pd.DataFrame()
    return pd.read_csv(caminho, encoding="utf-8-sig", parse_dates=["DataRef"])


@st.cache_data
def carregar_capital():
    caminho = DATA_DIR / "indicadores_capital.csv"
    if not caminho.exists():
        return pd.DataFrame()
    return pd.read_csv(caminho, encoding="utf-8-sig", parse_dates=["DataRef"])


@st.cache_data
def carregar_score():
    caminho = DATA_DIR / "score_estresse.csv"
    if not caminho.exists():
        return pd.DataFrame()
    return pd.read_csv(caminho, encoding="utf-8-sig", parse_dates=["DataRef"])


def calcular_semaforo(df_resumo, df_capital):
    """
    Calcula semáforo de risco para cada banco/trimestre.
    Verde = normal, Amarelo = atenção, Vermelho = crítico.
    """
    indicadores = []

    for _, row in df_resumo.iterrows():
        banco = row["NomeBanco"]
        periodo = row["AnoTri"]
        sinais = {}

        if "Alavancagem" in row and pd.notna(row["Alavancagem"]):
            tri_data = df_resumo[df_resumo["AnoMes"] == row["AnoMes"]]
            med = tri_data["Alavancagem"].median()
            std = tri_data["Alavancagem"].std()
            if std and std > 0:
                z = (row["Alavancagem"] - med) / std
                if z > 2:
                    sinais["Alavancagem"] = 2
                elif z > 1:
                    sinais["Alavancagem"] = 1
                else:
                    sinais["Alavancagem"] = 0

        if "CrescAtivo_YoY" in row and pd.notna(row["CrescAtivo_YoY"]):
            tri_data = df_resumo[df_resumo["AnoMes"] == row["AnoMes"]]
            med = tri_data["CrescAtivo_YoY"].median()
            std = tri_data["CrescAtivo_YoY"].std()
            if std and std > 0:
                z = (row["CrescAtivo_YoY"] - med) / std
                if z > 2:
                    sinais["Crescimento"] = 2
                elif z > 1:
                    sinais["Crescimento"] = 1
                else:
                    sinais["Crescimento"] = 0

        if "CoberturaCapt" in row and pd.notna(row["CoberturaCapt"]):
            if row["CoberturaCapt"] > 0.84:
                sinais["Captações"] = 2
            elif row["CoberturaCapt"] > 0.75:
                sinais["Captações"] = 1
            else:
                sinais["Captações"] = 0

        indicadores.append({
            "NomeBanco": banco,
            "AnoTri": periodo,
            "AnoMes": row["AnoMes"],
            "DataRef": row["DataRef"],
            **{f"Sinal_{k}": v for k, v in sinais.items()},
        })

    df_sinais = pd.DataFrame(indicadores)

    if not df_capital.empty and "Basileia" in df_capital.columns:
        for _, row in df_capital.iterrows():
            mask = (df_sinais["NomeBanco"] == row["NomeBanco"]) & (df_sinais["AnoMes"] == row["AnoMes"])
            if mask.any():
                basileia = row["Basileia"]
                if basileia <= 0.11:
                    df_sinais.loc[mask, "Sinal_Basileia"] = 2
                elif basileia <= 0.13:
                    df_sinais.loc[mask, "Sinal_Basileia"] = 1
                else:
                    df_sinais.loc[mask, "Sinal_Basileia"] = 0

    return df_sinais


def main():
    st.title("Sinais de Alerta — Red Flags")

    df_resumo = carregar_resumo()
    df_capital = carregar_capital()
    df_score = carregar_score()

    if df_resumo.empty:
        st.error("Dados não encontrados. Execute o pipeline primeiro.")
        return

    df_sinais = calcular_semaforo(df_resumo, df_capital)

    # --- Heatmap de sinais ---
    st.subheader("Heatmap de Sinais de Alerta")
    st.markdown("""
    Cada célula representa um indicador para um banco em um trimestre.
    - 🟢 **Verde (0)**: Dentro do normal
    - 🟡 **Amarelo (1)**: Acima de 1 desvio padrão dos pares
    - 🔴 **Vermelho (2)**: Acima de 2 desvios padrão ou violação regulatória
    """)

    sinal_cols = [c for c in df_sinais.columns if c.startswith("Sinal_")]

    for banco in sorted(df_sinais["NomeBanco"].unique()):
        dados_banco = df_sinais[df_sinais["NomeBanco"] == banco].sort_values("AnoMes")

        if dados_banco.empty or not sinal_cols:
            continue

        matriz = dados_banco.set_index("AnoTri")[sinal_cols].T
        matriz = matriz.fillna(-1)

        labels_y = [c.replace("Sinal_", "") for c in sinal_cols]

        fig = go.Figure(data=go.Heatmap(
            z=matriz.values,
            x=matriz.columns,
            y=labels_y,
            colorscale=[
                [0, "#cccccc"],
                [0.33, "#2ecc71"],
                [0.66, "#f39c12"],
                [1.0, "#e74c3c"],
            ],
            zmin=-1,
            zmax=2,
            showscale=False,
            hovertemplate="Trimestre: %{x}<br>Indicador: %{y}<br>Nível: %{z}<extra></extra>",
        ))

        cor_titulo = "#e74c3c" if banco == "Banco Master" else "#333333"
        fig.update_layout(
            title=f"{banco}",
            title_font_color=cor_titulo,
            height=250,
            margin=dict(l=10, r=10, t=40, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

    # --- Score de estresse ---
    st.markdown("---")
    st.subheader("Score de Estresse Financeiro")

    if not df_score.empty and "ScoreEstresse" in df_score.columns:
        st.markdown("""
        Score composto (0-100) combinando:
        - Proximidade do Basileia ao mínimo (0-25 pts)
        - Alavancagem vs mediana dos pares (0-25 pts)
        - Crescimento anormal do ativo (0-25 pts)
        - Concentração de captações (0-25 pts)
        """)

        cores_bancos = {
            "Banco Master": "#e74c3c",
            "Banco Inter": "#3498db",
            "Banco Pine": "#2ecc71",
            "Banco Original": "#f39c12",
            "Banco Daycoval": "#9b59b6",
        }

        fig_score = px.line(
            df_score,
            x="DataRef",
            y="ScoreEstresse",
            color="NomeBanco",
            color_discrete_map=cores_bancos,
            title="Score de Estresse ao Longo do Tempo",
            markers=True,
        )

        fig_score.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="Crítico")
        fig_score.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="Alto")
        fig_score.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Moderado")

        fig_score.update_layout(
            xaxis_title="",
            yaxis_title="Score (0-100)",
            height=500,
        )

        st.plotly_chart(fig_score, use_container_width=True)

        # Decomposição do score para Master
        st.subheader("Decomposição do Score — Banco Master")
        master_score = df_score[df_score["NomeBanco"] == "Banco Master"]
        if not master_score.empty:
            componentes = ["ScoreBasileia", "ScoreAlavancagem", "ScoreCrescimento", "ScoreCaptacoes"]
            componentes_disp = [c for c in componentes if c in master_score.columns]

            if componentes_disp:
                fig_decomp = go.Figure()
                cores_comp = ["#e74c3c", "#f39c12", "#3498db", "#9b59b6"]
                for i, comp in enumerate(componentes_disp):
                    fig_decomp.add_trace(go.Bar(
                        x=master_score["AnoTri"],
                        y=master_score[comp],
                        name=comp.replace("Score", ""),
                        marker_color=cores_comp[i % len(cores_comp)],
                    ))

                fig_decomp.update_layout(
                    barmode="stack",
                    title="Composição do Score de Estresse — Banco Master",
                    yaxis_title="Pontos",
                    height=400,
                )
                st.plotly_chart(fig_decomp, use_container_width=True)
    else:
        st.info("Score de estresse não calculado. Execute src/preparacao/indicadores.py")


if __name__ == "__main__":
    main()
