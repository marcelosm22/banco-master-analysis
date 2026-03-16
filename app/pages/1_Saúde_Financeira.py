"""
Página 1: Saúde Financeira — Gráficos comparativos.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Saúde Financeira", layout="wide")

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "preparado"
CORES_BANCOS = {
    "Banco Master": "#e74c3c",
    "Banco Inter": "#3498db",
    "Banco Pine": "#2ecc71",
    "Banco Original": "#f39c12",
    "Banco Daycoval": "#9b59b6",
}


@st.cache_data
def carregar_resumo():
    caminho = DATA_DIR / "indicadores_resumo.csv"
    if not caminho.exists():
        return pd.DataFrame()
    df = pd.read_csv(caminho, encoding="utf-8-sig", parse_dates=["DataRef"])
    return df


@st.cache_data
def carregar_capital():
    caminho = DATA_DIR / "indicadores_capital.csv"
    if not caminho.exists():
        return pd.DataFrame()
    df = pd.read_csv(caminho, encoding="utf-8-sig", parse_dates=["DataRef"])
    return df


def plot_indicador(df, coluna, titulo, formato="R$ {:,.0f}", yaxis_title=None, linha_ref=None):
    """Cria gráfico de linha comparativo entre bancos."""
    if coluna not in df.columns:
        st.warning(f"Coluna '{coluna}' não encontrada")
        return

    dados = df.dropna(subset=[coluna])
    if dados.empty:
        st.warning(f"Sem dados para '{coluna}'")
        return

    fig = px.line(
        dados,
        x="DataRef",
        y=coluna,
        color="NomeBanco",
        color_discrete_map=CORES_BANCOS,
        title=titulo,
        markers=True,
    )

    if linha_ref is not None:
        fig.add_hline(
            y=linha_ref["valor"],
            line_dash="dash",
            line_color="red",
            annotation_text=linha_ref.get("texto", ""),
        )

    fig.update_layout(
        xaxis_title="",
        yaxis_title=yaxis_title or coluna,
        legend_title="Banco",
        hovermode="x unified",
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("Saúde Financeira — Comparativo")

    df_resumo = carregar_resumo()
    df_capital = carregar_capital()

    if df_resumo.empty:
        st.error("Dados não encontrados. Execute o pipeline de coleta e preparação primeiro.")
        return

    # Filtros
    st.sidebar.header("Filtros")
    bancos_disp = sorted(df_resumo["NomeBanco"].unique())
    bancos_sel = st.sidebar.multiselect(
        "Bancos", bancos_disp, default=bancos_disp
    )

    df_r = df_resumo[df_resumo["NomeBanco"].isin(bancos_sel)]
    df_c = df_capital[df_capital["NomeBanco"].isin(bancos_sel)] if not df_capital.empty else pd.DataFrame()

    # --- Gráficos ---
    st.subheader("Evolução do Ativo Total")
    col1, col2 = st.columns(2)
    with col1:
        plot_indicador(df_r, "AtivoTotal", "Ativo Total (R$)", yaxis_title="R$")
    with col2:
        plot_indicador(
            df_r, "AtivoBase100",
            "Ativo Total — Base 100 (Q1/2019 = 100)",
            yaxis_title="Índice (base 100)",
        )

    st.markdown("---")

    st.subheader("Alavancagem e Captações")
    col3, col4 = st.columns(2)
    with col3:
        plot_indicador(df_r, "Alavancagem", "Alavancagem (Ativo / PL)", yaxis_title="x")
    with col4:
        plot_indicador(
            df_r, "CoberturaCapt",
            "Captações / Ativo Total",
            yaxis_title="Proporção",
        )

    st.markdown("---")

    st.subheader("Patrimônio Líquido e Rentabilidade")
    col5, col6 = st.columns(2)
    with col5:
        plot_indicador(df_r, "PatrimonioLiquido", "Patrimônio Líquido (R$)", yaxis_title="R$")
    with col6:
        plot_indicador(df_r, "ROE", "ROE — Retorno sobre Patrimônio", yaxis_title="ROE")

    st.markdown("---")

    st.subheader("Índice de Basileia")
    if not df_c.empty and "Basileia" in df_c.columns:
        plot_indicador(
            df_c, "Basileia",
            "Índice de Basileia",
            yaxis_title="Índice",
            linha_ref={"valor": 0.11, "texto": "Mínimo regulatório (11%)"},
        )
    else:
        st.info("Dados de Capital/Basileia não disponíveis")

    st.markdown("---")

    st.subheader("Crescimento Anual (Year-over-Year)")
    col7, col8 = st.columns(2)
    with col7:
        plot_indicador(
            df_r, "CrescAtivo_YoY",
            "Crescimento Ativo Total (% a.a.)",
            yaxis_title="Variação %",
        )
    with col8:
        plot_indicador(
            df_r, "CrescPL_YoY",
            "Crescimento Patrimônio Líquido (% a.a.)",
            yaxis_title="Variação %",
        )


if __name__ == "__main__":
    main()
