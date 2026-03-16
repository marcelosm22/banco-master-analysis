"""
Página 3: Timeline de Notícias.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Timeline de Notícias", layout="wide")

DATA_DIR_BRUTO = Path(__file__).resolve().parents[2] / "data" / "bruto"
DATA_DIR_PREP = Path(__file__).resolve().parents[2] / "data" / "preparado"

CORES_CATEGORIA = {
    "contexto": "#95a5a6",
    "financeiro": "#3498db",
    "aquisicao": "#f39c12",
    "regulatorio": "#e67e22",
    "criminal": "#e74c3c",
    "politico": "#9b59b6",
    "analise": "#2ecc71",
}


@st.cache_data
def carregar_noticias():
    caminho = DATA_DIR_BRUTO / "noticias_timeline.csv"
    if not caminho.exists():
        return pd.DataFrame()
    df = pd.read_csv(caminho, encoding="utf-8-sig", parse_dates=["data"])
    return df


@st.cache_data
def carregar_resumo():
    caminho = DATA_DIR_PREP / "indicadores_resumo.csv"
    if not caminho.exists():
        return pd.DataFrame()
    return pd.read_csv(caminho, encoding="utf-8-sig", parse_dates=["DataRef"])


def main():
    st.title("Timeline de Notícias — Caso Banco Master")

    df_noticias = carregar_noticias()
    df_resumo = carregar_resumo()

    if df_noticias.empty:
        st.error("Timeline não encontrada. Execute: python src/coleta/noticias_scraper.py")
        return

    # Filtros
    st.sidebar.header("Filtros")
    categorias = sorted(df_noticias["categoria"].unique())
    cats_sel = st.sidebar.multiselect("Categorias", categorias, default=categorias)
    df_filtered = df_noticias[df_noticias["categoria"].isin(cats_sel)]

    # --- Gráfico: Ativo do Master com anotações de eventos-chave ---
    st.subheader("Evolução Financeira x Eventos")

    fig = go.Figure()

    if not df_resumo.empty and "AtivoTotal" in df_resumo.columns:
        master = df_resumo[df_resumo["NomeBanco"] == "Banco Master"].sort_values("DataRef")
        if not master.empty:
            fig.add_trace(go.Scatter(
                x=master["DataRef"],
                y=master["AtivoTotal"],
                mode="lines+markers",
                name="Ativo Total (Master)",
                line=dict(color="#3498db", width=3),
                marker=dict(size=6),
            ))

            # Eventos-chave como linhas verticais com anotação limpa
            eventos_chave = df_filtered.sort_values("data").reset_index(drop=True)
            y_max = master["AtivoTotal"].max()

            for i, (_, row) in enumerate(eventos_chave.iterrows()):
                cor = CORES_CATEGORIA.get(row["categoria"], "#888888")

                # Linha vertical no ponto do evento
                fig.add_vline(
                    x=row["data"],
                    line_dash="dot",
                    line_color=cor,
                    line_width=1,
                    opacity=0.5,
                )

                # Anotação curta (só para eventos principais)
                # Alterna posição vertical para evitar sobreposição
                titulo_curto = row["titulo"][:40]
                y_pos = y_max * (0.95 - (i % 5) * 0.12)

                fig.add_annotation(
                    x=row["data"],
                    y=y_pos,
                    text=f"<b>{titulo_curto}...</b>",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=0.8,
                    arrowcolor=cor,
                    font=dict(size=9, color=cor),
                    bgcolor="rgba(0,0,0,0.7)",
                    bordercolor=cor,
                    borderwidth=1,
                    ax=30,
                    ay=-20,
                )

    fig.update_layout(
        title="Ativo Total do Banco Master x Eventos do Caso",
        xaxis_title="",
        yaxis_title="Ativo Total (R$)",
        height=600,
        showlegend=True,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption("Passe o mouse sobre as anotações para mais detalhes. Linhas verticais indicam eventos relevantes.")

    # --- Timeline detalhada ---
    st.markdown("---")
    st.subheader("Timeline Detalhada")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Eventos", len(df_filtered))
    col2.metric("Período", f"{df_filtered['data'].min().strftime('%m/%Y')} a {df_filtered['data'].max().strftime('%m/%Y')}")
    col3.metric("Categorias", len(df_filtered["categoria"].unique()))

    fig_cat = px.bar(
        df_filtered["categoria"].value_counts().reset_index(),
        x="categoria",
        y="count",
        color="categoria",
        color_discrete_map=CORES_CATEGORIA,
        title="Eventos por Categoria",
    )
    fig_cat.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_cat, use_container_width=True)

    # Lista de eventos
    st.markdown("---")
    st.subheader("Eventos")

    for _, row in df_filtered.sort_values("data").iterrows():
        cor = CORES_CATEGORIA.get(row["categoria"], "#333")
        data_str = row["data"].strftime("%d/%m/%Y")
        cat = row["categoria"].upper()

        url_html = ""
        if pd.notna(row.get("url")) and row["url"]:
            url_html = f'<br><a href="{row["url"]}" target="_blank">Link</a>'

        st.markdown(
            f'<div style="border-left: 4px solid {cor}; padding: 8px 16px; margin: 8px 0;">'
            f'<strong>[{data_str}] [{cat}]</strong><br>'
            f'{row["titulo"]}<br>'
            f'<small>Fonte: {row["fonte"]}</small>'
            f'{url_html}'
            f'</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
