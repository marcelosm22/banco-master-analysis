"""
Página 4: Predição — Modelo de Classificação de Risco.
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

FEATURES_NOMES = {
    "Alavancagem": "Alavancagem (Ativo/PL)",
    "CoberturaCapt": "Captações / Ativo",
    "CreditoSobreAtivo": "Crédito / Ativo",
    "ROE": "ROE",
    "CrescAtivo_QoQ": "Crescimento Trimestral",
    "Basileia": "Índice de Basileia",
    "MargemBasileia": "Margem Basileia",
}


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
def carregar_predicao():
    caminho = DATA_DIR / "predicao_risco.csv"
    if not caminho.exists():
        return pd.DataFrame()
    return pd.read_csv(caminho, encoding="utf-8-sig", parse_dates=["DataRef"])


def carregar_modelo():
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from predicao.modelo_risco import executar_predicao
        return executar_predicao()
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return {}


def main():
    st.title("Predição — Modelo de Risco Bancário")

    df_resumo = carregar_resumo()
    df_capital = carregar_capital()
    df_pred = carregar_predicao()

    if df_resumo.empty:
        st.error("Dados não encontrados. Execute o pipeline primeiro.")
        return

    # Carregar modelo
    resultado = carregar_modelo()

    # =====================================================================
    # 1. DECISION TREE + FEATURE IMPORTANCE
    # =====================================================================
    st.header("1. Modelo de Classificação — Decision Tree")

    st.markdown("""
    Treinamos uma **Decision Tree** para identificar quais indicadores financeiros
    mais distinguem o Banco Master dos pares saudáveis.
    """)

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

        # Permutation Importance (mais confiável que Gini)
        from sklearn.inspection import permutation_importance
        modelo = resultado["modelo"]
        features = resultado["features"]

        # Montar X, y a partir do dataset
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from predicao.modelo_risco import carregar_dados, preparar_dataset
        df_r, df_c = carregar_dados()
        df_ds, feats_ds = preparar_dataset(df_r, df_c)
        X_perm = df_ds[feats_ds].values
        y_perm = df_ds["Target"].values

        perm = permutation_importance(
            modelo, X_perm, y_perm,
            n_repeats=30, random_state=42, scoring="f1",
        )

        fi = pd.DataFrame({
            "Feature": [FEATURES_NOMES.get(f, f) for f in features],
            "Importância": perm.importances_mean,
            "Desvio": perm.importances_std,
        }).sort_values("Importância", ascending=True)

        fig_fi = px.bar(
            fi, x="Importância", y="Feature", orientation="h",
            error_x="Desvio",
            title="Importância das Features (Permutation Importance)",
            color="Importância",
            color_continuous_scale="Reds",
        )
        fig_fi.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True)

        st.caption(
            "Permutation Importance mede quanto o F1 cai ao embaralhar cada feature. "
            "Mais confiável que Gini Importance, que superestima features usadas em múltiplos splits."
        )

    # =====================================================================
    # 2. RADAR — PERFIL COMPARATIVO
    # =====================================================================
    st.markdown("---")
    st.header("2. Perfil Comparativo — Gráfico Radar")

    st.markdown("""
    Comparação do "perfil de risco" do Banco Master com a **média dos pares**.
    Use o seletor de período para explorar como o perfil mudou ao longo do tempo.
    """)

    radar_feats = ["Alavancagem", "CoberturaCapt", "CreditoSobreAtivo", "ROE", "CrescAtivo_QoQ"]
    radar_feats = [f for f in radar_feats if f in df_resumo.columns]

    if radar_feats:
        # Seletor de período
        trimestres_disp = sorted(df_resumo["AnoTri"].unique())
        opcoes_periodo = {
            "Período completo (2019–2025)": trimestres_disp,
        }
        # Agrupar por ano
        for ano in sorted(df_resumo["Ano"].unique()):
            tris_ano = sorted(df_resumo[df_resumo["Ano"] == ano]["AnoTri"].unique())
            if tris_ano:
                opcoes_periodo[f"Ano {ano}"] = tris_ano

        periodo_sel = st.selectbox(
            "Selecione o período para o radar:",
            list(opcoes_periodo.keys()),
            index=0,
        )
        tris_filtro = opcoes_periodo[periodo_sel]

        df_filtrado = df_resumo[df_resumo["AnoTri"].isin(tris_filtro)]

        master_vals = df_filtrado[df_filtrado["NomeBanco"] == "Banco Master"][radar_feats].mean()
        pares_vals = df_filtrado[df_filtrado["NomeBanco"] != "Banco Master"][radar_feats].mean()

        # Normalizar entre 0-1 (min-max do dataset INTEIRO para manter escala consistente)
        all_vals = df_resumo[radar_feats]
        mins = all_vals.min()
        maxs = all_vals.max()
        ranges = (maxs - mins).replace(0, 1)

        master_norm = (master_vals - mins) / ranges
        pares_norm = (pares_vals - mins) / ranges

        labels = [FEATURES_NOMES.get(f, f) for f in radar_feats]

        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=list(pares_norm.values) + [pares_norm.values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name="Média dos Pares",
            fillcolor="rgba(52, 152, 219, 0.15)",
            line=dict(color="#3498db", width=2),
        ))

        fig_radar.add_trace(go.Scatterpolar(
            r=list(master_norm.values) + [master_norm.values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name="Banco Master",
            fillcolor="rgba(231, 76, 60, 0.15)",
            line=dict(color="#e74c3c", width=3),
        ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=f"Perfil de Risco — Master vs Média dos Pares ({periodo_sel})",
            height=500,
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Tabela de valores reais
        st.markdown(f"**Valores reais ({periodo_sel}):**")
        tabela = pd.DataFrame({
            "Indicador": labels,
            "Banco Master": [f"{v:.4f}" for v in master_vals.values],
            "Média dos Pares": [f"{v:.4f}" for v in pares_vals.values],
            "Diferença": [f"{m - p:+.4f}" for m, p in zip(master_vals.values, pares_vals.values)],
        })
        st.dataframe(tabela, use_container_width=True, hide_index=True)

        st.caption("Normalização min-max usa o dataset inteiro (2019–2025) para manter a escala consistente entre períodos.")

    # =====================================================================
    # 3. ANÁLISE "E SE?" (WHAT-IF)
    # =====================================================================
    st.markdown("---")
    st.header("3. Análise 'E se?' — Simulador Interativo")

    st.markdown("""
    Ajuste os indicadores financeiros e veja em tempo real como o modelo
    classifica o banco. Isso demonstra **como a árvore de decisão funciona**
    e quais limiares disparam o alerta de risco.
    """)

    if resultado:
        modelo = resultado["modelo"]
        features = resultado["features"]

        # Sliders
        col1, col2 = st.columns(2)

        slider_vals = {}
        with col1:
            slider_vals["Alavancagem"] = st.slider(
                "Alavancagem (Ativo / PL)",
                min_value=2.0, max_value=30.0, value=15.0, step=0.5,
            )
            slider_vals["CoberturaCapt"] = st.slider(
                "Captações / Ativo Total",
                min_value=0.50, max_value=0.95, value=0.85, step=0.01,
            )
            slider_vals["CreditoSobreAtivo"] = st.slider(
                "Crédito / Ativo Total",
                min_value=0.10, max_value=0.80, value=0.45, step=0.01,
            )
            if "ROE" in features:
                slider_vals["ROE"] = st.slider(
                    "ROE (Retorno sobre PL)",
                    min_value=-0.50, max_value=0.30, value=0.05, step=0.01,
                )

        with col2:
            if "CrescAtivo_QoQ" in features:
                slider_vals["CrescAtivo_QoQ"] = st.slider(
                    "Crescimento Trimestral (%)",
                    min_value=-0.10, max_value=0.50, value=0.10, step=0.01,
                )
            if "Basileia" in features:
                slider_vals["Basileia"] = st.slider(
                    "Índice de Basileia",
                    min_value=0.02, max_value=0.50, value=0.12, step=0.005,
                    help="Mínimo regulatório: 11%",
                )
            if "MargemBasileia" in features:
                # Auto-calculado a partir do Basileia
                slider_vals["MargemBasileia"] = slider_vals.get("Basileia", 0.12) - 0.11

        # Montar vetor de features na ordem correta
        X_input = np.array([[slider_vals.get(f, 0.0) for f in features]])

        pred = modelo.predict(X_input)[0]
        proba = modelo.predict_proba(X_input)[0]
        idx_pos = list(modelo.classes_).index(1) if 1 in modelo.classes_ else 0
        prob_risco = proba[idx_pos]

        # Resultado
        st.markdown("---")
        r1, r2, r3 = st.columns(3)

        if pred == 1:
            r1.error(f"**Classificação: RISCO**")
        else:
            r1.success(f"**Classificação: NORMAL**")

        r2.metric("Probabilidade de Risco", f"{prob_risco:.1%}")
        r3.metric("Margem Basileia", f"{slider_vals.get('MargemBasileia', 0):.2%}",
                   delta="Abaixo do mínimo" if slider_vals.get("MargemBasileia", 0) < 0 else "Acima do mínimo",
                   delta_color="inverse" if slider_vals.get("MargemBasileia", 0) < 0 else "normal")

        # Dica: valores do Master
        with st.expander("Comparar com valores reais do Banco Master"):
            if not df_resumo.empty:
                master_ult = df_resumo[df_resumo["NomeBanco"] == "Banco Master"].sort_values("AnoMes").iloc[-1]
                st.markdown(f"""
                **Último trimestre disponível do Master ({master_ult.get('AnoTri', '?')}):**
                - Alavancagem: **{master_ult.get('Alavancagem', 0):.1f}x**
                - Captações/Ativo: **{master_ult.get('CoberturaCapt', 0):.2%}**
                - Crédito/Ativo: **{master_ult.get('CreditoSobreAtivo', 0):.2%}**
                - ROE: **{master_ult.get('ROE', 0):.2%}**
                - Crescimento QoQ: **{master_ult.get('CrescAtivo_QoQ', 0):.2%}**
                """)

    # =====================================================================
    # 4. MATRIZ DE CONFUSÃO
    # =====================================================================
    st.markdown("---")
    st.header("4. Validação — Matriz de Confusão")

    st.markdown("""
    A matriz de confusão mostra onde o modelo **acerta e erra**. Cada célula
    representa um grupo de trimestres classificados.
    """)

    if not df_pred.empty:
        from sklearn.metrics import confusion_matrix

        y_true = df_pred["Target"].values
        y_pred = df_pred["PredicaoRisco"].values

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Plotar matriz
        labels_cm = ["Normal (0)", "Risco (1)"]
        fig_cm = go.Figure(data=go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Predito: Normal", "Predito: Risco"],
            y=["Real: Normal", "Real: Risco"],
            text=[[f"VN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"VP\n{tp}"]],
            texttemplate="%{text}",
            textfont=dict(size=18),
            colorscale=[[0, "#2ecc71"], [0.5, "#f9e79f"], [1, "#e74c3c"]],
            showscale=False,
        ))

        fig_cm.update_layout(
            title="Matriz de Confusão",
            height=400,
            xaxis_title="Classificação do Modelo",
            yaxis_title="Valor Real",
            yaxis_autorange="reversed",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # Métricas detalhadas
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Verdadeiros Negativos", f"{tn}", "Bancos normais classificados corretamente")
        col2.metric("Verdadeiros Positivos", f"{tp}", "Master classificado como risco")
        col3.metric("Falsos Positivos", f"{fp}", "Bancos normais alertados como risco")
        col4.metric("Falsos Negativos", f"{fn}", "Master não detectado")

        # Detalhar falsos positivos
        if fp > 0:
            st.subheader("Detalhamento dos Falsos Positivos")
            fps = df_pred[(df_pred["Target"] == 0) & (df_pred["PredicaoRisco"] == 1)]
            st.dataframe(
                fps[["NomeBanco", "AnoTri", "ProbRisco"]].rename(columns={
                    "NomeBanco": "Banco",
                    "AnoTri": "Trimestre",
                    "ProbRisco": "Probabilidade",
                }),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("""
            **Análise dos falsos positivos:** Esses trimestres representam momentos em que
            bancos saudáveis apresentaram indicadores temporariamente similares ao padrão
            do Master (alta dependência de captações ou crédito baixo). O modelo não
            consegue distinguir estresse temporário de risco estrutural — uma limitação
            esperada com apenas 1 caso positivo no dataset.
            """)

    # =====================================================================
    # 5. LIMITAÇÕES
    # =====================================================================
    st.markdown("---")
    st.header("5. Limitações")
    st.warning("""
    - Com apenas **1 banco liquidado entre 5**, o modelo aprende a separar o Master
      dos pares — não detecta risco bancário em geral.
    - A árvore classifica Master como risco em **todos os trimestres** porque a feature
      principal (dependência de CDBs > 84%) era **estrutural**, presente desde 2019.
    - Os **falsos positivos** mostram que indicadores isolados não são suficientes —
      o contexto (crescimento explosivo + fraude contábil) é o que tornou o Master único.
    - Para generalizar, seria necessário um dataset com dezenas de bancos liquidados
      ao longo de décadas.
    """)


if __name__ == "__main__":
    main()
