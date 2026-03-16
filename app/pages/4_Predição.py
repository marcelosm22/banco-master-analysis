"""
Página 4: Predição — Dois modelos de classificação de risco.
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
def carregar_predicao():
    caminho = DATA_DIR / "predicao_risco.csv"
    if not caminho.exists():
        return pd.DataFrame()
    return pd.read_csv(caminho, encoding="utf-8-sig", parse_dates=["DataRef"])


@st.cache_resource
def treinar_modelos():
    """Treina Decision Tree e Logistic Regression. Cacheia para não retreinar a cada refresh."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from predicao.modelo_risco import carregar_dados, preparar_dataset

    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import accuracy_score
    from sklearn.inspection import permutation_importance

    df_resumo, df_capital = carregar_dados()
    df, features = preparar_dataset(df_resumo, df_capital)

    if df.empty or len(features) < 2:
        return None

    X = df[features].values
    y = df["Target"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Decision Tree ---
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42, class_weight="balanced")
    dt_f1 = cross_val_score(dt, X, y, cv=skf, scoring="f1")
    dt.fit(X, y)
    dt_acc = accuracy_score(y, dt.predict(X))
    dt_perm = permutation_importance(dt, X, y, n_repeats=30, random_state=42, scoring="f1")

    # --- Logistic Regression ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
    lr_f1 = cross_val_score(lr, X_scaled, y, cv=skf, scoring="f1")
    lr.fit(X_scaled, y)
    lr_acc = accuracy_score(y, lr.predict(X_scaled))
    lr_perm = permutation_importance(lr, X_scaled, y, n_repeats=30, random_state=42, scoring="f1")

    # Correlação
    corr_vals = [abs(df[f].corr(df["Target"])) for f in features]

    return {
        "features": features,
        "df": df,
        "X": X,
        "y": y,
        "scaler": scaler,
        # Decision Tree
        "dt_modelo": dt,
        "dt_acc": dt_acc,
        "dt_f1_mean": float(np.nanmean(dt_f1)),
        "dt_f1_std": float(np.nanstd(dt_f1)),
        "dt_arvore": export_text(dt, feature_names=features, max_depth=10),
        "dt_perm_mean": dt_perm.importances_mean,
        "dt_perm_std": dt_perm.importances_std,
        # Logistic Regression
        "lr_modelo": lr,
        "lr_acc": lr_acc,
        "lr_f1_mean": float(np.nanmean(lr_f1)),
        "lr_f1_std": float(np.nanstd(lr_f1)),
        "lr_coefs": lr.coef_[0],
        "lr_perm_mean": lr_perm.importances_mean,
        "lr_perm_std": lr_perm.importances_std,
        # Geral
        "corr_vals": corr_vals,
        "n_amostras": len(y),
        "n_positivos": int(y.sum()),
    }


def main():
    st.title("Predição — Modelos de Risco Bancário")

    df_resumo = carregar_resumo()
    df_pred = carregar_predicao()

    if df_resumo.empty:
        st.error("Dados não encontrados. Execute o pipeline primeiro.")
        return

    res = treinar_modelos()
    if res is None:
        st.error("Erro ao treinar modelos.")
        return

    features = res["features"]

    # =====================================================================
    # 1. COMPARATIVO DOS DOIS MODELOS
    # =====================================================================
    st.header("1. Dois Modelos Complementares")

    st.markdown("""
    Treinamos dois modelos para analisar o risco sob perspectivas diferentes:

    - **Decision Tree** — identifica **regras claras** (limiares) que separam Master dos pares
    - **Logistic Regression** — atribui **peso a todas as features**, mostrando quanto cada uma contribui
    """)

    col_dt, col_lr = st.columns(2)

    with col_dt:
        st.subheader("Decision Tree")
        m1, m2 = st.columns(2)
        m1.metric("Acurácia", f"{res['dt_acc']:.1%}")
        m2.metric("F1 (CV)", f"{res['dt_f1_mean']:.1%} ± {res['dt_f1_std']:.1%}")
        st.code(res["dt_arvore"], language="text")
        st.caption("Mostra regras de decisão — quais limiares disparam o alerta.")

    with col_lr:
        st.subheader("Logistic Regression")
        m3, m4 = st.columns(2)
        m3.metric("Acurácia", f"{res['lr_acc']:.1%}")
        m4.metric("F1 (CV)", f"{res['lr_f1_mean']:.1%} ± {res['lr_f1_std']:.1%}")

        # Coeficientes como tabela
        coef_df = pd.DataFrame({
            "Feature": [FEATURES_NOMES.get(f, f) for f in features],
            "Coeficiente": res["lr_coefs"],
        }).sort_values("Coeficiente", key=abs, ascending=False)

        st.dataframe(coef_df, use_container_width=True, hide_index=True)
        st.caption("Coeficientes padronizados — valor positivo aumenta risco, negativo diminui.")

    # =====================================================================
    # 2. IMPORTÂNCIA DAS FEATURES (3 perspectivas)
    # =====================================================================
    st.markdown("---")
    st.header("2. Importância das Features — Três Perspectivas")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        fi_dt = pd.DataFrame({
            "Feature": [FEATURES_NOMES.get(f, f) for f in features],
            "Importância": res["dt_perm_mean"],
        }).sort_values("Importância", ascending=True)

        fig_dt = px.bar(fi_dt, x="Importância", y="Feature", orientation="h",
                        title="Decision Tree (Permutation)",
                        color="Importância", color_continuous_scale="Reds")
        fig_dt.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_dt, use_container_width=True)
        st.caption("O que a árvore usa nos splits. Features não usadas ficam zeradas.")

    with col_b:
        fi_lr = pd.DataFrame({
            "Feature": [FEATURES_NOMES.get(f, f) for f in features],
            "Importância": res["lr_perm_mean"],
        }).sort_values("Importância", ascending=True)

        fig_lr = px.bar(fi_lr, x="Importância", y="Feature", orientation="h",
                        title="Logistic Regression (Permutation)",
                        color="Importância", color_continuous_scale="Oranges")
        fig_lr.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_lr, use_container_width=True)
        st.caption("Todas as features contribuem — nenhuma é ignorada.")

    with col_c:
        fi_corr = pd.DataFrame({
            "Feature": [FEATURES_NOMES.get(f, f) for f in features],
            "Correlação": res["corr_vals"],
        }).sort_values("Correlação", ascending=True)

        fig_corr = px.bar(fi_corr, x="Correlação", y="Feature", orientation="h",
                          title="Correlação com Target",
                          color="Correlação", color_continuous_scale="Blues")
        fig_corr.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Independente de qualquer modelo — relevância geral de cada indicador.")

    # =====================================================================
    # 3. RADAR — PERFIL COMPARATIVO
    # =====================================================================
    st.markdown("---")
    st.header("3. Perfil Comparativo — Gráfico Radar")

    st.markdown("""
    Comparação do "perfil de risco" do Banco Master com a **média dos pares**.
    Use o seletor de período para explorar como o perfil mudou ao longo do tempo.
    """)

    radar_feats = ["Alavancagem", "CoberturaCapt", "CreditoSobreAtivo", "ROE", "CrescAtivo_QoQ"]
    radar_feats = [f for f in radar_feats if f in df_resumo.columns]

    if radar_feats:
        trimestres_disp = sorted(df_resumo["AnoTri"].unique())
        opcoes_periodo = {"Período completo (2019–2025)": trimestres_disp}
        for ano in sorted(df_resumo["Ano"].unique()):
            tris_ano = sorted(df_resumo[df_resumo["Ano"] == ano]["AnoTri"].unique())
            if tris_ano:
                opcoes_periodo[f"Ano {ano}"] = tris_ano

        periodo_sel = st.selectbox("Selecione o período para o radar:", list(opcoes_periodo.keys()), index=0)
        tris_filtro = opcoes_periodo[periodo_sel]
        df_filtrado = df_resumo[df_resumo["AnoTri"].isin(tris_filtro)]

        master_vals = df_filtrado[df_filtrado["NomeBanco"] == "Banco Master"][radar_feats].mean()
        pares_vals = df_filtrado[df_filtrado["NomeBanco"] != "Banco Master"][radar_feats].mean()

        all_vals = df_resumo[radar_feats]
        mins = all_vals.min()
        ranges = (all_vals.max() - mins).replace(0, 1)
        master_norm = (master_vals - mins) / ranges
        pares_norm = (pares_vals - mins) / ranges
        labels = [FEATURES_NOMES.get(f, f) for f in radar_feats]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=list(pares_norm.values) + [pares_norm.values[0]],
            theta=labels + [labels[0]], fill="toself", name="Média dos Pares",
            fillcolor="rgba(52, 152, 219, 0.15)", line=dict(color="#3498db", width=2),
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=list(master_norm.values) + [master_norm.values[0]],
            theta=labels + [labels[0]], fill="toself", name="Banco Master",
            fillcolor="rgba(231, 76, 60, 0.15)", line=dict(color="#e74c3c", width=3),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=f"Perfil de Risco — Master vs Média dos Pares ({periodo_sel})",
            height=500, showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        tabela = pd.DataFrame({
            "Indicador": labels,
            "Banco Master": [f"{v:.4f}" for v in master_vals.values],
            "Média dos Pares": [f"{v:.4f}" for v in pares_vals.values],
            "Diferença": [f"{m - p:+.4f}" for m, p in zip(master_vals.values, pares_vals.values)],
        })
        st.dataframe(tabela, use_container_width=True, hide_index=True)
        st.caption("Normalização min-max usa o dataset inteiro (2019–2025) para manter a escala consistente entre períodos.")

    # =====================================================================
    # 4. SIMULADOR "E SE?" (usa Logistic Regression — todos os sliders funcionam)
    # =====================================================================
    st.markdown("---")
    st.header("4. Análise 'E se?' — Simulador Interativo")

    st.markdown("""
    Ajuste os indicadores financeiros e veja em tempo real como o modelo classifica.
    Este simulador usa a **Logistic Regression** — todos os indicadores influenciam o resultado.
    """)

    lr_modelo = res["lr_modelo"]
    scaler = res["scaler"]

    col1, col2 = st.columns(2)
    slider_vals = {}

    with col1:
        slider_vals["Alavancagem"] = st.slider(
            "Alavancagem (Ativo / PL)", min_value=2.0, max_value=30.0, value=15.0, step=0.5)
        slider_vals["CoberturaCapt"] = st.slider(
            "Captações / Ativo Total", min_value=0.50, max_value=0.95, value=0.85, step=0.01)
        slider_vals["CreditoSobreAtivo"] = st.slider(
            "Crédito / Ativo Total", min_value=0.10, max_value=0.80, value=0.45, step=0.01)
        slider_vals["ROE"] = st.slider(
            "ROE (Retorno sobre PL)", min_value=-0.50, max_value=0.30, value=0.05, step=0.01)

    with col2:
        slider_vals["CrescAtivo_QoQ"] = st.slider(
            "Crescimento Trimestral (%)", min_value=-0.10, max_value=0.50, value=0.10, step=0.01)
        slider_vals["Basileia"] = st.slider(
            "Índice de Basileia", min_value=0.02, max_value=0.50, value=0.12, step=0.005,
            help="Mínimo regulatório: 11%")
        slider_vals["MargemBasileia"] = slider_vals["Basileia"] - 0.11
        st.metric("Margem Basileia (auto)", f"{slider_vals['MargemBasileia']:.2%}",
                   delta="Abaixo do mínimo" if slider_vals["MargemBasileia"] < 0 else "Acima do mínimo",
                   delta_color="inverse" if slider_vals["MargemBasileia"] < 0 else "normal")

    # Predição com Logistic Regression
    X_input = np.array([[slider_vals.get(f, 0.0) for f in features]])
    X_input_scaled = scaler.transform(X_input)

    lr_pred = lr_modelo.predict(X_input_scaled)[0]
    lr_proba = lr_modelo.predict_proba(X_input_scaled)[0]
    idx_pos = list(lr_modelo.classes_).index(1) if 1 in lr_modelo.classes_ else 0
    lr_prob_risco = lr_proba[idx_pos]

    # Predição com Decision Tree (para comparar)
    dt_modelo = res["dt_modelo"]
    dt_pred = dt_modelo.predict(X_input)[0]
    dt_proba = dt_modelo.predict_proba(X_input)[0]
    dt_prob_risco = dt_proba[idx_pos]

    st.markdown("---")
    st.markdown("**Resultado:**")
    r1, r2, r3, r4 = st.columns(4)

    if lr_pred == 1:
        r1.error("**Logistic Regression: RISCO**")
    else:
        r1.success("**Logistic Regression: NORMAL**")
    r2.metric("Prob. Risco (LR)", f"{lr_prob_risco:.1%}")

    if dt_pred == 1:
        r3.error("**Decision Tree: RISCO**")
    else:
        r3.success("**Decision Tree: NORMAL**")
    r4.metric("Prob. Risco (DT)", f"{dt_prob_risco:.1%}")

    # Contribuição de cada feature na LR
    with st.expander("Detalhamento: contribuição de cada feature (Logistic Regression)"):
        contrib = X_input_scaled[0] * res["lr_coefs"]
        contrib_df = pd.DataFrame({
            "Feature": [FEATURES_NOMES.get(f, f) for f in features],
            "Valor Informado": [slider_vals.get(f, 0) for f in features],
            "Contribuição ao Risco": contrib,
        }).sort_values("Contribuição ao Risco", ascending=False)

        fig_contrib = px.bar(
            contrib_df, x="Contribuição ao Risco", y="Feature", orientation="h",
            title="Contribuição de Cada Feature para a Classificação",
            color="Contribuição ao Risco",
            color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
        )
        fig_contrib.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_contrib, use_container_width=True)
        st.caption("Barras vermelhas (positivas) empurram para RISCO. Barras azuis (negativas) empurram para NORMAL.")

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
    # 5. MATRIZ DE CONFUSÃO (Decision Tree — modelo principal)
    # =====================================================================
    st.markdown("---")
    st.header("5. Validação — Matriz de Confusão")

    st.markdown("""
    A matriz de confusão mostra onde o modelo (Decision Tree) **acerta e erra**.
    """)

    if not df_pred.empty:
        from sklearn.metrics import confusion_matrix

        y_true = df_pred["Target"].values
        y_pred_dt = df_pred["PredicaoRisco"].values

        cm = confusion_matrix(y_true, y_pred_dt)
        tn, fp, fn, tp = cm.ravel()

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
            title="Matriz de Confusão (Decision Tree)",
            height=400, xaxis_title="Classificação do Modelo",
            yaxis_title="Valor Real", yaxis_autorange="reversed",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Verdadeiros Negativos", f"{tn}", "Normais corretos")
        col2.metric("Verdadeiros Positivos", f"{tp}", "Master detectado")
        col3.metric("Falsos Positivos", f"{fp}", "Alertas indevidos")
        col4.metric("Falsos Negativos", f"{fn}", "Não detectado")

        if fp > 0:
            st.subheader("Detalhamento dos Falsos Positivos")
            fps = df_pred[(df_pred["Target"] == 0) & (df_pred["PredicaoRisco"] == 1)]
            st.dataframe(
                fps[["NomeBanco", "AnoTri", "ProbRisco"]].rename(columns={
                    "NomeBanco": "Banco", "AnoTri": "Trimestre", "ProbRisco": "Probabilidade",
                }),
                use_container_width=True, hide_index=True,
            )
            st.markdown("""
            **Análise:** Esses trimestres representam momentos em que bancos saudáveis
            apresentaram indicadores temporariamente similares ao Master. O modelo não
            distingue estresse temporário de risco estrutural.
            """)

    # =====================================================================
    # 6. LIMITAÇÕES
    # =====================================================================
    st.markdown("---")
    st.header("6. Limitações")
    st.warning("""
    - Com apenas **1 banco liquidado entre 5**, ambos os modelos aprendem a separar o Master
      dos pares — não detectam risco bancário em geral.
    - A **Decision Tree** ignora Alavancagem, ROE e Crescimento (resolve tudo com CoberturaCapt + Basileia).
    - A **Logistic Regression** usa todas as features, mas assume relação linear entre indicadores e risco.
    - Os **falsos positivos** mostram que indicadores isolados não são suficientes —
      o contexto (crescimento explosivo + fraude contábil) é o que tornou o Master único.
    - Para generalizar, seria necessário um dataset com dezenas de bancos liquidados ao longo de décadas.
    """)


if __name__ == "__main__":
    main()
