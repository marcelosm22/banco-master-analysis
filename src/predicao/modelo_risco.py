"""
Módulo de predição: modelo simples de classificação de risco bancário.

Abordagem: Decision Tree Classifier
  - Target: banco sofreu intervenção/liquidação (1) ou opera normalmente (0)
  - Features: indicadores financeiros trimestrais
  - Treino: dados de todos os bancos alvo (5 bancos, ~80 amostras)
  - Validação: StratifiedKFold 5-fold

Limitações conhecidas (documentadas propositalmente):
  - Apenas 1 banco positivo (Master) entre 5 — o modelo aprende a separar
    Master dos pares, não a detectar risco bancário em geral.
  - 100% de acurácia é esperado nesse cenário: com 1 caso positivo,
    qualquer feature que distinga Master dos demais resolve o problema.
  - Para um modelo generalizável, seria necessário um dataset com dezenas
    de bancos liquidados vs saudáveis ao longo de décadas.

O objetivo não é construir um modelo de produção, mas demonstrar
que dados bem preparados alimentam decisões e identificam padrões.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PREPARADO_DIR = DATA_DIR / "preparado"

BANCOS_LIQUIDADOS = {"Banco Master"}

# Features do Resumo (sem CrescAtivo_YoY para preservar mais amostras)
FEATURES = [
    "Alavancagem",
    "CoberturaCapt",
    "CreditoSobreAtivo",
    "ROE",
    "CrescAtivo_QoQ",
]

FEATURES_CAPITAL = [
    "Basileia",
    "MargemBasileia",
]


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega dados preparados de indicadores."""
    resumo = PREPARADO_DIR / "indicadores_resumo.csv"
    capital = PREPARADO_DIR / "indicadores_capital.csv"

    df_resumo = pd.DataFrame()
    df_capital = pd.DataFrame()

    if resumo.exists():
        df_resumo = pd.read_csv(resumo, encoding="utf-8-sig", parse_dates=["DataRef"])
    if capital.exists():
        df_capital = pd.read_csv(capital, encoding="utf-8-sig", parse_dates=["DataRef"])

    return df_resumo, df_capital


def preparar_dataset(
    df_resumo: pd.DataFrame,
    df_capital: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Monta dataset para o modelo combinando indicadores de resumo e capital.
    Retorna DataFrame com features + target + metadados.
    """
    if df_resumo.empty:
        return pd.DataFrame(), []

    cols_resumo = ["NomeBanco", "AnoMes", "DataRef", "AnoTri"]
    cols_resumo += [c for c in FEATURES if c in df_resumo.columns]
    df = df_resumo[cols_resumo].copy()

    features_usadas = [c for c in FEATURES if c in df.columns]

    if not df_capital.empty:
        cols_capital = ["NomeBanco", "AnoMes"]
        cols_capital += [c for c in FEATURES_CAPITAL if c in df_capital.columns]
        df = df.merge(df_capital[cols_capital], on=["NomeBanco", "AnoMes"], how="left")
        features_usadas += [c for c in FEATURES_CAPITAL if c in df.columns]

    df["Target"] = df["NomeBanco"].isin(BANCOS_LIQUIDADOS).astype(int)

    # Preencher NaN pontuais com último valor do banco (forward fill)
    # Isso preserva trimestres de 2025 onde CreditoSobreAtivo ou CrescAtivo_QoQ
    # podem estar ausentes por atraso na publicação do BCB.
    df = df.sort_values(["NomeBanco", "AnoMes"])
    for feat in features_usadas:
        df[feat] = df.groupby("NomeBanco")[feat].ffill()

    df_limpo = df.dropna(subset=features_usadas)
    logger.info(
        f"Dataset: {len(df_limpo)} amostras, {len(features_usadas)} features, "
        f"{df_limpo['Target'].sum()} positivos, "
        f"{len(df_limpo) - df_limpo['Target'].sum()} negativos"
    )

    return df_limpo, features_usadas


def treinar_modelo(
    df: pd.DataFrame,
    features: list[str],
    max_depth: int = 3,
) -> tuple[DecisionTreeClassifier, dict]:
    """
    Treina Decision Tree e retorna modelo + métricas.
    """
    X = df[features].values
    y = df["Target"].values

    modelo = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )

    # Cross-validation estratificada
    try:
        skf = StratifiedKFold(n_splits=min(5, len(y)), shuffle=True, random_state=42)
        scores = cross_val_score(modelo, X, y, cv=skf, scoring="f1")
    except ValueError:
        scores = np.array([0.0])

    modelo.fit(X, y)
    y_pred = modelo.predict(X)

    metricas = {
        "acuracia": accuracy_score(y, y_pred),
        "cv_f1_mean": float(np.nanmean(scores)),
        "cv_f1_std": float(np.nanstd(scores)),
        "feature_importance": dict(zip(features, modelo.feature_importances_)),
        "arvore_texto": export_text(modelo, feature_names=features, max_depth=10),
        "n_amostras": len(y),
        "n_positivos": int(y.sum()),
        "n_negativos": int(len(y) - y.sum()),
    }

    logger.info(f"Modelo treinado: acurácia={metricas['acuracia']:.2%}")
    logger.info(f"CV F1: {metricas['cv_f1_mean']:.2%} +/- {metricas['cv_f1_std']:.2%}")
    return modelo, metricas


def classificar_por_trimestre(
    modelo: DecisionTreeClassifier,
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Classifica cada banco em cada trimestre."""
    df = df.copy()
    X = df[features].values

    df["PredicaoRisco"] = modelo.predict(X)
    proba = modelo.predict_proba(X)
    idx_positivo = list(modelo.classes_).index(1) if 1 in modelo.classes_ else 0
    df["ProbRisco"] = proba[:, idx_positivo]

    return df[["NomeBanco", "AnoMes", "AnoTri", "DataRef", "Target", "PredicaoRisco", "ProbRisco"]]


def executar_predicao(dir_saida: Optional[Path] = None) -> dict:
    """Pipeline completo de predição."""
    if dir_saida is None:
        dir_saida = PREPARADO_DIR

    df_resumo, df_capital = carregar_dados()
    df, features = preparar_dataset(df_resumo, df_capital)

    if df.empty or len(features) < 2:
        logger.error("Dados insuficientes para treinar modelo")
        return {}

    modelo, metricas = treinar_modelo(df, features)
    classificacao = classificar_por_trimestre(modelo, df, features)
    classificacao.to_csv(
        dir_saida / "predicao_risco.csv", index=False, encoding="utf-8-sig"
    )

    return {
        "modelo": modelo,
        "metricas": metricas,
        "features": features,
        "classificacao": classificacao,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Predição: Modelo de Risco Bancário")
    print("=" * 60)

    resultado = executar_predicao()

    if not resultado:
        print("Erro: dados insuficientes.")
    else:
        metricas = resultado["metricas"]

        print(f"\nAmostras: {metricas['n_amostras']} ({metricas['n_positivos']} risco, {metricas['n_negativos']} normal)")
        print(f"Acurácia: {metricas['acuracia']:.2%}")
        print(f"CV F1: {metricas['cv_f1_mean']:.2%} +/- {metricas['cv_f1_std']:.2%}")

        print("\nImportância das features:")
        for feat, imp in sorted(
            metricas["feature_importance"].items(), key=lambda x: -x[1]
        ):
            barra = "#" * int(imp * 40)
            print(f"  {feat:25s}: {imp:.3f} {barra}")

        print("\nÁrvore de decisão:")
        print(metricas["arvore_texto"])

        print("\nLIMITAÇÕES:")
        print("  - Apenas 1 banco liquidado entre 5 analisados")
        print("  - 100% de acurácia é esperado nesse cenário (overfitting ao caso)")
        print("  - O modelo identifica o padrão do Master, não risco bancário geral")
        print("  - Para generalizar, seria necessário um dataset com múltiplos casos")

        print("\nClassificação por trimestre (Banco Master):")
        classif = resultado["classificacao"]
        master = classif[classif["NomeBanco"] == "Banco Master"].sort_values("AnoMes")
        for _, row in master.iterrows():
            flag = "*** RISCO ***" if row["PredicaoRisco"] == 1 else "Normal"
            print(f"  {row['AnoTri']}  Prob={row['ProbRisco']:.2%}  {flag}")

    print("\nConcluído!")
