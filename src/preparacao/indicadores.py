"""
Módulo de preparação: cálculo de indicadores derivados.

Transforma os dados extraídos em indicadores financeiros
comparáveis entre bancos e ao longo do tempo.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PREPARADO_DIR = DATA_DIR / "preparado"

# Minimo regulatorio de Basileia no Brasil
BASILEIA_MINIMO = 0.11


def carregar_preparado(nome: str) -> pd.DataFrame:
    """Carrega um CSV do diretorio preparado."""
    caminho = PREPARADO_DIR / f"{nome}.csv"
    if not caminho.exists():
        logger.warning(f"Nao encontrado: {caminho}")
        return pd.DataFrame()
    df = pd.read_csv(caminho, encoding="utf-8-sig", parse_dates=["DataRef"])
    return df


def calcular_indicadores_resumo(df_resumo: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores derivados a partir do Resumo.

    Indicadores:
      - Alavancagem = Ativo Total / Patrimonio Liquido
      - Cobertura de Captacoes = Captacoes / Ativo Total
      - Credito sobre Ativo = Carteira de Credito / Ativo Total
      - Crescimento Ativo (% trimestral)
      - Crescimento Ativo (% anual - YoY)
      - Ativo Base 100 (indexado ao primeiro trimestre)
    """
    df = df_resumo.copy()

    # Padronizar nomes de colunas (a API retorna com acentos)
    # Usar mapeamento exato para evitar duplicatas
    col_map = {
        "Ativo Total": "AtivoTotal",
        "Patrimônio Líquido": "PatrimonioLiquido",
        "Captações": "Captacoes",
        "Carteira de Crédito Classificada": "CarteiraCredito",
        "Carteira de Crédito": "CarteiraCreditoSimples",
        "Lucro Líquido": "LucroLiquido",
        "Passivo Circulante e Exigível a Longo Prazo e Resultados de Exercícios Futuros": "PassivoTotal",
        "Passivo Exigível": "PassivoExigivel",
        "Títulos e Valores Mobiliários": "TVM",
    }
    # Só renomear colunas que existem
    col_map = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=col_map)

    # Indicadores derivados
    if "AtivoTotal" in df.columns and "PatrimonioLiquido" in df.columns:
        df["Alavancagem"] = df["AtivoTotal"] / df["PatrimonioLiquido"].replace(0, np.nan)

    if "Captacoes" in df.columns and "AtivoTotal" in df.columns:
        df["CoberturaCapt"] = df["Captacoes"] / df["AtivoTotal"].replace(0, np.nan)

    if "CarteiraCredito" in df.columns and "AtivoTotal" in df.columns:
        df["CreditoSobreAtivo"] = df["CarteiraCredito"] / df["AtivoTotal"].replace(0, np.nan)

    if "LucroLiquido" in df.columns and "PatrimonioLiquido" in df.columns:
        df["ROE"] = df["LucroLiquido"] / df["PatrimonioLiquido"].replace(0, np.nan)

    # Crescimento trimestral e anual por banco
    df = df.sort_values(["NomeBanco", "DataRef"])

    if "AtivoTotal" in df.columns:
        df["CrescAtivo_QoQ"] = df.groupby("NomeBanco")["AtivoTotal"].pct_change()
        df["CrescAtivo_YoY"] = df.groupby("NomeBanco")["AtivoTotal"].pct_change(4)

        # Base 100 (primeiro trimestre de cada banco = 100)
        primeiro = df.groupby("NomeBanco")["AtivoTotal"].transform("first")
        df["AtivoBase100"] = (df["AtivoTotal"] / primeiro) * 100

    if "PatrimonioLiquido" in df.columns:
        df["CrescPL_QoQ"] = df.groupby("NomeBanco")["PatrimonioLiquido"].pct_change()
        df["CrescPL_YoY"] = df.groupby("NomeBanco")["PatrimonioLiquido"].pct_change(4)

    logger.info(f"Indicadores Resumo calculados: {len(df)} linhas, {len(df.columns)} colunas")
    return df


def calcular_indicadores_capital(df_capital: pd.DataFrame) -> pd.DataFrame:
    """
    Processa indicadores de capital e adiciona sinais de alerta.

    Indicadores:
      - Basileia normalizado
      - Margem sobre minimo regulatorio
      - Sinal de alerta (semaforo)
    """
    df = df_capital.copy()

    # O BCB mudou a fórmula do Basileia ao longo do tempo:
    #   Até ~Q2/2022: "Índice de Basileia (m) = (e) / (i)"
    #   A partir de ~Q3/2022: "Índice de Basileia (n) = (e) / (j)"
    # Precisamos combinar ambas em uma única coluna.
    cols_basileia = [c for c in df.columns if "Basileia" in c and "Índice" in c]

    if not cols_basileia:
        logger.warning("Coluna de Basileia não encontrada")
        return df

    # Combinar: usar a primeira coluna não-NaN de cada linha
    df["Basileia"] = np.nan
    for col in cols_basileia:
        df["Basileia"] = df["Basileia"].fillna(df[col])

    logger.info(f"Basileia: combinadas {len(cols_basileia)} colunas, {df['Basileia'].notna().sum()}/{len(df)} valores")

    # Margem sobre minimo regulatorio
    df["MargemBasileia"] = df["Basileia"] - BASILEIA_MINIMO

    # Semaforo de risco
    df["SemaforoBasileia"] = pd.cut(
        df["MargemBasileia"],
        bins=[-np.inf, 0, 0.02, 0.05, np.inf],
        labels=["VERMELHO", "AMARELO", "ATENCAO", "VERDE"],
    )

    logger.info(f"Indicadores Capital calculados: {len(df)} linhas")
    return df


def calcular_score_estresse(
    df_resumo: pd.DataFrame,
    df_capital: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula um Score de Estresse Financeiro combinando multiplos indicadores.

    O score varia de 0 (saudavel) a 100 (critico).
    Componentes (peso igual):
      1. Proximidade do Basileia ao minimo (0-25 pts)
      2. Alavancagem vs mediana dos pares (0-25 pts)
      3. Crescimento do ativo vs pares (0-25 pts)
      4. Concentracao de captacoes (0-25 pts)
    """
    if df_resumo.empty or df_capital.empty:
        return pd.DataFrame()

    # Preparar dados
    resumo = df_resumo[["NomeBanco", "AnoMes", "DataRef", "AnoTri"]].copy()

    # 1. Score Basileia (0-25)
    if "Basileia" in df_capital.columns:
        cap = df_capital[["NomeBanco", "AnoMes", "Basileia"]].copy()
        resumo = resumo.merge(cap, on=["NomeBanco", "AnoMes"], how="left")
        # Quanto mais perto de 11%, maior o score
        resumo["ScoreBasileia"] = np.clip(
            25 * (1 - (resumo["Basileia"] - BASILEIA_MINIMO) / 0.10), 0, 25
        )
    else:
        resumo["ScoreBasileia"] = 0

    # 2. Score Alavancagem (0-25)
    if "Alavancagem" in df_resumo.columns:
        alav = df_resumo[["NomeBanco", "AnoMes", "Alavancagem"]].copy()
        resumo = resumo.merge(alav, on=["NomeBanco", "AnoMes"], how="left")
        # Mediana dos pares por trimestre
        mediana_alav = resumo.groupby("AnoMes")["Alavancagem"].transform("median")
        desvio = (resumo["Alavancagem"] - mediana_alav) / mediana_alav.replace(0, np.nan)
        resumo["ScoreAlavancagem"] = np.clip(desvio * 25, 0, 25)
    else:
        resumo["ScoreAlavancagem"] = 0

    # 3. Score Crescimento Anormal (0-25)
    if "CrescAtivo_YoY" in df_resumo.columns:
        cresc = df_resumo[["NomeBanco", "AnoMes", "CrescAtivo_YoY"]].copy()
        resumo = resumo.merge(cresc, on=["NomeBanco", "AnoMes"], how="left")
        mediana_cresc = resumo.groupby("AnoMes")["CrescAtivo_YoY"].transform("median")
        desvio_cresc = (resumo["CrescAtivo_YoY"] - mediana_cresc).fillna(0)
        resumo["ScoreCrescimento"] = np.clip(desvio_cresc * 50, 0, 25)
    else:
        resumo["ScoreCrescimento"] = 0

    # 4. Score Captacoes (0-25)
    if "CoberturaCapt" in df_resumo.columns:
        capt = df_resumo[["NomeBanco", "AnoMes", "CoberturaCapt"]].copy()
        resumo = resumo.merge(capt, on=["NomeBanco", "AnoMes"], how="left")
        mediana_capt = resumo.groupby("AnoMes")["CoberturaCapt"].transform("median")
        desvio_capt = (resumo["CoberturaCapt"] - mediana_capt).fillna(0)
        resumo["ScoreCaptacoes"] = np.clip(desvio_capt * 50, 0, 25)
    else:
        resumo["ScoreCaptacoes"] = 0

    # Score total
    resumo["ScoreEstresse"] = (
        resumo["ScoreBasileia"]
        + resumo["ScoreAlavancagem"]
        + resumo["ScoreCrescimento"]
        + resumo["ScoreCaptacoes"]
    )

    # Classificacao
    resumo["ClasseRisco"] = pd.cut(
        resumo["ScoreEstresse"],
        bins=[-np.inf, 20, 40, 60, np.inf],
        labels=["BAIXO", "MODERADO", "ALTO", "CRITICO"],
    )

    logger.info(f"Score de estresse calculado: {len(resumo)} linhas")
    return resumo


def preparar_todos(dir_saida: Optional[Path] = None) -> dict[str, pd.DataFrame]:
    """
    Pipeline completo de preparacao.
    Carrega dados extraidos, calcula indicadores, salva resultados.
    """
    if dir_saida is None:
        dir_saida = PREPARADO_DIR

    dir_saida.mkdir(parents=True, exist_ok=True)
    resultados = {}

    # Carregar dados extraidos
    df_resumo = carregar_preparado("resumo")
    df_capital = carregar_preparado("capital")

    # Calcular indicadores
    if not df_resumo.empty:
        df_resumo = calcular_indicadores_resumo(df_resumo)
        df_resumo.to_csv(dir_saida / "indicadores_resumo.csv", index=False, encoding="utf-8-sig")
        resultados["resumo"] = df_resumo
        logger.info(f"Salvo: indicadores_resumo.csv")

    if not df_capital.empty:
        df_capital = calcular_indicadores_capital(df_capital)
        df_capital.to_csv(dir_saida / "indicadores_capital.csv", index=False, encoding="utf-8-sig")
        resultados["capital"] = df_capital
        logger.info(f"Salvo: indicadores_capital.csv")

    if not df_resumo.empty and not df_capital.empty:
        df_score = calcular_score_estresse(df_resumo, df_capital)
        df_score.to_csv(dir_saida / "score_estresse.csv", index=False, encoding="utf-8-sig")
        resultados["score"] = df_score
        logger.info(f"Salvo: score_estresse.csv")

    return resultados


if __name__ == "__main__":
    print("=" * 60)
    print("Preparacao: Calculo de Indicadores")
    print("=" * 60)

    resultados = preparar_todos()

    for nome, df in resultados.items():
        print(f"\n{nome}: {df.shape}")
        if "NomeBanco" in df.columns:
            print(f"  Bancos: {sorted(df['NomeBanco'].unique())}")

    print("\nConcluido!")
