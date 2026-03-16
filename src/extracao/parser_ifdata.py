"""
Módulo de extração e normalização dos dados brutos do IF.data.

Transforma os dados brutos (formato longo da API OData) em
DataFrames estruturados prontos para análise.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
BRUTO_DIR = DATA_DIR / "bruto"
PREPARADO_DIR = DATA_DIR / "preparado"


def carregar_dados_brutos(arquivo: str = "ifdata_consolidado.csv") -> pd.DataFrame:
    """Carrega CSV bruto gerado pelo crawler."""
    caminho = BRUTO_DIR / arquivo
    if not caminho.exists():
        logger.error(f"Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

    df = pd.read_csv(caminho, encoding="utf-8-sig")
    logger.info(f"Carregado {caminho.name}: {len(df)} registros")
    return df


def normalizar_periodo(df: pd.DataFrame) -> pd.DataFrame:
    """Converte AnoMes (YYYYMM) para datetime e adiciona colunas auxiliares."""
    df = df.copy()
    df["AnoMes"] = df["AnoMes"].astype(str)
    df["DataRef"] = pd.to_datetime(df["AnoMes"], format="%Y%m")
    df["Ano"] = df["DataRef"].dt.year
    df["Trimestre"] = df["DataRef"].dt.quarter
    df["AnoTri"] = df["Ano"].astype(str) + "Q" + df["Trimestre"].astype(str)
    return df


def pivotar_resumo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma dados do relatorio Resumo de formato longo para largo.
    Cada linha = um banco em um trimestre.
    Colunas = indicadores financeiros.
    """
    resumo = df[df["NomeRelatorio"] == "Resumo"].copy()

    if resumo.empty:
        logger.warning("Sem dados de Resumo para pivotar")
        return pd.DataFrame()

    resumo = normalizar_periodo(resumo)

    pivot = resumo.pivot_table(
        values="Saldo",
        index=["NomeBanco", "CodInst", "AnoMes", "DataRef", "Ano", "Trimestre", "AnoTri"],
        columns="NomeColuna",
        aggfunc="first",
    ).reset_index()

    pivot.columns.name = None
    logger.info(f"Resumo pivotado: {len(pivot)} linhas x {len(pivot.columns)} colunas")
    return pivot


def pivotar_capital(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma dados do relatorio Capital de formato longo para largo.
    Inclui Indice de Basileia, RWA, Patrimonio de Referencia, etc.
    """
    capital = df[df["NomeRelatorio"].str.contains("Capital", case=False, na=False)].copy()

    if capital.empty:
        logger.warning("Sem dados de Capital para pivotar")
        return pd.DataFrame()

    capital = normalizar_periodo(capital)

    pivot = capital.pivot_table(
        values="Saldo",
        index=["NomeBanco", "CodInst", "AnoMes", "DataRef", "Ano", "Trimestre", "AnoTri"],
        columns="NomeColuna",
        aggfunc="first",
    ).reset_index()

    pivot.columns.name = None
    logger.info(f"Capital pivotado: {len(pivot)} linhas x {len(pivot.columns)} colunas")
    return pivot


def pivotar_ativo(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma dados do relatorio Ativo."""
    ativo = df[df["NomeRelatorio"] == "Ativo"].copy()
    if ativo.empty:
        return pd.DataFrame()

    ativo = normalizar_periodo(ativo)
    pivot = ativo.pivot_table(
        values="Saldo",
        index=["NomeBanco", "CodInst", "AnoMes", "DataRef", "Ano", "Trimestre", "AnoTri"],
        columns="NomeColuna",
        aggfunc="first",
    ).reset_index()

    pivot.columns.name = None
    return pivot


def pivotar_passivo(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma dados do relatorio Passivo."""
    passivo = df[df["NomeRelatorio"] == "Passivo"].copy()
    if passivo.empty:
        return pd.DataFrame()

    passivo = normalizar_periodo(passivo)
    pivot = passivo.pivot_table(
        values="Saldo",
        index=["NomeBanco", "CodInst", "AnoMes", "DataRef", "Ano", "Trimestre", "AnoTri"],
        columns="NomeColuna",
        aggfunc="first",
    ).reset_index()

    pivot.columns.name = None
    return pivot


def pivotar_dre(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma dados da Demonstracao de Resultado."""
    dre = df[df["NomeRelatorio"].str.contains("Resultado", case=False, na=False)].copy()
    if dre.empty:
        return pd.DataFrame()

    dre = normalizar_periodo(dre)
    pivot = dre.pivot_table(
        values="Saldo",
        index=["NomeBanco", "CodInst", "AnoMes", "DataRef", "Ano", "Trimestre", "AnoTri"],
        columns="NomeColuna",
        aggfunc="first",
    ).reset_index()

    pivot.columns.name = None
    return pivot


def extrair_todos(
    arquivo_consolidado: str = "ifdata_consolidado.csv",
    dir_saida: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """
    Processa todos os relatorios e salva DataFrames pivotados.
    Retorna dicionario com os DataFrames.
    """
    if dir_saida is None:
        dir_saida = PREPARADO_DIR

    dir_saida.mkdir(parents=True, exist_ok=True)

    df = carregar_dados_brutos(arquivo_consolidado)
    if df.empty:
        return {}

    resultados = {}

    processadores = {
        "resumo": pivotar_resumo,
        "capital": pivotar_capital,
        "ativo": pivotar_ativo,
        "passivo": pivotar_passivo,
        "dre": pivotar_dre,
    }

    for nome, func in processadores.items():
        logger.info(f"Processando: {nome}")
        resultado = func(df)
        if not resultado.empty:
            arquivo = dir_saida / f"{nome}.csv"
            resultado.to_csv(arquivo, index=False, encoding="utf-8-sig")
            logger.info(f"  Salvo: {arquivo.name} ({len(resultado)} linhas)")
            resultados[nome] = resultado

    return resultados


if __name__ == "__main__":
    print("=" * 60)
    print("Extracao e normalizacao dos dados IF.data")
    print("=" * 60)

    resultados = extrair_todos()

    for nome, df in resultados.items():
        print(f"\n{nome}: {df.shape}")
        if "NomeBanco" in df.columns:
            print(f"  Bancos: {sorted(df['NomeBanco'].unique())}")
            print(f"  Periodos: {df['AnoTri'].min()} a {df['AnoTri'].max()}")

    print("\nConcluido!")
