"""
Modulo de coleta de series temporais do BCB via SGS (Sistema Gerenciador de Series).

Fonte: Banco Central do Brasil - SGS
Acesso: API REST publica
Documentacao: https://www3.bcb.gov.br/sgspub/
Licenca: Dados publicos

Series coletadas:
  - CDI acumulado mensal (codigo 4391)
  - Selic Meta (codigo 432)
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"

SERIES = {
    4391: "CDI_acumulado_mensal",
    432: "Selic_meta",
}


def coletar_serie_sgs(
    codigo: int,
    data_inicio: str = "01/01/2019",
    data_fim: Optional[str] = None,
) -> pd.DataFrame:
    """
    Coleta uma serie temporal do SGS/BCB.

    Parametros:
        codigo: Codigo da serie no SGS
        data_inicio: Data inicial no formato DD/MM/YYYY
        data_fim: Data final (padrao: hoje)

    Retorna:
        DataFrame com colunas: data, valor
    """
    if data_fim is None:
        data_fim = datetime.now().strftime("%d/%m/%Y")

    url = SGS_URL.format(codigo=codigo)
    params = {
        "formato": "json",
        "dataInicial": data_inicio,
        "dataFinal": data_fim,
    }

    logger.info(f"SGS: serie {codigo} ({SERIES.get(codigo, '?')}) de {data_inicio} a {data_fim}")

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    dados = response.json()
    if not dados:
        logger.warning(f"Nenhum dado para serie {codigo}")
        return pd.DataFrame()

    df = pd.DataFrame(dados)
    df.columns = ["data", "valor"]
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df["serie"] = SERIES.get(codigo, str(codigo))

    logger.info(f"  -> {len(df)} registros")
    return df


def coletar_todas_series(
    data_inicio: str = "01/01/2019",
    dir_saida: Optional[Path] = None,
) -> pd.DataFrame:
    """Coleta todas as series definidas e salva em CSV."""
    if dir_saida is None:
        dir_saida = Path(__file__).resolve().parents[2] / "data" / "bruto"

    dir_saida.mkdir(parents=True, exist_ok=True)
    todos = []

    for codigo, nome in SERIES.items():
        df = coletar_serie_sgs(codigo, data_inicio=data_inicio)
        if not df.empty:
            df.to_csv(
                dir_saida / f"sgs_{nome}.csv", index=False, encoding="utf-8-sig"
            )
            todos.append(df)

    if not todos:
        return pd.DataFrame()

    consolidado = pd.concat(todos, ignore_index=True)
    consolidado.to_csv(
        dir_saida / "sgs_series_temporais.csv", index=False, encoding="utf-8-sig"
    )
    logger.info(f"Salvo sgs_series_temporais.csv ({len(consolidado)} registros)")
    return consolidado


if __name__ == "__main__":
    print("=" * 60)
    print("Coleta de series temporais BCB/SGS")
    print("=" * 60)

    df = coletar_todas_series()

    if not df.empty:
        for serie in df["serie"].unique():
            sub = df[df["serie"] == serie]
            print(f"\n{serie}: {len(sub)} registros")
            print(f"  Periodo: {sub['data'].min().date()} a {sub['data'].max().date()}")
            print(f"  Ultimo valor: {sub.iloc[-1]['valor']}")

    print("\nConcluido!")
