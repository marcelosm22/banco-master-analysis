"""
Modulo de coleta de dados do BCB IF.data via API OData.

Fonte: Banco Central do Brasil - Portal de Dados Abertos
URL Base: https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata/
Documentacao: https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/swagger-ui3
Licenca: Dados publicos (Lei de Acesso a Informacao)

Mapeamento de TipoInstituicao no IfDataValores:
  1 = Conglomerado prudencial (CodConglomeradoPrudencial, ex: C0080367)
  2 = Conglomerado financeiro (CodConglomeradoFinanceiro, ex: C0050201)
  3 = Instituicao individual (CNPJ raiz 8 digitos, ex: 33923798)
  4 = Consolidado bancario (agregado, sem dados individuais)

Estrategia:
  - Relatorios 1-4 (Resumo, Ativo, Passivo, DRE): usar TipoInstituicao=3
  - Relatorio 5 (Capital/Basileia): usar TipoInstituicao=1 (congl. prudencial)
  - A API não suporta $filter — baixamos tudo e filtramos no pandas.
"""

import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata"

# -- Bancos alvo ---------------------------------------------------------------
# CodInst (CNPJ raiz) para TipoInstituicao=3
BANCOS_INDIVIDUAL = {
    "Banco Master":   "33923798",
    "Banco Inter":    "00416968",
    "Banco Pine":     "62144175",
    "Banco Original": "92894922",
    "Banco Daycoval": "62232889",
}

# CodConglomeradoPrudencial para TipoInstituicao=1 (relatorio de Capital)
BANCOS_CONGL_PRUDENCIAL = {
    "Banco Master":   "C0080367",
    "Banco Inter":    "C0080996",
    "Banco Pine":     "C0080374",
    "Banco Original": "C0080903",
    "Banco Daycoval": "C0081744",
}

# -- Relatorios ----------------------------------------------------------------
RELATORIOS_INDIVIDUAL = {
    "1": "Resumo",
    "2": "Ativo",
    "3": "Passivo",
    "4": "Demonstracao de Resultado",
}

RELATORIOS_CAPITAL = {
    "5": "Informacoes de Capital",
}

TIPO_INDIVIDUAL = 3
TIPO_CONGL_PRUDENCIAL = 1


def gerar_trimestres(ano_inicio: int, ano_fim: int) -> list[str]:
    """Gera lista de AnoMes trimestrais no formato YYYYMM."""
    trimestres = []
    for ano in range(ano_inicio, ano_fim + 1):
        for mes in ["03", "06", "09", "12"]:
            trimestres.append(f"{ano}{mes}")
    return trimestres


def _paginar_odata(url: str, params: dict, max_total: int = 100000) -> list[dict]:
    """Faz paginacao automatica da API OData usando $skip/$top."""
    todos = []
    page_size = 10000
    skip = 0

    while skip < max_total:
        p = {**params, "$top": str(page_size), "$skip": str(skip)}
        response = requests.get(url, params=p, timeout=120)
        response.raise_for_status()

        registros = response.json().get("value", [])
        if not registros:
            break

        todos.extend(registros)
        skip += len(registros)

        if len(registros) < page_size:
            break

    return todos


def consultar_ifdata_valores(
    ano_mes: str,
    tipo_instituicao: int,
    relatorio: str,
    cod_inst_list: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Consulta o endpoint IfDataValores da API OData do BCB.
    Baixa o relatorio completo e filtra localmente por CodInst.
    """
    url = (
        f"{BASE_URL}/IfDataValores("
        f"AnoMes=@AnoMes,"
        f"TipoInstituicao=@TipoInstituicao,"
        f"Relatorio=@Relatorio)"
    )

    params = {
        "@AnoMes": ano_mes,
        "@TipoInstituicao": str(tipo_instituicao),
        "@Relatorio": f"'{relatorio}'",
        "$format": "json",
    }

    logger.info(f"API: {ano_mes} Tipo={tipo_instituicao} Rel={relatorio}")

    registros = _paginar_odata(url, params)

    if not registros:
        logger.warning(f"Vazio: {ano_mes}/{relatorio}")
        return pd.DataFrame()

    df = pd.DataFrame(registros)

    if cod_inst_list:
        df = df[df["CodInst"].isin(cod_inst_list)]

    logger.info(f"  -> {len(df)} registros")
    return df


def consultar_cadastro(ano_mes: str) -> pd.DataFrame:
    """Consulta o cadastro de instituicoes financeiras para um periodo."""
    url = f"{BASE_URL}/IfDataCadastro(AnoMes=@AnoMes)"
    params = {"@AnoMes": ano_mes, "$format": "json", "$top": "2000"}
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return pd.DataFrame(response.json().get("value", []))


def coletar_relatorio_trimestral(
    trimestre: str,
    relatorio: str,
    tipo_instituicao: int,
    bancos: dict,
    pausa: float = 1.0,
) -> pd.DataFrame:
    """Coleta um relatorio/trimestre e filtra pelos bancos alvo."""
    cod_inst_list = list(bancos.values())
    nome_por_cod = {v: k for k, v in bancos.items()}

    try:
        df = consultar_ifdata_valores(
            ano_mes=trimestre,
            tipo_instituicao=tipo_instituicao,
            relatorio=relatorio,
            cod_inst_list=cod_inst_list,
        )
        if not df.empty:
            df["NomeBanco"] = df["CodInst"].map(nome_por_cod)
        time.sleep(pausa)
        return df

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP {trimestre}/{relatorio}: {e}")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        logger.error(f"Conexao: {e}")
        return pd.DataFrame()


def coletar_todos_dados(
    trimestres: Optional[list[str]] = None,
    dir_saida: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Coleta completa: relatorios 1-4 (individual) + 5 (capital/prudencial).
    Salva CSVs brutos separados e consolidado.
    """
    if trimestres is None:
        trimestres = gerar_trimestres(2019, 2025)
    if dir_saida is None:
        dir_saida = Path(__file__).resolve().parents[2] / "data" / "bruto"

    dir_saida.mkdir(parents=True, exist_ok=True)
    todos = []

    # -- Relatorios 1-4: TipoInstituicao=3 (individual, por CNPJ) --
    total_indiv = len(trimestres) * len(RELATORIOS_INDIVIDUAL)
    i = 0
    for trimestre in trimestres:
        for cod_rel, nome_rel in RELATORIOS_INDIVIDUAL.items():
            i += 1
            logger.info(f"[{i}/{total_indiv}] {trimestre} - {nome_rel}")
            df = coletar_relatorio_trimestral(
                trimestre=trimestre,
                relatorio=cod_rel,
                tipo_instituicao=TIPO_INDIVIDUAL,
                bancos=BANCOS_INDIVIDUAL,
            )
            if not df.empty:
                todos.append(df)

    # Salvar parcial
    if todos:
        df_indiv = pd.concat(todos, ignore_index=True)
        df_indiv.to_csv(
            dir_saida / "ifdata_individual.csv", index=False, encoding="utf-8-sig"
        )
        logger.info(f"Salvo ifdata_individual.csv ({len(df_indiv)} registros)")

    # -- Relatorio 5: TipoInstituicao=1 (congl. prudencial) --
    todos_cap = []
    for j, trimestre in enumerate(trimestres, 1):
        logger.info(f"[Capital {j}/{len(trimestres)}] {trimestre}")
        df = coletar_relatorio_trimestral(
            trimestre=trimestre,
            relatorio="5",
            tipo_instituicao=TIPO_CONGL_PRUDENCIAL,
            bancos=BANCOS_CONGL_PRUDENCIAL,
        )
        if not df.empty:
            todos_cap.append(df)

    if todos_cap:
        df_cap = pd.concat(todos_cap, ignore_index=True)
        df_cap.to_csv(
            dir_saida / "ifdata_capital.csv", index=False, encoding="utf-8-sig"
        )
        logger.info(f"Salvo ifdata_capital.csv ({len(df_cap)} registros)")
        todos.extend(todos_cap)

    # Consolidado
    if not todos:
        logger.warning("Nenhum dado coletado!")
        return pd.DataFrame()

    consolidado = pd.concat(todos, ignore_index=True)
    consolidado.to_csv(
        dir_saida / "ifdata_consolidado.csv", index=False, encoding="utf-8-sig"
    )
    logger.info(f"Salvo ifdata_consolidado.csv ({len(consolidado)} registros)")
    return consolidado


def listar_relatorios() -> pd.DataFrame:
    """Consulta a lista de relatorios disponiveis na API."""
    url = f"{BASE_URL}/ListaDeRelatorio()"
    params = {"$format": "json"}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return pd.DataFrame(response.json().get("value", []))


# -- Execucao direta para teste rapido ----------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # Coleta completa (2019-2025)
        print("Iniciando coleta completa...")
        df = coletar_todos_dados()
        print(f"Total: {len(df)} registros")
        sys.exit(0)

    # Teste rapido: 2 trimestres x Resumo + Capital
    print("=" * 60)
    print("TESTE RAPIDO: Q4/2023 e Q4/2024")
    print("=" * 60)

    trimestres_teste = ["202312", "202412"]
    todos = []

    for tri in trimestres_teste:
        # Resumo (tipo 3 - individual)
        df = coletar_relatorio_trimestral(
            trimestre=tri, relatorio="1",
            tipo_instituicao=TIPO_INDIVIDUAL,
            bancos=BANCOS_INDIVIDUAL,
        )
        if not df.empty:
            todos.append(df)

        # Capital (tipo 1 - congl. prudencial)
        df_cap = coletar_relatorio_trimestral(
            trimestre=tri, relatorio="5",
            tipo_instituicao=TIPO_CONGL_PRUDENCIAL,
            bancos=BANCOS_CONGL_PRUDENCIAL,
        )
        if not df_cap.empty:
            todos.append(df_cap)

    if todos:
        df_all = pd.concat(todos, ignore_index=True)
        print(f"\nTotal registros: {len(df_all)}")

        # Ativo Total por banco e trimestre
        ativo = df_all[df_all["NomeColuna"].str.contains("Ativo Total", na=False)]
        if not ativo.empty:
            print("\n--- Ativo Total ---")
            pivot = ativo.pivot_table(
                values="Saldo", index="NomeBanco", columns="AnoMes", aggfunc="first"
            )
            for banco in pivot.index:
                vals = "  ".join(
                    f"{col}: R$ {pivot.loc[banco, col]:>18,.2f}"
                    for col in pivot.columns
                    if pd.notna(pivot.loc[banco, col])
                )
                print(f"  {banco:20s} {vals}")

        # Basileia por banco
        basileia = df_all[
            df_all["NomeColuna"].str.contains("Basileia", na=False)
        ]
        if not basileia.empty:
            print("\n--- Indice de Basileia ---")
            pivot_b = basileia.pivot_table(
                values="Saldo", index="NomeBanco", columns="AnoMes", aggfunc="first"
            )
            for banco in pivot_b.index:
                vals = "  ".join(
                    f"{col}: {pivot_b.loc[banco, col]:.2%}"
                    for col in pivot_b.columns
                    if pd.notna(pivot_b.loc[banco, col])
                )
                print(f"  {banco:20s} {vals}")

    print("\n" + "=" * 60)
    print("TESTE CONCLUIDO")
    print("=" * 60)
