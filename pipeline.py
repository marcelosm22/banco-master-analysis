"""
Pipeline completo: Coleta -> Extracao -> Preparacao -> Predicao

Uso:
    python -X utf8 pipeline.py          # Pipeline completo (coleta + processamento)
    python -X utf8 pipeline.py --skip-coleta  # Pula coleta (usa dados ja baixados)
"""

import sys
import logging
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def etapa_coleta():
    """Etapa 1: Coleta de dados de todas as fontes."""
    print("\n" + "=" * 60)
    print("ETAPA 1: COLETA DE DADOS")
    print("=" * 60)

    # 1a. BCB IF.data
    print("\n--- BCB IF.data (API OData) ---")
    from coleta.bcb_ifdata import coletar_todos_dados
    df_ifdata = coletar_todos_dados()
    print(f"IF.data: {len(df_ifdata)} registros coletados")

    # 1b. BCB SGS (series temporais)
    print("\n--- BCB SGS (Series Temporais) ---")
    from coleta.bcb_sgs import coletar_todas_series
    df_sgs = coletar_todas_series()
    print(f"SGS: {len(df_sgs)} registros coletados")

    # 1c. Noticias
    print("\n--- Noticias (Timeline) ---")
    from coleta.noticias_scraper import salvar_timeline
    df_noticias = salvar_timeline()
    print(f"Noticias: {len(df_noticias)} eventos documentados")


def etapa_extracao():
    """Etapa 2: Extracao e normalizacao dos dados brutos."""
    print("\n" + "=" * 60)
    print("ETAPA 2: EXTRACAO E NORMALIZACAO")
    print("=" * 60)

    from extracao.parser_ifdata import extrair_todos
    resultados = extrair_todos()

    for nome, df in resultados.items():
        print(f"  {nome}: {df.shape}")


def etapa_preparacao():
    """Etapa 3: Calculo de indicadores derivados."""
    print("\n" + "=" * 60)
    print("ETAPA 3: PREPARACAO (INDICADORES)")
    print("=" * 60)

    from preparacao.indicadores import preparar_todos
    resultados = preparar_todos()

    for nome, df in resultados.items():
        print(f"  {nome}: {df.shape}")


def etapa_predicao():
    """Etapa 4: Modelo de predicao."""
    print("\n" + "=" * 60)
    print("ETAPA 4: PREDICAO")
    print("=" * 60)

    from predicao.modelo_risco import executar_predicao
    resultado = executar_predicao()

    if resultado:
        metricas = resultado["metricas"]
        print(f"  Acuracia: {metricas['acuracia']:.2%}")
        print(f"  CV F1: {metricas['cv_f1_mean']:.2%} +/- {metricas['cv_f1_std']:.2%}")
        print(f"  Features: {resultado['features']}")
    else:
        print("  AVISO: Dados insuficientes para predicao")


def main():
    skip_coleta = "--skip-coleta" in sys.argv

    print("=" * 60)
    print("PIPELINE: Analise Banco Master")
    print("=" * 60)
    print(f"Modo: {'Processamento apenas' if skip_coleta else 'Completo (coleta + processamento)'}")

    if not skip_coleta:
        etapa_coleta()

    etapa_extracao()
    etapa_preparacao()
    etapa_predicao()

    print("\n" + "=" * 60)
    print("PIPELINE CONCLUIDO")
    print("=" * 60)
    print("\nPara visualizar os resultados:")
    print("  streamlit run app/home.py")


if __name__ == "__main__":
    main()
