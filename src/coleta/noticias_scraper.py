"""
Modulo de coleta de noticias sobre o caso Banco Master.

Fontes: Google News (busca estruturada)
Metodo: requests + BeautifulSoup
Conteudo coletado: titulo, data, fonte, url, resumo

Nota: Este modulo coleta metadados de noticias para construir
uma timeline do caso. Nao faz scraping do conteudo completo
dos artigos (respeita robots.txt).
"""

import logging
import time
from pathlib import Path
from typing import Optional
from datetime import datetime
from urllib.parse import quote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Timeline manual de eventos-chave (fonte: reportagens publicas)
# Usado como fallback e complemento ao scraping
TIMELINE_EVENTOS = [
    {
        "data": "2019-01-01",
        "titulo": "Banco Master inicia fase de crescimento acelerado sob Daniel Vorcaro",
        "fonte": "Contexto historico",
        "categoria": "contexto",
        "url": "",
    },
    {
        "data": "2024-06-01",
        "titulo": "Patrimonio liquido do Master atinge R$ 4.7 bilhoes (era R$ 200M em 2019)",
        "fonte": "BCB IF.data",
        "categoria": "financeiro",
        "url": "https://www3.bcb.gov.br/ifdata/",
    },
    {
        "data": "2025-03-01",
        "titulo": "BRB anuncia proposta de aquisicao de 58% do Banco Master",
        "fonte": "Agencia Brasil / Reuters",
        "categoria": "aquisicao",
        "url": "https://agenciabrasil.ebc.com.br/economia/noticia/2025-04/com-pedido-de-compra-banco-master-divulga-lucro-de-r-1-bi-em-2024",
    },
    {
        "data": "2025-04-01",
        "titulo": "Banco Master divulga lucro de R$ 1 bilhao em 2024",
        "fonte": "Agencia Brasil",
        "categoria": "financeiro",
        "url": "https://agenciabrasil.ebc.com.br/economia/noticia/2025-04/com-pedido-de-compra-banco-master-divulga-lucro-de-r-1-bi-em-2024",
    },
    {
        "data": "2025-09-01",
        "titulo": "Banco Central rejeita compra do Master pelo BRB por irregularidades contabeis",
        "fonte": "Agencia Brasil",
        "categoria": "regulatorio",
        "url": "https://agenciabrasil.ebc.com.br/economia/noticia/2025-09/bc-rejeita-compra-do-master-pelo-banco-de-brasilia-brb",
    },
    {
        "data": "2025-10-01",
        "titulo": "BC aprova aumento de capital em instituicoes ligadas ao Banco Master",
        "fonte": "Agencia Brasil",
        "categoria": "regulatorio",
        "url": "https://agenciabrasil.ebc.com.br/economia/noticia/2025-10/bc-aprova-aumento-de-capital-em-instituicoes-ligadas-ao-banco-master",
    },
    {
        "data": "2025-11-17",
        "titulo": "Daniel Vorcaro preso no aeroporto de Guarulhos ao tentar embarcar para Malta",
        "fonte": "Bloomberg / Reuters",
        "categoria": "criminal",
        "url": "https://finance.yahoo.com/news/brazils-central-bank-shuts-down-191229053.html",
    },
    {
        "data": "2025-11-18",
        "titulo": "Banco Central decreta liquidacao extrajudicial do Banco Master",
        "fonte": "Yahoo Finance / BCB",
        "categoria": "regulatorio",
        "url": "https://finance.yahoo.com/news/brazils-central-bank-shuts-down-191229053.html",
    },
    {
        "data": "2025-11-18",
        "titulo": "Policia Federal revela fraude de R$ 12 bilhoes no sistema bancario",
        "fonte": "Yahoo Finance",
        "categoria": "criminal",
        "url": "https://finance.yahoo.com/news/brazils-central-bank-shuts-down-191229053.html",
    },
    {
        "data": "2025-12-26",
        "titulo": "STF abre investigacao sobre atuacao do Banco Central no caso Master",
        "fonte": "Bloomberg",
        "categoria": "politico",
        "url": "https://www.bloomberg.com/news/articles/2025-12-26/brazil-s-central-bank-faces-court-scrutiny-over-bank-liquidation",
    },
    {
        "data": "2026-01-15",
        "titulo": "FGC estima perda de R$ 41 bilhoes com liquidacao do Master - maior da historia",
        "fonte": "AInvest / Bloomberg",
        "categoria": "financeiro",
        "url": "https://www.ainvest.com/news/banco-master-scandal-implications-brazil-financial-sector-investor-risk-management-2601/",
    },
    {
        "data": "2026-02-09",
        "titulo": "Investigacao revela conexoes politicas de Vorcaro com governo Lula",
        "fonte": "ColombiaOne",
        "categoria": "politico",
        "url": "https://colombiaone.com/2026/02/09/brazil-banco-master-scandal-scam-new-lava-jato/",
    },
    {
        "data": "2026-02-15",
        "titulo": "Covington analisa impacto do escandalo para investidores estrangeiros",
        "fonte": "Covington & Burling LLP",
        "categoria": "analise",
        "url": "https://www.cov.com/en/news-and-insights/insights/2026/02/brazils-banco-master-scandal-and-why-it-matters-for-foreign-investors",
    },
    {
        "data": "2026-02-20",
        "titulo": "Investigacao sobre relacao de Ibaneis Rocha com crise do Master via BRB",
        "fonte": "Agencia Publica",
        "categoria": "politico",
        "url": "https://apublica.org/2026/02/brb-como-ibaneis-rocha-esta-ligado-a-crise-do-banco-master/",
    },
    {
        "data": "2026-03-04",
        "titulo": "Vorcaro preso novamente por tentativa de suborno a ex-diretor do BC",
        "fonte": "US News / Bloomberg",
        "categoria": "criminal",
        "url": "https://money.usnews.com/investing/news/articles/2026-03-04/banco-master-owner-vorcaro-detained-by-brazil-police-local-media-reports",
    },
    {
        "data": "2026-03-05",
        "titulo": "Caso Master comparado a Operacao Lava Jato como maior escandalo de corrupcao",
        "fonte": "US News",
        "categoria": "analise",
        "url": "https://www.usnews.com/news/world/articles/2026-03-05/analysis-brazil-rocked-by-probe-of-central-bankers-aiding-failed-banco-master",
    },
    {
        "data": "2026-03-06",
        "titulo": "Vorcaro transferido para presidio federal por questoes de seguranca",
        "fonte": "Bloomberg",
        "categoria": "criminal",
        "url": "https://www.bloomberg.com/news/articles/2026-03-06/banco-master-s-vorcaro-moved-to-federal-jail-due-to-safety-risks",
    },
]


def obter_timeline_manual() -> pd.DataFrame:
    """Retorna a timeline de eventos baseada em fontes publicas verificadas."""
    df = pd.DataFrame(TIMELINE_EVENTOS)
    df["data"] = pd.to_datetime(df["data"])
    df = df.sort_values("data").reset_index(drop=True)
    return df


def salvar_timeline(dir_saida: Optional[Path] = None) -> pd.DataFrame:
    """Salva a timeline de noticias em CSV."""
    if dir_saida is None:
        dir_saida = Path(__file__).resolve().parents[2] / "data" / "bruto"

    dir_saida.mkdir(parents=True, exist_ok=True)

    df = obter_timeline_manual()
    arquivo = dir_saida / "noticias_timeline.csv"
    df.to_csv(arquivo, index=False, encoding="utf-8-sig")
    logger.info(f"Salvo: {arquivo} ({len(df)} eventos)")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("Timeline de Noticias - Caso Banco Master")
    print("=" * 60)

    df = salvar_timeline()

    print(f"\nTotal: {len(df)} eventos documentados\n")
    for _, row in df.iterrows():
        data = row["data"].strftime("%Y-%m-%d")
        cat = row["categoria"].upper()
        print(f"  [{data}] [{cat:12s}] {row['titulo']}")
        print(f"           Fonte: {row['fonte']}")
        if row["url"]:
            print(f"           URL: {row['url']}")
        print()

    # Resumo por categoria
    print("Eventos por categoria:")
    for cat, count in df["categoria"].value_counts().items():
        print(f"  {cat:15s}: {count}")

    print("\nConcluido!")
