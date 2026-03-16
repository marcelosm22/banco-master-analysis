"""
Página 5: Documentação — Metodologia, limitações e referências.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Documentação", layout="wide")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def main():
    st.title("Documentação do Projeto")

    # --- Pipeline ---
    st.header("1. Pipeline de Dados")
    st.markdown("""
    ```
    DOCUMENTAR ORIGEM → DOWNLOAD → EXTRAÇÃO → PREPARAÇÃO → EXIBIÇÃO → PREDIÇÃO
    ```
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1.1 Fontes de Dados")
        st.markdown("""
        | Fonte | Método | Dados Coletados |
        |-------|--------|-----------------|
        | **BCB IF.data** (API OData) | Crawler REST com paginação | Balanços trimestrais: Resumo, Ativo, Passivo, DRE, Capital |
        | **BCB SGS** (API REST) | Biblioteca `requests` | Séries temporais CDI e Selic |
        | **Notícias** (fontes públicas) | Curadoria manual com URLs verificadas | 17 eventos-chave do caso (2019–2026) |
        """)

        st.info("""
        **Nota técnica:** A API OData do BCB não suporta `$filter` em endpoints
        parametrizados. O crawler baixa o relatório completo (~10.000 registros por query)
        e filtra localmente por `CodInst` no pandas.
        """)

    with col2:
        st.subheader("1.2 Volumes Coletados")

        bruto_dir = DATA_DIR / "bruto"
        prep_dir = DATA_DIR / "preparado"

        @st.cache_data
        def _carregar_consolidado():
            return pd.read_csv(bruto_dir / "ifdata_consolidado.csv", encoding="utf-8-sig")

        if (bruto_dir / "ifdata_consolidado.csv").exists():
            df = _carregar_consolidado()
            n_tri = df["AnoMes"].nunique()
            n_bancos = df["NomeBanco"].dropna().nunique()
            n_rels = df["NomeRelatorio"].nunique()
            st.metric("Registros Brutos", f"{len(df):,}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Trimestres", n_tri)
            c2.metric("Bancos", n_bancos)
            c3.metric("Relatórios", n_rels)
        else:
            st.warning("Dados brutos não encontrados")

    st.markdown("---")

    # --- Indicadores ---
    st.header("2. Indicadores Calculados")

    st.markdown("""
    | Indicador | Fórmula | O que Revela |
    |-----------|---------|--------------|
    | **Alavancagem** | Ativo Total / Patrimônio Líquido | Quanto do banco é financiado por terceiros |
    | **Cobertura de Captações** | Captações / Ativo Total | Dependência de depósitos (CDBs) — **feature mais discriminante** |
    | **Crédito sobre Ativo** | Carteira de Crédito / Ativo Total | Concentração em operações de crédito |
    | **ROE** | Lucro Líquido / Patrimônio Líquido | Rentabilidade sobre capital próprio |
    | **Crescimento Trimestral** | Variação % do Ativo (QoQ) | Velocidade de crescimento |
    | **Ativo Base 100** | Indexado ao Q1/2019 | Comparação normalizada entre bancos |
    | **Índice de Basileia** | PR / RWA (via BCB) | Solvência — mínimo regulatório de 11% |
    | **Margem Basileia** | Basileia - 11% | Folga regulatória |
    | **Score de Estresse** | Composto (0-100 pts) | Combinação ponderada dos sinais de risco |
    """)

    st.markdown("---")

    # --- Principais Achados ---
    st.header("3. Principais Achados")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔴 Crescimento Anômalo")
        st.markdown("""
        O Ativo Total do Banco Master cresceu **22x** em 6 anos
        (R\$ 3,1B → R\$ 69B), enquanto os pares cresceram em média 2-3x.
        """)

        st.subheader("🔴 Dependência Extrema de Captações")
        st.markdown("""
        O Master manteve captações (CDBs) acima de **84% do ativo total**
        durante todo o período — muito acima dos pares (68-80%). Esta é a
        feature mais discriminante no modelo preditivo.
        """)

    with col2:
        st.subheader("🔴 Basileia no Limite")
        st.markdown("""
        O Índice de Basileia do Master oscilou entre **10,3% e 14,1%**,
        frequentemente próximo do mínimo regulatório de 11%.

        *Nota: Banco Original também teve Basileia abaixo de 11% em 6
        trimestres (2020-2021) sem ser liquidado.*
        """)

        st.subheader("🔴 Alavancagem Elevada")
        st.markdown("""
        Alavancagem média de **16x** (Ativo/PL), contra 5-11x dos pares.
        O Master operava com muito mais capital de terceiros em proporção ao próprio.
        """)

    st.markdown("---")

    # --- Modelo Preditivo ---
    st.header("4. Modelo Preditivo")

    st.markdown("""
    | Aspecto | Detalhe |
    |---------|---------|
    | **Algoritmo** | Decision Tree Classifier (scikit-learn) |
    | **Profundidade máxima** | 3 |
    | **Target** | Banco liquidado (1) vs operação normal (0) |
    | **Features** | 7 indicadores financeiros trimestrais |
    | **Amostras** | ~79 (15 positivas, 64 negativas) |
    | **Validação** | StratifiedKFold 5-fold |
    | **Resultado** | 100% de acurácia e F1 |
    """)

    st.subheader("4.1 Árvore de Decisão")
    st.code("""
    |--- Captações/Ativo <= 0.84
    |   |--- classe: 0 (normal)
    |--- Captações/Ativo >  0.84
    |   |--- classe: 1 (risco)
    """, language="text")

    st.markdown("""
    **Interpretação:** O modelo identifica que a dependência extrema de captações
    (CDBs representando mais de 84% do ativo) é o sinal mais forte de risco para
    o Banco Master. Isso é factualmente correto — o esquema fraudulento dependia
    da emissão massiva de CDBs com taxas acima do mercado para financiar o crescimento.
    """)

    st.markdown("---")

    # --- Limitações ---
    st.header("5. Limitações e Ressalvas")

    st.subheader("5.1 Sobre o Modelo Preditivo")
    st.warning("""
    **Estas limitações são documentadas propositalmente — transparência é parte do projeto.**
    """)

    st.markdown("""
    1. **Apenas 1 caso positivo:** Com somente o Banco Master como exemplo de liquidação,
       o modelo aprende a separar *este banco específico* dos pares — não risco bancário em geral.

    2. **100% de acurácia é esperado, não excepcional:** Com 1 caso positivo, qualquer
       feature que distinga Master dos demais resolve o problema perfeitamente. Não indica
       poder preditivo generalizável.

    3. **A árvore usa 1 split principal:** `CoberturaCapt > 0.84` separa Master de todos
       os pares. Isso é factualmente correto, mas trivial em termos de machine learning.

    4. **Para um modelo generalizável**, seria necessário um dataset com dezenas de bancos
       liquidados vs saudáveis ao longo de décadas (ex: dados do FGC sobre todas as
       liquidações desde 1990).
    """)

    st.subheader("5.2 Sobre os Dados")
    st.markdown("""
    1. **IF.data é autodeclarado:** Os dados financeiros são reportados pelas próprias
       instituições ao BCB. No caso do Master, parte desses dados pode refletir a
       contabilidade fraudulenta.

    2. **Q4/2025 não disponível:** O último trimestre disponível é Q3/2025. A liquidação
       ocorreu em Nov/2025, então o trimestre da crise em si não está nos dados.

    3. **Precatórios não visíveis:** A concentração em precatórios (um dos red flags
       principais do caso) não aparece diretamente nos relatórios do IF.data no nível
       de detalhe disponível na API.
    """)

    st.markdown("---")

    # --- Referências ---
    st.header("6. Referências")

    st.subheader("Fontes de Dados")
    st.markdown("""
    - [BCB IF.data — API OData (Swagger)](https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/swagger-ui3)
    - [BCB IF.data — Portal Interativo](https://www3.bcb.gov.br/ifdata/)
    - [BCB SGS — Sistema Gerenciador de Séries](https://www3.bcb.gov.br/sgspub/)
    - [BancoData.com.br — Banco Master](https://bancodata.com.br/relatorio/master)
    """)

    st.subheader("Reportagens e Análises")
    st.markdown("""
    - [Covington & Burling — Banco Master Scandal and Foreign Investors (Fev/2026)](https://www.cov.com/en/news-and-insights/insights/2026/02/brazils-banco-master-scandal-and-why-it-matters-for-foreign-investors)
    - [Bloomberg — Banco Master Fraud: Daniel Vorcaro's Connections (2026)](https://www.bloomberg.com/graphics/2026-banco-master-fraud-case/)
    - [Yahoo Finance — Brazil shuts down $16B bank (Nov/2025)](https://finance.yahoo.com/news/brazils-central-bank-shuts-down-191229053.html)
    - [ColombiaOne — Banco Master: New Lava Jato? (Fev/2026)](https://colombiaone.com/2026/02/09/brazil-banco-master-scandal-scam-new-lava-jato/)
    - [US News — Central Bankers Aiding Failed Banco Master (Mar/2026)](https://www.usnews.com/news/world/articles/2026-03-05/analysis-brazil-rocked-by-probe-of-central-bankers-aiding-failed-banco-master)
    - [Agência Brasil — BC rejeita compra do Master pelo BRB (Set/2025)](https://agenciabrasil.ebc.com.br/economia/noticia/2025-09/bc-rejeita-compra-do-master-pelo-banco-de-brasilia-brb)
    - [Agência Pública — BRB e Ibaneis Rocha (Fev/2026)](https://apublica.org/2026/02/brb-como-ibaneis-rocha-esta-ligado-a-crise-do-banco-master/)
    - [Signature Litigation — Banco Master Collapse (2026)](https://www.signaturelitigation.com/ioannis-alexopoulos-duncan-grieve-pietro-grassi-and-nikara-rangesh-examine-the-banco-master-fallout-in-thomson-reuters-regulatory-intelligence/)
    """)

    st.subheader("Ferramentas e Bibliotecas")
    st.markdown("""
    - [python-bcb](https://pypi.org/project/python-bcb/) — Acesso ao BCB via Python
    - [bacen-ifdata-scraper](https://github.com/alexcamargos/bacen-ifdata-scraper) — Referência de scraper
    - [Streamlit](https://streamlit.io/) — Framework do dashboard
    - [Plotly](https://plotly.com/) — Visualizações interativas
    - [scikit-learn](https://scikit-learn.org/) — Modelo preditivo
    - [pandas](https://pandas.pydata.org/) — Manipulação de dados
    """)

    st.markdown("---")
    st.caption("Projeto acadêmico — Extração e Preparação de Dados | Dados: BCB IF.data (dados públicos)")


if __name__ == "__main__":
    main()
