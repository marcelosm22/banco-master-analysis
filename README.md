# Caso Banco Master: Análise da Saúde Financeira

**Disciplina:** Extração e Preparação de Dados
**Tipo:** Projeto em grupo
**Tema:** Análise da deterioração financeira do Banco Master (2019–2025)

---

## Resumo

Este projeto implementa um pipeline completo de dados — da coleta à predição — para analisar a **saúde financeira do Banco Master** e identificar os sinais de alerta que antecederam sua liquidação extrajudicial em novembro de 2025.

O Banco Master, sob a gestão de Daniel Vorcaro, cresceu de R$ 3 bilhões para R$ 69 bilhões em ativos entre 2019 e 2025, financiado majoritariamente por CDBs com taxas acima do mercado (até 130% do CDI). Em novembro de 2025, o Banco Central decretou sua liquidação após a Polícia Federal revelar uma fraude de R$ 12 bilhões. O caso é considerado o maior escândalo financeiro do Brasil desde a Operação Lava Jato.

O projeto compara os indicadores financeiros do Master com 4 bancos de porte similar (Inter, Pine, Original, Daycoval) para evidenciar o desvio do padrão e construir um modelo preditivo de risco.

---

## Pipeline de Dados

```
1. DOCUMENTAR ORIGEM → 2. DOWNLOAD → 3. EXTRAÇÃO → 4. PREPARAÇÃO → 5. EXIBIÇÃO → 6. PREDIÇÃO
```

### 1. Documentar Origem (`data/fontes.md`)
Registro completo das fontes de dados: URLs, formatos, frequência, licenças e método de acesso.

### 2. Download dos Dados (`src/coleta/`)
| Módulo | Fonte | Método | Dados |
|--------|-------|--------|-------|
| `bcb_ifdata.py` | BCB IF.data | API OData REST | Balanços trimestrais (Resumo, Ativo, Passivo, DRE, Capital) |
| `bcb_sgs.py` | BCB SGS | API REST | Séries temporais CDI e Selic |
| `noticias_scraper.py` | Fontes públicas | Curadoria manual com URLs verificadas | Timeline de 17 eventos-chave do caso |

**Nota técnica:** A API OData do BCB não suporta `$filter` em endpoints parametrizados. O crawler baixa o relatório completo (~10.000 registros por query) e filtra localmente por CodInst no pandas.

### 3. Extração (`src/extracao/parser_ifdata.py`)
- Converte dados brutos (formato longo da API) para DataFrames pivotados
- Normaliza períodos (AnoMes → datetime, trimestre, ano)
- Gera 5 tabelas: `resumo.csv`, `capital.csv`, `ativo.csv`, `passivo.csv`, `dre.csv`

### 4. Preparação (`src/preparacao/indicadores.py`)
Calcula indicadores derivados:

| Indicador | Fórmula | Significado |
|-----------|---------|-------------|
| Alavancagem | Ativo Total / Patrimônio Líquido | Quanto do banco é financiado por terceiros |
| Cobertura de Captações | Captações / Ativo Total | Dependência de depósitos (CDBs) |
| Crédito sobre Ativo | Carteira de Crédito / Ativo Total | Concentração em operações de crédito |
| ROE | Lucro Líquido / Patrimônio Líquido | Rentabilidade sobre capital próprio |
| Crescimento Trimestral | Variação % do Ativo (QoQ) | Velocidade de crescimento |
| Crescimento Anual | Variação % do Ativo (YoY) | Tendência de longo prazo |
| Ativo Base 100 | Indexado ao Q1/2019 | Comparação normalizada entre bancos |
| Margem Basileia | Basileia - 11% (mínimo) | Folga regulatória |
| Score de Estresse | Composto 0-100 pts | Combinação ponderada dos sinais de risco |

### 5. Exibição (`app/`)
Dashboard Streamlit com 4 páginas interativas:
- **Saúde Financeira** — Gráficos comparativos (Plotly) de todos os indicadores
- **Sinais de Alerta** — Heatmap semáforo + Score de Estresse ao longo do tempo
- **Notícias** — Timeline de eventos sobrepostos à evolução financeira
- **Predição** — Resultado do modelo + probabilidade de risco por trimestre

### 6. Predição (`src/predicao/modelo_risco.py`)
- **Modelo:** Decision Tree Classifier (max_depth=3)
- **Target:** Banco liquidado (1) vs operação normal (0)
- **Features:** 7 indicadores financeiros trimestrais
- **Resultado:** O modelo identifica Master como alto risco desde Q2/2019

**Limitações importantes (ver seção abaixo).**

---

## Dados Coletados

| Fonte | Registros | Período | Frequência |
|-------|-----------|---------|------------|
| BCB IF.data (5 relatórios) | 13.255 | Q1/2019 – Q3/2025 | Trimestral |
| BCB SGS (CDI, Selic) | 2.719 | 2019 – 2026 | Diária |
| Timeline de notícias | 17 eventos | 2019 – Mar/2026 | — |

### Bancos Analisados

| Banco | CodInst (CNPJ) | Porte | Justificativa |
|-------|----------------|-------|---------------|
| **Banco Master** | 33923798 | Médio | **Alvo da análise** — liquidado em Nov/2025 |
| Banco Inter | 00416968 | Médio-grande | Crescimento acelerado mas saudável (controle) |
| Banco Pine | 62144175 | Médio | Tradicional, conservador |
| Banco Original | 92894922 | Médio | Digital, porte similar |
| Banco Daycoval | 62232889 | Médio-grande | Conservador, referência de estabilidade |

---

## Principais Achados

### 1. Crescimento Anômalo
O Ativo Total do Banco Master cresceu **22x** em 6 anos (R$ 3,1B → R$ 69B), enquanto os pares cresceram em média 2-3x no mesmo período.

### 2. Dependência Extrema de Captações
O Master manteve captações (CDBs) acima de **84% do ativo total** durante todo o período — muito acima dos pares (68-80%). Esta é a feature mais discriminante no modelo preditivo.

### 3. Basileia no Limite
O Índice de Basileia do Master oscilou entre **10,3% e 14,1%**, frequentemente próximo do mínimo regulatório de 11%. Nota: Banco Original também teve Basileia abaixo de 11% em 6 trimestres (2020-2021) sem ser liquidado — indicando que Basileia baixa sozinha não é suficiente para prever liquidação.

### 4. Alavancagem Elevada
Alavancagem média de 16x (Ativo/PL), contra 5-11x dos pares. O Master operava com muito mais capital de terceiros em proporção ao próprio.

---

## Limitações e Ressalvas

### Sobre o Modelo Preditivo
1. **Apenas 1 caso positivo:** Com somente o Banco Master como exemplo de liquidação, o modelo aprende a separar *este banco específico* dos pares — não risco bancário em geral.
2. **100% de acurácia é esperado, não excepcional:** Com 1 caso positivo, qualquer feature que distinga Master dos demais resolve o problema perfeitamente. Não indica poder preditivo generalizável.
3. **A árvore de decisão usa 1 split principal:** `CoberturaCapt > 0.84` separa Master de todos os pares. Isso é factualmente correto (a dependência de CDBs era o motor do esquema), mas trivial em termos de ML.
4. **Para um modelo generalizável**, seria necessário um dataset com dezenas de bancos liquidados vs saudáveis ao longo de décadas (ex: dados do FGC sobre todas as liquidações desde 1990).

### Sobre os Dados
1. **IF.data é autodeclarado:** Os dados financeiros são reportados pelas próprias instituições ao BCB. No caso do Master, parte desses dados pode refletir a contabilidade fraudulenta.
2. **Q4/2025 não disponível:** O último trimestre disponível é Q3/2025. A liquidação ocorreu em Nov/2025, então o trimestre da crise em si não está nos dados.
3. **Precatórios não visíveis:** A concentração em precatórios (um dos red flags principais) não aparece diretamente nos relatórios do IF.data no nível de detalhe disponível na API.

---

## Como Executar

### Requisitos
```
Python 3.10+
pip install -r requirements.txt
```

### Pipeline Completo (com coleta ~2h)
```bash
python -X utf8 pipeline.py
```

### Apenas Processamento (usa dados já coletados)
```bash
python -X utf8 pipeline.py --skip-coleta
```

### Dashboard
```bash
python -m streamlit run app/home.py
```
Acesse `http://localhost:8501`

---

## Estrutura do Projeto

```
banco-master-analysis/
├── app/                          # Dashboard Streamlit
│   ├── home.py                   # Página inicial
│   └── pages/
│       ├── 1_saude_financeira.py # Gráficos comparativos
│       ├── 2_sinais_alerta.py    # Heatmap + Score de Estresse
│       ├── 3_noticias.py         # Timeline de eventos
│       └── 4_predicao.py         # Modelo preditivo
├── src/
│   ├── coleta/                   # Download dos dados
│   │   ├── bcb_ifdata.py         # Crawler BCB IF.data (API OData)
│   │   ├── bcb_sgs.py            # Séries temporais CDI/Selic
│   │   └── noticias_scraper.py   # Timeline de notícias
│   ├── extracao/
│   │   └── parser_ifdata.py      # Parsing e normalização
│   ├── preparacao/
│   │   └── indicadores.py        # Cálculo de indicadores derivados
│   └── predicao/
│       └── modelo_risco.py       # Decision Tree Classifier
├── data/
│   ├── bruto/                    # Dados como vieram da fonte
│   ├── preparado/                # Dados limpos e prontos
│   └── fontes.md                 # Documentação de origem
├── pipeline.py                   # Orquestrador do pipeline completo
├── requirements.txt              # Dependências Python
└── README.md                     # Este arquivo
```

---

## Referências

### Fontes de Dados
- [BCB IF.data — API OData](https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/swagger-ui3)
- [BCB IF.data — Portal Interativo](https://www3.bcb.gov.br/ifdata/)
- [BCB SGS — Sistema Gerenciador de Séries](https://www3.bcb.gov.br/sgspub/)
- [BancoData.com.br — Banco Master](https://bancodata.com.br/relatorio/master)

### Reportagens e Análises
- [Covington & Burling — Banco Master Scandal and Foreign Investors (Fev/2026)](https://www.cov.com/en/news-and-insights/insights/2026/02/brazils-banco-master-scandal-and-why-it-matters-for-foreign-investors)
- [Bloomberg — Banco Master Fraud: Daniel Vorcaro's Connections (2026)](https://www.bloomberg.com/graphics/2026-banco-master-fraud-case/)
- [Yahoo Finance — Brazil shuts down $16B bank (Nov/2025)](https://finance.yahoo.com/news/brazils-central-bank-shuts-down-191229053.html)
- [ColombiaOne — Banco Master: New Lava Jato? (Fev/2026)](https://colombiaone.com/2026/02/09/brazil-banco-master-scandal-scam-new-lava-jato/)
- [US News — Central Bankers Aiding Failed Banco Master (Mar/2026)](https://www.usnews.com/news/world/articles/2026-03-05/analysis-brazil-rocked-by-probe-of-central-bankers-aiding-failed-banco-master)
- [Agência Brasil — BC rejeita compra do Master pelo BRB (Set/2025)](https://agenciabrasil.ebc.com.br/economia/noticia/2025-09/bc-rejeita-compra-do-master-pelo-banco-de-brasilia-brb)
- [Agência Pública — BRB e Ibaneis Rocha (Fev/2026)](https://apublica.org/2026/02/brb-como-ibaneis-rocha-esta-ligado-a-crise-do-banco-master/)
- [Signature Litigation — Banco Master Collapse (2026)](https://www.signaturelitigation.com/ioannis-alexopoulos-duncan-grieve-pietro-grassi-and-nikara-rangesh-examine-the-banco-master-fallout-in-thomson-reuters-regulatory-intelligence/)

### Ferramentas e Bibliotecas
- [python-bcb — Acesso ao BCB via Python](https://pypi.org/project/python-bcb/)
- [bacen-ifdata-scraper — Referência de scraper](https://github.com/alexcamargos/bacen-ifdata-scraper)
- [Streamlit](https://streamlit.io/) | [Plotly](https://plotly.com/) | [scikit-learn](https://scikit-learn.org/)
