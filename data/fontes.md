# Fontes de Dados

## 1. BCB IF.data - API OData (Fonte Principal)
- **URL**: https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata/
- **Swagger**: https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/swagger-ui3
- **Formato**: JSON (OData Protocol)
- **Frequencia**: Trimestral (03, 06, 09, 12)
- **Periodo coletado**: Q1/2019 a Q3/2025 (28 trimestres)
- **Licenca**: Dados publicos (Lei de Acesso a Informacao)

### Endpoints utilizados:
| Endpoint | Descricao |
|----------|-----------|
| IfDataCadastro | Cadastro de instituicoes financeiras |
| IfDataValores | Valores dos relatorios financeiros |
| ListaDeRelatorio | Lista de relatorios disponiveis |

### Relatorios coletados:
| Codigo | Nome | TipoInstituicao |
|--------|------|-----------------|
| 1 | Resumo | 3 (Individual) |
| 2 | Ativo | 3 (Individual) |
| 3 | Passivo | 3 (Individual) |
| 4 | Demonstracao de Resultado | 3 (Individual) |
| 5 | Informacoes de Capital | 1 (Congl. Prudencial) |

### Instituicoes coletadas:
| Banco | CodInst (Individual) | CodInst (Congl. Prudencial) |
|-------|---------------------|-----------------------------|
| Banco Master | 33923798 | C0080367 |
| Banco Inter | 00416968 | C0080996 |
| Banco Pine | 62144175 | C0080374 |
| Banco Original | 92894922 | C0080903 |
| Banco Daycoval | 62232889 | C0081744 |

### Nota tecnica:
A API OData nao suporta $filter em endpoints parametrizados.
Estrategia: baixar relatorio completo e filtrar localmente por CodInst.
TipoInstituicao=3 retorna dados por CNPJ individual.
TipoInstituicao=1 retorna dados por conglomerado prudencial.

## 2. BCB SGS - Series Temporais (a implementar)
- **Acesso**: Biblioteca python-bcb
- **Series**: CDI (codigo 12), Selic Meta (codigo 432)
- **Formato**: JSON
- **Frequencia**: Diaria

## 3. BancoData.com.br (a implementar)
- **URL**: https://bancodata.com.br/relatorio/master
- **Metodo**: Web scraping (BeautifulSoup)
- **Conteudo**: Indicadores consolidados, Basileia historico, ratings
- **Origem dos dados**: IF.data do BCB (reprocessados)

## 4. Noticias (a implementar)
- **Fontes**: Folha de S.Paulo, Valor Economico, Agencia Brasil, Bloomberg
- **Metodo**: Web scraping com BeautifulSoup
- **Conteudo**: Titulo, data, resumo, URL
- **Periodo**: 2024-2026
