"""Merge de todas as partes coletadas no consolidado final."""
import pandas as pd
from pathlib import Path

DIR = Path(__file__).resolve().parent / "data" / "bruto"

arquivos = [
    "ifdata_consolidado.csv",   # Original (Q1/2019 - Q3/2023, rel 1-4)
    "ifdata_parte1.csv",        # Q4/2023 - Q2/2024, rel 1-4
    "ifdata_parte2.csv",        # Q1/2025 - Q2/2025, rel 1-4
    "ifdata_parte3_capital.csv", # Capital (se existir)
    "ifdata_faltantes.csv",     # Tudo que faltava (com retry)
]

dfs = []
for arq in arquivos:
    caminho = DIR / arq
    if caminho.exists():
        df = pd.read_csv(caminho, encoding="utf-8-sig")
        print(f"{arq}: {len(df)} registros")
        dfs.append(df)
    else:
        print(f"{arq}: NAO ENCONTRADO")

if dfs:
    consolidado = pd.concat(dfs, ignore_index=True)
    # Remover duplicatas (mesmo banco/trimestre/conta/relatorio)
    antes = len(consolidado)
    consolidado = consolidado.drop_duplicates(
        subset=["CodInst", "AnoMes", "NomeRelatorio", "NomeColuna"],
        keep="last",
    )
    print(f"\nDuplicatas removidas: {antes - len(consolidado)}")
    print(f"Total final: {len(consolidado)} registros")
    print(f"Trimestres: {sorted(consolidado['AnoMes'].unique())}")
    print(f"Relatorios: {sorted(consolidado['NomeRelatorio'].unique())}")
    print(f"Bancos: {sorted(consolidado['NomeBanco'].dropna().unique())}")

    consolidado.to_csv(DIR / "ifdata_consolidado.csv", index=False, encoding="utf-8-sig")
    print(f"\nSalvo ifdata_consolidado.csv")
