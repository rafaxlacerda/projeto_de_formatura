"""
Unificação dos Arquivos .dat do SONDA/INPE
Projeto de Formatura - Poli USP
Autora: Ana Júlia
Etapa 1.4 - Pré-processamento dos dados de irradiância

Descrição:
    Este script lê todos os arquivos .dat exportados do portal SONDA/INPE
    (estação Cachoeira Paulista — CPA) e os unifica em um único arquivo
    SONDA_SP.csv pronto para ser processado pelo script processar_sonda.py.

    A coluna utilizada é 'glo_avg' (irradiância global horizontal média
    em W/m²), que corresponde à variável GHI necessária para o cálculo
    do índice de claridade.

Arquivo de entrada:
    Coloque todos os arquivos .dat na mesma pasta deste script.
    Exemplo: CPA_2023.dat, CPA_2024.dat, CPA_2025.dat
    (o nome exato não importa — o script lê TODOS os .dat da pasta)

Arquivo de saída:
    SONDA_SP.csv — com colunas Data;Hora;Irrad_Wm2
"""

import pandas as pd
import numpy as np
import os
import glob
import sys

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

# Construir caminho absoluto para a pasta dados_sonda (relativa ao script)
PASTA_DADOS = os.path.join(os.path.dirname(__file__), "dados_sonda")
PADRAO_ARQUIVOS = "*.dat"

ARQUIVO_SAIDA = "SONDA_SP.csv"

# Coluna de irradiância global horizontal no arquivo SONDA
COLUNA_GHI = "glo_avg"

# Resolução temporal esperada dos dados SONDA (em minutos)
# Os arquivos do SONDA CPA são tipicamente em 1 ou 10 minutos.
# O script reamostra para resolução horária automaticamente.
RESOLUCAO_SAIDA_MIN = 60   # horário

# Correção de fuso horário registrados em UTC. Para converter para o
# horário local de Cachoeira Paulista (UTC-3, fuso de Brasília), é necessário subtrair 3 horas. 
UTC_OFFSET_HORAS = -3   # UTC-3 (Brasília)

# =============================================================================
# LEITURA E UNIFICAÇÃO DOS ARQUIVOS .dat
# =============================================================================

def ler_arquivo_dat(caminho):
    
    try:
        # Lê pulando a linha de metadados da estação (linha 0)
        # e a linha de unidades (linha 2), mantendo apenas o cabeçalho (linha 1)
        df = pd.read_csv(
            caminho,
            skiprows=[0, 2],      # pula: metadados da estação e linha de unidades
            sep=",",
            low_memory=False,
            encoding="utf-8",
            encoding_errors="replace"
        )

        df.columns = df.columns.str.strip()

        # Verificar se as colunas necessárias existem
        if "timestamp" not in df.columns:
            print(f"  [AVISO] Coluna 'timestamp' não encontrada em {os.path.basename(caminho)}")
            print(f"          Colunas disponíveis: {df.columns.tolist()}")
            return None

        if COLUNA_GHI not in df.columns:
            print(f"  [AVISO] Coluna '{COLUNA_GHI}' não encontrada em {os.path.basename(caminho)}")
            print(f"          Colunas disponíveis: {df.columns.tolist()}")
            return None

        # Converter timestamp e aplicar correção de fuso horário
        # Os dados do SONDA são registrados em UTC; converte para horário
        # local (UTC-3) subtraindo o offset definido em UTC_OFFSET_HORAS.
        df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["datetime"] = df["datetime"] + pd.Timedelta(hours=UTC_OFFSET_HORAS)
        df = df.dropna(subset=["datetime"])
        df = df.set_index("datetime").sort_index()

        # Converter GHI para numérico
        df[COLUNA_GHI] = pd.to_numeric(df[COLUNA_GHI], errors="coerce")

        # Valores negativos e fisicamente impossíveis → NaN
        # (valores negativos pequenos são ruído do sensor à noite)
        df.loc[df[COLUNA_GHI] < 0, COLUNA_GHI] = 0.0
        df.loc[df[COLUNA_GHI] > 1400, COLUNA_GHI] = np.nan   # acima da constante solar

        n_validos = df[COLUNA_GHI].notna().sum()
        print(f"  OK  {os.path.basename(caminho):30s} "
              f"| {len(df):7,} registros "
              f"| {n_validos:7,} GHI válidos "
              f"| {df.index[0].date()} → {df.index[-1].date()} "
              f"(UTC{UTC_OFFSET_HORAS:+d})")

        return df[[COLUNA_GHI]].rename(columns={COLUNA_GHI: "ghi"})

    except Exception as e:
        print(f"  [ERRO] Falha ao ler {os.path.basename(caminho)}: {e}")
        return None


def reamostrar_para_horario(df):
    """
    Reamostra os dados para resolução horária, calculando a média de
    todos os registros dentro de cada hora.

    Horas sem nenhum dado válido ficam como NaN e são preenchidas
    com 0 (noite) se o índice solar for zero, ou interpoladas
    linearmente se for período diurno.
    """
    df_hora = df.resample("h").mean()

    # Para horas completamente sem dado, preenche com 0 se for noite
    # (entre 21h e 5h) e interpola para período diurno
    hora_do_dia = df_hora.index.hour
    mascara_noite = (hora_do_dia >= 21) | (hora_do_dia <= 5)
    df_hora.loc[mascara_noite & df_hora["ghi"].isna(), "ghi"] = 0.0
    df_hora["ghi"] = df_hora["ghi"].interpolate(method="linear", limit=3)
    df_hora["ghi"] = df_hora["ghi"].fillna(0.0)
    df_hora["ghi"] = df_hora["ghi"].clip(lower=0.0)

    return df_hora


if __name__ == "__main__":

    print("=" * 62)
    print("  UNIFICAÇÃO DE ARQUIVOS .dat — SONDA/INPE (CPA)")
    print("=" * 62)

    # Buscar arquivos .dat na pasta dados_sonda
    caminho_busca = os.path.join(PASTA_DADOS, PADRAO_ARQUIVOS)
    arquivos = sorted(glob.glob(caminho_busca))

    if not arquivos:
        print(f"\n[ERRO] Nenhum arquivo .dat encontrado na pasta:")
        print(f"       {os.path.abspath(PASTA_DADOS)}")
        print("\n  Certifique-se de que os arquivos .dat do SONDA estão")
        print("  na mesma pasta que este script e tente novamente.")
        sys.exit(1)

    print(f"\n  Pasta de busca: {os.path.abspath(PASTA_DADOS)}")
    print(f"  Arquivos encontrados: {len(arquivos)}")
    for a in arquivos:
        print(f"    → {os.path.basename(a)}")

    # Ler todos os arquivos
    print(f"\n  Lendo e validando arquivos...")
    print(f"  (timestamps convertidos de UTC para UTC{UTC_OFFSET_HORAS:+d} — horário de Brasília)\n")
    dfs = []
    for caminho in arquivos:
        df = ler_arquivo_dat(caminho)
        if df is not None and len(df) > 0:
            dfs.append(df)

    if not dfs:
        print("\n[ERRO] Nenhum arquivo foi lido com sucesso.")
        sys.exit(1)

    # Concatenar e remover duplicatas (sobreposição entre arquivos)
    print(f"\n  Concatenando {len(dfs)} arquivo(s)...")
    df_completo = pd.concat(dfs)
    n_antes = len(df_completo)
    df_completo = df_completo[~df_completo.index.duplicated(keep="first")]
    df_completo = df_completo.sort_index()
    n_depois = len(df_completo)

    if n_antes != n_depois:
        print(f"  Removidos {n_antes - n_depois:,} registros duplicados.")

    print(f"\n  Período unificado : {df_completo.index[0]} → {df_completo.index[-1]}")
    print(f"  Total de registros: {len(df_completo):,}")

    # Reamostrar para resolução horária
    resolucao_min = int(
        df_completo.index.to_series().diff().dt.total_seconds().median() / 60
    )
    print(f"  Resolução original: ~{resolucao_min} minutos")

    if resolucao_min != RESOLUCAO_SAIDA_MIN:
        print(f"  Reamostrando para resolução horária ({RESOLUCAO_SAIDA_MIN} min)...")
        df_completo = reamostrar_para_horario(df_completo)
        print(f"  Registros após reamostragem: {len(df_completo):,}")
    else:
        print("  Dados já estão em resolução horária.")

    # Montar o DataFrame de saída no formato esperado por processar_sonda.py
    df_saida = pd.DataFrame({
        "Data"      : df_completo.index.strftime("%d/%m/%Y"),
        "Hora"      : df_completo.index.strftime("%H:%M"),
        "Irrad_Wm2" : df_completo["ghi"].round(2).values
    })

    # Exportar na pasta dados_sonda
    caminho_saida = os.path.join(PASTA_DADOS, ARQUIVO_SAIDA)
    df_saida.to_csv(caminho_saida, index=False, sep=";", decimal=".")
    print(f"\n  Arquivo exportado : {os.path.abspath(caminho_saida)}")
    print(f"  Linhas exportadas : {len(df_saida):,}")

    # Resumo final
    n_horas_total = len(df_saida)
    n_horas_zero  = (df_saida["Irrad_Wm2"] == 0).sum()
    n_horas_dados = n_horas_total - n_horas_zero

    print(f"\n  Horas com GHI > 0 : {n_horas_dados:,} ({n_horas_dados/n_horas_total*100:.1f}%)")
    print(f"  Horas noturnas    : {n_horas_zero:,} ({n_horas_zero/n_horas_total*100:.1f}%)")

    print("\n" + "=" * 62)
    print("  PRÓXIMO PASSO:")
    print(f"  Execute: python processar_sonda.py")
    print("  O arquivo SONDA_SP.csv já está na pasta correta.")
    print("=" * 62)