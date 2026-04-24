"""
Processamento de Dados SONDA/INPE — Extração de Perfis Típicos de Irradiância
Projeto de Formatura - Poli USP
Autora: Ana Júlia
Etapa 1.4 - Geração de cenários de inserção de sistemas FV e BESS

Descrição:
    Este script processa o arquivo de dados de irradiância solar horizontal
    global (GHI) obtido do portal SONDA/INPE (sonda.ccst.inpe.br) para a
    estação de Cachoeira Paulista. A partir dos dados históricos horários, classifica
    cada dia em um de três tipos — céu aberto, parcialmente nublado e nublado
    — com base no índice de claridade diário (kt), e extrai o perfil médio
    horário normalizado de cada tipo. Os perfis resultantes são exportados
    para uso direto no script de geração de cenários Monte Carlo.

Fundamentação metodológica:
    O índice de claridade diário (kt) é definido como a razão entre a
    irradiância global horizontal medida na superfície (GHI) e a irradiância
    extraterrestre teórica (G0) para o mesmo dia e localização. Valores de
    kt próximos de 1 indicam atmosfera transparente (céu aberto), enquanto
    valores baixos indicam forte atenuação por nuvens. A classificação em
    três classes por limiares de kt é o método padrão da literatura
    (Skartveit & Olseth, 1992, Solar Energy; Liu & Jordan, 1960).

    Limiares adotados (Liu & Jordan, 1960; amplamente replicados):
        kt > 0.60  → Céu aberto       (Clear sky)
        0.30 < kt ≤ 0.60 → Parcialmente nublado (Partly cloudy)
        kt ≤ 0.30  → Nublado / Encoberto (Overcast)

Arquivo de entrada necessário:
    Coloque na mesma pasta deste script o arquivo CSV exportado do portal
    SONDA, com o nome exato: SONDA_SP.csv

    Formato esperado do arquivo SONDA (separador ponto-e-vírgula):
        Coluna 1: Data       — formato DD/MM/YYYY
        Coluna 2: Hora       — formato HH:MM  (ou HH:MM:SS)
        Coluna 3: Irrad_Wm2  — Irradiância Global Horizontal em W/m²

    ATENÇÃO: O portal SONDA pode exportar arquivos com cabeçalhos em
    português e separadores variados. O script tenta detectar
    automaticamente o formato. Caso ocorra erro de leitura, consulte
    a seção "Adaptação do formato" nos comentários abaixo.

Saídas geradas:
    perfis_tipicos_irradiancia.csv — perfis normalizados dos 3 tipos de dia
    sonda_fig1_classificacao_kt.png — histograma da distribuição de kt
    sonda_fig2_perfis_tipicos.png   — perfis médios dos 3 tipos com bandas
    sonda_resumo_classificacao.txt  — estatísticas da classificação

Referências:
    - Liu, B.Y.H.; Jordan, R.C. (1960). The interrelationship and
      characteristic distribution of direct, diffuse and total solar
      radiation. Solar Energy, 4(3), 1–19.
    - Skartveit, A.; Olseth, J.A. (1992). The probability density of
      diffuse irradiance. Solar Energy, 49(5), 403–410.
    - SONDA/INPE: sonda.ccst.inpe.br
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

# Caminho do arquivo de entrada na pasta dados_sonda
PASTA_DADOS = os.path.join(os.path.dirname(__file__), "dados_sonda")
ARQUIVO_ENTRADA = os.path.join(PASTA_DADOS, "SONDA_SP.csv")

# Pasta de saída
PASTA_SAIDA = "resultados_sonda"

# Localização geográfica da estação SONDA de Cachoeira Paulista
LATITUDE  = -22.69  # graus (negativo = Sul)
LONGITUDE = -45.006  # graus (negativo = Oeste)

# Limiares do índice de claridade diário (kt)
# Fonte: Liu & Jordan (1960), Skartveit & Olseth (1992)
KT_CEU_ABERTO_MIN        = 0.60   # kt > 0.60
KT_PARCIALMENTE_NUBLADO_MAX = 0.60  # 0.30 < kt <= 0.60
KT_PARCIALMENTE_NUBLADO_MIN = 0.30
KT_NUBLADO_MAX           = 0.30   # kt <= 0.30

# Constante solar (irradiância no topo da atmosfera)
G_SC = 1361.0  # W/m²

# Número mínimo de horas diurnas com irradiância > 10 W/m² para que o
# dia seja considerado válido (evita dias com dados faltantes)
MIN_HORAS_DIURNAS = 4


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def irradiancia_extraterrestre_diaria(doy, lat_graus):
    """
    Calcula a irradiância extraterrestre diária integrada (Wh/m²/dia)
    para um dado dia do ano e latitude.

    Parâmetros
    ----------
    doy : int — dia do ano (1–365)
    lat_graus : float — latitude em graus (negativo para Sul)

    Retorna
    -------
    H0 : float — irradiância extraterrestre diária em Wh/m²
    """
    lat  = math.radians(lat_graus)
    decl = math.radians(23.45 * math.sin(math.radians(360 / 365 * (doy - 81))))

    # Ângulo horário ao nascer/pôr do sol (em radianos)
    cos_ws = -math.tan(lat) * math.tan(decl)
    cos_ws = max(-1.0, min(1.0, cos_ws))
    ws     = math.acos(cos_ws)

    # Fator de correção da distância Terra-Sol
    dr = 1 + 0.033 * math.cos(math.radians(360 * doy / 365))

    H0 = (24 / math.pi) * G_SC * dr * (
        ws * math.sin(lat) * math.sin(decl)
        + math.cos(lat) * math.cos(decl) * math.sin(ws)
    )
    return max(0.0, H0)


def detectar_formato_csv(caminho):
    """
    Detecta o separador e as colunas relevantes do CSV do SONDA.
    O portal SONDA pode exportar com ; ou , como separador, e os
    nomes das colunas podem variar ligeiramente entre estações.

    Retorna (separador, nome_col_data, nome_col_hora, nome_col_ghi)
    """
    with open(caminho, "r", encoding="utf-8", errors="replace") as f:
        cabecalho = f.readline().strip()

    sep = ";" if cabecalho.count(";") >= cabecalho.count(",") else ","

    # Lê apenas o cabeçalho para identificar colunas
    df_cabecalho = pd.read_csv(caminho, sep=sep, nrows=0,
                               encoding="utf-8", encoding_errors="replace")
    colunas = [c.strip().lower() for c in df_cabecalho.columns]

    # Mapeamento flexível de nomes de coluna
    mapa_data = ["data", "date", "dt"]
    mapa_hora = ["hora", "hour", "time", "hh:mm", "horario"]
    mapa_ghi  = ["irrad_wm2", "ghi", "irradiancia", "irradiance",
                 "radiacao_global", "global", "gh", "irrad"]

    col_data = next((df_cabecalho.columns[i]
                     for i, c in enumerate(colunas)
                     if any(m in c for m in mapa_data)), None)
    col_hora = next((df_cabecalho.columns[i]
                     for i, c in enumerate(colunas)
                     if any(m in c for m in mapa_hora)), None)
    col_ghi  = next((df_cabecalho.columns[i]
                     for i, c in enumerate(colunas)
                     if any(m in c for m in mapa_ghi)), None)

    return sep, col_data, col_hora, col_ghi, df_cabecalho.columns.tolist()


def carregar_dados_sonda(caminho):
    """
    Lê o arquivo SONDA e retorna um DataFrame com índice DatetimeIndex
    e coluna 'ghi' em W/m².
    """
    print(f"[1/5] Lendo arquivo: {caminho}")

    sep, col_data, col_hora, col_ghi, todas_colunas = detectar_formato_csv(caminho)

    print(f"      Separador detectado : '{sep}'")
    print(f"      Colunas encontradas : {todas_colunas}")
    print(f"      Coluna data         : {col_data}")
    print(f"      Coluna hora         : {col_hora}")
    print(f"      Coluna GHI          : {col_ghi}")

    if col_data is None or col_hora is None or col_ghi is None:
        print("\n[ERRO] Não foi possível identificar automaticamente as colunas.")
        print("       Colunas disponíveis no arquivo:", todas_colunas)
        print("       Edite as variáveis col_data, col_hora, col_ghi ")
        print("       manualmente na função carregar_dados_sonda().")
        sys.exit(1)

    df = pd.read_csv(caminho, sep=sep, encoding="utf-8", encoding_errors="replace",
                     low_memory=False)
    df.columns = df.columns.str.strip()

    # Converter GHI para numérico (tratar vírgula decimal se necessário)
    df[col_ghi] = (df[col_ghi].astype(str)
                   .str.replace(",", ".", regex=False)
                   .str.strip())
    df[col_ghi] = pd.to_numeric(df[col_ghi], errors="coerce")

    # Montar timestamp
    df["datetime"] = pd.to_datetime(
        df[col_data].astype(str).str.strip() + " "
        + df[col_hora].astype(str).str.strip(),
        dayfirst=True, errors="coerce"
    )
    df = df.dropna(subset=["datetime", col_ghi])
    df = df.set_index("datetime").sort_index()
    df = df.rename(columns={col_ghi: "ghi"})
    df["ghi"] = df["ghi"].clip(lower=0)

    print(f"      Período dos dados   : {df.index[0]} → {df.index[-1]}")
    print(f"      Total de registros  : {len(df):,}")
    return df[["ghi"]]


# =============================================================================
# CLASSIFICAÇÃO DOS DIAS POR ÍNDICE DE CLARIDADE
# =============================================================================

def classificar_dias(df_horario):
    """
    Para cada dia com dados suficientes, calcula kt diário e classifica
    o dia em um dos três tipos.

    Retorna DataFrame diário com colunas: doy, H_medido, H0, kt, tipo_dia
    """
    print("\n[2/5] Calculando índice de claridade diário (kt)...")

    registros = []
    for data, grupo in df_horario.resample("D"):
        grupo = grupo.dropna()

        # Filtro de qualidade: exige mínimo de horas com radiação positiva
        horas_diurnas = (grupo["ghi"] > 10).sum()
        if horas_diurnas < MIN_HORAS_DIURNAS:
            continue

        doy      = data.day_of_year
        H_medido = grupo["ghi"].sum()   # integral horária ≈ Wh/m²/dia
        H0       = irradiancia_extraterrestre_diaria(doy, LATITUDE)

        if H0 < 100:   # proteção para dias de inverno em latitudes extremas
            continue

        kt = H_medido / H0

        if kt > KT_CEU_ABERTO_MIN:
            tipo = "ceu_aberto"
        elif kt > KT_PARCIALMENTE_NUBLADO_MIN:
            tipo = "parcialmente_nublado"
        else:
            tipo = "nublado"

        registros.append({
            "data"    : data,
            "doy"     : doy,
            "H_medido": round(H_medido, 1),
            "H0"      : round(H0, 1),
            "kt"      : round(kt, 4),
            "tipo_dia": tipo
        })

    df_dias = pd.DataFrame(registros).set_index("data")

    # Estatísticas da classificação
    contagem = df_dias["tipo_dia"].value_counts()
    total    = len(df_dias)
    print(f"\n      Total de dias válidos: {total}")
    for tipo in ["ceu_aberto", "parcialmente_nublado", "nublado"]:
        n    = contagem.get(tipo, 0)
        pct  = n / total * 100 if total > 0 else 0
        print(f"      {tipo:25s}: {n:4d} dias ({pct:.1f}%)")

    return df_dias


# =============================================================================
# EXTRAÇÃO DOS PERFIS HORÁRIOS MÉDIOS POR TIPO DE DIA
# =============================================================================

def extrair_perfis(df_horario, df_dias):
    """
    Para cada tipo de dia, extrai todos os perfis diários horários
    normalizados e calcula o perfil médio (vetor de 24 valores em [0,1]).

    Normalização: divide cada hora pelo máximo do dia — assim o pico
    diário é sempre 1.0, independente da estação do ano.

    Retorna dicionário {tipo: {"perfil_medio": array(24), "n_dias": int,
                                "perfil_p10": array, "perfil_p90": array}}
    """
    print("\n[3/5] Extraindo perfis horários médios por tipo de dia...")

    TIPOS = ["ceu_aberto", "parcialmente_nublado", "nublado"]
    resultados = {}

    for tipo in TIPOS:
        dias_tipo = df_dias[df_dias["tipo_dia"] == tipo].index
        perfis_normalizados = []

        for data in dias_tipo:
            # Selecionar dados horários do dia
            try:
                dia_data = df_horario.loc[
                    data.strftime("%Y-%m-%d") : data.strftime("%Y-%m-%d")
                ]["ghi"]
            except Exception:
                continue

            # Reamostrar para resolução horária exata (0–23h)
            dia_hora = pd.Series(index=range(24), dtype=float)
            for h in range(24):
                hora_ts = data + pd.Timedelta(hours=h)
                if hora_ts in df_horario.index:
                    dia_hora[h] = df_horario.loc[hora_ts, "ghi"]
                else:
                    dia_hora[h] = 0.0
            dia_hora = dia_hora.fillna(0.0)

            ghi_max = dia_hora.max()
            if ghi_max < 50:   # descarta dias com pico insignificante
                continue

            perfil_norm = (dia_hora / ghi_max).values
            perfis_normalizados.append(perfil_norm)

        if len(perfis_normalizados) == 0:
            print(f"      AVISO: nenhum perfil válido para '{tipo}'")
            resultados[tipo] = {
                "perfil_medio": np.zeros(24),
                "perfil_p10"  : np.zeros(24),
                "perfil_p90"  : np.zeros(24),
                "n_dias"      : 0
            }
            continue

        matriz = np.array(perfis_normalizados)   # shape: (n_dias, 24)
        perfil_medio = np.mean(matriz, axis=0)
        perfil_p10   = np.percentile(matriz, 10, axis=0)
        perfil_p90   = np.percentile(matriz, 90, axis=0)

        resultados[tipo] = {
            "perfil_medio": perfil_medio,
            "perfil_p10"  : perfil_p10,
            "perfil_p90"  : perfil_p90,
            "n_dias"      : len(perfis_normalizados)
        }
        print(f"      {tipo:25s}: {len(perfis_normalizados):4d} perfis extraídos")

    return resultados


# =============================================================================
# EXPORTAÇÃO DOS RESULTADOS
# =============================================================================

def exportar_resultados(perfis, df_dias, pasta_saida):
    """
    Exporta:
      1. CSV com os três perfis normalizados (prontos para o Monte Carlo)
      2. Figura 1: histograma da distribuição de kt
      3. Figura 2: perfis médios por tipo com bandas de percentil
      4. Arquivo texto com resumo e probabilidades observadas
    """
    os.makedirs(pasta_saida, exist_ok=True)

    LABELS = {
        "ceu_aberto"           : "Céu Aberto",
        "parcialmente_nublado" : "Parc. Nublado",
        "nublado"              : "Nublado"
    }
    CORES = {
        "ceu_aberto"           : "#FF8C00",
        "parcialmente_nublado" : "#4169E1",
        "nublado"              : "#708090"
    }
    horas = np.arange(24)

    # ------------------------------------------------------------------
    # CSV: perfis prontos para o Monte Carlo
    # ------------------------------------------------------------------
    print("\n[4/5] Exportando resultados...")

    linhas_csv = []
    for tipo, res in perfis.items():
        linha = {"tipo_dia": tipo}
        for h in range(24):
            linha[f"hora_{h:02d}"] = round(float(res["perfil_medio"][h]), 4)
        linhas_csv.append(linha)
    df_perfis = pd.DataFrame(linhas_csv)
    df_perfis.to_csv(os.path.join(pasta_saida, "perfis_tipicos_irradiancia.csv"),
                     index=False, sep=";", decimal=",")

    # ------------------------------------------------------------------
    # Figura 1: Histograma do índice de claridade kt
    # ------------------------------------------------------------------
    fig1, ax = plt.subplots(figsize=(9, 5))
    kt_vals = df_dias["kt"].values
    n_total = len(kt_vals)

    cores_hist = []
    for k in kt_vals:
        if k > KT_CEU_ABERTO_MIN:
            cores_hist.append(CORES["ceu_aberto"])
        elif k > KT_PARCIALMENTE_NUBLADO_MIN:
            cores_hist.append(CORES["parcialmente_nublado"])
        else:
            cores_hist.append(CORES["nublado"])

    # Histogramas coloridos por faixa
    for tipo, (kt_min, kt_max) in [
        ("ceu_aberto",           (KT_CEU_ABERTO_MIN,         1.1)),
        ("parcialmente_nublado", (KT_PARCIALMENTE_NUBLADO_MIN, KT_PARCIALMENTE_NUBLADO_MAX)),
        ("nublado",              (-0.01, KT_NUBLADO_MAX)),
    ]:
        vals = kt_vals[(kt_vals > kt_min) & (kt_vals <= kt_max)] \
               if kt_min > 0 else kt_vals[kt_vals <= kt_max]
        n    = len(vals)
        pct  = n / n_total * 100 if n_total > 0 else 0
        ax.hist(vals, bins=40, color=CORES[tipo], edgecolor="white",
                alpha=0.75, label=f"{LABELS[tipo]} ({pct:.1f}%)")

    ax.axvline(KT_PARCIALMENTE_NUBLADO_MIN, color="gray", linestyle="--",
               linewidth=1.2, label=f"kt = {KT_PARCIALMENTE_NUBLADO_MIN}")
    ax.axvline(KT_CEU_ABERTO_MIN, color="gray", linestyle=":",
               linewidth=1.2, label=f"kt = {KT_CEU_ABERTO_MIN}")
    ax.set_xlabel("Índice de Claridade Diário (kt)", fontsize=11)
    ax.set_ylabel("Número de dias", fontsize=11)
    ax.set_title(
        "Distribuição do Índice de Claridade — Dados SONDA/INPE Cachoeira Paulista\n"
        f"Total: {n_total} dias válidos",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(pasta_saida, "sonda_fig1_classificacao_kt.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figura 2: Perfis típicos com bandas de percentil
    # ------------------------------------------------------------------
    fig2, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig2.suptitle(
        "Perfis Horários Típicos de Irradiância — Dados SONDA/INPE Cachoeira Paulista",
        fontsize=12, fontweight="bold"
    )
    fig2.subplots_adjust(top=0.88, wspace=0.08)

    for idx, tipo in enumerate(["ceu_aberto", "parcialmente_nublado", "nublado"]):
        res = perfis[tipo]
        ax  = axes[idx]
        cor = CORES[tipo]

        ax.fill_between(horas, res["perfil_p10"], res["perfil_p90"],
                        alpha=0.22, color=cor, label="P10–P90")
        ax.plot(horas, res["perfil_medio"], color=cor, linewidth=2.5,
                label="Perfil médio")
        ax.set_title(
            f"{LABELS[tipo]}\n"
            f"n = {res['n_dias']} dias",
            fontsize=10
        )
        ax.set_xlabel("Hora do dia")
        if idx == 0:
            ax.set_ylabel("Irradiância Global Horizontal\nNormalizada pelo Pico Diário (p.u.)")
        ax.set_xticks(range(0, 24, 3))
        ax.set_ylim(-0.02, 1.08)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig2.savefig(os.path.join(pasta_saida, "sonda_fig2_perfis_tipicos.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ------------------------------------------------------------------
    # Arquivo texto com probabilidades e perfis numéricos
    # ------------------------------------------------------------------
    contagem = df_dias["tipo_dia"].value_counts()
    n_total  = len(df_dias)

    linhas_txt = [
        "=" * 62,
        "  RESULTADOS DO PROCESSAMENTO DOS DADOS SONDA/INPE",
        "=" * 62,
        f"  Estação: Cachoeira Paulista (lat={LATITUDE}, lon={LONGITUDE})",
        f"  Total de dias válidos: {n_total}",
        "",
        "  PROBABILIDADES OBSERVADAS (usar no Monte Carlo):",
    ]
    for tipo in ["ceu_aberto", "parcialmente_nublado", "nublado"]:
        n   = contagem.get(tipo, 0)
        pct = n / n_total if n_total > 0 else 0
        linhas_txt.append(
            f"    {tipo:25s}: {n:4d} dias  ({pct*100:.1f}%)  → p = {pct:.3f}"
        )

    linhas_txt += [
        "",
        "  PERFIS MÉDIOS NORMALIZADOS (substituir no monte_carlo_sorteio.py):",
        "  (copiar os valores abaixo para o dicionário TIPOS_DIA)",
        ""
    ]
    for tipo in ["ceu_aberto", "parcialmente_nublado", "nublado"]:
        pm = perfis[tipo]["perfil_medio"]
        linhas_txt.append(f"  {tipo}:")
        # Formatar em grupos de 6 para facilitar cópia
        grupos = [pm[i:i+6] for i in range(0, 24, 6)]
        for i, g in enumerate(grupos):
            inicio = i * 6
            fim    = min(inicio + 5, 23)
            vals   = ", ".join(f"{v:.4f}" for v in g)
            linhas_txt.append(f"    # {inicio:02d}h–{fim:02d}h: {vals},")
        linhas_txt.append("")

    linhas_txt.append("=" * 62)
    resumo_txt = "\n".join(linhas_txt)

    with open(os.path.join(pasta_saida, "sonda_resumo_classificacao.txt"), "w",
              encoding="utf-8") as f:
        f.write(resumo_txt)

    print(resumo_txt)
    print(f"\n[OK] Todos os arquivos salvos em: {os.path.abspath(pasta_saida)}/")

    return df_perfis


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    # Verifica se o arquivo de entrada existe
    if not os.path.exists(ARQUIVO_ENTRADA):
        print("=" * 62)
        print("  ARQUIVO DE ENTRADA NÃO ENCONTRADO")
        print("=" * 62)
        print(f"\n  Arquivo esperado: {os.path.abspath(ARQUIVO_ENTRADA)}")
        sys.exit(0)

    print("=" * 62)
    print("  PROCESSAMENTO DADOS SONDA — PERFIS TÍPICOS DE IRRADIÂNCIA")
    print(f"  Arquivo: {ARQUIVO_ENTRADA}")
    print("=" * 62)

    # 1. Carregar dados
    df_horario = carregar_dados_sonda(ARQUIVO_ENTRADA)

    # 2. Classificar dias por kt
    df_dias = classificar_dias(df_horario)

    # 3. Extrair perfis por tipo
    perfis = extrair_perfis(df_horario, df_dias)

    # 4. Exportar tudo
    print("\n[5/5] Salvando figuras e arquivos...")
    df_perfis = exportar_resultados(perfis, df_dias, PASTA_SAIDA)

    print("\n" + "=" * 62)
    print("  PRÓXIMO PASSO:")
    print("  Abra o arquivo 'resultados_sonda/sonda_resumo_classificacao.txt'")
    print("  e copie os perfis médios e probabilidades para o dicionário")
    print("  TIPOS_DIA no script monte_carlo_sorteio.py.")
    print("=" * 62)
