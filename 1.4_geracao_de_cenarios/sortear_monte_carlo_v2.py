"""
Geração de Cenários por Simulação de Monte Carlo
Projeto de Formatura - Poli USP
Etapa: Geração de cenários de inserção de sistemas FV e BESS (Atualizado IEEE 34)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =============================================================================
# SEMENTE ALEATÓRIA
# =============================================================================
SEMENTE = 42
np.random.seed(SEMENTE)

# =============================================================================
# PARÂMETROS DA REDE IEEE 34 BARRAS
# =============================================================================
N_BARRAS = 34
BARRAS_SISTEMA = np.arange(1, N_BARRAS + 1)
CARGA_PICO_TOTAL_MW = 1.769   # Típico da IEEE 34 (Ajuste se necessário)

# =============================================================================
# PARÂMETROS DO SORTEIO MONTE CARLO
# =============================================================================
N_REALIZACOES = 50      # Sorteios por nível de penetração
N_HORAS = 24            # Resolução temporal

# Range de Penetração FV: de 0% a 200% em passos de 10%
# (Devido a questões de ponto flutuante, usamos np.round)
PV_PENETRACAO_NIVEIS = np.round(np.arange(0.0, 2.1, 0.1), 2)

# --- BESS ---
# BESS inserido: 1 unidade para cada 5% de penetração
BESS_CAP_MIN_KWH = 100
BESS_CAP_MAX_KWH = 500
BESS_KW_MIN = 50
BESS_KW_MAX = 250

# Perfil fixo de carga e descarga do BESS
# Exemplo: Carrega (-1) das 10h às 14h, Descarrega (+1) das 18h às 21h
PERFIL_BESS_FIXO = np.zeros(N_HORAS)
PERFIL_BESS_FIXO[10:15] = -1.0 
PERFIL_BESS_FIXO[18:22] = 1.0  

# =============================================================================
# IRRADIÂNCIA SOLAR — PESOS IGUAIS
# =============================================================================
TIPOS_DIA = {
    "ceu_aberto": {
        "label": "Céu Aberto", "probabilidade": 1/3, "desvio_rel": 0.12, "cor": "#FF8C00",
        "perfil_medio": np.array([0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0030, 0.0440, 0.2110, 0.4546, 0.7038, 0.8908, 0.9835, 0.9823, 0.8985, 0.7267, 0.5005, 0.2370, 0.0463, 0.0021, 0.0003, 0.0000, 0.0000, 0.0000, 0.0001]),
    },
    "parcialmente_nublado": {
        "label": "Parc. Nublado", "probabilidade": 1/3, "desvio_rel": 0.12, "cor": "#4169E1",
        "perfil_medio": np.array([0.0010, 0.0009, 0.0011, 0.0011, 0.0008, 0.0042, 0.0534, 0.1890, 0.3924, 0.6090, 0.7773, 0.8749, 0.8760, 0.7534, 0.5791, 0.3876, 0.1909, 0.0517, 0.0053, 0.0008, 0.0006, 0.0004, 0.0004, 0.0004]),
    },
    "nublado": {
        "label": "Nublado", "probabilidade": 1/3, "desvio_rel": 0.12, "cor": "#708090",
        "perfil_medio": np.array([0.0005, 0.0000, 0.0000, 0.0001, 0.0000, 0.0118, 0.0876, 0.2579, 0.4096, 0.5360, 0.6646, 0.6854, 0.6593, 0.5984, 0.4774, 0.3438, 0.1885, 0.0524, 0.0040, 0.0000, 0.0002, 0.0016, 0.0016, 0.0018]),
    },
}
_NOMES_TIPOS = list(TIPOS_DIA.keys())
_PROBS_TIPOS = np.array([TIPOS_DIA[t]["probabilidade"] for t in _NOMES_TIPOS])

# =============================================================================
# PERFIS DE CARGA (Residencial, Comercial, Industrial)
# =============================================================================
# Dividindo as 34 barras uniformemente (você pode alterar depois)
BARRAS_RES = BARRAS_SISTEMA[:11]     # Barras 1 a 11
BARRAS_COM = BARRAS_SISTEMA[11:22]   # Barras 12 a 22
BARRAS_IND = BARRAS_SISTEMA[22:]     # Barras 23 a 34

PERFIS_BASE_CARGA = {
    "residencial": np.array([0.4, 0.3, 0.3, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.6, 0.8, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]),
    "comercial":   np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.6, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2]),
    "industrial":  np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8]),
}
CARGA_DESVIO_TEMPORAL = 0.05

# =============================================================================
# FUNÇÕES DE ALOCAÇÃO
# =============================================================================

def alocar_pv_por_barras(target_potencia_kw):
    """
    Aloca PV em barras diferentes até atingir o alvo de potência.
    Cada barra recebe uma única 'unidade PV' com capacidade variável.
    Retorna: dicts {barra: potencia_kw}
    """
    pv_alocacao = {}
    tamanhos_possiveis = np.arange(50, 1050, 50)  # 50 a 1000 kW
    
    # Pesos dando preferência a instalações menores
    pesos = np.linspace(10, 1, len(tamanhos_possiveis))
    pesos = pesos / np.sum(pesos)
    
    potencia_acumulada = 0.0
    barras_disponiveis = list(BARRAS_SISTEMA)
    
    while potencia_acumulada < target_potencia_kw and barras_disponiveis:
        # Sorteia tamanho da unidade PV
        tamanho = np.random.choice(tamanhos_possiveis, p=pesos)
        
        # Ajusta para não ultrapassar meta
        if potencia_acumulada + tamanho > target_potencia_kw:
            tamanho = target_potencia_kw - potencia_acumulada
            if tamanho < 10:  # Ignora restos muito pequenos
                break
        
        # Sorteia uma barra que ainda não tem PV
        if not barras_disponiveis:
            break
        barra = np.random.choice(barras_disponiveis)
        barras_disponiveis.remove(barra)
        
        pv_alocacao[barra] = tamanho
        potencia_acumulada += tamanho
    
    return pv_alocacao


def alocar_bess_por_barras(penetracao_pct, barras_com_pv=None):
    """
    Aloca BESS em barras diferentes (preferencialmente diferentes de PV).
    Uma unidade BESS por barra.
    Número total: 1 unidade a cada 5% de penetração PV
    Retorna: dict {barra: {'potencia_kw': ..., 'capacidade_kwh': ...}}
    """
    if barras_com_pv is None:
        barras_com_pv = set()
    else:
        barras_com_pv = set(barras_com_pv)
    
    n_unidades_bess = int(penetracao_pct / 5)
    if n_unidades_bess == 0:
        return {}
    
    bess_alocacao = {}
    barras_disponiveis = [b for b in BARRAS_SISTEMA if b not in barras_com_pv]
    
    # Se não houver barras suficientes sem PV, usa todas
    if len(barras_disponiveis) < n_unidades_bess:
        barras_disponiveis = list(BARRAS_SISTEMA)
    
    # Seleciona barras aleatoriamente (sem reposição)
    barras_sorteadas = np.random.choice(
        barras_disponiveis, 
        size=min(n_unidades_bess, len(barras_disponiveis)), 
        replace=False
    ).tolist()
    
    for barra in barras_sorteadas:
        potencia_kw = np.random.uniform(BESS_KW_MIN, BESS_KW_MAX)
        capacidade_kwh = np.random.uniform(BESS_CAP_MIN_KWH, BESS_CAP_MAX_KWH)
        bess_alocacao[barra] = {
            "potencia_kw": potencia_kw,
            "capacidade_kwh": capacidade_kwh,
        }
    
    return bess_alocacao

def gerar_perfis_carga():
    # Gera perfis estocásticos diários para cada barra
    perfis_rede = {}
    for barra in BARRAS_SISTEMA:
        if barra in BARRAS_RES:
            base = PERFIS_BASE_CARGA["residencial"]
            tipo = "residencial"
        elif barra in BARRAS_COM:
            base = PERFIS_BASE_CARGA["comercial"]
            tipo = "comercial"
        else:
            base = PERFIS_BASE_CARGA["industrial"]
            tipo = "industrial"
            
        ruido = np.random.normal(0.0, CARGA_DESVIO_TEMPORAL, size=N_HORAS)
        perfil_final = np.clip(base * (1.0 + ruido), 0.1, 1.2)
        perfis_rede[barra] = {
            "tipo": tipo,
            "perfil": round_list(perfil_final, 4)
        }
    return perfis_rede

def round_list(lst, decimals=2):
    return [round(float(x), decimals) for x in lst]

# =============================================================================
# FUNÇÃO PRINCIPAL DE SORTEIO
# =============================================================================

def gerar_realizacoes_por_nivel(nivel_penetracao, n_realizacoes):
    realizacoes = []
    target_pv_kw = nivel_penetracao * CARGA_PICO_TOTAL_MW * 1000
    penetracao_pct = nivel_penetracao * 100

    for i in range(n_realizacoes):
        # 1. Dia e Irradiância
        idx_tipo = np.random.choice(len(_NOMES_TIPOS), p=_PROBS_TIPOS)
        nome_tipo_dia = _NOMES_TIPOS[idx_tipo]
        config_tipo = TIPOS_DIA[nome_tipo_dia]
        ruido_irr = np.random.normal(0.0, config_tipo["desvio_rel"], size=N_HORAS)
        perfil_irradiancia = np.clip(config_tipo["perfil_medio"] * (1.0 + ruido_irr), 0.0, 1.0)

        # 2. PV e BESS (Estocásticos e Independentes)
        pv_alocacao = alocar_pv_por_barras(target_pv_kw)
        bess_alocacao = alocar_bess_por_barras(penetracao_pct, barras_com_pv=pv_alocacao.keys())

        # 3. Perfis de Carga
        perfis_carga_barras = gerar_perfis_carga()

        realizacao = {
            "id_realizacao": i + 1,
            "tipo_dia": nome_tipo_dia,
            "perfil_irradiancia": round_list(perfil_irradiancia, 4),
            "penetracao_pct": penetracao_pct,
            
            "pv_unidades": len(pv_alocacao),
            "pv_alocacao": {int(b): round(float(p), 2) for b, p in pv_alocacao.items()},
            
            "bess_unidades": len(bess_alocacao),
            "bess_alocacao": {
                int(b): {
                    "potencia_kw": round(float(d["potencia_kw"]), 2),
                    "capacidade_kwh": round(float(d["capacidade_kwh"]), 2),
                } for b, d in bess_alocacao.items()
            },
            
            "perfis_carga": perfis_carga_barras,
        }
        realizacoes.append(realizacao)

    return realizacoes

# =============================================================================
# EXPORTAÇÃO "OPENDSS READY" - ESTRUTURADA POR NÍVEL DE PENETRAÇÃO
# =============================================================================

def exportar_realizacoes_por_nivel(realizacoes, nivel_penetracao_pct, pasta_saida):
    """
    Exporta toda a informação de uma nível de penetração em estrutura de arquivos.
    Cada barra recebe NO MÁXIMO uma unidade PV e uma unidade BESS.
    """
    os.makedirs(pasta_saida, exist_ok=True)
    colunas_hora = [f"h{h:02d}" for h in range(N_HORAS)]
    
    # =========================================================================
    # CSV 1: RESUMO DE CONFIGURAÇÕES (uma linha por realizacao)
    # =========================================================================
    linhas_resumo = []
    for r in realizacoes:
        linhas_resumo.append({
            "id_realizacao": r["id_realizacao"],
            "tipo_dia": r["tipo_dia"],
            "penetracao_pct": r["penetracao_pct"],
            "pv_unidades": r["pv_unidades"],
            "pv_potencia_total_kw": sum(r["pv_alocacao"].values()) if r["pv_alocacao"] else 0,
            "bess_unidades": r["bess_unidades"],
        })
    df_resumo = pd.DataFrame(linhas_resumo)
    df_resumo.to_csv(os.path.join(pasta_saida, "01_resumo_configuracoes.csv"),
                     index=False, sep=";", decimal=",")
    
    # =========================================================================
    # CSV 2: PERFIS DE IRRADIÂNCIA (N_REALIZACOES x N_HORAS)
    # =========================================================================
    dados_irr = []
    for r in realizacoes:
        linha = {"id_realizacao": r["id_realizacao"], "tipo_dia": r["tipo_dia"]}
        linha.update({col: val for col, val in zip(colunas_hora, r["perfil_irradiancia"])})
        dados_irr.append(linha)
    df_irr = pd.DataFrame(dados_irr)
    df_irr.to_csv(os.path.join(pasta_saida, "02_perfis_irradiancia.csv"),
                  index=False, sep=";", decimal=",")
    
    # =========================================================================
    # CSV 3: PERFIS DE CARGA POR BARRA (cada barra recebe sua curva 24h)
    # =========================================================================
    dados_carga_por_barra = []
    for r in realizacoes:
        for barra in BARRAS_SISTEMA:
            linha = {
                "id_realizacao": r["id_realizacao"],
                "barra": barra,
                "tipo_carga": r["perfis_carga"][barra]["tipo"],
            }
            linha.update({col: val for col, val in zip(colunas_hora, r["perfis_carga"][barra]["perfil"])})
            dados_carga_por_barra.append(linha)
    df_carga = pd.DataFrame(dados_carga_por_barra)
    df_carga.to_csv(os.path.join(pasta_saida, "03_perfis_carga_por_barra.csv"),
                    index=False, sep=";", decimal=",")
    
    # =========================================================================
    # CSV 4: UNIDADES PV (uma linha por barra com PV)
    # =========================================================================
    linhas_pv = []
    for r in realizacoes:
        for barra, potencia_kw in r["pv_alocacao"].items():
            linhas_pv.append({
                "id_realizacao": r["id_realizacao"],
                "tipo_dia": r["tipo_dia"],
                "barra": barra,
                "tipo_carga": r["perfis_carga"][barra]["tipo"],
                "potencia_kw": potencia_kw,
            })
    if linhas_pv:
        df_pv = pd.DataFrame(linhas_pv)
        df_pv.to_csv(os.path.join(pasta_saida, "04_unidades_pv.csv"),
                     index=False, sep=";", decimal=",")
    
    # =========================================================================
    # CSV 5: UNIDADES BESS (uma linha por barra com BESS)
    # =========================================================================
    linhas_bess = []
    for r in realizacoes:
        for barra, config_bess in r["bess_alocacao"].items():
            razao_h = config_bess["capacidade_kwh"] / config_bess["potencia_kw"] if config_bess["potencia_kw"] > 0 else 0
            linhas_bess.append({
                "id_realizacao": r["id_realizacao"],
                "tipo_dia": r["tipo_dia"],
                "barra": barra,
                "tipo_carga": r["perfis_carga"][barra]["tipo"],
                "potencia_kw": config_bess["potencia_kw"],
                "capacidade_kwh": config_bess["capacidade_kwh"],
                "razao_armazenamento_h": round(razao_h, 3),
            })
    if linhas_bess:
        df_bess = pd.DataFrame(linhas_bess)
        df_bess.to_csv(os.path.join(pasta_saida, "05_unidades_bess.csv"),
                       index=False, sep=";", decimal=",")
    
    # =========================================================================
    # CSV 6: ELEMENTOS OPENDSS
    # =========================================================================
    linhas_dss = []
    for r in realizacoes:
        id_r = r["id_realizacao"]
        tipo_dia = r["tipo_dia"]
        
        # Elementos PV
        for barra, potencia_kw in r["pv_alocacao"].items():
            linhas_dss.append({
                "id_realizacao": id_r,
                "tipo_dia": tipo_dia,
                "classe": "PVSystem",
                "nome": f"PV_barra{barra}",
                "barra": barra,
                "potencia_kw": potencia_kw,
                "capacidade_kwh": "",
                "perfil": f"PV_irr_real{id_r}",
            })
        
        # Elementos BESS
        for barra, config_bess in r["bess_alocacao"].items():
            linhas_dss.append({
                "id_realizacao": id_r,
                "tipo_dia": tipo_dia,
                "classe": "Storage",
                "nome": f"BESS_barra{barra}",
                "barra": barra,
                "potencia_kw": config_bess["potencia_kw"],
                "capacidade_kwh": config_bess["capacidade_kwh"],
                "perfil": "BESS_fixo",
            })
    
    if linhas_dss:
        df_dss = pd.DataFrame(linhas_dss)
        df_dss.to_csv(os.path.join(pasta_saida, "06_elementos_opendss.csv"),
                      index=False, sep=";", decimal=",")
    
    # =========================================================================
    # TXT: ARQUIVO INFORMATIVO SOBRE O NÍVEL
    # =========================================================================
    n_real = len(realizacoes)
    n_pv_total = sum(r["pv_unidades"] for r in realizacoes)
    n_bess_total = sum(r["bess_unidades"] for r in realizacoes)
    
    info = f"""
NÍVEL DE PENETRAÇÃO: {nivel_penetracao_pct}%
{'=' * 70}

DEFINIÇÃO DE UNIDADE:
  - Uma UNIDADE PV = capacidade total de PV instalada em uma barra
  - Uma UNIDADE BESS = armazenamento total BESS instalado em uma barra
  - Cada barra recebe NO MÁXIMO 1 unidade de cada (PV e BESS independentes)

Características gerais:
  - Número de realizações: {n_real}
  - Total de unidades PV (todas realizações): {n_pv_total}
  - Total de unidades BESS (todas realizações): {n_bess_total}
  - Número de horas: {N_HORAS}
  - Número de barras: {N_BARRAS}

Distribuição de tipos de carga:
  - Barras residenciais (1-11): {list(BARRAS_RES)}
  - Barras comerciais (12-22): {list(BARRAS_COM)}
  - Barras industriais (23-34): {list(BARRAS_IND)}

Tipos de dia (pesos iguais):
  - Céu Aberto (33.33%)
  - Parcialmente Nublado (33.33%)
  - Nublado (33.33%)

Perfil BESS fixo de carga/descarga:
  - Carregamento: horas 10-15 (valor: -1.0)
  - Descarregamento: horas 18-22 (valor: +1.0)

Arquivos gerados:
  01_resumo_configuracoes.csv ............ Resumo de cada realizacao
  02_perfis_irradiancia.csv ............. Perfis de irradancia 24h
  03_perfis_carga_por_barra.csv ......... Perfis de carga por barra
  04_unidades_pv.csv .................... Unidades PV (uma por barra)
  05_unidades_bess.csv .................. Unidades BESS (uma por barra)
  06_elementos_opendss.csv .............. Elementos para integração OpenDSS
"""
    
    with open(os.path.join(pasta_saida, "00_informacoes.txt"), "w", encoding="utf-8") as f:
        f.write(info)
    
    return df_resumo, df_irr, df_carga, df_pv if linhas_pv else None, df_bess if linhas_bess else None

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PASTA_BASE = os.path.join(BASE_DIR, "resultados_monte_carlo_v2")
    
    print("\n" + "=" * 80)
    print("  GERAÇÃO DE CENÁRIOS MONTE CARLO - OPENDSS READY")
    print("=" * 80)
    print(f"  Rede: IEEE {N_BARRAS} Barras")
    print(f"  Realizar realizações por nível: {N_REALIZACOES}")
    print(f"  Número de níveis de penetração: {len(PV_PENETRACAO_NIVEIS)}")
    print(f"  Estrutura de carga:")
    print(f"    - Residencial (11 barras): 1-11")
    print(f"    - Comercial (11 barras): 12-22")
    print(f"    - Industrial (12 barras): 23-34")
    print(f"  Tipos de dia: Pesos iguais (33.33% cada)")
    print(f"\n  DEFINIÇÃO: Cada barra recebe NO MÁXIMO 1 unidade PV e 1 unidade BESS")
    print("=" * 80)
    
    # Inicializa estrutura para rastrear estatísticas globais
    stats_globais = {
        "niveis": [],
        "realizacoes_por_nivel": [],
        "unidades_pv_por_nivel": [],
        "unidades_bess_por_nivel": [],
    }
    
    for nivel in PV_PENETRACAO_NIVEIS:
        pen_pct = int(nivel * 100)
        pasta_nivel = os.path.join(PASTA_BASE, f"pen_{pen_pct:03d}pct")
        
        print(f"\n➤ Processando: {pen_pct}% de Penetração PV", end=" ")
        print(f"(alvo: {nivel * CARGA_PICO_TOTAL_MW * 1000:.1f} kW)")
        
        realizacoes = gerar_realizacoes_por_nivel(nivel, N_REALIZACOES)
        
        # Calcula estatísticas deste nível
        n_pv_total = sum(r["pv_unidades"] for r in realizacoes)
        n_bess_total = sum(r["bess_unidades"] for r in realizacoes)
        tipos_dia_count = {}
        for r in realizacoes:
            tipos_dia_count[r["tipo_dia"]] = tipos_dia_count.get(r["tipo_dia"], 0) + 1
        
        # Exporta arquivos estruturados
        df_resumo, df_irr, df_carga, df_pv, df_bess = exportar_realizacoes_por_nivel(
            realizacoes, pen_pct, pasta_nivel
        )
        
        # Registra estatísticas
        stats_globais["niveis"].append(pen_pct)
        stats_globais["realizacoes_por_nivel"].append(N_REALIZACOES)
        stats_globais["unidades_pv_por_nivel"].append(n_pv_total)
        stats_globais["unidades_bess_por_nivel"].append(n_bess_total)
        
        # Log detalhado
        print(f"  ✓ {N_REALIZACOES} realizações geradas")
        print(f"  ✓ Total de {n_pv_total} unidades PV (média: {n_pv_total/N_REALIZACOES:.1f}/real)")
        print(f"  ✓ Total de {n_bess_total} unidades BESS (média: {n_bess_total/N_REALIZACOES:.1f}/real)")
        print(f"  ✓ Distribuição tipos de dia: {tipos_dia_count}")
        print(f"  ✓ Arquivos salvos em: {os.path.abspath(pasta_nivel)}")
    
    # Log de conclusão
    print("\n" + "=" * 80)
    print("  ✅ GERAÇÃO FINALIZADA COM SUCESSO!")
    print("=" * 80)
    print(f"\n  Diretório de resultados: {os.path.abspath(PASTA_BASE)}")
    print(f"  Total de níveis processados: {len(PV_PENETRACAO_NIVEIS)}")
    print(f"  Total de realizações geradas: {len(PV_PENETRACAO_NIVEIS) * N_REALIZACOES}")
    print("\n  Estrutura de saída (por nível de penetração):")
    print(f"    - pen_000pct/  [0% PV]")
    print(f"    - pen_010pct/  [10% PV]")
    print(f"    - ...")
    print(f"    - pen_200pct/  [200% PV]")
    print("\n  Cada pasta contém:")
    print(f"    - 00_informacoes.txt")
    print(f"    - 01_resumo_configuracoes.csv")
    print(f"    - 02_perfis_irradiancia.csv")
    print(f"    - 03_perfis_carga_por_barra.csv")
    print(f"    - 04_unidades_pv.csv")
    print(f"    - 05_unidades_bess.csv")
    print(f"    - 06_elementos_opendss.csv")
    print("=" * 80 + "\n")