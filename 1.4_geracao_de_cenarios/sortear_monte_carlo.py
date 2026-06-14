"""
Geração de Cenários por Simulação de Monte Carlo
Projeto de Formatura - Poli USP
Etapa: Geração de cenários de inserção de sistemas FV e BESS - IEEE 34 bus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# SEMENTE ALEATÓRIA

SEMENTE = 42
np.random.seed(SEMENTE)

# PARÂMETROS DA REDE IEEE 34 BARRAS
N_BARRAS = 32
BARRAS_SISTEMA = [800,802,806,808,810,812,814,850,816,818,820,822,824,826,828,830,854,832,858,834,860,836,862,838,842,844,846,848,852,856,888,890]
BARRAS_TRIFASICAS = [860, 840, 844, 848, 890]  # Barras com cargas trifásicas
CARGA_PICO_TOTAL_MW = 1.769

# PARÂMETROS DO SORTEIO MONTE CARLO
N_REALIZACOES = 500     # Sorteios por nível de penetração
N_HORAS = 24            # Resolução temporal

# Range de Penetração FV: de 0% a 150% em passos de 10%
PV_PENETRACAO_NIVEIS = np.round(np.arange(0.0, 1.1, 0.1), 2)

# --- BESS ---
# Potência total de BESS como fração da potência FV instalada
BESS_FRACAO_PV = 0.35
# Desvio relativo da perturbação gaussiana na alocação de BESS entre barras
BESS_DESVIO_ALOCACAO = 0.10
# Intervalo da razão de armazenamento τ (em horas)
BESS_TAU_MIN_H = 2.0
BESS_TAU_MAX_H = 4.0

# Perfil fixo de carga e descarga do BESS
# Carrega (-1) das 10h às 14h, Descarrega (+1) das 18h às 21h
PERFIL_BESS_FIXO = np.zeros(N_HORAS)
PERFIL_BESS_FIXO[10:15] = -1.0 
PERFIL_BESS_FIXO[18:22] = 1.0  

# IRRADIÂNCIA SOLAR
TIPOS_DIA = {
    "ceu_aberto": {
        "label": "Céu Aberto", "probabilidade": 0.493, "desvio_rel": 0.07, "cor": "#FF8C00",
        "perfil_medio": np.array([0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0030, 0.0440, 0.2110, 0.4546, 0.7038, 0.8908, 0.9835, 0.9823, 0.8985, 0.7267, 0.5005, 0.2370, 0.0463, 0.0021, 0.0003, 0.0000, 0.0000, 0.0000, 0.0001]),
    },
    "parcialmente_nublado": {
        "label": "Parc. Nublado", "probabilidade": 0.372, "desvio_rel": 0.12, "cor": "#4169E1",
        "perfil_medio": np.array([0.0010, 0.0009, 0.0011, 0.0011, 0.0008, 0.0042, 0.0534, 0.1890, 0.3924, 0.6090, 0.7773, 0.8749, 0.8760, 0.7534, 0.5791, 0.3876, 0.1909, 0.0517, 0.0053, 0.0008, 0.0006, 0.0004, 0.0004, 0.0004]),
    },
    "nublado": {
        "label": "Nublado", "probabilidade": 0.135, "desvio_rel": 0.17, "cor": "#708090", 
        "perfil_medio": np.array([0.0005, 0.0000, 0.0000, 0.0001, 0.0000, 0.0118, 0.0876, 0.2579, 0.4096, 0.5360, 0.6646, 0.6854, 0.6593, 0.5984, 0.4774, 0.3438, 0.1885, 0.0524, 0.0040, 0.0000, 0.0002, 0.0016, 0.0016, 0.0018]),
    },
}
_NOMES_TIPOS = list(TIPOS_DIA.keys())
_PROBS_TIPOS = np.array([TIPOS_DIA[t]["probabilidade"] for t in _NOMES_TIPOS])

# DESVIO DOS FATORES DE INCERTEZA DE CARGA (multiplicadores hora a hora)
CARGA_DESVIO_INCERTEZA = 0.10

# Desvio relativo da perturbação gaussiana sobre a parcela igualitária de cada barra.
PV_DESVIO_ALOCACAO = 0.10

# FUNÇÕES DE ALOCAÇÃO
def alocar_pv_por_barras(target_potencia_kw):
    """
    Aloca PV em todas as barras trifásicas com distribuição proporcional
    e perturbação gaussiana controlada.
    Retorna: dict {barra: potencia_kw}
    """
    if target_potencia_kw <= 0:
        return {}
 
    n_barras = len(BARRAS_TRIFASICAS)
 
    # Sorteia fatores multiplicativos para cada barra e trunca
    fatores = np.random.normal(1.0, PV_DESVIO_ALOCACAO, size=n_barras)
    fatores = np.clip(fatores, 0.5, 1.5)
 
    # Normaliza para que a soma dos pesos seja 1
    pesos = fatores / fatores.sum()
 
    # Calcula potência alocada em cada barra
    potencias = pesos * target_potencia_kw
 
    pv_alocacao = {
        barra: round(float(pot), 2)
        for barra, pot in zip(BARRAS_TRIFASICAS, potencias)
        if pot >= 1.0  # descarta parcelas residuais irrelevantes
    }
 
    return pv_alocacao

def alocar_bess_por_barras(target_pv_kw):
    """
    Aloca BESS em barras trifásicas.
    A potência total de BESS é BESS_FRACAO_PV * target_pv_kw.
    Distribui essa potência entre todas as barras trifásicas usando perturbação gaussiana.
    A capacidade de energia é calculada como: Ej = Pj * τj, onde τj ~ U(BESS_TAU_MIN_H, BESS_TAU_MAX_H).
    Retorna: dict {barra: {'potencia_kw': ..., 'capacidade_kwh': ...}}
    """
    target_bess_kw = BESS_FRACAO_PV * target_pv_kw
    
    if target_bess_kw <= 0:
        return {}
    
    n_barras = len(BARRAS_TRIFASICAS)
    
    # Sorteia fatores multiplicativos para cada barra e trunca
    fatores = np.random.normal(1.0, BESS_DESVIO_ALOCACAO, size=n_barras)
    fatores = np.clip(fatores, 0.5, 1.5)
    
    # Normaliza para que a soma dos pesos seja 1
    pesos = fatores / fatores.sum()
    
    # Calcula potência alocada em cada barra
    potencias = pesos * target_bess_kw
    
    # Calcula capacidade de energia para cada barra
    bess_alocacao = {}
    for barra, pot in zip(BARRAS_TRIFASICAS, potencias):
        if pot >= 1.0:  # descarta parcelas residuais irrelevantes
            tau_h = np.random.uniform(BESS_TAU_MIN_H, BESS_TAU_MAX_H)
            capacidade_kwh = pot * tau_h
            bess_alocacao[barra] = {
                "potencia_kw": round(float(pot), 2),
                "capacidade_kwh": round(float(capacidade_kwh), 2),
            }
    
    return bess_alocacao

def gerar_fatores_incerteza_carga():
    """
    Gera APENAS fatores de incerteza (multiplicadores) para cada barra e hora.
    O perfil horário base já está definido no arquivo .dss via LoadShape.
    """
    fatores_rede = {}
    for barra in BARRAS_SISTEMA:
        # Sorteia fatores multiplicativos hora a hora (em torno de 1.0)
        fatores = np.random.normal(1.0, CARGA_DESVIO_INCERTEZA, size=N_HORAS)
        # Limita entre 0.8 e 1.2 para manter realismo
        fatores = np.clip(fatores, 0.8, 1.2)
        
        fatores_rede[barra] = round_list(fatores, 4)
    return fatores_rede

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
        bess_alocacao = alocar_bess_por_barras(target_pv_kw)

        # 3. Fatores de Incerteza de Carga
        fatores_incerteza_barras = gerar_fatores_incerteza_carga()

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
            
            "fatores_incerteza_carga": fatores_incerteza_barras,
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
        pv_total = sum(r["pv_alocacao"].values()) if r["pv_alocacao"] else 0
        bess_total = sum(config["potencia_kw"] for config in r["bess_alocacao"].values()) if r["bess_alocacao"] else 0
        razao_bess_pv = bess_total / pv_total if pv_total > 0 else 0
        linhas_resumo.append({
            "id_realizacao": r["id_realizacao"],
            "tipo_dia": r["tipo_dia"],
            "penetracao_pct": r["penetracao_pct"],
            "pv_unidades": r["pv_unidades"],
            "pv_potencia_total_kw": pv_total,
            "bess_unidades": r["bess_unidades"],
            "bess_potencia_total_kw": round(bess_total, 2),
            "razao_bess_pv": round(razao_bess_pv, 4),
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
    # CSV 3: FATORES DE INCERTEZA DE CARGA POR BARRA (multiplicadores 24h)
    # =========================================================================
    dados_incerteza_carga = []
    for r in realizacoes:
        for barra in BARRAS_SISTEMA: 
            linha = {
                "id_realizacao": r["id_realizacao"],
                "barra": barra,
            }
            linha.update({col: val for col, val in zip(colunas_hora, r["fatores_incerteza_carga"][barra])})
            dados_incerteza_carga.append(linha)
    df_incerteza = pd.DataFrame(dados_incerteza_carga)
    df_incerteza.to_csv(os.path.join(pasta_saida, "03_fatores_incerteza_carga.csv"),
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

Características gerais:
  - Número de realizações: {n_real}
  - Total de unidades PV (todas realizações): {n_pv_total}
  - Total de unidades BESS (todas realizações): {n_bess_total}
  - Número de horas: {N_HORAS}
  - Número de barras: {N_BARRAS}

Perfis base de carga (hora a hora):
  - Estão definidos em IEEE34_2.dss via LoadShape
  - Cada carga referencia um LoadShape via atributo daily

Fatores de incerteza de carga:
  - Multiplicadores estocásticos (média 1.0, desvio ±{CARGA_DESVIO_INCERTEZA*100:.0f}%)
  - Aplicados sobre o perfil base em IEEE34_2.dss
  - Variam hora a hora e são salvos em 03_fatores_incerteza_carga.csv

Tipos de dia:
  - Céu Aberto (49.30%)
  - Parcialmente Nublado (37.20%)
  - Nublado (13.50%)

Perfil BESS fixo de carga/descarga:
  - Carregamento: horas 10-15 (valor: -1.0)
  - Descarregamento: horas 18-22 (valor: +1.0)

Arquivos gerados:
  01_resumo_configuracoes.csv ............ Resumo de cada realizacao
  02_perfis_irradiancia.csv ............. Perfis de irradancia 24h (estocástico)
  03_fatores_incerteza_carga.csv ........ Fatores multiplicativos de carga (1.0 ± {CARGA_DESVIO_INCERTEZA*100:.0f}%)
  04_unidades_pv.csv .................... Unidades PV (uma por barra)
  05_unidades_bess.csv .................. Unidades BESS (uma por barra)
  06_elementos_opendss.csv .............. Elementos para integração OpenDSS

"""
    
    with open(os.path.join(pasta_saida, "00_informacoes.txt"), "w", encoding="utf-8") as f:
        f.write(info)
    
    return df_resumo, df_irr, df_incerteza, df_pv if linhas_pv else None, df_bess if linhas_bess else None

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PASTA_BASE = os.path.join(BASE_DIR, "resultados_monte_carlo")
    
    print("\n" + "=" * 80)
    print("  GERAÇÃO DE CENÁRIOS MONTE CARLO - OPENDSS READY")
    print("=" * 80)
    print(f"  Rede: IEEE {N_BARRAS} Barras")
    print(f"  Realizar realizações por nível: {N_REALIZACOES}")
    print(f"  Número de níveis de penetração: {len(PV_PENETRACAO_NIVEIS)}")
    print(f"  Número de barras: {len(BARRAS_SISTEMA)}")
    print(f"  Tipos de dia: Pesos iguais (33.33% cada)")
    print(f"  Perfis base de carga: Definidos em IEEE34_2.dss (via LoadShape)")
    print(f"  Fatores de incerteza: Sorteados (média 1.0, desvio ±{CARGA_DESVIO_INCERTEZA*100:.0f}%)")
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
        pasta_nivel = os.path.join(PASTA_BASE, "realizacoes_sorteadas", f"pen_{pen_pct:03d}pct")
        
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
    print(f"    realizacoes_sorteadas/")
    print(f"      - pen_000pct/  [0% PV]")
    print(f"      - pen_010pct/  [10% PV]")
    print(f"      - ...")
    print(f"      - pen_200pct/  [200% PV]")
    print("\n  Cada pasta contém:")
    print(f"    - 00_informacoes.txt")
    print(f"    - 01_resumo_configuracoes.csv")
    print(f"    - 02_perfis_irradiancia.csv")
    print(f"    - 03_fatores_incerteza_carga.csv")
    print(f"    - 04_unidades_pv.csv")
    print(f"    - 05_unidades_bess.csv")
    print(f"    - 06_elementos_opendss.csv")
    print("=" * 80 + "\n")