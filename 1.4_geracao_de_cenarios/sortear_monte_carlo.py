"""
Geração de Cenários por Simulação de Monte Carlo
Projeto de Formatura - Poli USP
Autora: Ana Júlia
Etapa 1.4 - Geração de cenários de inserção de sistemas FV e BESS

Descrição:
    Este script realiza exclusivamente os sorteios Monte Carlo das variáveis
    de entrada do problema (perfis operacionais e nível de penetração FV/BESS).

    A topologia da rede é FIXA:
      - 6 unidades FV e 6 BESS co-localizados nas barras: 6, 10, 14, 18, 25, 32

    Variáveis sorteadas por Monte Carlo:
      1. Tipo de dia de irradiância (céu aberto, parcialmente nublado, nublado)
      2. Perfil de irradiância com perturbação gaussiana sobre o perfil médio
         do tipo de dia sorteado
      3. Nível de penetração FV total (distribuição Uniforme, 20%-100%)
      4. Razão de armazenamento de cada BESS (distribuição Uniforme, 1-4 h)
      5. Perfil de carga com perturbação gaussiana sobre o perfil médio diário

Referências:
    - Baran & Wu (1989): parâmetros nominais da rede IEEE 33 barras
    - Skartveit & Olseth (1992, Solar Energy): classificação de dias por tipo
      de céu e modelos estocásticos de irradiância
    - Atwa et al. (2010, IEEE Trans. Power Syst.): modelo estocástico de FV
    - Parada Contzen (2024, J. Energy Storage): configuração FV/BESS co-localizada
    - Wong et al. (2019, J. Energy Storage): intervalo de razão de armazenamento BESS
    - ONS (2023): curvas de carga típicas para redes de distribuição brasileiras
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =============================================================================
# SEMENTE ALEATÓRIA — FIXAR PARA REPRODUTIBILIDADE
# =============================================================================
SEMENTE = 42
np.random.seed(SEMENTE)

# =============================================================================
# PARÂMETROS DA REDE IEEE 33 BARRAS (Baran & Wu, 1989)
# =============================================================================
CARGA_PICO_TOTAL_MW   = 3.715   # MW  — carga ativa de pico total da rede
CARGA_PICO_TOTAL_MVAr = 2.300   # MVAr — carga reativa de pico total da rede
N_BARRAS = 32

# --- Topologia FIXA de instalação FV e BESS ---
# Barras selecionadas por critério técnico: extremidades dos três alimentadores
# principais da IEEE 33 barras, onde as quedas de tensão são mais severas
# no caso base sem geração distribuída.
BARRAS_FV_BESS = [6, 10, 14, 18, 25, 32]
N_UNIDADES     = len(BARRAS_FV_BESS)   # = 6

# =============================================================================
# PARÂMETROS DO SORTEIO MONTE CARLO
# =============================================================================
N_REALIZACOES = 500   # número de realizações — verificar convergência
N_HORAS       = 24    # resolução temporal: uma amostra por hora

# --- Penetração FV ---
# Intervalo: 20% a 100% da carga de pico total.
# Justificativa: abaixo de ~20% o impacto na tensão é marginal em redes de
# distribuição de média tensão; acima de 100% representa cenário extremo mas
# tecnicamente plausível para redes modernas (Parada Contzen, 2024).
PV_PENETRACAO_MIN = 0.20   # fração da carga de pico (20%)
PV_PENETRACAO_MAX = 1.00   # fração da carga de pico (100%)

# --- BESS ---
# Razão de armazenamento: capacidade (kWh) / potência nominal (kW).
# Intervalo 1-4 h é padrão na literatura (Wong et al., 2019).
BESS_RAZAO_MIN = 1.0   # horas
BESS_RAZAO_MAX = 4.0   # horas

# =============================================================================
# IRRADIÂNCIA SOLAR — TRÊS TIPOS DE DIA TÍPICO
#
# Fundamentação: Skartveit & Olseth (1992, Solar Energy) demonstraram que
# a variabilidade da irradiância pode ser modelada por três regimes distintos
# correspondentes a céu aberto, parcialmente nublado e nublado, com
# probabilidades de ocorrência associadas à climatologia local.
#
# Perfis: vetores de 24 valores em [0, 1] (índice = hora do dia).
# =============================================================================

TIPOS_DIA = {
    "ceu_aberto": {
        "label"        : "Céu Aberto",
        "probabilidade": 0.493,  # 49.3%: valor observado na estação SONDA de Cachoeira Paulista
        "desvio_rel"   : 0.07,   # ±7%: baixa variabilidade em dias claros
        "cor"          : "#FF8C00", 
        "perfil_medio" : np.array([
            0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0030,
            0.0440, 0.2110, 0.4546, 0.7038, 0.8908, 0.9835,
            0.9823, 0.8985, 0.7267, 0.5005, 0.2370, 0.0463,
            0.0021, 0.0003, 0.0000, 0.0000, 0.0000, 0.0001,
        ]),
    },
    "parcialmente_nublado": {
        "label"        : "Parc. Nublado",
        "probabilidade": 0.372,  # 37.2%: valor observado na estação SONDA de Cachoeira Paulista
        "desvio_rel"   : 0.18,   # ±18%: alta variabilidade por passagem de nuvens
        "cor"          : "#4169E1",
        "perfil_medio" : np.array([
            0.0010, 0.0009, 0.0011, 0.0011, 0.0008, 0.0042,
            0.0534, 0.1890, 0.3924, 0.6090, 0.7773, 0.8749,
            0.8760, 0.7534, 0.5791, 0.3876, 0.1909, 0.0517,
            0.0053, 0.0008, 0.0006, 0.0004, 0.0004, 0.0004,
        ]),
    },
    "nublado": {
        "label"        : "Nublado",
        "probabilidade": 0.135,  # 13.5%: valor observado na estação SONDA de Cachoeira Paulista
        "desvio_rel"   : 0.12,   # ±12%: variabilidade moderada em dias encobertos
        "cor"          : "#708090",
        "perfil_medio" : np.array([
            0.0005, 0.0000, 0.0000, 0.0001, 0.0000, 0.0118,
            0.0876, 0.2579, 0.4096, 0.5360, 0.6646, 0.6854,
            0.6593, 0.5984, 0.4774, 0.3438, 0.1885, 0.0524,
            0.0040, 0.0000, 0.0002, 0.0016, 0.0016, 0.0018,
        ]),
    },
}

_NOMES_TIPOS = list(TIPOS_DIA.keys())
_PROBS_TIPOS = np.array([TIPOS_DIA[t]["probabilidade"] for t in _NOMES_TIPOS])
assert abs(_PROBS_TIPOS.sum() - 1.0) < 1e-9

# =============================================================================
# PERFIL DE CARGA
# Fonte: adaptado de ONS (2023) — Curvas de carga típicas.
# =============================================================================
PERFIL_MEDIO_CARGA = np.array([
    0.55, 0.50, 0.48, 0.47, 0.48, 0.52,
    0.60, 0.72, 0.82, 0.88, 0.90, 0.92,
    0.95, 0.98, 1.00, 0.98, 0.97, 0.99,
    1.00, 0.95, 0.85, 0.75, 0.67, 0.60
])
CARGA_DESVIO_BARRAMENTO = 0.05   # variabilidade espacial entre barras
CARGA_DESVIO_TEMPORAL   = 0.02   # variabilidade temporal horária


# =============================================================================
# FUNÇÃO PRINCIPAL DE SORTEIO
# =============================================================================

def gerar_realizacoes(n_realizacoes, n_horas, semente=SEMENTE):
    np.random.seed(semente)
    realizacoes = []

    for i in range(n_realizacoes):

        # 1. Tipo de dia (sorteio ponderado pelas probabilidades)
        idx_tipo      = np.random.choice(len(_NOMES_TIPOS), p=_PROBS_TIPOS)
        nome_tipo_dia = _NOMES_TIPOS[idx_tipo]
        config_tipo   = TIPOS_DIA[nome_tipo_dia]

        # 2. Perfil de irradiância com perturbação gaussiana
        ruido_irr = np.random.normal(0.0, config_tipo["desvio_rel"], size=n_horas)
        perfil_irradiancia = np.clip(
            config_tipo["perfil_medio"] * (1.0 + ruido_irr), 0.0, 1.0
        )

        # 3. Nível de penetração FV e distribuição entre unidades
        penetracao_pv        = np.random.uniform(PV_PENETRACAO_MIN, PV_PENETRACAO_MAX)
        potencia_pv_total_mw = penetracao_pv * CARGA_PICO_TOTAL_MW
        pesos_pv             = np.random.dirichlet(np.ones(N_UNIDADES))
        potencias_pv_kw      = pesos_pv * potencia_pv_total_mw * 1000

        # 4. Capacidade dos BESS (co-localizados com FV)
        razoes_bess          = np.random.uniform(BESS_RAZAO_MIN, BESS_RAZAO_MAX,
                                                 size=N_UNIDADES)
        capacidades_bess_kwh = potencias_pv_kw * razoes_bess

        # 5. Perfil de carga
        fator_carga_barras   = np.clip(
            np.random.normal(1.0, CARGA_DESVIO_BARRAMENTO, size=N_BARRAS),
            0.80, 1.20
        )
        ruido_carga_temporal = np.random.normal(0.0, CARGA_DESVIO_TEMPORAL,
                                                size=n_horas)
        perfil_carga = np.clip(
            PERFIL_MEDIO_CARGA * (1.0 + ruido_carga_temporal), 0.40, 1.10
        )

        realizacao = {
            "id_realizacao"        : i + 1,
            "barras_fv_bess"       : BARRAS_FV_BESS,
            "n_unidades"           : N_UNIDADES,
            "tipo_dia"             : nome_tipo_dia,
            "perfil_irradiancia"   : perfil_irradiancia.tolist(),
            "penetracao_pv_fracao" : round(float(penetracao_pv), 4),
            "penetracao_pv_pct"    : round(float(penetracao_pv) * 100, 2),
            "potencia_pv_total_mw" : round(float(potencia_pv_total_mw), 4),
            "potencias_pv_kw"      : [round(float(p), 2) for p in potencias_pv_kw],
            "razoes_bess_h"        : [round(float(r), 3) for r in razoes_bess],
            "potencias_bess_kw"    : [round(float(p), 2) for p in potencias_pv_kw],
            "capacidades_bess_kwh" : [round(float(c), 2) for c in capacidades_bess_kwh],
            "fator_carga_barras"   : [round(float(f), 4) for f in fator_carga_barras],
            "perfil_carga"         : [round(float(v), 4) for v in perfil_carga],
        }
        realizacoes.append(realizacao)

    return realizacoes


# =============================================================================
# EXPORTAÇÃO EM CSV
# =============================================================================

def exportar_csv(realizacoes, pasta_saida="."):
    os.makedirs(pasta_saida, exist_ok=True)
    colunas_hora = [f"hora_{h:02d}" for h in range(N_HORAS)]

    # CSV 1: Resumo escalar
    linhas = []
    for r in realizacoes:
        linhas.append({
            "id"                   : r["id_realizacao"],
            "tipo_dia"             : r["tipo_dia"],
            "penetracao_pct"       : r["penetracao_pv_pct"],
            "potencia_pv_total_mw" : r["potencia_pv_total_mw"],
            "potencias_pv_kw"      : str(r["potencias_pv_kw"]),
            "razoes_bess_h"        : str(r["razoes_bess_h"]),
            "capacidades_bess_kwh" : str(r["capacidades_bess_kwh"]),
            "fator_carga_barras"   : str(r["fator_carga_barras"]),
        })
    df_resumo = pd.DataFrame(linhas)
    df_resumo.to_csv(os.path.join(pasta_saida, "mc_resumo_configuracoes.csv"),
                     index=False, sep=";", decimal=",")

    # CSV 2: Perfis de irradiância
    dados_irr = [r["perfil_irradiancia"] for r in realizacoes]
    df_irr = pd.DataFrame(dados_irr, columns=colunas_hora)
    df_irr.insert(0, "tipo_dia", [r["tipo_dia"] for r in realizacoes])
    df_irr.insert(0, "id",       [r["id_realizacao"] for r in realizacoes])
    df_irr.to_csv(os.path.join(pasta_saida, "mc_perfis_irradiancia.csv"),
                  index=False, sep=";", decimal=",")

    # CSV 3: Perfis de carga
    dados_carga = [r["perfil_carga"] for r in realizacoes]
    df_carga = pd.DataFrame(dados_carga, columns=colunas_hora)
    df_carga.insert(0, "id", [r["id_realizacao"] for r in realizacoes])
    df_carga.to_csv(os.path.join(pasta_saida, "mc_perfis_carga.csv"),
                    index=False, sep=";", decimal=",")

    # CSV 4: Configurações por unidade (formato longo)
    linhas_u = []
    for r in realizacoes:
        for j, barra in enumerate(BARRAS_FV_BESS):
            linhas_u.append({
                "id"                 : r["id_realizacao"],
                "tipo_dia"           : r["tipo_dia"],
                "unidade"            : j + 1,
                "barra"              : barra,
                "potencia_pv_kw"     : r["potencias_pv_kw"][j],
                "potencia_bess_kw"   : r["potencias_bess_kw"][j],
                "razao_bess_h"       : r["razoes_bess_h"][j],
                "capacidade_bess_kwh": r["capacidades_bess_kwh"][j],
                "fator_carga"        : r["fator_carga_barras"][j],
            })
    df_unidades = pd.DataFrame(linhas_u)
    df_unidades.to_csv(os.path.join(pasta_saida, "mc_configuracoes_por_unidade.csv"),
                       index=False, sep=";", decimal=",")

    print(f"[OK] CSVs exportados para: {os.path.abspath(pasta_saida)}")
    return df_resumo, df_irr, df_carga, df_unidades


# =============================================================================
# FIGURAS DE ANÁLISE EXPLORATÓRIA
# =============================================================================

def gerar_figuras(realizacoes, df_resumo, df_irr, df_carga, pasta_saida="."):

    horas      = np.arange(N_HORAS)
    carga_vals = df_carga.drop(columns="id").values

    # -------------------------------------------------------------------------
    # FIGURA 1: Distribuições das variáveis sorteadas
    # -------------------------------------------------------------------------
    fig1, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig1.suptitle(
        f"Análise dos Sorteios Monte Carlo — N = {N_REALIZACOES} realizações\n"
        f"IEEE 33 barras | Barras FV/BESS: {BARRAS_FV_BESS} | Semente = {SEMENTE}",
        fontsize=12, fontweight="bold"
    )
    fig1.subplots_adjust(hspace=0.42, wspace=0.35)

    # Painel (0,0): Frequência dos tipos de dia
    ax = axes[0, 0]
    nomes_ord = list(TIPOS_DIA.keys())
    labels_ord = [TIPOS_DIA[t]["label"] for t in nomes_ord]
    cores_ord  = [TIPOS_DIA[t]["cor"]   for t in nomes_ord]
    contagens  = [(df_resumo["tipo_dia"] == t).sum() for t in nomes_ord]
    ax.bar(labels_ord, contagens, color=cores_ord, edgecolor="white", alpha=0.88)
    ax.set_ylabel("Frequência")
    ax.set_title("Frequência dos Tipos de Dia")
    for k, (n, t) in enumerate(zip(contagens, nomes_ord)):
        p_esp = TIPOS_DIA[t]["probabilidade"]
        ax.text(k, n + 3, f"{n}\n({p_esp*100:.0f}% esp.)", ha="center", fontsize=8)

    # Painel (0,1): Distribuição da penetração FV
    ax = axes[0, 1]
    pen = df_resumo["penetracao_pct"].values
    ax.hist(pen, bins=25, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(pen.mean(), color="navy", linestyle="--",
               label=f"Média = {pen.mean():.1f}%")
    ax.set_xlabel("Penetração FV (%)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição da Penetração FV")
    ax.legend(fontsize=9)

    # Painel (0,2): Distribuição da potência FV total
    ax = axes[0, 2]
    pot = df_resumo["potencia_pv_total_mw"].values
    ax.hist(pot, bins=25, color="#FF9800", edgecolor="white", alpha=0.85)
    ax.axvline(pot.mean(), color="darkorange", linestyle="--",
               label=f"Média = {pot.mean():.2f} MW")
    ax.set_xlabel("Potência FV Total (MW)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição da Potência FV Total")
    ax.legend(fontsize=9)

    # Painel (1,0): Distribuição das razões BESS
    ax = axes[1, 0]
    todas_razoes = [r for real in realizacoes for r in real["razoes_bess_h"]]
    ax.hist(todas_razoes, bins=25, color="#9C27B0", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(todas_razoes), color="indigo", linestyle="--",
               label=f"Média = {np.mean(todas_razoes):.2f} h")
    ax.set_xlabel("Razão de Armazenamento BESS (h)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição das Razões BESS\n(todas as unidades)")
    ax.legend(fontsize=9)

    # Painel (1,1): Distribuição das capacidades BESS
    ax = axes[1, 1]
    todas_cap = [c for real in realizacoes for c in real["capacidades_bess_kwh"]]
    ax.hist(todas_cap, bins=30, color="#F44336", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(todas_cap), color="darkred", linestyle="--",
               label=f"Média = {np.mean(todas_cap):.0f} kWh")
    ax.set_xlabel("Capacidade BESS (kWh)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição das Capacidades BESS\n(todas as unidades)")
    ax.legend(fontsize=9)

    # Painel (1,2): Perfis de carga com bandas de percentil
    ax = axes[1, 2]
    p10c = np.percentile(carga_vals, 10, axis=0)
    p50c = np.percentile(carga_vals, 50, axis=0)
    p90c = np.percentile(carga_vals, 90, axis=0)
    ax.fill_between(horas, p10c, p90c, alpha=0.25, color="#4CAF50", label="P10–P90")
    ax.plot(horas, p50c, color="#4CAF50", linewidth=2, label="Mediana (P50)")
    ax.plot(horas, PERFIL_MEDIO_CARGA, "k--", linewidth=1.5, label="Perfil base")
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Carga norm. (p.u.)")
    ax.set_title("Perfis de Carga Sorteados")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig1.savefig(os.path.join(pasta_saida, "mc_fig1_distribuicoes.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("[OK] Figura 1 salva: mc_fig1_distribuicoes.png")

    # -------------------------------------------------------------------------
    # FIGURA 2: Perfis de irradiância por tipo de dia
    # -------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig2.suptitle(
        "Perfis de Irradiância Sorteados por Tipo de Dia — Bandas de Percentil\n"
        f"N = {N_REALIZACOES} realizações | Semente = {SEMENTE}",
        fontsize=12, fontweight="bold"
    )
    fig2.subplots_adjust(wspace=0.10)

    for idx_ax, (nome_tipo, config) in enumerate(TIPOS_DIA.items()):
        ax   = axes2[idx_ax]
        cor  = config["cor"]
        mask = df_irr["tipo_dia"] == nome_tipo
        irr_tipo = df_irr[mask].drop(columns=["id", "tipo_dia"]).values

        if len(irr_tipo) > 0:
            p10 = np.percentile(irr_tipo, 10, axis=0)
            p25 = np.percentile(irr_tipo, 25, axis=0)
            p50 = np.percentile(irr_tipo, 50, axis=0)
            p75 = np.percentile(irr_tipo, 75, axis=0)
            p90 = np.percentile(irr_tipo, 90, axis=0)
            ax.fill_between(horas, p10, p90, alpha=0.18, color=cor, label="P10–P90")
            ax.fill_between(horas, p25, p75, alpha=0.32, color=cor, label="P25–P75")
            ax.plot(horas, p50, color=cor, linewidth=2.5,  label="Mediana (P50)")

        ax.plot(horas, config["perfil_medio"], "k--", linewidth=1.5,
                label="Perfil médio base")
        ax.set_title(
            f"{config['label']}\n"
            f"(p = {config['probabilidade']*100:.0f}%,  n = {mask.sum()} real.)",
            fontsize=10
        )
        ax.set_xlabel("Hora do dia")
        if idx_ax == 0:
            ax.set_ylabel("Irradiância normalizada (p.u.)")
        ax.set_xticks(range(0, 24, 3))
        ax.set_ylim(-0.02, 1.08)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig2.savefig(os.path.join(pasta_saida, "mc_fig2_irradiancia_por_tipo.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("[OK] Figura 2 salva: mc_fig2_irradiancia_por_tipo.png")
    
    # ---CONVERGêNCIA---

def plotar_convergencia(realizacoes, df_resumo, pasta_saida="."):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    print("[INFO] Gerando análise de convergência do Monte Carlo...")

    N = len(realizacoes)
    n_vals = np.arange(1, N + 1)

    # =========================
    # Variáveis de interesse
    # =========================
    pen = df_resumo["penetracao_pct"].values
    pot = df_resumo["potencia_pv_total_mw"].values
    razoes = np.array([r for real in realizacoes for r in real["razoes_bess_h"]])

    # =========================
    # Função auxiliar
    # =========================
    def media_acumulada(valores):
        medias = []
        for i in range(1, len(valores)+1):
            medias.append(np.mean(valores[:i]))
        return np.array(medias)

    # =========================
    # Cálculo das médias acumuladas
    # =========================
    pen_media = media_acumulada(pen)
    pot_media = media_acumulada(pot)
    raz_media = media_acumulada(razoes[:N])  # aproximação alinhada com N

    # Valores finais (referência)
    pen_final = np.mean(pen)
    pot_final = np.mean(pot)
    raz_final = np.mean(razoes)

    # =========================
    # Erro relativo
    # =========================
    def erro_rel(medias, final):
        return np.abs((medias - final) / final)

    pen_erro = erro_rel(pen_media, pen_final)
    pot_erro = erro_rel(pot_media, pot_final)
    raz_erro = erro_rel(raz_media, raz_final)

    # =========================
    # FIGURA
    # =========================
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Convergência dos Sorteios Monte Carlo\nN = {N} realizações",
        fontsize=12, fontweight="bold"
    )

    # -------- Penetração FV --------
    axes[0,0].plot(n_vals, pen_media)
    axes[0,0].axhline(pen_final, linestyle="--", label="Valor final")
    axes[0,0].set_title("Média acumulada — Penetração FV (%)")
    axes[0,0].set_xlabel("N")
    axes[0,0].set_ylabel("%")
    axes[0,0].legend()
    axes[0,0].grid(True)

    axes[1,0].plot(n_vals, pen_erro)
    axes[1,0].set_yscale("log")
    axes[1,0].set_title("Erro relativo — Penetração FV")
    axes[1,0].set_xlabel("N")
    axes[1,0].set_ylabel("Erro (log)")
    axes[1,0].grid(True)

    # -------- Potência FV --------
    axes[0,1].plot(n_vals, pot_media)
    axes[0,1].axhline(pot_final, linestyle="--", label="Valor final")
    axes[0,1].set_title("Média acumulada — Potência FV (MW)")
    axes[0,1].set_xlabel("N")
    axes[0,1].legend()
    axes[0,1].grid(True)

    axes[1,1].plot(n_vals, pot_erro)
    axes[1,1].set_yscale("log")
    axes[1,1].set_title("Erro relativo — Potência FV")
    axes[1,1].set_xlabel("N")
    axes[1,1].grid(True)

    # -------- Razão BESS --------
    axes[0,2].plot(n_vals, raz_media)
    axes[0,2].axhline(raz_final, linestyle="--", label="Valor final")
    axes[0,2].set_title("Média acumulada — Razão BESS (h)")
    axes[0,2].set_xlabel("N")
    axes[0,2].legend()
    axes[0,2].grid(True)

    axes[1,2].plot(n_vals, raz_erro)
    axes[1,2].set_yscale("log")
    axes[1,2].set_title("Erro relativo — Razão BESS")
    axes[1,2].set_xlabel("N")
    axes[1,2].grid(True)

    # =========================
    # Salvar figura
    # =========================
    os.makedirs(pasta_saida, exist_ok=True)
    caminho = os.path.join(pasta_saida, "mc_convergencia.png")
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Figura de convergência salva em: {caminho}")

    # =========================
    # Diagnóstico numérico
    # =========================
    print("\n[DIAGNÓSTICO DE CONVERGÊNCIA]")
    print(f"  Penetração FV  → erro final = {pen_erro[-1]*100:.3f}%")
    print(f"  Potência FV    → erro final = {pot_erro[-1]*100:.3f}%")
    print(f"  Razão BESS     → erro final = {raz_erro[-1]*100:.3f}%")

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    PASTA_SAIDA = "resultados_monte_carlo"

    print("=" * 62)
    print("  GERAÇÃO DE CENÁRIOS — SIMULAÇÃO DE MONTE CARLO")
    print(f"  N = {N_REALIZACOES} realizações | Semente = {SEMENTE}")
    print(f"  Topologia fixa: {N_UNIDADES} FV + {N_UNIDADES} BESS")
    print(f"  Barras: {BARRAS_FV_BESS}")
    print("=" * 62)

    print("\n[1/3] Realizando sorteios...")
    realizacoes = gerar_realizacoes(N_REALIZACOES, N_HORAS, SEMENTE)
    print(f"      {len(realizacoes)} realizações geradas.")

    print("\n[2/3] Exportando resultados para CSV...")
    df_resumo, df_irr, df_carga, df_unidades = exportar_csv(realizacoes, PASTA_SAIDA)

    print("\n[3/3] Gerando figuras de análise exploratória...")
    gerar_figuras(realizacoes, df_resumo, df_irr, df_carga, PASTA_SAIDA)

    print("\n" + "=" * 62)
    print("  RESUMO ESTATÍSTICO DOS SORTEIOS")
    print("=" * 62)

    print("\n  Tipos de dia sorteados:")
    for nome, config in TIPOS_DIA.items():
        n     = (df_resumo["tipo_dia"] == nome).sum()
        p_esp = config["probabilidade"]
        print(f"    {config['label']:20s}: {n:4d} real. "
              f"({n/N_REALIZACOES*100:.1f}% obtido | {p_esp*100:.0f}% esperado)")

    pen = df_resumo["penetracao_pct"]
    print(f"\n  Penetração FV (%): "
          f"Mín={pen.min():.1f}  Máx={pen.max():.1f}  "
          f"Média={pen.mean():.1f}  DP={pen.std():.1f}")

    pot = df_resumo["potencia_pv_total_mw"]
    print(f"  Potência FV Total (MW): "
          f"Mín={pot.min():.3f}  Máx={pot.max():.3f}  "
          f"Média={pot.mean():.3f}  DP={pot.std():.3f}")

    todas_razoes = [r for real in realizacoes for r in real["razoes_bess_h"]]
    print(f"  Razão BESS (h): "
          f"Mín={min(todas_razoes):.2f}  Máx={max(todas_razoes):.2f}  "
          f"Média={np.mean(todas_razoes):.2f}  DP={np.std(todas_razoes):.2f}")

    todas_cap = [c for real in realizacoes for c in real["capacidades_bess_kwh"]]
    print(f"  Capacidade BESS (kWh): "
          f"Mín={min(todas_cap):.1f}  Máx={max(todas_cap):.1f}  "
          f"Média={np.mean(todas_cap):.1f}  DP={np.std(todas_cap):.1f}")
    
    plotar_convergencia(realizacoes, df_resumo, PASTA_SAIDA)

    print(f"\n  Arquivos gerados em: {PASTA_SAIDA}/")
    print("=" * 62)