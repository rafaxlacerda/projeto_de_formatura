"""
Análise diagnóstica do Controle Secundário por Consenso (BESS)
==============================================================
Reaplica controle primário VoltVar + controle secundário por consenso
para uma única realização, expondo hora a hora:
  - Tensões antes de qualquer controle (referência MC)
  - Tensões após controle primário VoltVar
  - Tensões após primário + secundário (consenso BESS)
  - ΔP de correção do consenso por BESS
  - Violações de tensão nas três etapas

Mesmo cenário de 1.5_controle_primario/5_primario_integrado.py
(pen=90%, real=1) para comparação direta no relatório.

Reutiliza as funções de controle_secundario.py (1.4_geracao_de_cenarios)
sem duplicar a lógica de controle.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from dss._cffi_api_util import DSSException
from opendssdirect import dss

# ---------------------------------------------------------------------------
# Imports da pipeline Monte Carlo
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "1.4_geracao_de_cenarios"))

from controle_primario import (       # noqa: E402
    V_DEADBAND_HIGH, V_DEADBAND_LOW,
    V_SAT_HIGH, V_SAT_LOW,
    MAX_VOLTVAR_ITER, Q_MAX_PU,
    _calcular_q_refs, volt_var_curve,
)
from controle_secundario import (     # noqa: E402
    construir_grafo_comunicacao,
    calcular_delta_p_consenso,
    _calcular_q_para_delta_p,
    CONSENSO_MAX_ITER, CONSENSO_TOL,
    CONSENSO_ALPHA, CONSENSO_GAMMA,
)
from simulacao_opendss import (       # noqa: E402
    carregar_cargas_base,
    criar_elementos_simulacao,
    editar_load,
    extrair_tensoes_por_barra,
    ler_elementos_opendss,
    ler_fatores_incerteza_carga,
    ler_perfis_irradiancia,
    obter_multiplicadores_shapes,
    redirecionar_circuito,
)
from simulacoes_config import BESS_PERFIL, BARRAS_EXCLUIDAS_ANALISE  # noqa: E402


# ---------------------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------------------
FIG_DIR   = os.path.join(SCRIPT_DIR, "figuras")
CSV_DIR   = os.path.join(SCRIPT_DIR, "resultados")
BASE_DSS  = os.path.join(SCRIPT_DIR, "..", "IEEE34bus",
                          "IEEE34_original_with_loadshapes.dss")
MC_DIR    = os.path.join(SCRIPT_DIR, "..", "1.4_geracao_de_cenarios",
                          "resultados_monte_carlo", "realizacoes_sorteadas")
ANALISE_DIR = os.path.join(SCRIPT_DIR, "..", "1.4_geracao_de_cenarios",
                            "resultados_monte_carlo", "analise_opendss")

PEN_PCT        = 90
ID_REALIZACAO  = 1
TOTAL_HOURS    = 24
MODO_SECUNDARIO = "sempre_ativo"   # "sempre_ativo" ou "condicional"

# Constantes de estilo — iguais a 1.4_geracao_de_cenarios/graficos_opendss.py
FONTSIZE_TITULO  = 16
FONTSIZE_EIXO    = 13
FONTSIZE_TICK    = 11
FONTSIZE_LEGENDA = 11


# ---------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ---------------------------------------------------------------------------

def preparar_cenario_load(pen_pct: int, id_realizacao: int) -> tuple:
    """
    Lê CSVs da realização Monte Carlo, cria Load elements no circuito DSS
    e retorna (pv_real_df, bess_real_df, fatores_irradiancia, fatores_carga).
    """
    pasta = os.path.join(MC_DIR, f"pen_{pen_pct:03d}pct")

    perfis_irr   = ler_perfis_irradiancia(pasta)
    irr_row      = perfis_irr[perfis_irr["id_realizacao"] == id_realizacao].iloc[0]
    fatores_irr  = [float(irr_row[f"h{h:02d}"]) for h in range(24)]

    fat_df   = ler_fatores_incerteza_carga(pasta)
    fat_real = fat_df[fat_df["id_realizacao"] == id_realizacao]
    fatores_carga = {
        int(linha["barra"]): [float(linha[f"h{h:02d}"]) for h in range(24)]
        for _, linha in fat_real.iterrows()
    }

    elementos = ler_elementos_opendss(pasta)
    pv_real   = elementos[elementos["id_realizacao"] == id_realizacao]
    bess_real = pv_real[pv_real["classe"] == "Storage"]

    criar_elementos_simulacao(pv_real, bess_real, id_realizacao)

    n_pv   = len(pv_real[pv_real["classe"] == "PVSystem"])
    n_bess = len(bess_real)
    print(f"[Cenário] pen={pen_pct}% | real={id_realizacao} | "
          f"PV={n_pv} | BESS={n_bess} | cargas com fator={len(fatores_carga)}")
    return pv_real, bess_real, fatores_irr, fatores_carga


def count_voltage_violations(tensoes: dict) -> tuple:
    """Retorna (n_violacoes, {bus: info}) para barras fora de [0.95, 1.05] p.u."""
    violations = {}
    for barra, fases in tensoes.items():
        v_min = min(fases)
        v_max = max(fases)
        under = v_min < 0.95
        over  = v_max > 1.05001
        if under or over:
            violations[barra] = {
                "v_min": round(v_min, 6),
                "v_max": round(v_max, 6),
                "under": under,
                "over":  over,
            }
    return len(violations), violations


def carregar_violacoes_mc(pen_pct: int, id_realizacao: int) -> list:
    """
    Lê tensoes_opendss_completas.csv do MC (sem controle) e conta,
    por hora, quantas barras de carga têm violação em [0.95, 1.05].
    Retorna lista de 24 inteiros.
    """
    csv_path = os.path.join(ANALISE_DIR, f"pen_{pen_pct:03d}pct",
                             "tensoes_opendss_completas.csv")
    df = pd.read_csv(csv_path, sep=";", decimal=",")
    df = df[df["id_realizacao"] == id_realizacao].copy()
    df = df[~df["barra"].astype(str).str.lower().isin(BARRAS_EXCLUIDAS_ANALISE)]

    fase_cols  = ["tensao_fase_1_pu", "tensao_fase_2_pu", "tensao_fase_3_pu"]
    violacoes  = []
    for hora in range(TOTAL_HOURS):
        df_h = df[df["hora"] == hora]
        n = 0
        for _, row in df_h.iterrows():
            n_fases = int(row["n_fases"])
            fases   = [row[c] for c in fase_cols[:n_fases]]
            if any(v < 0.95 or v > 1.05001 for v in fases):
                n += 1
        violacoes.append(n)
    return violacoes


# ---------------------------------------------------------------------------
# FUNÇÕES DE VISUALIZAÇÃO
# ---------------------------------------------------------------------------

def plot_violations_3way(df_hora: pd.DataFrame, n_viols_mc: list):
    """
    Barras triplas por hora: sem controle (MC) / com primário / com primário+secundário.
    Setas indicam melhora (verde) ou piora (vermelho) em relação à etapa anterior.
    """
    horas     = df_hora["hora"].tolist()
    n_mc      = n_viols_mc
    n_prim    = df_hora["n_viols_primario"].tolist()
    n_sec     = df_hora["n_viols_secundario"].tolist()
    x         = np.arange(len(horas))
    width     = 0.25

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width,     n_mc,   width, label="Sem controle (MC)",         color="steelblue",  edgecolor="white")
    ax.bar(x,             n_prim, width, label="Com controle primário",      color="darkorange", edgecolor="white")
    ax.bar(x + width,     n_sec,  width, label="Com primário + secundário",  color="#2ca02c",    edgecolor="white")

    arrow_props = dict(arrowstyle="-|>", lw=2.0, mutation_scale=12)
    for xi, pre, pos in zip(x, n_prim, n_sec):
        if pos == pre:
            continue
        color = "green" if pos < pre else "red"
        ax.annotate("",
                    xy=(xi + width, pos), xytext=(xi + width, pre),
                    arrowprops={**arrow_props, "color": color})

    ax.set_xticks(x)
    ax.set_xticklabels(horas, fontsize=FONTSIZE_TICK)
    ax.tick_params(axis="y", labelsize=FONTSIZE_TICK)
    ax.set_xlabel("Hora do dia", fontsize=FONTSIZE_EIXO)
    ax.set_ylabel("Barras com violação de tensão", fontsize=FONTSIZE_EIXO)
    ax.set_title(
        f"Violações de tensão — sem controle vs primário vs primário+secundário\n"
        f"penetração={PEN_PCT}% / realização={ID_REALIZACAO}  |  "
        f"Modo secundário: {MODO_SECUNDARIO}  |  Limites: V < 0,95 p.u. ou V > 1,05 p.u.",
        fontsize=FONTSIZE_TITULO,
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    y_top = max(max(n_mc, default=0), max(n_prim, default=0), max(n_sec, default=0)) + 2
    ax.set_ylim(0, max(y_top, 2))
    ax.legend(fontsize=FONTSIZE_LEGENDA, loc="upper right", framealpha=0.9)

    path = os.path.join(FIG_DIR, "fig_violations_3way.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Violações 3-vias salvo em: {path}")


def plot_bess_delta_p(df_bess: pd.DataFrame):
    """
    Heatmap de ΔP (kW) do consenso por BESS × hora.
    Azul = mais descarga / menos carga; vermelho = menos descarga / mais carga.
    Branco = sem correção (BESS ocioso ou consenso não ativado).
    """
    pivot      = df_bess.pivot_table(index="der", columns="hora",
                                     values="delta_p_kw", aggfunc="first")
    der_labels = pivot.index.tolist()
    hours      = pivot.columns.tolist()
    vals       = pivot.values.astype(float)
    dp_abs     = np.nanmax(np.abs(vals)) if np.any(np.abs(vals) > 1e-6) else 1.0

    fig, ax = plt.subplots(
        figsize=(max(10, len(hours) * 0.6), max(4, len(der_labels) * 0.7))
    )
    im = ax.imshow(vals, cmap="RdBu", vmin=-dp_abs, vmax=dp_abs, aspect="auto")

    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels(hours, fontsize=FONTSIZE_TICK)
    ax.set_yticks(range(len(der_labels)))
    ax.set_yticklabels(der_labels, fontsize=FONTSIZE_TICK)
    ax.set_xlabel("Hora do dia", fontsize=FONTSIZE_EIXO)
    ax.set_ylabel("BESS", fontsize=FONTSIZE_EIXO)
    ax.set_title(
        f"ΔP do consenso por BESS — pen={PEN_PCT}% / real={ID_REALIZACAO}\n"
        f"Azul = mais injeção (↑P)  |  Vermelho = menos injeção (↓P)  |  "
        f"Branco = BESS ocioso ou sem correção",
        fontsize=FONTSIZE_TITULO,
    )
    for i in range(len(der_labels)):
        for j, h in enumerate(hours):
            val = vals[i, j]
            if not np.isnan(val) and abs(val) > 1e-6:
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                        fontsize=8, color="black")

    fig.colorbar(im, ax=ax, label="ΔP (kW)  (+ mais injeção / − menos injeção)")
    path = os.path.join(FIG_DIR, "fig_bess_delta_p_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Heatmap ΔP consenso salvo em: {path}")


def plot_bess_kw_profile(df_bess: pd.DataFrame):
    """
    Perfil de potência ativa dos BESS: referência (BESS_PERFIL) vs pós-secundário.
    Negativo = carga (absorção da rede); positivo = descarga (injeção).
    """
    bess_names = sorted(df_bess["der"].unique())
    n = len(bess_names)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, bess in zip(axes, bess_names):
        sub  = df_bess[df_bess["der"] == bess].sort_values("hora")
        horas = sub["hora"].tolist()
        kw_ref = sub["kw_ref_kw"].tolist()
        kw_pos = sub["kw_pos_secundario_kw"].tolist()

        ax.plot(horas, kw_ref, color="steelblue", linewidth=2,
                marker="o", markersize=4, linestyle="--", label="Referência (perfil fixo)")
        ax.plot(horas, kw_pos, color="#2ca02c", linewidth=2,
                marker="s", markersize=4, label="Pós-secundário (após consenso)")
        ax.fill_between(horas, kw_ref, kw_pos, alpha=0.20, color="#2ca02c")

        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_ylabel("P BESS (kW)", fontsize=FONTSIZE_EIXO)
        ax.set_title(f"BESS: {bess}", fontsize=FONTSIZE_EIXO)
        ax.tick_params(axis="both", labelsize=FONTSIZE_TICK)
        ax.legend(fontsize=FONTSIZE_LEGENDA)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Hora do dia", fontsize=FONTSIZE_EIXO)
    axes[0].set_title(
        f"Perfil de potência ativa dos BESS — pen={PEN_PCT}% / real={ID_REALIZACAO}\n"
        f"Negativo = carga (absorção)  |  Positivo = descarga (injeção)\n"
        + axes[0].get_title(),
        fontsize=FONTSIZE_TITULO,
    )
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_bess_kw_profile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Perfil kW BESS salvo em: {path}")


def plot_q_activation(df_bess: pd.DataFrame):
    """Heatmap de kVAr do controle primário por BESS × hora."""
    pivot      = df_bess.pivot_table(index="der", columns="hora",
                                     values="q_primario_kvar", aggfunc="first")
    der_labels = pivot.index.tolist()
    hours      = pivot.columns.tolist()
    vals       = pivot.values.astype(float)
    q_abs      = np.nanmax(np.abs(vals)) if np.any(np.abs(vals) > 1e-6) else 1.0

    fig, ax = plt.subplots(
        figsize=(max(10, len(hours) * 0.6), max(4, len(der_labels) * 0.7))
    )
    im = ax.imshow(vals, cmap="RdBu", vmin=-q_abs, vmax=q_abs, aspect="auto")

    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels(hours, fontsize=FONTSIZE_TICK)
    ax.set_yticks(range(len(der_labels)))
    ax.set_yticklabels(der_labels, fontsize=FONTSIZE_TICK)
    ax.set_xlabel("Hora do dia", fontsize=FONTSIZE_EIXO)
    ax.set_ylabel("BESS", fontsize=FONTSIZE_EIXO)
    ax.set_title(
        f"kVAr aplicado pelo controle primário VoltVar — pen={PEN_PCT}% / real={ID_REALIZACAO}\n"
        f"Azul = capacitivo (↑V, q>0)  |  Vermelho = indutivo (↓V, q<0)  |  Branco = zona morta",
        fontsize=FONTSIZE_TITULO,
    )
    for i in range(len(der_labels)):
        for j, h in enumerate(hours):
            val = vals[i, j]
            if not np.isnan(val) and abs(val) > 1e-6:
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                        fontsize=8, color="black")

    fig.colorbar(im, ax=ax, label="kVAr  (+ capacitivo / − indutivo)")
    path = os.path.join(FIG_DIR, "fig_q_activation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Heatmap Q primário salvo em: {path}")


def _plot_bus_voltage_profile(bus_records: list, col: str, limit: float,
                               limit_label: str, limit_color: str,
                               title_suffix: str, filename: str):
    df_v  = pd.DataFrame(bus_records)
    buses = sorted(df_v["bus"].unique())
    hours = sorted(df_v["hora"].unique())
    cmap  = plt.cm.get_cmap("rainbow", len(buses))

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, bus in enumerate(buses):
        sub = df_v[df_v["bus"] == bus].sort_values("hora")
        ax.plot(sub["hora"], sub[col], color=cmap(i), linewidth=1.0,
                marker=".", markersize=3, label=bus)

    ax.axhline(limit, color=limit_color, linewidth=1.8, linestyle="--",
               label=f"{limit_label} = {limit} p.u.")
    ax.axhline(1.00, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)

    ax.set_xticks(hours)
    ax.tick_params(axis="x", labelsize=FONTSIZE_TICK)
    ax.tick_params(axis="y", labelsize=FONTSIZE_TICK)
    ax.set_xlabel("Hora do dia", fontsize=FONTSIZE_EIXO)
    ax.set_ylabel("Tensão (p.u.)", fontsize=FONTSIZE_EIXO)
    ax.set_title(
        f"Tensão {title_suffix} nas barras da rede\n"
        f"pen={PEN_PCT}% / real={ID_REALIZACAO}  |  "
        f"Controle primário VoltVar + secundário por consenso (modo: {MODO_SECUNDARIO})",
        fontsize=FONTSIZE_TITULO,
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=6,
              ncol=1, framealpha=0.9, title="Barras")

    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Tensão {title_suffix} salvo em: {path}")


# ---------------------------------------------------------------------------
# LOOP PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    # --- Preparar rede e cenário ---
    redirecionar_circuito(BASE_DSS)
    cargas_base = carregar_cargas_base()
    pv_real, bess_real, fatores_irr, fatores_carga = preparar_cenario_load(
        PEN_PCT, ID_REALIZACAO
    )
    mult_perfil_carga = obter_multiplicadores_shapes()

    # --- Grafo de consenso e potências nominais ---
    barras_bess = list(bess_real["barra"].astype(int).unique())
    L, barras_ordenadas = construir_grafo_comunicacao(barras_bess)
    potencias_nominais = {
        int(k): float(v)
        for k, v in bess_real.groupby("barra")["potencia_kw"].sum().items()
    }
    print(f"[RegControls] {dss.RegControls.AllNames()}")
    print(f"[BESS barras] {barras_bess}")
    print(f"[Grafo]       ordem topológica={barras_ordenadas}")

    records_hora = []
    records_bess = []
    records_bus  = []

    n_viols_mc = carregar_violacoes_mc(PEN_PCT, ID_REALIZACAO)

    # =======================================================================
    # Loop 24h
    # =======================================================================
    for hora in range(TOTAL_HOURS):
        print(f"\n{'='*60}")
        print(f"Hora {hora:02d}/{TOTAL_HOURS - 1}")
        print(f"{'='*60}")

        # --- Cargas base ---
        for nome_load, carga in cargas_base.items():
            barra      = carga["barra"]
            nome_shape = carga["shape"].lower()
            f_incert   = fatores_carga.get(barra, [1.0] * 24)[hora]
            f_perfil   = mult_perfil_carga.get(nome_shape, [1.0] * 24)[hora]
            editar_load(nome_load,
                        carga["kW"]   * f_perfil * f_incert,
                        carga["kvar"] * f_perfil * f_incert)

        # --- PV ---
        kw_pv = {}
        for idx, (_, linha) in enumerate(pv_real.iterrows()):
            nome = f"PV_{ID_REALIZACAO}_{idx}_barra{linha['barra']}"
            kw   = -float(linha["potencia_kw"]) * fatores_irr[hora]
            kw_pv[nome] = (kw, int(linha["barra"]))
            editar_load(nome, kw, 0.0)

        # --- BESS (referência) ---
        kw_bess = {}
        for idx, (_, linha) in enumerate(bess_real.iterrows()):
            nome = f"BESS_{ID_REALIZACAO}_{idx}_barra{linha['barra']}"
            kw   = -float(linha["potencia_kw"]) * BESS_PERFIL[hora]
            kw_bess[nome] = (kw, int(linha["barra"]))
            editar_load(nome, kw, 0.0)

        kw_bess_ref = {n: kw for n, (kw, _) in kw_bess.items()}

        # ---------------------------------------------------------------
        # ETAPA 1: Solve base (reguladores livres, sem controle)
        # ---------------------------------------------------------------
        try:
            dss.Text.Command("Set ControlMode=STATIC")
            dss.Solution.Solve()
        except DSSException as e:
            if e.args[0] == 485 or "Max Control Iterations" in str(e):
                print(f"  [AVISO] hora {hora:02d}: Max Control Iter — hora ignorada.")
                continue
            raise

        tensoes_pre = extrair_tensoes_por_barra()
        n_viols_pre, _ = count_voltage_violations(tensoes_pre)

        # ---------------------------------------------------------------
        # ETAPA 2: Controle Primário VoltVar (congela taps)
        # ---------------------------------------------------------------
        q_refs_pv   = _calcular_q_refs(kw_pv,   tensoes_pre)
        q_refs_bess = _calcular_q_refs(kw_bess, tensoes_pre)
        voltvar_ativo = any(abs(q) > 1e-6
                            for q in {**q_refs_pv, **q_refs_bess}.values())

        tensoes_pos_prim = tensoes_pre
        q_aplicado_bess  = {n: 0.0 for n in kw_bess}

        if voltvar_ativo:
            dss.Text.Command("Set ControlMode=OFF")
            for _ in range(MAX_VOLTVAR_ITER):
                q_aplicado_bess = dict(q_refs_bess)
                for nome, (kw, _) in kw_pv.items():
                    editar_load(nome, kw, -q_refs_pv[nome])
                for nome, (kw, _) in kw_bess.items():
                    editar_load(nome, kw, -q_refs_bess[nome])

                try:
                    dss.Solution.Solve()
                except DSSException as e:
                    if e.args[0] == 485 or "Max Control Iterations" in str(e):
                        break
                    raise

                tensoes_pos_prim = extrair_tensoes_por_barra()
                q_refs_pv   = _calcular_q_refs(kw_pv,   tensoes_pos_prim)
                q_refs_bess = _calcular_q_refs(kw_bess, tensoes_pos_prim)
                if not any(abs(q) > 1e-6
                           for q in {**q_refs_pv, **q_refs_bess}.values()):
                    break
            dss.Text.Command("Set ControlMode=STATIC")

        n_viols_prim, _ = count_voltage_violations(tensoes_pos_prim)

        # ---------------------------------------------------------------
        # ETAPA 3: Controle Secundário por Consenso (P dos BESS)
        # ---------------------------------------------------------------
        bess_operando = abs(BESS_PERFIL[hora]) > 1e-6
        if MODO_SECUNDARIO == "condicional":
            ativar_consenso = (n_viols_prim > 0) and bess_operando and len(barras_ordenadas) >= 2
        else:
            ativar_consenso = bess_operando and len(barras_ordenadas) >= 2

        tensoes_pos_sec = tensoes_pos_prim
        delta_p_final   = {n: 0.0 for n in kw_bess}

        if ativar_consenso:
            dss.Text.Command("Set ControlMode=OFF")
            tensoes_consenso = tensoes_pos_prim

            for _ in range(CONSENSO_MAX_ITER):
                delta_p = calcular_delta_p_consenso(
                    kw_bess, barras_bess, potencias_nominais,
                    L, barras_ordenadas, tensoes_consenso,
                )
                delta_q = _calcular_q_para_delta_p(delta_p, kw_bess, q_refs_bess)

                for nome, (kw, barra) in kw_bess.items():
                    kw_novo = kw + delta_p[nome]
                    q_base  = -q_refs_bess.get(nome, 0.0)
                    q_novo  = q_base + delta_q[nome]
                    kw_bess[nome] = (kw_novo, barra)
                    editar_load(nome, kw_novo, q_novo)

                try:
                    dss.Solution.Solve()
                except DSSException as e:
                    if e.args[0] == 485 or "Max Control Iterations" in str(e):
                        break
                    raise

                tensoes_consenso = extrair_tensoes_por_barra()
                q_refs_bess = _calcular_q_refs(kw_bess, tensoes_consenso)

                if max(abs(dp) for dp in delta_p.values()) < CONSENSO_TOL:
                    break

            delta_p_final = {
                n: kw_bess[n][0] - kw_bess_ref[n]
                for n in kw_bess
            }
            tensoes_pos_sec = tensoes_consenso
            dss.Text.Command("Set ControlMode=STATIC")

        n_viols_sec, _ = count_voltage_violations(tensoes_pos_sec)

        print(f"  [Violações] MC={n_viols_mc[hora]} | pré={n_viols_pre} | "
              f"pós-primário={n_viols_prim} | pós-secundário={n_viols_sec} | "
              f"consenso={'ON' if ativar_consenso else 'OFF'}")

        # --- Registros por hora ---
        records_hora.append({
            "hora":               hora,
            "n_viols_mc":         n_viols_mc[hora],
            "n_viols_pre":        n_viols_pre,
            "n_viols_primario":   n_viols_prim,
            "n_viols_secundario": n_viols_sec,
            "voltvar_ativo":      voltvar_ativo,
            "consenso_ativo":     ativar_consenso,
        })

        # --- Registros por BESS ---
        for nome, (kw_final, barra) in kw_bess.items():
            p_nom = potencias_nominais.get(barra, 1.0)
            records_bess.append({
                "hora":                  hora,
                "der":                   nome,
                "barra":                 barra,
                "kw_ref_kw":             round(kw_bess_ref[nome], 4),
                "kw_pos_secundario_kw":  round(kw_final, 4),
                "delta_p_kw":            round(delta_p_final[nome], 4),
                "q_primario_kvar":       round(q_aplicado_bess.get(nome, 0.0), 4),
                "p_nom_kw":              round(p_nom, 2),
            })
            if abs(delta_p_final[nome]) > 0.1:
                print(f"  [BESS {nome}] ΔP={delta_p_final[nome]:+.2f} kW | "
                      f"Q_prim={q_aplicado_bess.get(nome, 0.0):+.2f} kVAr")

        # --- Registros de tensão das barras (estado final pós-secundário) ---
        for bus, fases in tensoes_pos_sec.items():
            records_bus.append({
                "hora":  hora,
                "bus":   bus,
                "v_min": round(min(fases), 6),
                "v_max": round(max(fases), 6),
            })

    # =======================================================================
    # Exportar CSVs
    # =======================================================================
    tag      = f"pen{PEN_PCT:03d}_real{ID_REALIZACAO:04d}_modo{MODO_SECUNDARIO}"
    df_hora  = pd.DataFrame(records_hora)
    df_bess  = pd.DataFrame(records_bess)

    df_hora.to_csv(os.path.join(CSV_DIR, f"secundario_{tag}_por_hora.csv"),  index=False)
    df_bess.to_csv(os.path.join(CSV_DIR, f"secundario_{tag}_por_bess.csv"),  index=False)
    print(f"\n[OK] CSVs salvos em {CSV_DIR}")

    # =======================================================================
    # Gerar gráficos
    # =======================================================================
    if not df_hora.empty:
        plot_violations_3way(df_hora, n_viols_mc)

    if not df_bess.empty:
        plot_bess_delta_p(df_bess)
        plot_bess_kw_profile(df_bess)
        plot_q_activation(df_bess)

    if records_bus:
        _plot_bus_voltage_profile(
            records_bus, col="v_min", limit=0.95,
            limit_label="V_min limite", limit_color="red",
            title_suffix="mínima", filename="fig_all_bus_vmin.png",
        )
        _plot_bus_voltage_profile(
            records_bus, col="v_max", limit=1.05,
            limit_label="V_max limite", limit_color="darkorange",
            title_suffix="máxima", filename="fig_all_bus_vmax.png",
        )

    print("\n[FIM] Análise do controle secundário integrado concluída.")


if __name__ == "__main__":
    main()
