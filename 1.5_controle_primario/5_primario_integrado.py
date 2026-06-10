"""
Análise diagnóstica do Controle Primário VoltVar integrado à pipeline Monte Carlo
ABNT NBR 16149:2013 — Seção 4.7.3

Reaplica a lógica exata de 1.4_geracao_de_cenarios/controle_primario.py para uma única
realização, expondo hora a hora: tensões, taps dos reguladores pré e pós VoltVar,
reativos injetados por DER e violações antes/depois do controle.

Diferença em relação a 4_myVoltVarQ.py: usa Load elements (não PVSystem/Storage nativos)
para manter comparabilidade com os resultados do Monte Carlo.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dss._cffi_api_util import DSSException
from opendssdirect import dss

# ---------------------------------------------------------------------------
# Imports da pipeline Monte Carlo (evita duplicação de código)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "1.4_geracao_de_cenarios"))

from controle_primario import (  # noqa: E402
    V_DEADBAND_HIGH,
    V_DEADBAND_LOW,
    V_SAT_HIGH,
    V_SAT_LOW,
    MAX_VOLTVAR_ITER,
    Q_MAX_PU,
    _calcular_q_refs,
    volt_var_curve,
)
from simulacao_opendss import (  # noqa: E402
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
from simulacoes_config import BESS_PERFIL  # noqa: E402


# ---------------------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------------------
FIG_DIR  = os.path.join(SCRIPT_DIR, "figuras_primario_integrado")
CSV_DIR  = os.path.join(SCRIPT_DIR, "resultados")
BASE_DSS = os.path.join(SCRIPT_DIR, "..", "IEEE34bus", "IEEE34_original_with_loadshapes.dss")
MC_DIR   = os.path.join(SCRIPT_DIR, "..", "1.4_geracao_de_cenarios",
                         "resultados_monte_carlo", "realizacoes_sorteadas")

PEN_PCT       = 90
ID_REALIZACAO = 6
TOTAL_HOURS   = 24


# ---------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ---------------------------------------------------------------------------

def preparar_cenario_load(pen_pct: int, id_realizacao: int) -> tuple:
    """
    Lê os CSVs da realização Monte Carlo, cria os Load elements no circuito DSS,
    e retorna (pv_real_df, bess_real_df, fatores_irradiancia, fatores_carga).

    pv_real_df : todos os elementos da realização (PVSystem + Storage) — igual ao
                 que processar_nivel_controle_primario passa como pv_df.
    bess_real_df: apenas elementos Storage — passado como bess_df.
    """
    pasta = os.path.join(MC_DIR, f"pen_{pen_pct:03d}pct")

    perfis_irr = ler_perfis_irradiancia(pasta)
    irr_row    = perfis_irr[perfis_irr["id_realizacao"] == id_realizacao].iloc[0]
    fatores_irradiancia = [float(irr_row[f"h{h:02d}"]) for h in range(24)]

    fat_df   = ler_fatores_incerteza_carga(pasta)
    fat_real = fat_df[fat_df["id_realizacao"] == id_realizacao]
    fatores_carga = {
        int(linha["barra"]): [float(linha[f"h{h:02d}"]) for h in range(24)]
        for _, linha in fat_real.iterrows()
    }

    elementos  = ler_elementos_opendss(pasta)
    pv_real    = elementos[elementos["id_realizacao"] == id_realizacao]
    bess_real  = pv_real[pv_real["classe"] == "Storage"]

    criar_elementos_simulacao(pv_real, bess_real, id_realizacao)

    n_pv   = len(pv_real[pv_real["classe"] == "PVSystem"])
    n_bess = len(bess_real)
    print(f"[Cenário] pen={pen_pct}% | real={id_realizacao} | "
          f"PV={n_pv} | BESS={n_bess} | cargas com fator={len(fatores_carga)}")
    return pv_real, bess_real, fatores_irradiancia, fatores_carga


def capture_taps(reg_names: list) -> dict:
    """Lê os taps atuais de todos os reguladores após um solve."""
    tap_snapshot = {}
    for reg in reg_names:
        dss.RegControls.Name(reg)
        xfmr = dss.RegControls.Transformer()
        wdg  = dss.RegControls.Winding()
        dss.Transformers.Name(xfmr)
        dss.Transformers.Wdg(wdg)
        tap_snapshot[reg] = {"transformer": xfmr, "winding": wdg,
                              "tap": dss.Transformers.Tap()}
    return tap_snapshot


def count_voltage_violations(tensoes: dict) -> tuple:
    """
    Retorna (n_violacoes, {bus: info}) para barras fora de [0.95, 1.05] p.u.
    Usa o dict de extrair_tensoes_por_barra() — sem reler o DSS.
    """
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


# ---------------------------------------------------------------------------
# FUNÇÕES DE VISUALIZAÇÃO
# ---------------------------------------------------------------------------

def plot_volt_var_curve():
    """Curva Volt-VAr normalizada (Q/Qmax × V) — ABNT NBR 16149:2013."""
    v_range = np.linspace(0.85, 1.15, 500)
    q_norm  = [volt_var_curve(v, 1.0) / Q_MAX_PU for v in v_range]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(v_range, q_norm, color="steelblue", linewidth=2, label="Curva Volt-VAr")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    for v in [V_SAT_LOW, V_DEADBAND_LOW, V_DEADBAND_HIGH, V_SAT_HIGH]:
        ax.axvline(v, color="tomato", linewidth=1, linestyle=":", alpha=0.8)
        ax.text(v, 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="tomato")

    ax.fill_betweenx([-1.1, 1.1], V_DEADBAND_LOW, V_DEADBAND_HIGH,
                     alpha=0.1, color="gray", label="Zona morta")
    ax.set_xlabel("Tensão na barra (p.u.)")
    ax.set_ylabel("Q / Q$_{max}$ (p.u. de P$_{inst}$)")
    ax.set_title("Curva Volt-VAr — ABNT NBR 16149:2013")
    ax.set_ylim(-1.2, 1.2)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    path = os.path.join(FIG_DIR, "fig_volt_var_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Curva Volt-VAr salva em: {path}")


def plot_taps_por_hora(df_taps: pd.DataFrame):
    """Posição de tap por regulador ao longo das 24h (estado pré-VoltVar)."""
    regs = df_taps["regulador"].unique()
    fig, ax = plt.subplots(figsize=(14, 5))
    for reg in regs:
        sub = df_taps[df_taps["regulador"] == reg].sort_values("hora")
        ax.plot(sub["hora"], sub["tap_pre_voltvar"], marker="o", markersize=4,
                linewidth=1.5, label=reg)
    ax.set_xticks(range(TOTAL_HOURS))
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Posição de tap (p.u.)")
    ax.set_title(f"Posições de tap dos reguladores — pen={PEN_PCT}% / real={ID_REALIZACAO}\n"
                 f"Estado pré-VoltVar (após primeiro solve com reguladores livres)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    path = os.path.join(FIG_DIR, "fig_taps_por_hora.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Taps por hora salvo em: {path}")


def plot_taps_comparacao(df_taps: pd.DataFrame):
    """Pré vs pós VoltVar por regulador — coincidência confirma freeze dos taps."""
    regs  = sorted(df_taps["regulador"].unique())
    n_reg = len(regs)
    fig, axes = plt.subplots(n_reg, 1, figsize=(14, 3.5 * n_reg), sharex=True)
    if n_reg == 1:
        axes = [axes]

    for ax, reg in zip(axes, regs):
        sub = df_taps[df_taps["regulador"] == reg].sort_values("hora")
        ax.plot(sub["hora"], sub["tap_pre_voltvar"], color="steelblue",
                marker="o", markersize=5, linewidth=1.5, label="Pré-VoltVar")
        ax.plot(sub["hora"], sub["tap_pos_voltvar"], color="darkorange",
                marker="x", markersize=6, linewidth=1.5, linestyle="--", label="Pós-VoltVar")
        ax.set_ylabel("Tap (p.u.)")
        ax.set_title(f"Regulador: {reg}")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Hora do dia")
    axes[0].set_title(
        f"Tap pré vs pós VoltVar — pen={PEN_PCT}% / real={ID_REALIZACAO}\n"
        f"Coincidência confirma freeze dos taps durante controle primário (ControlMode=OFF)\n"
        + axes[0].get_title()
    )
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_taps_comparacao.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Comparação de taps salvo em: {path}")


def plot_q_activation(df_der: pd.DataFrame):
    """Heatmap de kVAr aplicado por DER × hora (azul = capacitivo, vermelho = indutivo)."""
    pivot      = df_der.pivot_table(index="der", columns="hora", values="q_kvar", aggfunc="first")
    der_labels = pivot.index.tolist()
    hours      = pivot.columns.tolist()
    vals       = pivot.values.astype(float)
    q_abs      = np.nanmax(np.abs(vals)) if np.any(np.abs(vals) > 1e-6) else 1.0

    fig, ax = plt.subplots(figsize=(max(10, len(hours) * 0.6), max(4, len(der_labels) * 0.7)))
    im = ax.imshow(vals, cmap="RdBu", vmin=-q_abs, vmax=q_abs, aspect="auto")

    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels(hours, fontsize=8)
    ax.set_yticks(range(len(der_labels)))
    ax.set_yticklabels(der_labels, fontsize=8)
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("DER")
    ax.set_title(
        f"kVAr aplicado pelo controle Volt-VAr — pen={PEN_PCT}% / real={ID_REALIZACAO}\n"
        f"Azul = capacitivo (↑V, q>0)  |  Vermelho = indutivo (↓V, q<0)  |  Branco = zona morta"
    )
    for i, der in enumerate(der_labels):
        for j, h in enumerate(hours):
            val = pivot.loc[der, h]
            if not np.isnan(val) and abs(val) > 1e-6:
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=6, color="black")

    fig.colorbar(im, ax=ax, label="kVAr  (+ capacitivo / − indutivo)")
    path = os.path.join(FIG_DIR, "fig_q_activation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Heatmap kVAr salvo em: {path}")


def plot_violations(df_hora: pd.DataFrame):
    """Barras duplas: violações de tensão pré e pós VoltVar por hora."""
    horas = df_hora["hora"].tolist()
    n_pre = df_hora["n_viols_pre"].tolist()
    n_pos = df_hora["n_viols_pos"].tolist()
    x     = np.arange(len(horas))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, n_pre, width, label="Pré-VoltVar",  color="steelblue",  edgecolor="white")
    ax.bar(x + width / 2, n_pos, width, label="Pós-VoltVar",  color="darkorange", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(horas)
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Barras com violação de tensão")
    ax.set_title(
        f"Violações de tensão pré e pós controle VoltVar\n"
        f"pen={PEN_PCT}% / real={ID_REALIZACAO}  |  Limites: V < 0.95 p.u. ou V > 1.05 p.u."
    )
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    y_top = max(max(n_pre, default=0), max(n_pos, default=0)) + 1
    ax.set_ylim(0, max(y_top, 1))

    path = os.path.join(FIG_DIR, "fig_violations.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Violações pré/pós VoltVar salvo em: {path}")


def plot_voltvar_efeito(df_der: pd.DataFrame):
    """Scatter V_pre × V_pos nas barras dos DERs — efeito do VoltVar nas tensões."""
    df_active = df_der[df_der["q_kvar"].abs() > 1e-6]

    fig, ax = plt.subplots(figsize=(8, 6))
    cores = {"PV": "steelblue", "BESS": "darkorange"}
    for tipo, sub in df_der.groupby("tipo"):
        ax.scatter(sub["v_pre"], sub["v_pos"], s=20, alpha=0.5,
                   color=cores.get(tipo, "gray"), label=f"{tipo} (todos)")
    if not df_active.empty:
        for tipo, sub in df_active.groupby("tipo"):
            ax.scatter(sub["v_pre"], sub["v_pos"], s=50, alpha=0.9,
                       color=cores.get(tipo, "gray"), edgecolors="black",
                       linewidths=0.5, label=f"{tipo} (VoltVar ativo)", zorder=5)

    vmin = min(df_der["v_pre"].min(), df_der["v_pos"].min()) - 0.005
    vmax = max(df_der["v_pre"].max(), df_der["v_pos"].max()) + 0.005
    ax.plot([vmin, vmax], [vmin, vmax], color="gray", linewidth=1,
            linestyle="--", label="Sem efeito (V_pre = V_pos)")
    for v, c in [(0.95, "red"), (1.05, "darkorange")]:
        ax.axvline(v, color=c, linewidth=0.8, linestyle=":", alpha=0.7)
        ax.axhline(v, color=c, linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel("Tensão pré-VoltVar (p.u.)")
    ax.set_ylabel("Tensão pós-VoltVar (p.u.)")
    ax.set_title(
        f"Efeito do VoltVar nas tensões dos DERs\n"
        f"pen={PEN_PCT}% / real={ID_REALIZACAO}  |  cada ponto = (DER, hora)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    path = os.path.join(FIG_DIR, "fig_voltvar_efeito_tensao.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Efeito VoltVar nas tensões salvo em: {path}")


def _plot_bus_voltage_profile(df_v: pd.DataFrame, col: str, limit: float,
                               limit_label: str, limit_color: str,
                               title_suffix: str, filename: str):
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
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Tensão (p.u.)")
    ax.set_title(
        f"Tensão {title_suffix} nas barras da rede\n"
        f"pen={PEN_PCT}% / real={ID_REALIZACAO}  |  Controle Primário VoltVar ativo"
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=6,
              ncol=1, framealpha=0.9, title="Barras")
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Tensão {title_suffix} de todas as barras salvo em: {path}")


def plot_all_bus_vmin(bus_records: list):
    df_v = pd.DataFrame(bus_records)
    _plot_bus_voltage_profile(df_v, col="v_min", limit=0.95,
                               limit_label="V_min limite", limit_color="red",
                               title_suffix="mínima", filename="fig_all_bus_vmin.png")


def plot_all_bus_vmax(bus_records: list):
    df_v = pd.DataFrame(bus_records)
    _plot_bus_voltage_profile(df_v, col="v_max", limit=1.05,
                               limit_label="V_max limite", limit_color="darkorange",
                               title_suffix="máxima", filename="fig_all_bus_vmax.png")


# ---------------------------------------------------------------------------
# LOOP PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    # --- Compilar rede e preparar cenário ---
    redirecionar_circuito(BASE_DSS)
    cargas_base = carregar_cargas_base()
    pv_real, bess_real, fatores_irradiancia, fatores_carga = preparar_cenario_load(
        PEN_PCT, ID_REALIZACAO
    )
    multiplicadores_perfil_carga = obter_multiplicadores_shapes()

    reg_names = dss.RegControls.AllNames() or []
    print(f"[Cargas base] {len(cargas_base)} Load(s)")
    print(f"[RegControls] {len(reg_names)} regulador(es): {reg_names}")

    plot_volt_var_curve()

    records_hora = []
    records_der  = []
    records_bus  = []
    records_taps = []

    # --- Loop 24h ---
    for hora in range(TOTAL_HOURS):

        print(f"\n{'='*60}")
        print(f"Hora {hora:02d}/{TOTAL_HOURS - 1}")
        print(f"{'='*60}")

        # Cargas base com LoadShape × incerteza
        for nome_load, carga in cargas_base.items():
            barra           = carga["barra"]
            nome_shape      = carga["shape"].lower()
            fator_incerteza = fatores_carga.get(barra, [1.0] * 24)[hora]
            fator_perfil    = multiplicadores_perfil_carga.get(nome_shape, [1.0] * 24)[hora]
            editar_load(nome_load,
                        carga["kW"]   * fator_perfil * fator_incerteza,
                        carga["kvar"] * fator_perfil * fator_incerteza)

        # PV loads — mesma lógica de simular_realizacao_controle_primario
        kw_pv = {}
        for idx, (_, linha) in enumerate(pv_real.iterrows()):
            nome = f"PV_{ID_REALIZACAO}_{idx}_barra{linha['barra']}"
            kw   = -float(linha["potencia_kw"]) * fatores_irradiancia[hora]
            kw_pv[nome] = (kw, int(linha["barra"]))
            editar_load(nome, kw, 0.0)

        # BESS loads
        kw_bess = {}
        for idx, (_, linha) in enumerate(bess_real.iterrows()):
            nome = f"BESS_{ID_REALIZACAO}_{idx}_barra{linha['barra']}"
            kw   = -float(linha["potencia_kw"]) * BESS_PERFIL[hora]
            kw_bess[nome] = (kw, int(linha["barra"]))
            editar_load(nome, kw, 0.0)

        # --- Solve principal com reguladores livres ---
        try:
            dss.Text.Command("Set ControlMode=STATIC")
            dss.Solution.Solve()
        except DSSException as e:
            if e.args[0] == 485 or "Max Control Iterations" in str(e):
                print(f"  [AVISO] hora {hora:02d}: Max Control Iterations — hora ignorada.")
                continue
            raise

        taps_pre    = capture_taps(reg_names)
        tensoes_pre = extrair_tensoes_por_barra()
        n_viols_pre, _ = count_voltage_violations(tensoes_pre)

        for reg, info in taps_pre.items():
            print(f"  [Tap pré ] {reg}: tap={info['tap']:.6f}")

        # Q_refs iniciais
        q_refs_pv   = _calcular_q_refs(kw_pv,   tensoes_pre)
        q_refs_bess = _calcular_q_refs(kw_bess, tensoes_pre)

        voltvar_ativo   = any(abs(q) > 1e-6 for q in {**q_refs_pv, **q_refs_bess}.values())
        iters_voltvar   = 0
        tensoes_final   = tensoes_pre

        # Q efetivamente aplicado (0 se VoltVar não ativou)
        q_aplicado_pv   = {n: 0.0 for n in kw_pv}
        q_aplicado_bess = {n: 0.0 for n in kw_bess}

        # --- Loop VoltVar com taps congelados ---
        if voltvar_ativo:
            dss.Text.Command("Set ControlMode=OFF")

            for iteration in range(MAX_VOLTVAR_ITER):
                iters_voltvar = iteration + 1

                # Salva os Q que serão aplicados nesta iteração
                q_aplicado_pv   = dict(q_refs_pv)
                q_aplicado_bess = dict(q_refs_bess)

                for nome, (kw, _) in kw_pv.items():
                    editar_load(nome, kw, -q_refs_pv[nome])   # negação: Load usa conv. de carga
                for nome, (kw, _) in kw_bess.items():
                    editar_load(nome, kw, -q_refs_bess[nome])

                try:
                    dss.Solution.Solve()
                except DSSException as e:
                    if e.args[0] == 485 or "Max Control Iterations" in str(e):
                        print(f"  [AVISO] VoltVar iter {iters_voltvar}: Max Control Iter — encerrando loop.")
                        break
                    raise

                tensoes_final = extrair_tensoes_por_barra()
                q_refs_pv   = _calcular_q_refs(kw_pv,   tensoes_final)
                q_refs_bess = _calcular_q_refs(kw_bess, tensoes_final)

                q_str = " | ".join(
                    f"{n}: {q_aplicado_pv.get(n, q_aplicado_bess.get(n, 0.0)):+.2f} kvar"
                    for n in list(kw_pv) + list(kw_bess)
                )
                print(f"  [VoltVar iter {iters_voltvar}] Q aplicado: {q_str}")

                if not any(abs(q) > 1e-6 for q in {**q_refs_pv, **q_refs_bess}.values()):
                    break

            dss.Text.Command("Set ControlMode=STATIC")

        taps_pos = capture_taps(reg_names)
        for reg, info in taps_pos.items():
            print(f"  [Tap pós ] {reg}: tap={info['tap']:.6f}")

        n_viols_pos, viols_pos = count_voltage_violations(tensoes_final)

        # --- Registros por hora ---
        records_hora.append({
            "hora":          hora,
            "n_viols_pre":   n_viols_pre,
            "n_viols_pos":   n_viols_pos,
            "voltvar_ativo": voltvar_ativo,
            "iters_voltvar": iters_voltvar,
        })

        # --- Registros por DER ---
        for nome, (kw, barra) in {**kw_pv, **kw_bess}.items():
            tipo    = "PV" if nome.startswith("PV_") else "BESS"
            barra_s = str(barra)
            v_pre   = min(tensoes_pre[barra_s])   if barra_s in tensoes_pre   else None
            v_pos   = min(tensoes_final[barra_s]) if barra_s in tensoes_final else None
            p_inst  = abs(kw)
            q_kvar  = q_aplicado_pv.get(nome, q_aplicado_bess.get(nome, 0.0))
            q_pct   = round(q_kvar / p_inst * 100, 2) if p_inst > 1e-6 else None

            records_der.append({
                "hora":        hora,
                "der":         nome,
                "tipo":        tipo,
                "barra":       barra,
                "v_pre":       round(v_pre, 6) if v_pre is not None else None,
                "v_pos":       round(v_pos, 6) if v_pos is not None else None,
                "q_kvar":      round(q_kvar, 4),
                "p_inst_kw":   round(p_inst, 4),
                "q_pct_pinst": q_pct,
            })

            status  = "ativo" if abs(q_kvar) > 1e-6 else "zona morta"
            q_str   = f"{q_pct:+.1f}%" if q_pct is not None else "N/A"
            v_p_str = f"{v_pre:.4f}" if v_pre is not None else "N/A"
            v_s_str = f"{v_pos:.4f}" if v_pos is not None else "N/A"
            print(f"  [{tipo:4s} {nome}] V_pre={v_p_str} | V_pos={v_s_str} | "
                  f"q={q_kvar:+.2f} kVAr ({q_str}) | {status}")

        # --- Registros de taps ---
        for reg in reg_names:
            records_taps.append({
                "hora":            hora,
                "regulador":       reg,
                "tap_pre_voltvar": taps_pre[reg]["tap"],
                "tap_pos_voltvar": taps_pos[reg]["tap"],
            })

        # --- Registros de tensão de todas as barras (estado final) ---
        for barra, fases in tensoes_final.items():
            records_bus.append({
                "hora":  hora,
                "bus":   barra,
                "v_min": round(min(fases), 6),
                "v_max": round(max(fases), 6),
            })

        vv_str = "ATIVO" if voltvar_ativo else "inativo"
        print(f"  [Violações] pré={n_viols_pre} → pós={n_viols_pos} | "
              f"VoltVar={vv_str} | iters={iters_voltvar}")
        if viols_pos:
            for b, info in sorted(viols_pos.items()):
                partes = []
                if info["under"]: partes.append(f"↓V_min={info['v_min']:.4f}")
                if info["over"]:  partes.append(f"↑V_max={info['v_max']:.4f}")
                print(f"    barra {b}: {', '.join(partes)}")

    # --- Exportar CSVs ---
    tag      = f"pen{PEN_PCT:03d}_real{ID_REALIZACAO:04d}"
    df_hora  = pd.DataFrame(records_hora)
    df_der   = pd.DataFrame(records_der)
    df_taps  = pd.DataFrame(records_taps)

    csv_hora = os.path.join(CSV_DIR, f"primario_integrado_{tag}_por_hora.csv")
    csv_der  = os.path.join(CSV_DIR, f"primario_integrado_{tag}_por_der.csv")
    csv_taps = os.path.join(CSV_DIR, f"primario_integrado_{tag}_taps.csv")

    df_hora.to_csv(csv_hora, index=False)
    df_der.to_csv(csv_der,   index=False)
    df_taps.to_csv(csv_taps, index=False)

    print(f"\n[OK] CSVs salvos:")
    print(f"  {csv_hora}")
    print(f"  {csv_der}")
    print(f"  {csv_taps}")

    # --- Gerar gráficos ---
    if not df_taps.empty:
        plot_taps_por_hora(df_taps)
        plot_taps_comparacao(df_taps)
    if not df_der.empty:
        plot_q_activation(df_der)
        plot_voltvar_efeito(df_der)
    if not df_hora.empty:
        plot_violations(df_hora)
    if records_bus:
        plot_all_bus_vmin(records_bus)
        plot_all_bus_vmax(records_bus)

    print("\n[FIM] Análise do controle primário integrado concluída.")


if __name__ == "__main__":
    main()
