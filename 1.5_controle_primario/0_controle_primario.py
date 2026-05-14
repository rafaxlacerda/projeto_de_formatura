"""
Controle Primário Volt-VAr para BESSs na rede IEEE 34 barras (OpenDSS)
Baseado na ABNT NBR 16149:2013 - Seção 4.7.3 e Figura 2

Curva Volt-VAr (Figura 2 da norma):
  - Zona morta: ±20% de Q/PN  →  sem injeção de reativo
  - Saturação:  ±43,58% de Q/PN (≈ tan(arccos(0,90))) → Q máximo
  - Entre zona morta e saturação: interpolação linear

Reguladores: taps congelados no valor do snapshot inicial.
             RegControl desabilitado durante o controle primário.
"""

from opendssdirect import dss
import pandas as pd
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------------------
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DSS      = os.path.join(SCRIPT_DIR, "..", "IEEE34bus", "IEEE34_original.dss")
MC_DIR        = os.path.join(SCRIPT_DIR, "..", "1.4_geracao_de_cenarios",
                              "resultados_monte_carlo_v2")

# Selecione o cenário Monte Carlo desejado:
PEN_PCT       = 50   # nível de penetração em % (ex.: 50 → pen_050pct)
ID_REALIZACAO = 1    # ID da realização (1 a 50)

TOTAL_HOURS   = 24          # número de passos horários
MAX_ITER      = 50          # iterações máximas do loop Volt-VAr por timestep
TOL_VMAG      = 1e-4        # tolerância de convergência (p.u.)
V_NOM_PU      = 1.0         # tensão nominal (p.u.) — referência para a curva

# Agregação de tensão nas fases: "min" usa a menor fase; "mean" usa a média
V_AGGREGATION = "min"

# Limites da curva Volt-VAr — ABNT NBR 16149:2013, Figura 2
# (expressos como variação em relação a V_NOM_PU)
V_DEADBAND_LOW  = 0.97      # p.u.  — início da zona morta inferior
V_DEADBAND_HIGH = 1.03      # p.u.  — início da zona morta superior
# OBS: ajuste esses pontos de acordo com o projeto; a norma não fixa os
# pontos de tensão — ela fixa os limites de Q. Use valores típicos de
# 0,97–1,03 p.u. para a zona morta e 0,90–1,10 p.u. para saturação.
V_SAT_LOW  = 0.90           # p.u.  — tensão de saturação capacitiva (Q > 0)
V_SAT_HIGH = 1.10           # p.u.  — tensão de saturação indutiva   (Q < 0)

# Limites de Q pela norma (em fração da potência nominal do BESS)
# Figura 2: Qmax/Pnom = 43,58 %  →  equivale a FP = 0,90
Q_MAX_PU = 0.4358           # p.u. de Pnom


# ---------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ---------------------------------------------------------------------------

def volt_var_curve(v_pu: float, p_nom_kw: float) -> float:
    """
    Retorna o Q_ref (kvar) para um dado v_pu medido na barra do BESS.

    Convenção de sinal (OpenDSS Storage):
      Q > 0  →  capacitivo (injeção de reativo → eleva tensão)
      Q < 0  →  indutivo   (absorção de reativo → reduz tensão)

    Curva (por regiões):
      v ≤ V_SAT_LOW          → Q = +Q_MAX (máximo capacitivo)
      V_SAT_LOW < v < V_DB_LOW → interpolação linear 0 → +Q_MAX (invertida)
      V_DB_LOW ≤ v ≤ V_DB_HIGH → zona morta, Q = 0
      V_DB_HIGH < v < V_SAT_HIGH → interpolação linear 0 → -Q_MAX
      v ≥ V_SAT_HIGH         → Q = -Q_MAX (máximo indutivo)
    """
    q_max_kvar = Q_MAX_PU * p_nom_kw  # kvar

    if v_pu <= V_SAT_LOW:
        return +q_max_kvar

    elif v_pu < V_DEADBAND_LOW:
        # interpola entre +Qmax (em V_SAT_LOW) e 0 (em V_DB_LOW)
        slope = -q_max_kvar / (V_DEADBAND_LOW - V_SAT_LOW)
        return q_max_kvar + slope * (v_pu - V_SAT_LOW)

    elif v_pu <= V_DEADBAND_HIGH:
        return 0.0

    elif v_pu < V_SAT_HIGH:
        # interpola entre 0 (em V_DB_HIGH) e -Qmax (em V_SAT_HIGH)
        slope = -q_max_kvar / (V_SAT_HIGH - V_DEADBAND_HIGH)
        return slope * (v_pu - V_DEADBAND_HIGH)

    else:  # v_pu >= V_SAT_HIGH
        return -q_max_kvar


def get_bus_vmag_pu(bus_name: str) -> float:
    """
    Retorna a tensão representativa em p.u. nas fases presentes da barra.
    V_AGGREGATION = "min"  → menor fase (mais conservador)
    V_AGGREGATION = "mean" → média das fases
    """
    dss.Circuit.SetActiveBus(bus_name)
    vmag_list  = dss.Bus.puVmagAngle()   # [mag1, ang1, mag2, ang2, ...]
    magnitudes = vmag_list[0::2]         # pega só as magnitudes
    valid = [v for v in magnitudes if v > 0.1]
    if not valid:
        return 1.0
    return min(valid) if V_AGGREGATION == "min" else sum(valid) / len(valid)


def enable_regulators(reg_names: list):
    """Reabilita todos os RegControls para que ajustem taps automaticamente."""
    for reg in reg_names:
        dss.Command(f"RegControl.{reg}.enabled=yes")


def disable_regulators(reg_names: list):
    """Desabilita todos os RegControls (congela taps na posição atual)."""
    for reg in reg_names:
        dss.Command(f"RegControl.{reg}.enabled=no")


def capture_taps(reg_names: list) -> dict:
    """
    Lê os taps atuais de todos os reguladores sem alterar seu estado.
    Deve ser chamada após um solve, enquanto os taps estão no ponto desejado.
    """
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


def restore_tap(tap_snapshot: dict):
    """
    Reaplica os taps congelados (chama após cada solve interno para garantir
    que o OpenDSS não mova os taps por outras razões).
    """
    for _, info in tap_snapshot.items():
        dss.Transformers.Name(info["transformer"])
        dss.Transformers.Wdg(info["winding"])
        dss.Transformers.Tap(info["tap"])


def discover_bess_buses() -> dict:
    """
    Descobre todos os elementos Storage presentes na rede e retorna
    {nome_storage: {bus, pnom_kw, qnom_kvar}}.
    Filtra apenas elementos em barras com 3 fases (trifásicos).
    """
    bess_info = {}
    all_elements = dss.Circuit.AllElementNames()

    for el in all_elements:
        if el.lower().startswith("storage."):
            dss.Circuit.SetActiveElement(el)
            name   = el.split(".")[1]
            bus    = dss.CktElement.BusNames()[0].split(".")[0].lower()
            phases = dss.CktElement.NumPhases()
            if phases < 3:
                print(f"[BESS] '{name}' em '{bus}' ignorado (monofásico/bifásico).")
                continue
            # Potência nominal (kW) — usa a propriedade kWrated do Storage
            dss.Command(f"? Storage.{name}.kWrated")
            p_nom = float(dss.Text.Result()) if dss.Text.Result() else 0.0
            bess_info[name] = {"bus": bus, "pnom_kw": p_nom}
            print(f"[BESS] '{name}' | bus='{bus}' | Pnom={p_nom:.1f} kW | fases={phases}")

    return bess_info


def set_bess_kvar(name: str, kvar: float):
    """
    Injeta Q no BESS via comando direto ao Storage element.
    Usa %kvar relativo à capacidade ou kvar absoluto, conforme disponibilidade.
    """
    dss.Command(f"Storage.{name}.kvar={kvar:.4f}")


def count_voltage_violations() -> tuple:
    """
    Retorna (n_violacoes, dict{bus: v_pu}) para todas as barras com
    V < 0.95 p.u. ou V > 1.05 p.u. após o solve atual.
    """
    violations = {}
    for bus in dss.Circuit.AllBusNames():
        v = get_bus_vmag_pu(bus)
        if v < 0.95 or v > 1.05:
            violations[bus] = round(v, 6)
    return len(violations), violations


def plot_voltvar_curve():
    """Gera e salva o gráfico da curva Volt-VAr configurada."""
    v_range = np.linspace(0.85, 1.15, 500)
    q_norm  = [volt_var_curve(v, 1.0) / Q_MAX_PU for v in v_range]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(v_range, q_norm, color="steelblue", linewidth=2, label="Curva Volt-VAr")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    breakpoints = {
        "V_SAT_LOW":       (V_SAT_LOW,       "+Q_MAX"),
        "V_DB_LOW":        (V_DEADBAND_LOW,   "Zona morta"),
        "V_DB_HIGH":       (V_DEADBAND_HIGH,  ""),
        "V_SAT_HIGH":      (V_SAT_HIGH,       "−Q_MAX"),
    }
    for v in [bp for bp, _ in breakpoints.values()]:
        ax.axvline(v, color="tomato", linewidth=1, linestyle=":", alpha=0.8)
        ax.text(v, 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="tomato")

    ax.fill_betweenx([-1.1, 1.1], V_DEADBAND_LOW, V_DEADBAND_HIGH,
                     alpha=0.1, color="gray", label="Zona morta")
    ax.set_xlabel("Tensão na barra (p.u.)")
    ax.set_ylabel("Q / Q$_{max}$ (p.u. de P$_{nom}$)")
    ax.set_title("Curva Volt-VAr — ABNT NBR 16149:2013")
    ax.set_ylim(-1.2, 1.2)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    path = os.path.join(SCRIPT_DIR, "fig_voltvar_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Curva Volt-VAr salva em: {path}")


def add_realizacao_elements(pen_pct: int, id_realizacao: int):
    """
    Lê os CSVs da realização Monte Carlo e adiciona PVSystem e Storage
    à rede já compilada. Retorna (n_pv, n_bess).
    """
    pasta = os.path.join(MC_DIR, f"pen_{pen_pct:03d}pct")

    # LoadShape de irradiância (estocástico, 24 h)
    df_irr  = pd.read_csv(os.path.join(pasta, "02_perfis_irradiancia.csv"),
                          sep=";", decimal=",")
    irr_row = df_irr[df_irr["id_realizacao"] == id_realizacao].iloc[0]
    irr_vals = [float(irr_row[f"h{h:02d}"]) for h in range(24)]
    irr_str  = " ".join(f"{v:.4f}" for v in irr_vals)
    irr_ls   = f"PV_irr_real{id_realizacao}"
    dss.Command(f"New LoadShape.{irr_ls} npts=24 interval=1 mult=[{irr_str}]")

    # LoadShape fixo de BESS (carga 10–14 h, descarga 18–21 h)
    bess_prof = [0.0] * 24
    for h in range(10, 15):
        bess_prof[h] = -1.0
    for h in range(18, 22):
        bess_prof[h] =  1.0
    bess_str = " ".join(f"{v:.1f}" for v in bess_prof)
    dss.Command(f"New LoadShape.BESS_fixo npts=24 interval=1 mult=[{bess_str}]")

    # Lê elementos da realização
    df_elem = pd.read_csv(os.path.join(pasta, "06_elementos_opendss.csv"),
                          sep=";", decimal=",")
    df_real = df_elem[df_elem["id_realizacao"] == id_realizacao]

    n_pv = n_bess = 0
    for _, row in df_real.iterrows():
        nome   = row["nome"]
        barra  = int(row["barra"])
        p_kw   = float(row["potencia_kw"])
        classe = row["classe"]

        if classe == "PVSystem":
            dss.Command(
                f"New PVSystem.{nome} Bus1={barra} Phases=3 Conn=Wye "
                f"kV=24.9 kVA={p_kw:.2f} Pmpp={p_kw:.2f} irradiance=1 daily={irr_ls}"
            )
            print(f"[PV]   '{nome}' | bus={barra} | Pmpp={p_kw:.1f} kW")
            n_pv += 1

        elif classe == "Storage":
            e_kwh = float(row["capacidade_kwh"])
            dss.Command(
                f"New Storage.{nome} Bus1={barra} Phases=3 Conn=Wye "
                f"kV=24.9 kWrated={p_kw:.2f} kWhrated={e_kwh:.2f} "
                f"%stored=50 State=Idling daily=BESS_fixo"
            )
            print(f"[BESS] '{nome}' | bus={barra} | Pnom={p_kw:.1f} kW | E={e_kwh:.1f} kWh")
            n_bess += 1

    dss.Command("CalcVoltageBases")
    print(f"[Cenário] pen={pen_pct}% | real={id_realizacao} | PV={n_pv} | BESS={n_bess}")
    return n_pv, n_bess


# ---------------------------------------------------------------------------
# COMPILAR A REDE
# ---------------------------------------------------------------------------
dss.Command(f"compile [{BASE_DSS}]")
add_realizacao_elements(PEN_PCT, ID_REALIZACAO)

# Garante que o modo diário está ativo
dss.Command("set mode=daily")
dss.Command("set stepsize=1h")
dss.Command("set number=1")

# Desativa o controle interno de potência reativa dos inversores (se houver InvControl)
# para que APENAS o nosso loop externo gerencie o Q
inv_controls = [e for e in dss.Circuit.AllElementNames() if e.lower().startswith("invcontrol.")]
for ic in inv_controls:
    ic_name = ic.split(".")[1]
    dss.Command(f"InvControl.{ic_name}.enabled=no")
    print(f"[InvControl] '{ic_name}' desabilitado.")

# ---------------------------------------------------------------------------
# PREPARAÇÃO PRÉ-LOOP
# ---------------------------------------------------------------------------
# Descobre os BESSs trifásicos
bess_dict = discover_bess_buses()
if not bess_dict:
    raise RuntimeError("Nenhum elemento Storage trifásico encontrado. "
                       "Verifique se os BESSs foram inseridos na rede.")

# Lista de nomes dos RegControls (pode ser vazia se não houver reguladores)
reg_names = dss.RegControls.AllNames() or []
print(f"[RegControl] {len(reg_names)} regulador(es) encontrado(s): {reg_names}")

# Gera o gráfico da curva Volt-VAr uma única vez
plot_voltvar_curve()

# Tolerância de convergência absoluta (kvar)
tol_kvar = TOL_VMAG * min(info["pnom_kw"] for info in bess_dict.values())

# ---------------------------------------------------------------------------
# ESTRUTURAS DE SAÍDA
# ---------------------------------------------------------------------------
records      = []   # uma linha por (hora, BESS)
hour_records = []   # uma linha por hora (violações)

# ---------------------------------------------------------------------------
# LOOP HORÁRIO PRINCIPAL
# ---------------------------------------------------------------------------
for hour in range(TOTAL_HOURS):

    print(f"\n{'='*60}")
    print(f"Hora {hour+1:02d}/{TOTAL_HOURS}")
    print(f"{'='*60}")

    # -----------------------------------------------------------------
    # PASSO NATURAL: reguladores livres, Q = 0
    # Captura os taps que o regulador escolheria sem controle primário
    # -----------------------------------------------------------------
    enable_regulators(reg_names)
    for name in bess_dict:
        set_bess_kvar(name, 0.0)
    dss.Command("solve")
    tap_snapshot = capture_taps(reg_names)
    disable_regulators(reg_names)

    if reg_names:
        for reg, info in tap_snapshot.items():
            print(f"  [Tap hora {hour+1:02d}] {reg}: tap={info['tap']:.6f}")

    # -----------------------------------------------------------------
    # LOOP INTERNO DE CONVERGÊNCIA VOLT-VAr
    # -----------------------------------------------------------------
    q_current = {name: 0.0 for name in bess_dict}
    converged = False

    for iteration in range(MAX_ITER):

        # 1) Aplica os Q da iteração anterior
        for name, q_kvar in q_current.items():
            set_bess_kvar(name, q_kvar)

        # 2) Resolve o fluxo de potência
        dss.Command("solve")

        # 3) Restaura taps naturais desta hora
        restore_tap(tap_snapshot)

        # 4) Calcula novo Q para cada BESS e verifica convergência
        q_new  = {}
        max_dq = 0.0
        for name, info in bess_dict.items():
            v_pu  = get_bus_vmag_pu(info["bus"])
            q_ref = volt_var_curve(v_pu, info["pnom_kw"])
            q_new[name] = q_ref
            max_dq = max(max_dq, abs(q_ref - q_current[name]))

        print(f"  iter {iteration+1:3d} | max ΔQ = {max_dq:.4f} kvar")
        q_current = q_new

        if max_dq < tol_kvar:
            print(f"  → Convergiu em {iteration+1} iterações.")
            converged = True
            break

    if not converged:
        print(f"  ⚠ Não convergiu em {MAX_ITER} iterações (hora {hour+1}).")

    # -----------------------------------------------------------------
    # APLICA Q FINAL E FAZ SOLVE DEFINITIVO DO TIMESTEP
    # -----------------------------------------------------------------
    for name, q_kvar in q_current.items():
        set_bess_kvar(name, q_kvar)
    dss.Command("solve")
    restore_tap(tap_snapshot)

    # -----------------------------------------------------------------
    # COLETA RESULTADOS POR BESS
    # -----------------------------------------------------------------
    for name, info in bess_dict.items():
        v_pu   = get_bus_vmag_pu(info["bus"])
        q_kvar = q_current[name]
        q_max  = Q_MAX_PU * info["pnom_kw"]
        records.append({
            "hour":    hour + 1,
            "bess":    name,
            "bus":     info["bus"],
            "v_pu":    round(v_pu, 6),
            "q_kvar":  round(q_kvar, 4),
            "q_pu":    round(q_kvar / q_max, 4) if q_max > 0 else 0.0,
            "q_active": q_kvar != 0.0,
            "pnom_kw": info["pnom_kw"],
        })
        print(f"  [BESS {name}] bus={info['bus']} | V={v_pu:.4f} p.u. | "
              f"Q={q_kvar:.2f} kvar ({'ativo' if q_kvar != 0 else 'zona morta'})")

    # -----------------------------------------------------------------
    # COLETA VIOLAÇÕES DE TENSÃO DESTA HORA
    # -----------------------------------------------------------------
    n_viol, viol_dict = count_voltage_violations()
    print(f"  [Violações] hora {hour+1:02d}: {n_viol} barra(s) com V fora de [0.95, 1.05] p.u.")
    if viol_dict:
        for b, v in sorted(viol_dict.items()):
            print(f"    barra {b}: {v:.4f} p.u.")
    hour_records.append({"hour": hour + 1, "n_violations": n_viol, "violations": viol_dict})

# ---------------------------------------------------------------------------
# FUNÇÕES DE VISUALIZAÇÃO PÓS-SIMULAÇÃO
# ---------------------------------------------------------------------------

def plot_q_activation(df: pd.DataFrame):
    """Heatmap de Q/Qmax por BESS e hora (azul = capacitivo, vermelho = indutivo)."""
    pivot = df.pivot_table(index="bess", columns="hour", values="q_pu", aggfunc="first")
    bess_labels = pivot.index.tolist()
    hours       = pivot.columns.tolist()

    fig, ax = plt.subplots(figsize=(max(10, len(hours) * 0.5), max(4, len(bess_labels) * 0.6)))
    im = ax.imshow(pivot.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels(hours, fontsize=8)
    ax.set_yticks(range(len(bess_labels)))
    ax.set_yticklabels(bess_labels, fontsize=8)
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("BESS")
    ax.set_title(f"Ativação do controle Volt-VAr — pen={PEN_PCT}% / real={ID_REALIZACAO}\n"
                 f"Azul = capacitivo (↑V)   |   Vermelho = indutivo (↓V)   |   Branco = zona morta")

    # Anotações com valor de Q/Qmax
    for i, bess in enumerate(bess_labels):
        for j, h in enumerate(hours):
            val = pivot.loc[bess, h]
            if not np.isnan(val) and val != 0.0:
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=6, color="black")

    fig.colorbar(im, ax=ax, label="Q / Q$_{max}$")
    path = os.path.join(SCRIPT_DIR, "fig_q_activation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Heatmap de ativação Q salvo em: {path}")


def plot_violations(hr: pd.DataFrame):
    """
    Gráfico de violações de tensão por hora com controle primário ativo.
    Cada barra mostra o número de barras violando [0.95, 1.05] p.u.
    O nome e a tensão de cada barra violadora são anotados dentro da barra.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    hours  = hr["hour"].tolist()
    counts = hr["n_violations"].tolist()

    bars = ax.bar(hours, counts, color="steelblue", edgecolor="white", linewidth=0.5)

    # Linhas de referência
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(hours)
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Barras com violação de tensão")
    ax.set_title(
        f"Violações de tensão com controle primário ativo\n"
        f"pen={PEN_PCT}% / real={ID_REALIZACAO}  |  V_AGGREGATION={V_AGGREGATION!r}\n"
        f"Limites: V < 0.95 p.u. ou V > 1.05 p.u."
    )

    # Anota cada barra com "barra: tensão" das barras violadoras
    for bar, (_, row) in zip(bars, hr.iterrows()):
        viol = row["violations"]
        if not viol:
            continue

        # Monta o texto: uma linha por barra violadora, ordenada por tensão
        linhas = [f"{b}: {v:.3f}" for b, v in sorted(viol.items(), key=lambda x: x[1])]
        texto  = "\n".join(linhas)

        bar_h = bar.get_height()
        bar_x = bar.get_x() + bar.get_width() / 2

        # Posição: dentro da barra se couber, acima caso contrário
        y_pos   = bar_h / 2
        v_align = "center"
        color   = "white" if bar_h >= 2 else "black"
        y_above = bar_h + 0.05
        if bar_h < 1:
            y_pos, v_align, color = y_above, "bottom", "black"

        ax.text(
            bar_x, y_pos, texto,
            ha="center", va=v_align,
            fontsize=6, color=color,
            rotation=90,
            linespacing=1.3,
        )

    # Margem extra no topo para anotações acima das barras
    if counts:
        ax.set_ylim(0, max(counts) * 1.4 + 1)

    path = os.path.join(SCRIPT_DIR, "fig_violations_with_control.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Violações com controle salvo em: {path}")


# ---------------------------------------------------------------------------
# EXPORTA RESULTADOS E GERA GRÁFICOS
# ---------------------------------------------------------------------------
df       = pd.DataFrame(records)
df_hours = pd.DataFrame(hour_records)

out_csv = os.path.join(SCRIPT_DIR, f"voltvar_pen{PEN_PCT:03d}_real{ID_REALIZACAO:04d}.csv")
df.to_csv(out_csv, index=False)
print(f"\n[OK] Resultados por BESS salvos em: {out_csv}")
print(df.to_string(index=False))

plot_q_activation(df)
plot_violations(df_hours)