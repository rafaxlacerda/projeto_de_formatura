"""
Controle Primário Volt-VAr para BESSs na rede IEEE 34 barras (OpenDSS)
Baseado na ABNT NBR 16149:2013 - Seção 4.7.3 e Figura 2

Variável de controle: fator de potência (pf) dos inversores.
Curva Volt-VAr: tensão (p.u.) → fator de potência alvo
  - Zona morta [0.97, 1.03] p.u.  →  pf = 1.0
  - Saturação capacitiva ≤ 0.90   →  pf = +0.90 (injeta Q → eleva tensão)
  - Saturação indutiva  ≥ 1.10    →  pf = -0.90 (absorve Q → reduz tensão)
  - Transição: interpolação linear entre zona morta e saturação

Lógica por hora:
  1. Solve daily (mode=daily, 1h) com reguladores livres (Set ControlMode default)
  2. Lê tensões pós-tap
  3. Se há problemas de tensão → ativa VoltVar:
       Set ControlMode=OFF  (impede novos movimentos de tap)
       Loop iterativo (máx. MAX_VOLTVAR_ITER):
         - Aplica pf nos inversores pela curva VoltVar
         - Re-solve no mesmo patamar (stepsize=0.01s, não avança 1h)
         - Lê tensões novamente
         - Para se tensões OK ou pf saturado
  4. Registra último pf aplicado (para uso no controle secundário)
"""

from opendssdirect import dss
import pandas as pd
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR    = os.path.join(SCRIPT_DIR, "figuras_myvoltvarQ")
CSV_DIR    = os.path.join(SCRIPT_DIR, "resultados")
BASE_DSS   = os.path.join(SCRIPT_DIR, "..", "IEEE34bus", "IEEE34_original_with_loadshapes.dss")
MC_DIR     = os.path.join(SCRIPT_DIR, "..", "1.4_geracao_de_cenarios",
                           "resultados_monte_carlo", "realizacoes_sorteadas")

PEN_PCT       = 60
ID_REALIZACAO = 1
TOTAL_HOURS   = 24

V_AGGREGATION   = "min"   # "min" → fase mais crítica; "mean" → média das fases

# Limites da curva Volt-VAr (ABNT NBR 16149:2013)
V_DEADBAND_LOW  = 0.97
V_DEADBAND_HIGH = 1.03
V_SAT_LOW       = 0.90
V_SAT_HIGH      = 1.10

PF_MIN          = 0.90    # FP de saturação (≡ Q_MAX_PU = tan(arccos(0.90)) ≈ 0.4358)

MAX_VOLTVAR_ITER = 3     # máximo de iterações VoltVar por hora

# Limites de Q pela norma (em fração da potência nominal do BESS)
# Figura 2: Qmax/Pnom = 43,58 %  →  equivale a FP = 0,90
Q_MAX_PU = 0.4358           # p.u. de Pnom



# ---------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ---------------------------------------------------------------------------

def volt_var_pf_curve(v_pu: float) -> float:
    """
    Retorna o fator de potência alvo para o inversor a partir da tensão medida.

    Convenção OpenDSS (PVSystem e Storage discharging — convenção de gerador):
      0 < pf < 1  → capacitivo: injeta Q → eleva tensão   (para subtensão)
      pf = 1      → zona morta (sem reativo)
      pf < 0      → indutivo: absorve Q → reduz tensão     (para sobretensão)

    Para Storage charging (P < 0) o Q resultante inverte com o mesmo pf.
    O chamador é responsável por corrigir o sinal ao aplicar em BESS carregando.

    Curva:
      V ≤ V_SAT_LOW  → pf = +PF_MIN  (saturação capacitiva)
      V_SAT_LOW < V < V_DB_LOW → interpolação: PF_MIN→1.0, capacitivo (positivo)
      V_DB_LOW ≤ V ≤ V_DB_HIGH → pf = 1.0 (zona morta)
      V_DB_HIGH < V < V_SAT_HIGH → interpolação: −1.0→−PF_MIN, indutivo (negativo)
      V ≥ V_SAT_HIGH → pf = −PF_MIN  (saturação indutiva)
    """
    if v_pu <= V_SAT_LOW:
        return PF_MIN

    elif v_pu < V_DEADBAND_LOW:
        # Capacitivo → pf positivo; magnitude de PF_MIN (V_SAT_LOW) até 1.0 (V_DB_LOW)
        t = (v_pu - V_SAT_LOW) / (V_DEADBAND_LOW - V_SAT_LOW)
        fp_mag = PF_MIN + (1.0 - PF_MIN) * t
        return fp_mag

    elif v_pu <= V_DEADBAND_HIGH:
        return 1.0

    elif v_pu < V_SAT_HIGH:
        # Indutivo → pf negativo; magnitude de 1.0 (V_DB_HIGH) até PF_MIN (V_SAT_HIGH)
        t = (v_pu - V_DEADBAND_HIGH) / (V_SAT_HIGH - V_DEADBAND_HIGH)
        return -(1.0 - (1.0 - PF_MIN) * t)

    else:
        return -PF_MIN


def get_bus_vmag_pu(bus_name: str) -> float:
    """Tensão representativa da barra: menor fase (min) ou média (mean)."""
    dss.Circuit.SetActiveBus(bus_name)
    magnitudes = dss.Bus.puVmagAngle()[0::2]
    valid = [v for v in magnitudes if v > 0.1]
    if not valid:
        return 1.0
    return min(valid) if V_AGGREGATION == "min" else sum(valid) / len(valid)


def get_bus_vmin_vmax_pu(bus_name: str) -> tuple:
    """Retorna (v_min, v_max) entre as fases válidas da barra."""
    dss.Circuit.SetActiveBus(bus_name)
    magnitudes = dss.Bus.puVmagAngle()[0::2]
    valid = [v for v in magnitudes if v > 0.1]
    if not valid:
        return 1.0, 1.0
    return min(valid), max(valid)


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


def restore_tap(tap_snapshot: dict):
    """Reaplica os taps congelados para evitar deriva em solves iterativos."""
    for _, info in tap_snapshot.items():
        dss.Transformers.Name(info["transformer"])
        dss.Transformers.Wdg(info["winding"])
        dss.Transformers.Tap(info["tap"])


def discover_bess_buses() -> dict:
    """
    Descobre todos os Storage trifásicos na rede.
    Retorna {nome: {bus, pnom_kw}}.
    """
    bess_info = {}
    for el in dss.Circuit.AllElementNames():
        if not el.lower().startswith("storage."):
            continue
        dss.Circuit.SetActiveElement(el)
        name   = el.split(".")[1]
        bus    = dss.CktElement.BusNames()[0].split(".")[0].lower()
        phases = dss.CktElement.NumPhases()
        if phases < 3:
            print(f"[BESS] '{name}' em '{bus}' ignorado (monofásico/bifásico).")
            continue
        dss.Command(f"? Storage.{name}.kWrated")
        p_nom = float(dss.Text.Result()) if dss.Text.Result() else 0.0
        bess_info[name] = {"bus": bus, "pnom_kw": p_nom}
        print(f"[BESS] '{name}' | bus='{bus}' | Pnom={p_nom:.1f} kW | fases={phases}")
    return bess_info


def discover_pv_buses() -> dict:
    """
    Descobre todos os PVSystem trifásicos na rede.
    Retorna {nome: {bus, pmpp_kw}}.
    """
    pv_info = {}
    for el in dss.Circuit.AllElementNames():
        if not el.lower().startswith("pvsystem."):
            continue
        dss.Circuit.SetActiveElement(el)
        name   = el.split(".")[1]
        bus    = dss.CktElement.BusNames()[0].split(".")[0].lower()
        phases = dss.CktElement.NumPhases()
        if phases < 3:
            print(f"[PV]   '{name}' em '{bus}' ignorado (monofásico/bifásico).")
            continue
        dss.Command(f"? PVSystem.{name}.Pmpp")
        p_nom = float(dss.Text.Result()) if dss.Text.Result() else 0.0
        pv_info[name] = {"bus": bus, "pmpp_kw": p_nom}
        print(f"[PV]   '{name}' | bus='{bus}' | Pmpp={p_nom:.1f} kW | fases={phases}")
    return pv_info


def set_bess_dispatch(name: str, mult: float, pnom_kw: float):
    """Define State e kW do BESS para a hora corrente."""
    if mult < 0:
        dss.Command(f"Storage.{name}.State=Charging kW={abs(mult) * pnom_kw:.4f}")
    elif mult > 0:
        dss.Command(f"Storage.{name}.State=Discharging kW={mult * pnom_kw:.4f}")
    else:
        dss.Command(f"Storage.{name}.State=Idling")


def carregar_cargas_base() -> dict:
    """Lê kW/kvar base de cada Load antes do primeiro solve."""
    cargas = {}
    nome = dss.Loads.First()
    while nome:
        dss.Circuit.SetActiveElement(f"Load.{nome}")
        bus_str = dss.CktElement.BusNames()[0].split(".")[0]
        try:
            bus_int = int(bus_str)
        except ValueError:
            bus_int = None
        cargas[nome] = {
            "kw":   dss.Loads.kW(),
            "kvar": dss.Loads.kvar(),
            "bus":  bus_int,
        }
        nome = dss.Loads.Next()
    return cargas


def aplicar_fatores_carga(hora: int, cargas_base: dict, fatores_carga: dict):
    """Escala cada Load pelo fator de incerteza da realização para a hora dada."""
    for nome, carga in cargas_base.items():
        fator = fatores_carga.get(carga["bus"], [1.0] * 24)[hora]
        dss.Command(
            f"Edit Load.{nome} kW={carga['kw'] * fator:.6f} kvar={carga['kvar'] * fator:.6f}"
        )


def get_bess_soc(name: str) -> tuple:
    """Retorna (%stored, state_str) do BESS após um solve."""
    dss.Command(f"? Storage.{name}.%stored")
    soc = float(dss.Text.Result()) if dss.Text.Result() else 0.0
    dss.Command(f"? Storage.{name}.State")
    state = dss.Text.Result().strip() if dss.Text.Result() else "Unknown"
    return soc, state


def count_voltage_violations() -> tuple:
    """
    Retorna (n_violacoes, dict{bus: info}) para barras fora de [0.95, 1.05] p.u.
    """
    violations = {}
    for bus in dss.Circuit.AllBusNames():
        v_min, v_max = get_bus_vmin_vmax_pu(bus)
        under = v_min < 0.95
        over  = v_max > 1.05001
        if under or over:
            violations[bus] = {
                "v_min": round(v_min, 6),
                "v_max": round(v_max, 6),
                "under": under,
                "over":  over,
            }
    return len(violations), violations


def add_realizacao_elements(pen_pct: int, id_realizacao: int) -> tuple:
    """
    Lê os CSVs da realização Monte Carlo e adiciona PVSystem e Storage à rede.
    Retorna (n_pv, n_bess, bess_prof, fatores_carga).
    """
    pasta = os.path.join(MC_DIR, f"pen_{pen_pct:03d}pct")

    df_irr  = pd.read_csv(os.path.join(pasta, "02_perfis_irradiancia.csv"),
                          sep=";", decimal=",")
    irr_row  = df_irr[df_irr["id_realizacao"] == id_realizacao].iloc[0]
    irr_vals = [float(irr_row[f"h{h:02d}"]) for h in range(24)]
    irr_str  = " ".join(f"{v:.4f}" for v in irr_vals)
    irr_ls   = f"PV_irr_real{id_realizacao}"
    dss.Command(f"New LoadShape.{irr_ls} npts=24 interval=1 mult=[{irr_str}]")

    bess_prof = [0.0] * 24
    for h in range(10, 15):
        bess_prof[h] = -1.0
    for h in range(18, 22):
        bess_prof[h] =  1.0
    bess_str = " ".join(f"{v:.1f}" for v in bess_prof)
    dss.Command(f"New LoadShape.BESS_fixo npts=24 interval=1 mult=[{bess_str}]")

    df_fat = pd.read_csv(os.path.join(pasta, "03_fatores_incerteza_carga.csv"),
                         sep=";", decimal=",")
    df_fat_real = df_fat[df_fat["id_realizacao"] == id_realizacao]
    fatores_carga = {
        int(row["barra"]): [float(row[f"h{h:02d}"]) for h in range(24)]
        for _, row in df_fat_real.iterrows()
    }

    df_elem = pd.read_csv(os.path.join(pasta, "06_elementos_opendss.csv"),
                          sep=";", decimal=",")
    df_real = df_elem[df_elem["id_realizacao"] == id_realizacao]

    n_pv = n_bess = 0
    for _, row in df_real.iterrows():
        nome   = row["nome"]
        barra  = int(row["barra"])
        p_kw   = float(row["potencia_kw"])
        classe = row["classe"]

        dss.Circuit.SetActiveBus(str(barra))
        kv_base_ln = dss.Bus.kVBase()
        kv_ll = kv_base_ln * (3 ** 0.5) if kv_base_ln > 0 else 24.9

        if classe == "PVSystem":
            dss.Command(
                f"New PVSystem.{nome} Bus1={barra} Phases=3 Conn=Wye "
                f"kV={kv_ll:.4f} kVA={p_kw:.2f} Pmpp={p_kw:.2f} irradiance=1 daily={irr_ls}"
            )
            print(f"[PV]   '{nome}' | bus={barra} | kV={kv_ll:.4f} | Pmpp={p_kw:.1f} kW")
            n_pv += 1

        elif classe == "Storage":
            e_kwh = float(row["capacidade_kwh"])
            dss.Command(
                f"New Storage.{nome} Bus1={barra} Phases=3 Conn=Wye "
                f"kV={kv_ll:.4f} kWrated={p_kw:.2f} kWhrated={e_kwh:.2f} "
                f"%stored=0 dispmode=follow daily=BESS_fixo"
            )
            print(f"[BESS] '{nome}' | bus={barra} | kV={kv_ll:.4f} | Pnom={p_kw:.1f} kW | E={e_kwh:.1f} kWh")
            n_bess += 1

    dss.Command("CalcVoltageBases")
    print(f"[Cenário] pen={pen_pct}% | real={id_realizacao} | PV={n_pv} | BESS={n_bess} "
          f"| cargas com fator={len(fatores_carga)}")
    return n_pv, n_bess, bess_prof, fatores_carga


# ---------------------------------------------------------------------------
# FUNÇÕES DE VISUALIZAÇÃO
# ---------------------------------------------------------------------------

def _pf_to_display(pf: float) -> float:
    """
    Transforma o FP em coordenada de exibição centrada em 0 (= FP 1,0).

    Convenção (nova, PVSystem / Storage discharging):
      Capacitivo: pf ∈ (PF_MIN ; 1,0)  →  display ∈ (+PF_MIN ; 0)
      Zona morta: pf = 1,0              →  display = 0
      Indutivo:   pf ∈ (−1,0 ; −PF_MIN) → display ∈ (0 ; −PF_MIN)
    """
    if abs(pf - 1.0) < 1e-9:
        return 0.0
    if pf > 0:  # capacitivo: pf ∈ (PF_MIN, 1) → display ∈ (0, +PF_MIN)
        return (1.0 - pf) / (1.0 - PF_MIN) * PF_MIN
    else:       # indutivo: pf ∈ (-1, -PF_MIN) → display ∈ (-PF_MIN, 0)
        return (pf + 1.0) / (1.0 - PF_MIN) * (-PF_MIN)

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

    path = os.path.join(FIG_DIR, "fig_voltvar_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Curva Volt-VAr salva em: {path}")


def plot_voltvar_pf_curve():
    """Gráfico da curva VoltVar: V (p.u.) × FP, com FP 1,0 no centro do eixo Y."""
    v_range = np.linspace(0.85, 1.15, 500)
    pf_vals = [volt_var_pf_curve(v) for v in v_range]

    # Converte para coordenadas de exibição (centro = 0 ≡ FP 1,0)
    dy_vals = [_pf_to_display(pf) for pf in pf_vals]

    margin = PF_MIN * 1.2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(v_range, dy_vals, color="steelblue", linewidth=2, label="Curva Volt-VAr")
    ax.axhline(0, color="gray", linewidth=1.0, linestyle="--")  # FP = 1,0

    for v in [V_SAT_LOW, V_DEADBAND_LOW, V_DEADBAND_HIGH, V_SAT_HIGH]:
        ax.axvline(v, color="tomato", linewidth=1, linestyle=":", alpha=0.8)
        ax.text(v, -margin * 0.06, f"{v:.2f}", ha="center", va="top",
                fontsize=8, color="tomato")

    ax.fill_betweenx([-margin, margin], V_DEADBAND_LOW, V_DEADBAND_HIGH,
                     alpha=0.1, color="gray", label="Zona morta")

    # Ticks customizados: posição real no eixo → label legível
    ax.set_yticks([-PF_MIN, 0, PF_MIN])
    ax.set_yticklabels([
        f"−{PF_MIN} (indutivo ↓V)",
        f"1,0 (zona morta)",
        f"+{PF_MIN} (capacitivo ↑V)",
    ])
    ax.set_ylim(-margin, margin)

    ax.set_xlabel("Tensão na barra (p.u.)")
    ax.set_ylabel("Fator de Potência")
    ax.set_title("Curva Volt-VAr — Tensão × Fator de Potência  (ABNT NBR 16149:2013)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    path = os.path.join(FIG_DIR, "fig_voltvar_pf_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Curva VoltVar (FP) salva em: {path}")


def plot_pf_activation(df: pd.DataFrame):
    """Heatmap do kVAr aplicado por DER (BESS e PV) e hora (azul = capacitivo, vermelho = indutivo)."""
    pivot = df.pivot_table(index="der", columns="hour", values="q_voltvar_kvar", aggfunc="first")
    der_labels = pivot.index.tolist()
    hours      = pivot.columns.tolist()

    vals   = pivot.values.astype(float)
    q_abs  = np.nanmax(np.abs(vals))
    clim   = q_abs if q_abs > 1e-6 else 1.0

    fig, ax = plt.subplots(figsize=(max(10, len(hours) * 0.5), max(4, len(der_labels) * 0.6)))
    # q > 0 = capacitivo → azul; q < 0 = indutivo → vermelho; 0 = zona morta → branco
    im = ax.imshow(vals, cmap="RdBu", vmin=-clim, vmax=clim, aspect="auto")

    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels(hours, fontsize=8)
    ax.set_yticks(range(len(der_labels)))
    ax.set_yticklabels(der_labels, fontsize=8)
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("DER")
    ax.set_title(f"kVAr aplicado pelo controle Volt-VAr (BESS + PV) — pen={PEN_PCT}% / real={ID_REALIZACAO}\n"
                 f"Azul = capacitivo (↑V, q>0)   |   Vermelho = indutivo (↓V, q<0)   |   Branco = zona morta (q=0)")

    for i, der in enumerate(der_labels):
        for j, h in enumerate(hours):
            val = pivot.loc[der, h]
            if not np.isnan(val) and abs(val) > 1e-6:
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                        fontsize=6, color="black")

    fig.colorbar(im, ax=ax, label="kVAr  (+ capacitivo / − indutivo)")
    path = os.path.join(FIG_DIR, "fig_pf_activation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Heatmap kVAr salvo em: {path}")


def plot_violations(hr: pd.DataFrame):
    """Gráfico de violações de tensão por hora."""
    fig, ax = plt.subplots(figsize=(14, 6))
    hours  = hr["hour"].tolist()
    counts = hr["n_violations"].tolist()
    bars   = ax.bar(hours, counts, color="steelblue", edgecolor="white", linewidth=1.0)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(hours)
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Barras com violação de tensão")
    ax.set_title(
        f"Violações de tensão com controle VoltVar (FP) ativo\n"
        f"pen={PEN_PCT}% / real={ID_REALIZACAO}  |  V_AGGREGATION={V_AGGREGATION!r}\n"
        f"Limites: V < 0.95 p.u. ou V > 1.05 p.u."
    )

    for bar, (_, row) in zip(bars, hr.iterrows()):
        viol = row["violations"]
        if not viol:
            continue
        n_viols = len(viol)
        bar_h   = bar.get_height()
        bar_x   = bar.get_x() + bar.get_width() / 2
        for i, (b, info) in enumerate(sorted(viol.items())):
            if info["under"] and info["over"]:
                bg = "orange"
            elif info["under"]:
                bg = "#fdd835"
            else:
                bg = "#e53935"
            if bar_h >= 1.0:
                y_pos = bar_h * (i + 0.5) / n_viols
                va    = "center"
            else:
                y_pos = bar_h + 0.05 + i * 0.28
                va    = "bottom"
            ax.text(bar_x, y_pos, str(b), ha="center", va=va, fontsize=6,
                    rotation=45, color="black",
                    bbox=dict(facecolor=bg, alpha=0.85, pad=1.2,
                              edgecolor="none", boxstyle="round,pad=0.25"))

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#fdd835", label="Subtensão  (V_min < 0.95 p.u.)"),
        Patch(facecolor="#e53935", label="Sobretensão (V_max > 1.05 p.u.)"),
        Patch(facecolor="orange",  label="Sub e sobretensão"),
    ], loc="upper right", fontsize=8, framealpha=0.9)
    if counts:
        ax.set_ylim(0, max(counts) + 1)

    path = os.path.join(FIG_DIR, "fig_violations_with_control.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Violações com controle salvo em: {path}")


def plot_bess_soc(df: pd.DataFrame):
    """SOC (%) por BESS ao longo das 24h, com marcadores nas horas de FP ativo."""
    df = df[df["der_type"] == "BESS"]
    bess_names = df["der"].unique().tolist()
    n = len(bess_names)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, bess in zip(axes, bess_names):
        sub   = df[df["der"] == bess].sort_values("hour")
        hours = sub["hour"].tolist()
        soc   = sub["soc_pct"].tolist()
        q     = sub["q_voltvar_kvar"].tolist()

        ax.plot(hours, soc, color="steelblue", linewidth=2, marker="o", markersize=4)
        ax.set_ylim(0, 110)
        ax.set_ylabel("SOC (%)")
        ax.set_title(f"BESS: {bess}")
        ax.grid(True, linestyle="--", alpha=0.4)

        for h, s, qi in zip(hours, soc, q):
            if qi > 1e-6:   # capacitivo: injeta Q → ↑V
                ax.plot(h, s, marker="^", color="blue", markersize=8, zorder=5)
            elif qi < -1e-6:  # indutivo: absorve Q → ↓V
                ax.plot(h, s, marker="v", color="red", markersize=8, zorder=5)

        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0], [0], color="steelblue", linewidth=2, label="SOC (%)"),
            Line2D([0], [0], marker="^", color="blue", linestyle="None", markersize=8, label="Q capacitivo (↑V)"),
            Line2D([0], [0], marker="v", color="red",  linestyle="None", markersize=8, label="Q indutivo (↓V)"),
        ], fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Hora do dia")
    axes[0].set_title(
        f"Estado de carga (SOC) dos BESSs — pen={PEN_PCT}% / real={ID_REALIZACAO}\n"
        f"Nota: FP não consome energia da bateria neste modelo\n"
        + axes[0].get_title()
    )
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_bess_soc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] SOC dos BESSs salvo em: {path}")


def _plot_bus_voltage_profile(df_v: pd.DataFrame, col: str, limit: float,
                               limit_label: str, limit_color: str,
                               title_suffix: str, filename: str):
    buses = sorted(df_v["bus"].unique())
    hours = sorted(df_v["hour"].unique())
    cmap  = plt.cm.get_cmap("rainbow", len(buses))
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, bus in enumerate(buses):
        sub = df_v[df_v["bus"] == bus].sort_values("hour")
        ax.plot(sub["hour"], sub[col], color=cmap(i), linewidth=1.0,
                marker=".", markersize=3, label=bus)
    ax.axhline(limit, color=limit_color, linewidth=1.8, linestyle="--",
               label=f"{limit_label} = {limit} p.u.")
    ax.axhline(1.00, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_xticks(hours)
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Tensão (p.u.)")
    ax.set_title(
        f"Tensão {title_suffix} nas barras da rede\n"
        f"pen={PEN_PCT}% / real={ID_REALIZACAO}  |  Controle VoltVar (FP) ativo"
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=6,
              ncol=1, framealpha=0.9, title="Barras")
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {title_suffix} de todas as barras salvo em: {path}")


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
# LOOP HORÁRIO PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    # --- Compilar rede ---
    dss.Command("Clear")
    dss.Command(f"compile [{BASE_DSS}]")
    _, _, bess_prof, fatores_carga = add_realizacao_elements(PEN_PCT, ID_REALIZACAO)

    dss.Command("set mode=daily")
    dss.Command("set stepsize=1h")
    dss.Command("set number=1")

    # Desabilita InvControls externos para que apenas nosso loop gerencie o FP
    for el in dss.Circuit.AllElementNames():
        if el.lower().startswith("invcontrol."):
            ic_name = el.split(".")[1]
            dss.Command(f"InvControl.{ic_name}.enabled=no")
            print(f"[InvControl] '{ic_name}' desabilitado.")

    # --- Preparação pré-loop ---
    bess_dict = discover_bess_buses()
    if not bess_dict:
        raise RuntimeError("Nenhum elemento Storage trifásico encontrado.")

    pv_dict = discover_pv_buses()

    # DER unificado: BESS + PV — VoltVar atua em ambos pelo mesmo FP
    der_dict: dict[str, dict] = {}
    for name, info in bess_dict.items():
        der_dict[name] = {"bus": info["bus"], "pnom_kw": info["pnom_kw"], "type": "BESS"}
    for name, info in pv_dict.items():
        der_dict[name] = {"bus": info["bus"], "pnom_kw": info["pmpp_kw"], "type": "PV"}

    cargas_base = carregar_cargas_base()
    reg_names   = dss.RegControls.AllNames() or []
    print(f"[Cargas] {len(cargas_base)} Load(s) | [RegControl] {len(reg_names)} regulador(es): {reg_names}")
    print(f"[DERs] {len(bess_dict)} BESS + {len(pv_dict)} PV = {len(der_dict)} total")

    # Último kVAr aplicado por DER — usado pelo controle secundário
    last_q = {name: 0.0 for name in der_dict}

    plot_voltvar_pf_curve()
    plot_voltvar_curve()

    records             = []
    hour_records        = []
    bus_voltage_records = []
    last_q_records     = []

    # --- Loop 24h ---
    for hour in range(TOTAL_HOURS):

        print(f"\n{'='*60}")
        print(f"Hora {hour+1:02d}/{TOTAL_HOURS}")
        print(f"{'='*60}")

        aplicar_fatores_carga(hour, cargas_base, fatores_carga)

        # -----------------------------------------------------------------
        # SOLVE DIÁRIO — reguladores livres (Set Controlmode=ON)
        # mode=daily mantém RegControls ativos por padrão
        # -----------------------------------------------------------------
        dss.Command("Set ControlMode=STATIC")
        dss.Command("solve")

        tap_snapshot = capture_taps(reg_names)
        for reg, info in tap_snapshot.items():
            print(f"  [Tap hora {hour+1:02d}] {reg}: tap={info['tap']:.6f}")

        # Lê tensões pós-tap nos barramentos de todos os DERs (BESS + PV)
        v_measured = {name: get_bus_vmag_pu(info["bus"]) for name, info in der_dict.items()}

        # Calcula FP alvo pela curva VoltVar para cada DER
        q_new = {
            name: volt_var_curve(v_measured[name], info["pnom_kw"])
            for name, info in der_dict.items()
        }
        
        #pf_new = {name: volt_var_pf_curve(v_measured[name]) for name in der_dict}
        #voltvar_active = any(pf != 1.0 for pf in pf_new.values())
        voltvar_active = any(abs(q) > 1e-6 for q in q_new.values())

        v_final        = dict(v_measured)
        n_viol_final, viol_final = count_voltage_violations()
        iter_count     = 0

        # -----------------------------------------------------------------
        # CONTROLE VOLTVAR ITERATIVO (se há problema de tensão)
        # -----------------------------------------------------------------
        if voltvar_active:
            # Set Controlmode=OFF — impede novos movimentos de tap
            dss.Command("Set ControlMode=OFF")
            
            tap_snapshot2 = capture_taps(reg_names)
            for reg, info in tap_snapshot2.items():
                print(f"  [Tap hora {hour+1:02d} controlmode off] {reg}: tap={info['tap']:.6f}")


            for iteration in range(MAX_VOLTVAR_ITER):
                iter_count = iteration + 1

                for name, q in q_new.items():
                    der_type = der_dict[name]["type"]
                    if der_type == "BESS":
                        dss.Command(f"Storage.{name}.kVAr={q:.4f}")
                    else:  # PV
                        dss.Command(f"PVSystem.{name}.kVAr={q:.4f}")
                    last_q[name] = q

                # Re-solve no mesmo patamar horário (stepsize=0.01s → não avança 1h)
                dss.Command("set stepsize=0.01s")
                dss.Command("solve")
                dss.Command("set stepsize=1h")

                # Lê tensões pós-iteração para todos os DERs
                v_new = {name: get_bus_vmag_pu(info["bus"]) for name, info in der_dict.items()}
                n_viol_new, viol_new = count_voltage_violations()

                print(f"  [VoltVar iter {iter_count}] violações={n_viol_new} | "
                      + " | ".join(f"{n}({der_dict[n]['type']}): V={v:.4f} q={q_new[n]:+.3f}"
                                   for n, v in v_new.items()))

                v_final      = v_new
                n_viol_final = n_viol_new
                viol_final   = viol_new

                # Critério de parada: sem violações ou Q saturado em todos os DERs
                q_new = {name: volt_var_curve(v_new[name], der_dict[name]["pnom_kw"]) for name in der_dict}
                #all_saturated = all(abs(q) <= Q_MIN + 0.001 for q in q_new.values())
                if n_viol_new == 0: #or all_saturated:
                    break

        # -----------------------------------------------------------------
        # Coleta tensões de TODAS as barras (estado final da hora)
        # -----------------------------------------------------------------
        tap_snapshot3 = capture_taps(reg_names)
        for reg, info in tap_snapshot3.items():
            print(f"  [Tap hora {hour+1:02d} pos VoltVar] {reg}: tap={info['tap']:.6f}")

        for bus in dss.Circuit.AllBusNames():
            dss.Circuit.SetActiveBus(bus)
            mags  = dss.Bus.puVmagAngle()[0::2]
            valid = [v for v in mags if v > 0.1]
            if valid:
                bus_voltage_records.append({
                    "hour":  hour + 1,
                    "bus":   bus,
                    "v_min": round(min(valid), 6),
                    "v_max": round(max(valid), 6),
                })

        # SOC vem do solve diário (o iterativo com 0.01s não atualiza energia)
        soc_info = {name: get_bess_soc(name) for name in bess_dict}

        # -----------------------------------------------------------------
        # Coleta resultados por DER (BESS e PV)
        # -----------------------------------------------------------------
        for name, info in der_dict.items():
            q_aplicado = last_q[name]
            der_type   = info["type"]
            pnom       = info["pnom_kw"]
            if der_type == "BESS":
                soc, bess_state = soc_info[name]
            else:
                soc, bess_state = None, "N/A"

            q_pct = round(q_aplicado / pnom * 100, 2) if pnom else None

            records.append({
                "hour":            hour + 1,
                "der":             name,
                "der_type":        der_type,
                "bus":             info["bus"],
                "v_pu_preVoltVar": round(v_measured[name], 6),
                "v_pu_posVoltVar": round(v_final[name], 6),
                "voltvar_active":  voltvar_active,
                "iter_count":      iter_count,
                "pnom_kw":         pnom,
                "q_voltvar_kvar":  round(q_aplicado, 4),
                "q/pn(%)":         q_pct,
                "soc_pct":         round(soc, 2) if soc is not None else None,
                "bess_state":      bess_state,
            })
            last_q_records.append({
                "hour":           hour + 1,
                "der":            name,
                "der_type":       der_type,
                "q_voltvar_kvar": round(q_aplicado, 4),
            })
            status = "ativo" if abs(q_aplicado) > 1e-6 else "zona morta"
            if der_type == "BESS":
                print(f"  [BESS {name}] V_pre={v_measured[name]:.4f} | V_pos={v_final[name]:.4f} | "
                      f"q={q_aplicado:+.2f} kVAr ({q_pct:+.1f}%) ({status}) | iter={iter_count} | SOC={soc:.1f}% [{bess_state}]")
            else:
                print(f"  [PV   {name}] V_pre={v_measured[name]:.4f} | V_pos={v_final[name]:.4f} | "
                      f"q={q_aplicado:+.2f} kVAr ({q_pct:+.1f}%) ({status}) | iter={iter_count}")

        # -----------------------------------------------------------------
        # Violações de tensão (estado final da hora)
        # -----------------------------------------------------------------
        print(f"  [Violações] hora {hour+1:02d}: {n_viol_final} barra(s) fora de [0.95, 1.05] p.u.")
        if viol_final:
            for b, info in sorted(viol_final.items()):
                partes = []
                if info["under"]: partes.append(f"↓V_min={info['v_min']:.4f}")
                if info["over"]:  partes.append(f"↑V_max={info['v_max']:.4f}")
                print(f"    barra {b}: {', '.join(partes)}")
        hour_records.append({
            "hour": hour + 1,
            "n_violations": n_viol_final,
            "violations":   viol_final,
        })

    # -----------------------------------------------------------------
    # Exporta resultados e gera gráficos
    # -----------------------------------------------------------------
    df       = pd.DataFrame(records)
    df_hours = pd.DataFrame(hour_records)
    df_q    = pd.DataFrame(last_q_records)

    out_csv = os.path.join(CSV_DIR, f"myvoltvar_pen{PEN_PCT:03d}_real{ID_REALIZACAO:04d}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Resultados por BESS salvos em: {out_csv}")
    print(df.to_string(index=False))

    out_q = os.path.join(CSV_DIR, f"last_q_pen{PEN_PCT:03d}_real{ID_REALIZACAO:04d}.csv")
    df_q.to_csv(out_q, index=False)
    print(f"[OK] Última Q por BESS/hora salvo em: {out_q}")

    plot_pf_activation(df)
    plot_violations(df_hours)
    plot_bess_soc(df)
    plot_all_bus_vmin(bus_voltage_records)
    plot_all_bus_vmax(bus_voltage_records)


if __name__ == "__main__":
    main()
