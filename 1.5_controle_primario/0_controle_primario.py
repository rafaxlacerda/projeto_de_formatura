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
#TODO: como influencia no tempo de carregamente/descarregamento do bess? comparar sem o controle primario
#TODO: o que acontece se pede injeção de potencia e a bateria não tem energia?

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
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
FIG_DIR       = os.path.join(SCRIPT_DIR, "figuras")
CSV_DIR       = os.path.join(SCRIPT_DIR, "resultados")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
BASE_DSS      = os.path.join(SCRIPT_DIR, "..", "IEEE34bus", "IEEE34_original_with_loadshapes.dss")
MC_DIR        = os.path.join(SCRIPT_DIR, "..", "1.4_geracao_de_cenarios",
                              "resultados_monte_carlo", "realizacoes_sorteadas")

# Selecione o cenário Monte Carlo desejado:
PEN_PCT       = 90   # nível de penetração em % (ex.: 50 → pen_050pct)
ID_REALIZACAO = 1    # ID da realização (1 a 50)

TOTAL_HOURS   = 24          # número de passos horários
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
    Tensão representativa da barra para a curva Volt-VAr.
    V_AGGREGATION = "min"  → menor fase (mais conservador para subtensão)
    V_AGGREGATION = "mean" → média das fases
    """
    dss.Circuit.SetActiveBus(bus_name)
    magnitudes = dss.Bus.puVmagAngle()[0::2]
    valid = [v for v in magnitudes if v > 0.1]
    if not valid:
        return 1.0
    return min(valid) if V_AGGREGATION == "min" else sum(valid) / len(valid)


def get_bus_vmin_vmax_pu(bus_name: str) -> tuple:
    """
    Retorna (v_min, v_max) entre as fases válidas da barra.
    Usado para detectar sub e sobretensão independentemente.
    """
    dss.Circuit.SetActiveBus(bus_name)
    magnitudes = dss.Bus.puVmagAngle()[0::2]
    valid = [v for v in magnitudes if v > 0.1]
    if not valid:
        return 1.0, 1.0
    return min(valid), max(valid)


def enable_regulators(reg_names: list):
    """Reabilita todos os RegControls para que ajustem taps automaticamente."""
    for reg in reg_names:
        #dss.Command(f"RegControl.{reg}.enabled=yes")
        dss.Command(f"RegControl.{reg}.maxtapchange = 16")



def disable_regulators(reg_names: list):
    """Desabilita todos os RegControls (congela taps na posição atual)."""
    for reg in reg_names:
        #dss.Command(f"RegControl.{reg}.enabled=no")
        dss.Command(f"RegControl.{reg}.maxtapchange = 0")



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


def set_bess_dispatch(name: str, mult: float, pnom_kw: float):
    """
    Define State e kW do BESS para a hora corrente antes do daily solve.
    O OpenDSS Storage não atualiza SOC automaticamente com State=Idling;
    o controle explícito aqui é necessário para o SOC evoluir corretamente.
      mult < 0 → Charging  (absorve da rede, SOC sobe)
      mult > 0 → Discharging (injeta na rede, SOC desce)
      mult = 0 → Idling
    """
    if mult < 0:
        dss.Command(f"Storage.{name}.State=Charging kW={abs(mult) * pnom_kw:.4f}")
    elif mult > 0:
        dss.Command(f"Storage.{name}.State=Discharging kW={mult * pnom_kw:.4f}")
    else:
        dss.Command(f"Storage.{name}.State=Idling")


def carregar_cargas_base() -> dict:
    """
    Lê kW/kvar base de cada Load logo após a compilação da rede.
    Deve ser chamada antes do primeiro solve para capturar os valores nominais.
    """
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
    """
    Escala cada Load pelo fator de incerteza da realização para a hora dada.
    O solve diário aplica o shape multiplier em cima desse valor base escalado:
      kW_efetivo = kW_base × fator_incerteza × shape_multiplier(hora)
    """
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
    Retorna (n_violacoes, dict{bus: info}) onde info contém v_min, v_max,
    e flags 'under' (v_min < 0.95) e 'over' (v_max > 1.05).
    Uma barra é violadora se qualquer fase estiver fora do intervalo [0.95, 1.05].
    """
    violations = {}
    for bus in dss.Circuit.AllBusNames():
        v_min, v_max = get_bus_vmin_vmax_pu(bus)
        under = v_min < 0.95
        over  = v_max > 1.05
        if under or over:
            violations[bus] = {
                "v_min": round(v_min, 6),
                "v_max": round(v_max, 6),
                "under": under,
                "over":  over,
            }
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

    path = os.path.join(FIG_DIR, "fig_voltvar_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Curva Volt-VAr salva em: {path}")


def add_realizacao_elements(pen_pct: int, id_realizacao: int) -> tuple:
    """
    Lê os CSVs da realização Monte Carlo e adiciona PVSystem e Storage
    à rede já compilada.
    Retorna (n_pv, n_bess, bess_prof, fatores_carga) onde:
      bess_prof    — lista 24h de multiplicadores de despacho (−1=carga, +1=descarga)
      fatores_carga — {barra_int: [fator_h00, ..., fator_h23]}
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

    # Perfil de despacho do BESS — idêntico a BESS_PERFIL em simulacoes_fluxo_de_potencia_realizações.py
    # carga h10–14 (mult = −1), descarga h18–21 (mult = +1), ocioso no restante
    bess_prof = [0.0] * 24
    for h in range(10, 15):
        bess_prof[h] = -1.0
    for h in range(18, 22):
        bess_prof[h] =  1.0
    bess_str = " ".join(f"{v:.1f}" for v in bess_prof)
    dss.Command(f"New LoadShape.BESS_fixo npts=24 interval=1 mult=[{bess_str}]")

    # Fatores de incerteza de carga por barra e hora
    df_fat = pd.read_csv(os.path.join(pasta, "03_fatores_incerteza_carga.csv"),
                         sep=";", decimal=",")
    df_fat_real = df_fat[df_fat["id_realizacao"] == id_realizacao]
    fatores_carga = {
        int(row["barra"]): [float(row[f"h{h:02d}"]) for h in range(24)]
        for _, row in df_fat_real.iterrows()
    }

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

        # Tensão de base linha-a-linha da barra (kVBase é L-N; multiplica por √3 para L-L)
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
# COMPILAR A REDE
# ---------------------------------------------------------------------------
dss.Command("Clear")
dss.Command(f"compile [{BASE_DSS}]")
n_pv, n_bess, bess_prof, fatores_carga = add_realizacao_elements(PEN_PCT, ID_REALIZACAO)

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

# Captura kW/kvar base de cada Load (antes do primeiro solve)
cargas_base = carregar_cargas_base()
print(f"[Cargas] {len(cargas_base)} Load(s) com fator de incerteza disponível para {len(fatores_carga)} barra(s).")

# Lista de nomes dos RegControls (pode ser vazia se não houver reguladores)
reg_names = dss.RegControls.AllNames() or []
print(f"[RegControl] {len(reg_names)} regulador(es) encontrado(s): {reg_names}")

# Gera o gráfico da curva Volt-VAr uma única vez
plot_voltvar_curve()

# ---------------------------------------------------------------------------
# ESTRUTURAS DE SAÍDA
# ---------------------------------------------------------------------------
records            = []
hour_records       = []
bus_voltage_records = []

# ---------------------------------------------------------------------------
# LOOP HORÁRIO PRINCIPAL — CENÁRIO C (dois solves por hora)
#
# Timescale físico dentro de cada hora:
#   1. Solve daily  (reguladores ON)  → taps ajustam em minutos; tempo avança
#   2. Mede tensão pós-tap → calcula Q → aplica Q no BESS
#   3. Solve snapshot (reguladores OFF) → BESS age em ms, sem avançar o relógio
#   4. Restaura tap para o estado capturado após o Solve 1
#   As violações e os resultados são coletados do estado pós-snapshot (Solve 2).
# ---------------------------------------------------------------------------
enable_regulators(reg_names)
dss.Command("set mode=daily")
dss.Command("set stepsize=1h")
dss.Command("set number=1")

controle_primario = False  # True se qualquer BESS estiver com Q ativo na hora anterior

for hour in range(TOTAL_HOURS):

    print(f"\n{'='*60}")
    print(f"Hora {hour:02d}/{TOTAL_HOURS}") #Opendss começa com hora 0
    print(f"{'='*60}")

    aplicar_fatores_carga(hour, cargas_base, fatores_carga)

    # -----------------------------------------------------------------
    # SOLVE 1: daily.
    # Reguladores só atuam se nenhum BESS estiver com controle local ativo.
    # Se houver controle primário em curso, manter taps congelados para
    # evitar interferência entre tap-changer e injeção de reativo do BESS.
    # -----------------------------------------------------------------
    if controle_primario:
        disable_regulators(reg_names)
        print(f"  [RegControl] Controle primário ativo — reguladores BLOQUEADOS.")
    else:
        enable_regulators(reg_names)
    dss.Command("solve")

    tap_snapshot = capture_taps(reg_names)
    if reg_names:
        for reg, info in tap_snapshot.items():
            print(f"  [Tap hora {hour+1:02d}] {reg}: tap={info['tap']:.6f}")

    # -----------------------------------------------------------------
    # Mede tensão pós-tap e calcula novo Q pelo Volt-VAr
    # -----------------------------------------------------------------
    v_measured = {name: get_bus_vmag_pu(info["bus"]) for name, info in bess_dict.items()}

    q_new = {
        name: volt_var_curve(v_measured[name], info["pnom_kw"])
        for name, info in bess_dict.items()
    }
    for name, q in q_new.items():
        set_bess_kvar(name, q)

    # Atualiza flag: True se qualquer BESS estiver fora da zona morta
    controle_primario = any(q != 0.0 for q in q_new.values())

    # -----------------------------------------------------------------
    # SOLVE 2: snapshot, reguladores desabilitados.
    # BESS injeta Q instantaneamente; o relógio NÃO avança.
    # O SOC não é atualizado pelo snapshot — o valor correto vem do Solve 1.
    # -----------------------------------------------------------------
    disable_regulators(reg_names)
    dss.Command("set stepsize=0.01s")  # passo pequeno para estabilidade do snapshot
    #dss.Command("set mode=snapshot")
    dss.Command("solve")
    dss.Command("set stepsize=1h")
    #dss.Command("set mode=daily")

    # Restaura tap para o estado pós-Solve 1 (snapshot pode mover internamente)
    restore_tap(tap_snapshot)

    #tensão nas barras com bess pós-snapshot (com Q do BESS)
    v_controlled = {name: get_bus_vmag_pu(info["bus"]) for name, info in bess_dict.items()}

    # registra v_min e v_max de TODAS as barras (pós-snapshot, com Q do BESS)
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        mags = dss.Bus.puVmagAngle()[0::2]
        valid = [v for v in mags if v > 0.1]
        if valid:
            bus_voltage_records.append({
                "hour":  hour + 1,
                "bus":   bus,
                "v_min": round(min(valid), 6),
                "v_max": round(max(valid), 6),
            })

    # SOC vem do Solve 1 (daily atualiza energia; snapshot não)
    soc_info = {name: get_bess_soc(name) for name in bess_dict}

    # -----------------------------------------------------------------
    # Coleta resultados por BESS
    # -----------------------------------------------------------------
    for name, info in bess_dict.items():
        q_kvar  = q_new[name]
        q_max   = Q_MAX_PU * info["pnom_kw"]
        soc, bess_state = soc_info[name]
        soc_warn = soc < 5.0 and q_kvar != 0.0
        if soc_warn:
            print(f"  [AVISO] BESS {name}: SOC={soc:.1f}% (bateria quase vazia) "
                  f"mas Q={q_kvar:.1f} kvar solicitado. "
                  f"Inversor ainda pode fornecer Q (sem consumo de kWh).")
        records.append({
            "hour":       hour + 1,
            "bess":       name,
            "bus":        info["bus"],
            "v_pu":       round(v_controlled[name], 6),
            "q_kvar":     round(q_kvar, 4),
            "q_pu":       round(q_kvar / q_max, 4) if q_max > 0 else 0.0,
            "q_active":   q_kvar != 0.0,
            "pnom_kw":    info["pnom_kw"],
            "soc_pct":    round(soc, 2),
            "bess_state": bess_state,
            "soc_warn":   soc_warn,
        })
        print(f"  [BESS {name}] V_pós-tap={v_measured[name]:.4f} | V_pós-snapshot={v_controlled[name]:.4f} | "
              f"Q={q_kvar:.2f} kvar ({'ativo' if q_kvar != 0 else 'zona morta'}) | "
              f"SOC={soc:.1f}% [{bess_state}]")

    # -----------------------------------------------------------------
    # violações de tensão (pós-snapshot, estado final da hora)
    # -----------------------------------------------------------------
    n_viol, viol_dict = count_voltage_violations()
    print(f"  [Violações] hora {hour+1:02d}: {n_viol} barra(s) fora de [0.95, 1.05] p.u.")
    if viol_dict:
        for b, info in sorted(viol_dict.items()):
            partes = []
            if info["under"]: partes.append(f"↓V_min={info['v_min']:.4f}")
            if info["over"]:  partes.append(f"↑V_max={info['v_max']:.4f}")
            print(f"    barra {b}: {', '.join(partes)}")
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
    path = os.path.join(FIG_DIR, "fig_q_activation.png")
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

    bars = ax.bar(hours, counts, color="steelblue", edgecolor="white", linewidth=1.0)

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

    # Anota cada barra com chips coloridos por tipo de violação
    # Vermelho = subtensão (v_min < 0.95), Amarelo = sobretensão (v_max > 1.05)
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
                val = f"{b}"
            elif info["under"]:
                bg  = "#fdd835"   # vermelho
                val = f"{b}"
            else:
                bg  = "#e53935"   # amarelo
                val = f"{b}"

            # empilha dentro da barra ou acima dela
            if bar_h >= 1.0:
                y_pos = bar_h * (i + 0.5) / n_viols
                va    = "center"
            else:
                y_pos = bar_h + 0.05 + i * 0.28
                va    = "bottom"

            ax.text(
                bar_x, y_pos, val,
                ha="center", va=va,
                fontsize=6, rotation=45, color="black",
                bbox=dict(facecolor=bg, alpha=0.85, pad=1.2,
                          edgecolor="none", boxstyle="round,pad=0.25"),
            )

    # Legenda de cores
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="#fdd835", label="Subtensão  (V_min < 0.95 p.u.)"),
            Patch(facecolor="#e53935", label="Sobretensão (V_max > 1.05 p.u.)"),
            Patch(facecolor="orange",  label="Sub e sobretensão"),
        ],
        loc="upper right", fontsize=8, framealpha=0.9,
    )

    # Margem extra no topo para anotações acima das barras
    if counts:
        ax.set_ylim(0, max(counts) + 1)

    path = os.path.join(FIG_DIR, "fig_violations_with_control.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Violações com controle salvo em: {path}")


def plot_bess_soc(df: pd.DataFrame):
    """
    Linha de SOC (%) por BESS ao longo das 24h.
    Marcadores indicam horas com controle Q ativo: azul (capacitivo) e vermelho (indutivo).
    Nota: no modelo OpenDSS, kvar não consome kWh — SOC com e sem controle é idêntico.
    """
    bess_names = df["bess"].unique().tolist()
    n = len(bess_names)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, bess in zip(axes, bess_names):
        sub = df[df["bess"] == bess].sort_values("hour")
        hours = sub["hour"].tolist()
        soc   = sub["soc_pct"].tolist()
        q_pu  = sub["q_pu"].tolist()

        ax.plot(hours, soc, color="steelblue", linewidth=2, marker="o", markersize=4, label="SOC (%)")
        ax.set_ylim(0, 110)
        ax.set_ylabel("SOC (%)")
        ax.set_title(f"BESS: {bess}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.axhline(0, color="gray", linewidth=0.6)

        # Marca horas com Q ativo
        for h, soc_h, q in zip(hours, soc, q_pu):
            if q > 0:
                ax.plot(h, soc_h, marker="^", color="blue", markersize=8, zorder=5)
            elif q < 0:
                ax.plot(h, soc_h, marker="v", color="red", markersize=8, zorder=5)

        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0], [0], color="steelblue", linewidth=2, label="SOC (%)"),
            Line2D([0], [0], marker="^", color="blue", linestyle="None", markersize=8, label="Q capacitivo (↑V)"),
            Line2D([0], [0], marker="v", color="red",  linestyle="None", markersize=8, label="Q indutivo (↓V)"),
        ], fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Hora do dia")
    axes[0].set_title(
        f"Estado de carga (SOC) dos BESSs — pen={PEN_PCT}% / real={ID_REALIZACAO}\n"
        f"Nota: injeção de Q não consome energia da bateria neste modelo\n"
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
    """
    Linha de tensão (col = 'v_min' ou 'v_max') de todas as barras ao longo das 24h.
    Cada barra da rede é representada por uma linha com cor distinta.
    """
    buses = sorted(df_v["bus"].unique())
    hours = sorted(df_v["hour"].unique())

    cmap   = plt.cm.get_cmap("rainbow", len(buses))
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, bus in enumerate(buses):
        sub = df_v[df_v["bus"] == bus].sort_values("hour")
        ax.plot(sub["hour"], sub[col], color=cmap(i), linewidth=1.0,
                marker=".", markersize=3, label=bus)

    ax.axhline(limit, color=limit_color, linewidth=1.8, linestyle="--",
               label=f"{limit_label} = {limit} p.u.")
    ax.axhline(1.00,  color="gray", linewidth=0.8, linestyle=":", alpha=0.7)

    ax.set_xticks(hours)
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Tensão (p.u.)")
    ax.set_title(
        f"Tensão {title_suffix} nas barras da rede\n"
        f"pen={PEN_PCT}% / real={ID_REALIZACAO}  |  Controle Volt-VAr ativo"
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=6,
              ncol=1, framealpha=0.9, title="Barras")

    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {title_suffix} de todas as barras salvo em: {path}")


def plot_all_bus_vmin(bus_records: list):
    """Tensão mínima (pior fase) de cada barra ao longo das 24h."""
    df_v = pd.DataFrame(bus_records)
    _plot_bus_voltage_profile(
        df_v, col="v_min", limit=0.95,
        limit_label="V_min limite", limit_color="red",
        title_suffix="mínima", filename="fig_all_bus_vmin.png"
    )


def plot_all_bus_vmax(bus_records: list):
    """Tensão máxima (pior fase) de cada barra ao longo das 24h."""
    df_v = pd.DataFrame(bus_records)
    _plot_bus_voltage_profile(
        df_v, col="v_max", limit=1.05,
        limit_label="V_max limite", limit_color="darkorange",
        title_suffix="máxima", filename="fig_all_bus_vmax.png"
    )


# ---------------------------------------------------------------------------
# EXPORTA RESULTADOS E GERA GRÁFICOS
# ---------------------------------------------------------------------------
df       = pd.DataFrame(records)
df_hours = pd.DataFrame(hour_records)

out_csv = os.path.join(CSV_DIR, f"voltvar_pen{PEN_PCT:03d}_real{ID_REALIZACAO:04d}.csv")
df.to_csv(out_csv, index=False)
print(f"\n[OK] Resultados por BESS salvos em: {out_csv}")
print(df.to_string(index=False))

plot_q_activation(df)
plot_violations(df_hours)
plot_bess_soc(df)
plot_all_bus_vmin(bus_voltage_records)
plot_all_bus_vmax(bus_voltage_records)