"""
Controle Primário Volt-VAr para BESSs na rede IEEE 34 barras (OpenDSS)
Usando InvControl nativo do OpenDSS com XYCurve (modo VOLTVAR)

Diferenças em relação a 0_controle_primario.py:
  - Curva Volt-VAr definida via XYCurve DSS + InvControl VOLTVAR
  - Sem loop Python de cálculo de Q — o InvControl itera internamente a cada solve
  - Apenas 1 solve por hora (RegControl e InvControl coiteram até convergência)
  - Reguladores livres em todos os passos — não há desacoplamento manual necessário

Interferência reguladores × InvControl:
  OpenDSS executa um loop de controle interno: power flow → todos os controles
  atualizam (InvControl ajusta Q, RegControl ajusta tap) → power flow → repete até
  convergir. Os dois controles coexistem sem coordenação manual, o que é
  fisicamente correto para o regime permanente.

Curva Volt-VAr (ABNT NBR 16149:2013, Figura 2) via XYCurve:
  xarray = [V_SAT_LOW, V_DB_LOW, V_DB_HIGH, V_SAT_HIGH]
  yarray = [+1.0,      0.0,      0.0,       -1.0       ]  (fração de kvarmax)

Nota sobre kvarmax:
  Com RefReactivePower=VARMAX, kvarmax = sqrt(kVA² - kW²).
  Difere do Q_MAX_PU=0.4358×Pnom fixo de 0_controle_primario.py porque aqui o
  limite de Q varia com a potência ativa corrente do BESS.

Resultados em pasta separada: figuras_invcontrol/
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
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
FIG_DIR       = os.path.join(SCRIPT_DIR, "figuras_invcontrol")
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
# Esses valores definem o XYCurve DSS e são os mesmos de 0_controle_primario.py
V_DEADBAND_LOW  = 0.97      # p.u.  — início da zona morta inferior
V_DEADBAND_HIGH = 1.03      # p.u.  — início da zona morta superior
V_SAT_LOW  = 0.90           # p.u.  — tensão de saturação capacitiva (Q > 0)
V_SAT_HIGH = 1.10           # p.u.  — tensão de saturação indutiva   (Q < 0)

# Referência de Q para o InvControl (VARMAX: fração de kvarmax disponível)
Q_MAX_PU = 0.4358           # usado apenas para normalização nos registros de saída


# ---------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ---------------------------------------------------------------------------

def get_bus_vmag_pu(bus_name: str) -> float:
    """
    Tensão representativa da barra para monitoramento.
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
        dss.Command(f"RegControl.{reg}.maxtapchange = 16")


def disable_regulators(reg_names: list):
    """Desabilita todos os RegControls (congela taps na posição atual)."""
    for reg in reg_names:
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
            dss.Command(f"? Storage.{name}.kWrated")
            p_nom = float(dss.Text.Result()) if dss.Text.Result() else 0.0
            bess_info[name] = {"bus": bus, "pnom_kw": p_nom}
            print(f"[BESS] '{name}' | bus='{bus}' | Pnom={p_nom:.1f} kW | fases={phases}")

    return bess_info


def get_bess_kvar_output(name: str) -> float:
    """
    Lê o kvar efetivamente injetado pelo BESS após o InvControl atuar.
    Valor positivo = capacitivo (injeção); negativo = indutivo (absorção).
    """
    dss.Command(f"? Storage.{name}.kvar")
    return float(dss.Text.Result()) if dss.Text.Result() else 0.0


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


def add_realizacao_elements(pen_pct: int, id_realizacao: int) -> tuple:
    """
    Lê os CSVs da realização Monte Carlo e adiciona PVSystem e Storage
    à rede já compilada. 
    
    #### IMPORTANTE: Para cada PVSystem e Storage, cria um InvControl VOLTVAR #######

    Retorna (n_pv, n_bess, bess_prof, fatores_carga).
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

    '''CURVA VOLT VAR'''
    # XYCurve para a curva Volt-VAr (definida uma única vez)
    # X: tensão em p.u. do rated  |  Y: fração de kvarmax (+capacitivo, -indutivo)
    dss.Command(
        f"New XYCurve.CurvaVoltVar npts=4 "
        f"xarray=[{V_SAT_LOW} {V_DEADBAND_LOW} {V_DEADBAND_HIGH} {V_SAT_HIGH}] "
        f"yarray=[1.0 0.0 0.0 -1.0]"
    )
    print(f"[XYCurve] CurvaVoltVar: x=[{V_SAT_LOW},{V_DEADBAND_LOW},{V_DEADBAND_HIGH},{V_SAT_HIGH}] "
          f"y=[1.0,0.0,0.0,-1.0]")

    # Lê elementos da realização
    df_elem = pd.read_csv(os.path.join(pasta, "06_elementos_opendss.csv"),
                          sep=";", decimal=",")
    df_real = df_elem[df_elem["id_realizacao"] == id_realizacao]

    n_pv = n_bess = 0
    elementos_por_barra = {}  # {barra: {"PVSystem": [nomes], "Storage": [nomes]}}

    for _, row in df_real.iterrows():
        nome   = row["nome"]
        barra  = int(row["barra"])
        p_kw   = float(row["potencia_kw"])
        classe = row["classe"]

        # Tensão de base linha-a-linha da barra (kVBase é L-N; multiplica por √3 para L-L)
        dss.Circuit.SetActiveBus(str(barra))
        kv_base_ln = dss.Bus.kVBase()
        kv_ll = kv_base_ln * (3 ** 0.5) if kv_base_ln > 0 else 24.9

        if barra not in elementos_por_barra:
            elementos_por_barra[barra] = {"PVSystem": [], "Storage": []}

        if classe == "PVSystem":
            dss.Command(
                f"New PVSystem.{nome} Bus1={barra} Phases=3 Conn=Wye "
                f"kV={kv_ll:.4f} kVA={p_kw:.2f} Pmpp={p_kw:.2f} "
                f"kvarmax={Q_MAX_PU * p_kw:.4f} "
                f"irradiance=1 daily={irr_ls}"
            )
            elementos_por_barra[barra]["PVSystem"].append(nome)
            print(f"[PV]   '{nome}' | bus={barra} | kV={kv_ll:.4f} | Pmpp={p_kw:.1f} kW")
            n_pv += 1

        elif classe == "Storage":
            e_kwh = float(row["capacidade_kwh"])
            dss.Command(
                f"New Storage.{nome} Bus1={barra} Phases=3 Conn=Wye "
                f"kV={kv_ll:.4f} kWrated={p_kw:.2f} kWhrated={e_kwh:.2f} "
                f"kvarmax={Q_MAX_PU * p_kw:.4f} "
                f"%stored=0 dispmode=follow daily=BESS_fixo"
            )
            elementos_por_barra[barra]["Storage"].append(nome)
            print(f"[BESS] '{nome}' | bus={barra} | kV={kv_ll:.4f} | Pnom={p_kw:.1f} kW | E={e_kwh:.1f} kWh")
            n_bess += 1

    # Um InvControl por barra agrupa todos os DERs daquela barra no mesmo controlador.
    # Quando BESS e PV compartilham a barra, ambos reagem à mesma leitura de tensão.
    for barra, grupos in elementos_por_barra.items():
        der_list = (
            [f"PVSystem.{n}" for n in grupos["PVSystem"]] +
            [f"Storage.{n}"  for n in grupos["Storage"]]
        )
        der_str = " ".join(der_list)
        ic_nome = f"IC_bus_{barra}"
        dss.Command(
            f"New InvControl.{ic_nome} mode=VOLTVAR "
            f"voltage_curvex_ref=rated vvc_curve1=CurvaVoltVar "
            f"DERList=[{der_str}] RefReactivePower=VARMAX"
        )
        print(f"[InvControl] {ic_nome} criado | DERList=[{der_str}]")

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

# Garante modo diário
dss.Command("set mode=daily")
dss.Command("set stepsize=1h")
dss.Command("set number=1")

# Os InvControls criados aqui (IC_{nome})
existing_inv = [e for e in dss.Circuit.AllElementNames() if e.lower().startswith("invcontrol.")]
print(f"[InvControl] {len(existing_inv)} InvControl(s) presentes após compilação: "
      f"{[e.split('.')[1] for e in existing_inv]}")

# ---------------------------------------------------------------------------
# PREPARAÇÃO PRÉ-LOOP
# ---------------------------------------------------------------------------
bess_dict = discover_bess_buses()
if not bess_dict:
    raise RuntimeError("Nenhum elemento Storage trifásico encontrado. "
                       "Verifique se os BESSs foram inseridos na rede.")

cargas_base = carregar_cargas_base()
print(f"[Cargas] {len(cargas_base)} Load(s) | {len(fatores_carga)} barra(s) com fator de incerteza.")

reg_names = dss.RegControls.AllNames() or []
print(f"[RegControl] {len(reg_names)} regulador(es): {reg_names}")

# ---------------------------------------------------------------------------
# ESTRUTURAS DE SAÍDA
# ---------------------------------------------------------------------------
records             = []
hour_records        = []
bus_voltage_records = []

# ---------------------------------------------------------------------------
# LOOP HORÁRIO PRINCIPAL — InvControl nativo (1 solve por hora)
#
# InvControl e RegControl coiteram internamente até convergência em cada solve.
# Não é necessário desacoplar manualmente os reguladores quando o BESS age.
# ---------------------------------------------------------------------------
enable_regulators(reg_names)
dss.Command("set mode=daily")
dss.Command("set stepsize=1h")
dss.Command("set number=1")
# InvControl + RegControl podem precisar de muitas iterações para convergir juntos
dss.Command("set MaxControlIter=100")

for hour in range(TOTAL_HOURS):

    print(f"\n{'='*60}")
    print(f"Hora {hour:02d}/{TOTAL_HOURS}")
    print(f"{'='*60}")

    aplicar_fatores_carga(hour, cargas_base, fatores_carga)

    # Reguladores sempre livres: InvControl e RegControl coiteram juntos
    enable_regulators(reg_names)

    # Solve único: InvControl ajusta Q, RegControl ajusta tap, iterativamente
    try:
        dss.Command("solve")
    except Exception as e:
        if "485" in str(e) or "Max Control" in str(e):
            # Warning não-fatal: o fluxo de potência convergiu mas os controles
            # não atingiram regime estacionário em MaxControlIter iterações.
            # O resultado é aproveitável; aumentar MaxControlIter se recorrente.
            print(f"  [AVISO controle] hora {hour+1:02d}: {e}")
        else:
            raise

    # Lê taps pós-solve para monitoramento
    #tap_snapshot = capture_taps(reg_names)
    #if reg_names:
    #    for reg, info in tap_snapshot.items():
    #        print(f"  [Tap hora {hour+1:02d}] {reg}: tap={info['tap']:.6f}")

    # Lê tensão pós-solve (monitoramento; InvControl já atuou)
    v_measured = {name: get_bus_vmag_pu(info["bus"]) for name, info in bess_dict.items()}

    # Lê kvar efetivo injetado pelo InvControl em cada BESS
    q_atual = {name: get_bess_kvar_output(name) for name in bess_dict}

    controle_primario = any(q != 0.0 for q in q_atual.values())
    #if controle_primario:
        #print(f"  [InvControl] Controle primário ATIVO em ≥1 BESS.")

    # Registra v_min e v_max de TODAS as barras
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

    soc_info = {name: get_bess_soc(name) for name in bess_dict}

    # -----------------------------------------------------------------
    # Coleta resultados por BESS
    # -----------------------------------------------------------------
    for name, info in bess_dict.items():
        q_kvar  = q_atual[name]
        q_max   = Q_MAX_PU * info["pnom_kw"]
        soc, bess_state = soc_info[name]
        soc_warn = soc < 5.0 and q_kvar != 0.0
        #if soc_warn:
            #print(f"  [AVISO] BESS {name}: SOC={soc:.1f}% (bateria quase vazia) "
                  #f"mas Q={q_kvar:.1f} kvar solicitado.")
        records.append({
            "hour":       hour + 1,
            "bess":       name,
            "bus":        info["bus"],
            "v_pu":       round(v_measured[name], 6),
            "q_kvar":     round(q_kvar, 4),
            "q_pu":       round(q_kvar / q_max, 4) if q_max > 0 else 0.0,
            "q_active":   q_kvar != 0.0,
            "pnom_kw":    info["pnom_kw"],
            "soc_pct":    round(soc, 2),
            "bess_state": bess_state,
            "soc_warn":   soc_warn,
        })
        #print(f"  [BESS {name}] V={v_measured[name]:.4f} pu | "
              #f"Q={q_kvar:.2f} kvar ({'ativo' if q_kvar != 0 else 'zona morta'}) | "
              #f"SOC={soc:.1f}% [{bess_state}]")

    # Violações de tensão
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
    ax.set_title(f"Ativação do controle Volt-VAr (InvControl nativo) — pen={PEN_PCT}% / real={ID_REALIZACAO}\n"
                 f"Azul = capacitivo (↑V)   |   Vermelho = indutivo (↓V)   |   Branco = zona morta")

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
    Gráfico de violações de tensão por hora com InvControl nativo.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    hours  = hr["hour"].tolist()
    counts = hr["n_violations"].tolist()

    bars = ax.bar(hours, counts, color="steelblue", edgecolor="white", linewidth=1.0)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(hours)
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Barras com violação de tensão")
    ax.set_title(
        f"Violações de tensão com InvControl nativo (VOLTVAR)\n"
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
                bg  = "#fdd835"
            else:
                bg  = "#e53935"

            if bar_h >= 1.0:
                y_pos = bar_h * (i + 0.5) / n_viols
                va    = "center"
            else:
                y_pos = bar_h + 0.05 + i * 0.28
                va    = "bottom"

            ax.text(
                bar_x, y_pos, b,
                ha="center", va=va,
                fontsize=6, rotation=45, color="black",
                bbox=dict(facecolor=bg, alpha=0.85, pad=1.2,
                          edgecolor="none", boxstyle="round,pad=0.25"),
            )

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="#fdd835", label="Subtensão  (V_min < 0.95 p.u.)"),
            Patch(facecolor="#e53935", label="Sobretensão (V_max > 1.05 p.u.)"),
            Patch(facecolor="orange",  label="Sub e sobretensão"),
        ],
        loc="upper right", fontsize=8, framealpha=0.9,
    )

    if counts:
        ax.set_ylim(0, max(counts) + 1)

    path = os.path.join(FIG_DIR, "fig_violations_with_control.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Violações com InvControl salvo em: {path}")


def plot_bess_soc(df: pd.DataFrame):
    """
    Linha de SOC (%) por BESS ao longo das 24h.
    Marcadores indicam horas com controle Q ativo: azul (capacitivo) e vermelho (indutivo).
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
        f"InvControl nativo (VOLTVAR)  |  Nota: injeção de Q não consome energia da bateria\n"
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
        f"pen={PEN_PCT}% / real={ID_REALIZACAO}  |  InvControl nativo (VOLTVAR)"
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
    _plot_bus_voltage_profile(
        df_v, col="v_min", limit=0.95,
        limit_label="V_min limite", limit_color="red",
        title_suffix="mínima", filename="fig_all_bus_vmin.png"
    )


def plot_all_bus_vmax(bus_records: list):
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

out_csv = os.path.join(CSV_DIR, f"invcontrol_pen{PEN_PCT:03d}_real{ID_REALIZACAO:04d}.csv")
df.to_csv(out_csv, index=False)
print(f"\n[OK] Resultados por BESS salvos em: {out_csv}")
print(df.to_string(index=False))

plot_q_activation(df)
plot_violations(df_hours)
plot_bess_soc(df)
plot_all_bus_vmin(bus_voltage_records)
plot_all_bus_vmax(bus_voltage_records)
