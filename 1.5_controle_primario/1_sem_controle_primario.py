"""
Simulação da rede IEEE 34 barras SEM controle primário Volt-VAr.
Serve como referência para comparar violações de tensão com/sem controle.

Mesma rede e cenário Monte Carlo de 0_controle_primario.py, porém:
  - Sem avaliação da curva Volt-VAr
  - Sem injeção de Q pelos BESSs
  - Despacho ativo do BESS e fatores de incerteza de carga aplicados igualmente
  - Reguladores livres em todos os passos
"""
from opendssdirect import dss
import pandas as pd
import os
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

PEN_PCT       = 120
ID_REALIZACAO = 1

TOTAL_HOURS   = 24


# ---------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ---------------------------------------------------------------------------

def get_bus_vmin_vmax_pu(bus_name: str) -> tuple:
    dss.Circuit.SetActiveBus(bus_name)
    magnitudes = dss.Bus.puVmagAngle()[0::2]
    valid = [v for v in magnitudes if v > 0.1]
    if not valid:
        return 1.0, 1.0
    return min(valid), max(valid)


def enable_regulators(reg_names: list):
    for reg in reg_names:
        dss.Command(f"RegControl.{reg}.enabled=yes")


def discover_bess_buses() -> dict:
    bess_info = {}
    for el in dss.Circuit.AllElementNames():
        if el.lower().startswith("storage."):
            dss.Circuit.SetActiveElement(el)
            name   = el.split(".")[1]
            bus    = dss.CktElement.BusNames()[0].split(".")[0].lower()
            phases = dss.CktElement.NumPhases()
            if phases < 3:
                continue
            dss.Command(f"? Storage.{name}.kWrated")
            p_nom = float(dss.Text.Result()) if dss.Text.Result() else 0.0
            bess_info[name] = {"bus": bus, "pnom_kw": p_nom}
    return bess_info


def set_bess_dispatch(name: str, mult: float, pnom_kw: float):
    """Define State e kW do BESS antes do solve para que o SOC atualize."""
    if mult < 0:
        dss.Command(f"Storage.{name}.State=Charging kW={abs(mult) * pnom_kw:.4f}")
    elif mult > 0:
        dss.Command(f"Storage.{name}.State=Discharging kW={mult * pnom_kw:.4f}")
    else:
        dss.Command(f"Storage.{name}.State=Idling")


def carregar_cargas_base() -> dict:
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
    for nome, carga in cargas_base.items():
        fator = fatores_carga.get(carga["bus"], [1.0] * 24)[hora]
        dss.Command(
            f"Edit Load.{nome} kW={carga['kw'] * fator:.6f} kvar={carga['kvar'] * fator:.6f}"
        )


def count_voltage_violations() -> tuple:
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
        title_suffix="mínima", filename="fig_no_controll_all_bus_vmin.png"
    )


def plot_all_bus_vmax(bus_records: list):
    """Tensão máxima (pior fase) de cada barra ao longo das 24h."""
    df_v = pd.DataFrame(bus_records)
    _plot_bus_voltage_profile(
        df_v, col="v_max", limit=1.05,
        limit_label="V_max limite", limit_color="darkorange",
        title_suffix="máxima", filename="fig_no_controll_all_bus_vmax.png"
    )



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


def plot_violations_no_control(hr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 6))

    hours  = hr["hour"].tolist()
    counts = hr["n_violations"].tolist()

    bars = ax.bar(hours, counts, color="tomato", edgecolor="white", linewidth=1.0)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(hours)
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Barras com violação de tensão")
    ax.set_title(
        f"Violações de tensão SEM controle primário\n"
        f"pen={PEN_PCT}% / real={ID_REALIZACAO}\n"
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

            ax.text(bar_x, y_pos, b, ha="center", va=va,
                    fontsize=6, rotation=45, color="black",
                    bbox=dict(facecolor=bg, alpha=0.85, pad=1.2,
                              edgecolor="none", boxstyle="round,pad=0.25"))

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

    path = os.path.join(FIG_DIR, "fig_violations_no_control.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Violações sem controle salvo em: {path}")

bus_voltage_records = []  # uma linha por (hora, barra) — v_min e v_max de toda a rede

# ---------------------------------------------------------------------------
# COMPILAR A REDE
# ---------------------------------------------------------------------------
dss.Command(f"compile [{BASE_DSS}]")
n_pv, n_bess, bess_prof, fatores_carga = add_realizacao_elements(PEN_PCT, ID_REALIZACAO)

dss.Command("set mode=daily")
dss.Command("set stepsize=1h")
dss.Command("set number=1")

inv_controls = [e for e in dss.Circuit.AllElementNames() if e.lower().startswith("invcontrol.")]
for ic in inv_controls:
    ic_name = ic.split(".")[1]
    dss.Command(f"InvControl.{ic_name}.enabled=no")

bess_dict  = discover_bess_buses()
cargas_base = carregar_cargas_base()
reg_names  = dss.RegControls.AllNames() or []
print(f"[RegControl] {len(reg_names)} regulador(es): {reg_names}")
print(f"[BESS] {len(bess_dict)} elemento(s) Storage trifásico(s) encontrado(s).")

# ---------------------------------------------------------------------------
# LOOP HORÁRIO PRINCIPAL — sem controle Volt-VAr
# ---------------------------------------------------------------------------
hour_records = []

for hour in range(TOTAL_HOURS):
    print(f"\n{'='*60}")
    print(f"Hora {hour:02d}/{TOTAL_HOURS}")
    print(f"{'='*60}")

    aplicar_fatores_carga(hour, cargas_base, fatores_carga)

    enable_regulators(reg_names)
    dss.Command("solve")

    tap_snapshot = {}
    for reg in reg_names:
        dss.RegControls.Name(reg)
        xfmr = dss.RegControls.Transformer()
        wdg  = dss.RegControls.Winding()
        dss.Transformers.Name(xfmr)
        dss.Transformers.Wdg(wdg)
        tap_snapshot[reg] = dss.Transformers.Tap()
    for reg, tap in tap_snapshot.items():
        print(f"  [Tap hora {hour+1:02d}] {reg}: tap={tap:.6f}")

    # registra v_min e v_max de TODAS as barras da rede nesta hora
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
# EXPORTA RESULTADOS E GERA GRÁFICO
# ---------------------------------------------------------------------------
df_hours = pd.DataFrame(hour_records)

out_csv = os.path.join(CSV_DIR, f"sem_controle_pen{PEN_PCT:03d}_real{ID_REALIZACAO:04d}.csv")
df_hours[["hour", "n_violations"]].to_csv(out_csv, index=False)
print(f"\n[OK] Resultados salvos em: {out_csv}")
print(df_hours[["hour", "n_violations"]].to_string(index=False))

plot_violations_no_control(df_hours)
plot_all_bus_vmin(bus_voltage_records)
plot_all_bus_vmax(bus_voltage_records)