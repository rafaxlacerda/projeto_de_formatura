from opendssdirect import dss
from dss import SolveModes
import pandas as pd
import math

dss.Basic.ClearAll()
dss.Command('Redirect "IEEE34bus/IEEE34_2.dss"')

# ── Tensões nas barras ────────────────────────────────────────────────────────
registros_tensao = []

for bus in dss.Circuit.AllBusNames():
    dss.Circuit.SetActiveBus(bus)
    kv_base   = dss.Bus.kVBase()
    pu_vals   = dss.Bus.puVmagAngle()  # [mag1,ang1, mag2,ang2, ...]
    num_nodes = dss.Bus.NumNodes()
    for i in range(num_nodes):
        registros_tensao.append({
            "Barra":      bus.upper(),
            "Fase":       i + 1,
            "V (p.u.)":   round(pu_vals[2 * i],     4),
            "Ângulo (°)": round(pu_vals[2 * i + 1], 2),
            "kV base":    round(kv_base,            3),
        })

df_tensoes = pd.DataFrame(registros_tensao)

# ── Correntes e potências nas linhas e fontes ─────────────────────────────────
registros_linhas = []

def _add_element(nome, prefix):
    dss.Circuit.SetActiveElement(f"{prefix}.{nome}")
    bus1     = dss.CktElement.BusNames()[0].split(".")[0].upper()
    bus2     = dss.CktElement.BusNames()[1].split(".")[0].upper()
    fases    = dss.CktElement.NumPhases()
    currents = dss.CktElement.CurrentsMagAng()  # [mag,ang,...] bus1 then bus2
    powers   = dss.CktElement.Powers()          # [P1,Q1, P2,Q2,...] kW/kVAR
    i_mag    = currents[0] if len(currents) > 0 else 0.0
    i_ang    = currents[1] if len(currents) > 1 else 0.0
    p_kw     = powers[0]   if len(powers)   > 0 else 0.0
    q_kvar   = powers[1]   if len(powers)   > 1 else 0.0
    s_kva    = math.sqrt(p_kw**2 + q_kvar**2)
    registros_linhas.append({
        "Linha":    nome,
        "De":       bus1,
        "Para":     bus2,
        "Fases":    fases,
        "I (A)":    round(i_mag,  2),
        "I (°)":    round(i_ang,  2),

        "P (kW)":   round(p_kw,   2),
        "Q (kVAR)": round(q_kvar, 2),
        "S (kVA)":  round(s_kva,  2),
    })

for nome in dss.Vsources.AllNames():
    _add_element(nome, "Vsource")

for nome in dss.Lines.AllNames():
    _add_element(nome, "Line")

df_linhas = pd.DataFrame(registros_linhas)

# ── Print no terminal ─────────────────────────────────────────────────────────
print("TENSÕES NAS BARRAS")
print(df_tensoes.to_string(index=False))
print("\nCORRENTES E POTÊNCIAS")
print(df_linhas.to_string(index=False))

# ── Exporta CSV ───────────────────────────────────────────────────────────────
df_tensoes.to_csv("IEEE34bus/resultados/tensoes_barras.csv", index=False)
df_linhas.to_csv("IEEE34bus/resultados/correntes_linhas.csv", index=False)
