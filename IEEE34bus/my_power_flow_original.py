import os
from opendssdirect import dss
import pandas as pd
import math
import matplotlib.pyplot as plt
import re

# 1. Inicialização e Carga do Circuito
dss.Basic.ClearAll()

# --- AJUSTE DE CAMINHO ---
# Pega a pasta onde este script está salvo
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
arquivo_dss = os.path.join(diretorio_atual, "IEEE34_original.dss")

# Correção da linha 7 (Sintaxe corrigida)
dss.Command(f'Redirect "{arquivo_dss}"')


dss.Solution.Solve() 

if not dss.Solution.Converged():
    print("ERRO: O fluxo de potência não convergiu!")
else:
    print("Sucesso: Fluxo de potência convergiu.\n")

# 2. Tensões nas barras
registros_tensao = []
for bus in dss.Circuit.AllBusNames():
    dss.Circuit.SetActiveBus(bus)
    kv_base = dss.Bus.kVBase()
    pu_vals = dss.Bus.puVmagAngle()
    nodes = dss.Bus.Nodes()
    
    for i, fase in enumerate(nodes):
        registros_tensao.append({
            "Barra":    bus.upper(),
            "Fase":     fase,
            "V (p.u.)": round(pu_vals[2 * i], 4),
            "Ângulo (°)": round(pu_vals[2 * i + 1], 2),
            "kV base":  round(kv_base, 3),
        })

df_tensoes = pd.DataFrame(registros_tensao)

# 3. Correntes e potências
registros_linhas = []

def _add_element(nome, prefix):
    full_name = f"{prefix}.{nome}"
    dss.Circuit.SetActiveElement(full_name)
    
    buses = dss.CktElement.BusNames()
    bus1 = buses[0].split(".")[0].upper()
    bus2 = buses[1].split(".")[0].upper() if len(buses) > 1 else "GROUND"
    
    n_phases = dss.CktElement.NumPhases()
    currents = dss.CktElement.CurrentsMagAng()
    powers = dss.CktElement.Powers() 
    
    # Soma das potências (P e Q estão em posições alternadas: P1, Q1, P2, Q2...)
    p_total = sum(powers[0:2*n_phases:2])
    q_total = sum(powers[1:2*n_phases:2])
    s_total = math.sqrt(p_total**2 + q_total**2)

    i_mag_f1 = currents[0] if len(currents) > 0 else 0.0

    registros_linhas.append({
        "Elemento": nome,
        "De":       bus1,
        "Para":     bus2,
        "Fases":    n_phases,
        "I_F1 (A)": round(i_mag_f1, 2),
        "P_total (kW)": round(p_total, 2),
        "Q_total (kVAR)": round(q_total, 2),
        "S_total (kVA)":  round(s_total, 2),
    })

for nome in dss.Vsources.AllNames():
    _add_element(nome, "Vsource")

for nome in dss.Lines.AllNames():
    _add_element(nome, "Line")

df_linhas = pd.DataFrame(registros_linhas)

# 4. Saída e Exportação
print("TENSÕES NAS BARRAS (Amostra):")
print(df_tensoes.head(10).to_string(index=False))

# Exportação para pasta resultados
caminho_tensoes = os.path.join(diretorio_atual, "resultados", "tensoes_barras.csv")
caminho_linhas = os.path.join(diretorio_atual, "resultados", "correntes_linhas.csv")

# Cria a pasta "resultados" se não existir
os.makedirs(os.path.dirname(caminho_tensoes), exist_ok=True)

df_tensoes.to_csv(caminho_tensoes, index=False)
df_linhas.to_csv(caminho_linhas, index=False)

print(f"\nArquivos CSV gerados em: {diretorio_atual}")