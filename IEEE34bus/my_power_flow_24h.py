import os
from opendssdirect import dss
import pandas as pd
import math

# 1. Inicialização e Carga do Circuito
dss.Basic.ClearAll()

# --- AJUSTE DE CAMINHO ---
# Pega a pasta onde este script está salvo
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
arquivo_dss = os.path.join(diretorio_atual, "IEEE34_original_with_loadshapes.dss")

# Carregar circuito com LoadShapes
dss.Command(f'Redirect "{arquivo_dss}"')

# Cria a pasta "resultados_24h" se não existir
pasta_resultados = os.path.join(diretorio_atual, "resultados_24h")
os.makedirs(pasta_resultados, exist_ok=True)

# Variáveis para armazenar resultados consolidados
todos_tensoes = []
todos_correntes = []

# 2. Simular 24 horas
print("Iniciando simulação de 24 horas...")
print("=" * 60)

for hora in range(1, 25):
    print(f"\nSimulando hora {hora}...")
    
    # Configurar o tempo de simulação para a hora específica
    dss.Solution.Hour(hora - 1)  # OpenDSS usa índice 0-23
    dss.Solution.Solve()
    
    if not dss.Solution.Converged():
        print(f"  AVISO: Solução não convergiu na hora {hora}!")
        continue
    
    # 2.1 Coletar Tensões nas barras
    registros_tensao_hora = []
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        kv_base = dss.Bus.kVBase()
        pu_vals = dss.Bus.puVmagAngle()
        nodes = dss.Bus.Nodes()
        
        for i, fase in enumerate(nodes):
            registros_tensao_hora.append({
                "Hora": hora,
                "Barra": bus.upper(),
                "Fase": fase,
                "V (p.u.)": round(pu_vals[2 * i], 4),
                "Ângulo (°)": round(pu_vals[2 * i + 1], 2),
                "kV base": round(kv_base, 3),
            })
    
    todos_tensoes.extend(registros_tensao_hora)
    
    # 2.2 Coletar Correntes e Potências
    registros_correntes_hora = []
    
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
        
        registros_correntes_hora.append({
            "Hora": hora,
            "Elemento": nome,
            "De": bus1,
            "Para": bus2,
            "Fases": n_phases,
            "I_F1 (A)": round(i_mag_f1, 2),
            "P_total (kW)": round(p_total, 2),
            "Q_total (kVAR)": round(q_total, 2),
            "S_total (kVA)": round(s_total, 2),
        })
    
    # Adicionar fontes de tensão
    for nome in dss.Vsources.AllNames():
        _add_element(nome, "Vsource")
    
    # Adicionar linhas
    for nome in dss.Lines.AllNames():
        _add_element(nome, "Line")
    
    todos_correntes.extend(registros_correntes_hora)
    print(f"  ✓ Hora {hora} processada com sucesso")

print("\n" + "=" * 60)

# 3. Consolidar e Exportar Resultados

# Converter para DataFrames
df_tensoes_24h = pd.DataFrame(todos_tensoes)
df_correntes_24h = pd.DataFrame(todos_correntes)

# Exportar arquivo consolidado de tensões (todas as 24 horas)
caminho_tensoes_24h = os.path.join(pasta_resultados, "tensoes_24h_consolidado.csv")
df_tensoes_24h.to_csv(caminho_tensoes_24h, index=False)
print(f"\n✓ Arquivo exportado: tensoes_24h_consolidado.csv")

# Exportar arquivo consolidado de correntes (todas as 24 horas)
caminho_correntes_24h = os.path.join(pasta_resultados, "correntes_24h_consolidado.csv")
df_correntes_24h.to_csv(caminho_correntes_24h, index=False)
print(f"✓ Arquivo exportado: correntes_24h_consolidado.csv")

# Exportar arquivos individuais por hora (opcional)
print("\nExportando resultados por hora...")
for hora in range(1, 25):
    # Tensões
    df_tensoes_hora = df_tensoes_24h[df_tensoes_24h['Hora'] == hora]
    caminho_tensoes_hora = os.path.join(pasta_resultados, f"tensoes_hora_{hora:02d}.csv")
    df_tensoes_hora.to_csv(caminho_tensoes_hora, index=False)
    
    # Correntes
    df_correntes_hora = df_correntes_24h[df_correntes_24h['Hora'] == hora]
    caminho_correntes_hora = os.path.join(pasta_resultados, f"correntes_hora_{hora:02d}.csv")
    df_correntes_hora.to_csv(caminho_correntes_hora, index=False)

print(f"✓ 24 arquivos de tensões exportados")
print(f"✓ 24 arquivos de correntes exportados")

# 4. Exibir Sumário
print("\n" + "=" * 60)
print("SUMÁRIO DA SIMULAÇÃO 24H")
print("=" * 60)

print(f"\nTotal de registros de tensão: {len(df_tensoes_24h)}")
print(f"Barras monitoradas: {df_tensoes_24h['Barra'].nunique()}")
print(f"Total de registros de corrente: {len(df_correntes_24h)}")
print(f"Elementos monitorados: {df_correntes_24h['Elemento'].nunique()}")

# Exibir algumas estatísticas por hora
print("\nTensão mínima e máxima por hora:")
stats_tensao = df_tensoes_24h.groupby('Hora')['V (p.u.)'].agg(['min', 'max']).round(4)
print(stats_tensao)

print("\nPotência ativa (kW) total por hora:")
stats_potencia = df_correntes_24h.groupby('Hora')['P_total (kW)'].sum().round(2)
print(stats_potencia)

print("\n" + "=" * 60)
print("Simulação concluída com sucesso!")
print(f"Resultados salvos em: {pasta_resultados}")
