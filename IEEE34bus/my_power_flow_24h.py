"""
Simulação de Fluxo de Potência 24h - IEEE 34 barras
Executa simulação para cada hora do dia e coleta:
- Tensões nas barras
- Correntes e potências nos elementos
- Análise de problemas de tensão (subtensão/sobretensão)
- Visualização gráfica dos resultados
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from opendssdirect import dss


# ============================================================================
# 1. CONFIGURAÇÃO INICIAL
# ============================================================================

def inicializar_dss(arquivo_dss):
    """Inicializa o OpenDSS e carrega o circuito."""
    dss.Basic.ClearAll()
    dss.Command(f'Redirect "{arquivo_dss}"')
    print("✓ Circuito carregado com sucesso")


# ============================================================================
# 2. COLETA DE DADOS
# ============================================================================

def coletar_tensoes(hora):
    """Coleta tensões em todas as barras para uma determinada hora."""
    registros = []
    
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        kv_base = dss.Bus.kVBase()
        pu_vals = dss.Bus.puVmagAngle()
        nodes = dss.Bus.Nodes()
        
        for i, fase in enumerate(nodes):
            registros.append({
                "Hora": hora,
                "Barra": bus.upper(),
                "Fase": fase,
                "V (p.u.)": round(pu_vals[2 * i], 4),
                "Ângulo (°)": round(pu_vals[2 * i + 1], 2),
                "kV base": round(kv_base, 3),
            })
    
    return registros


def coletar_correntes(hora):
    """Coleta correntes e potências em linhas e fontes para uma determinada hora."""
    registros = []
    
    def adicionar_elemento(nome, prefix):
        full_name = f"{prefix}.{nome}"
        dss.Circuit.SetActiveElement(full_name)
        
        buses = dss.CktElement.BusNames()
        bus1 = buses[0].split(".")[0].upper()
        bus2 = buses[1].split(".")[0].upper() if len(buses) > 1 else "GROUND"
        
        n_phases = dss.CktElement.NumPhases()
        currents = dss.CktElement.CurrentsMagAng()
        powers = dss.CktElement.Powers()
        
        # Soma das potências (P e Q estão em posições alternadas)
        p_total = sum(powers[0:2*n_phases:2])
        q_total = sum(powers[1:2*n_phases:2])
        s_total = math.sqrt(p_total**2 + q_total**2)
        
        i_mag_f1 = currents[0] if len(currents) > 0 else 0.0
        
        registros.append({
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
    
    # Adicionar dados de fontes de tensão
    for nome in dss.Vsources.AllNames():
        adicionar_elemento(nome, "Vsource")
    
    # Adicionar dados de linhas
    for nome in dss.Lines.AllNames():
        adicionar_elemento(nome, "Line")
    
    return registros


def simular_24h():
    """Executa simulação para as 24 horas do dia."""
    todos_tensoes = []
    todos_correntes = []
    
    print("\nExecutando simulação 24h...")
    
    for hora in range(1, 25):
        # Configurar hora e resolver
        dss.Solution.Hour(hora - 1)  # OpenDSS usa índice 0-23
        dss.Solution.Solve()
        
        if not dss.Solution.Converged():
            print(f"  ⚠ Solução não convergiu na hora {hora}")
            continue
        
        # Coletar dados
        todos_tensoes.extend(coletar_tensoes(hora))
        todos_correntes.extend(coletar_correntes(hora))
        
        print(f"  ✓ Hora {hora:2d} processada")
    
    return pd.DataFrame(todos_tensoes), pd.DataFrame(todos_correntes)


# ============================================================================
# 3. ANÁLISE DE PROBLEMAS DE TENSÃO
# ============================================================================

def analisar_problemas_tensao(df_tensoes, tensao_min=0.95, tensao_max=1.05):
    """Identifica barras com problemas de tensão por hora."""
    
    barras_subtensao = []
    barras_sobretensao = []
    horas = sorted(df_tensoes['Hora'].unique())
    
    print("\nAnalisando problemas de tensão...")
    
    for hora in horas:
        df_hora = df_tensoes[df_tensoes['Hora'] == hora]
        
        # Encontrar barras com subtensão
        subtensao = df_hora[df_hora['V (p.u.)'] < tensao_min]['Barra'].unique()
        n_subtensao = len(subtensao)
        barras_subtensao.append(n_subtensao)
        
        # Encontrar barras com sobretensão
        sobretensao = df_hora[df_hora['V (p.u.)'] > tensao_max]['Barra'].unique()
        n_sobretensao = len(sobretensao)
        barras_sobretensao.append(n_sobretensao)
        
        if n_subtensao > 0 or n_sobretensao > 0:
            print(f"  Hora {hora:2d}: {n_subtensao} barra(s) com subtensão, {n_sobretensao} barra(s) com sobretensão")
    
    df_problemas = pd.DataFrame({
        'Hora': horas,
        'Subtensão (<0.95 pu)': barras_subtensao,
        'Sobretensão (>1.05 pu)': barras_sobretensao,
    })
    
    return df_problemas


# ============================================================================
# 4. VISUALIZAÇÃO
# ============================================================================

def plotar_problemas_tensao(df_problemas):
    """Cria gráficos dos problemas de tensão."""
    
    horas = df_problemas['Hora'].values
    barras_subtensao = df_problemas['Subtensão (<0.95 pu)'].values
    barras_sobretensao = df_problemas['Sobretensão (>1.05 pu)'].values
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    x_pos = np.arange(len(horas))
    cores_sub = '#d62728'  # Vermelho
    cores_sob = '#ff7f0e'  # Laranja
    
    # Gráfico 1: Subtensão
    ax1.bar(x_pos, barras_subtensao, color=cores_sub, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Hora do Dia', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Número de Barras', fontsize=12, fontweight='bold')
    ax1.set_title('Barras com Problema de Subtensão (<0.95 pu)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{h:02d}h' for h in horas], fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)
    
    for i, v in enumerate(barras_subtensao):
        if v > 0:
            ax1.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Gráfico 2: Sobretensão
    ax2.bar(x_pos, barras_sobretensao, color=cores_sob, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Hora do Dia', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Número de Barras', fontsize=12, fontweight='bold')
    ax2.set_title('Barras com Problema de Sobretensão (>1.05 pu)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{h:02d}h' for h in horas], fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(bottom=0)
    
    for i, v in enumerate(barras_sobretensao):
        if v > 0:
            ax2.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    return fig


# ============================================================================
# 5. EXPORTAÇÃO
# ============================================================================

def exportar_resultados(df_tensoes, df_correntes, df_problemas, pasta_resultados):
    """Exporta todos os resultados para arquivos CSV."""
    
    print("\nExportando resultados...")
    
    # Arquivo consolidado de tensões
    arquivo_tensoes = os.path.join(pasta_resultados, "tensoes_24h_consolidado.csv")
    df_tensoes.to_csv(arquivo_tensoes, index=False)
    print(f"  ✓ {os.path.basename(arquivo_tensoes)}")
    
    # Arquivo consolidado de correntes
    arquivo_correntes = os.path.join(pasta_resultados, "correntes_24h_consolidado.csv")
    df_correntes.to_csv(arquivo_correntes, index=False)
    print(f"  ✓ {os.path.basename(arquivo_correntes)}")
    
    # Arquivo de problemas de tensão
    arquivo_problemas = os.path.join(pasta_resultados, "problemas_tensao_24h.csv")
    df_problemas.to_csv(arquivo_problemas, index=False)
    print(f"  ✓ {os.path.basename(arquivo_problemas)}")


def salvar_graficos(fig, pasta_resultados):
    """Salva os gráficos em arquivo PNG."""
    
    caminho_figura = os.path.join(pasta_resultados, "problemas_tensao_24h.png")
    fig.savefig(caminho_figura, dpi=300, bbox_inches='tight')
    print(f"  ✓ problemas_tensao_24h.png")


# ============================================================================
# 6. RELATÓRIO
# ============================================================================

def imprimir_relatorio(df_tensoes, df_correntes, df_problemas):
    """Imprime relatório resumido da simulação."""
    
    print("\n" + "=" * 70)
    print("RELATÓRIO DA SIMULAÇÃO 24h - IEEE 34 BARRAS")
    print("=" * 70)
    
    print(f"\n📊 DADOS GERAIS:")
    print(f"   Total de registros de tensão: {len(df_tensoes):,}")
    print(f"   Barras monitoradas: {df_tensoes['Barra'].nunique()}")
    print(f"   Total de registros de corrente: {len(df_correntes):,}")
    print(f"   Elementos monitorados: {df_correntes['Elemento'].nunique()}")
    
    print(f"\n📈 TENSÕES:")
    print(f"   Mínima: {df_tensoes['V (p.u.)'].min():.4f} p.u.")
    print(f"   Máxima: {df_tensoes['V (p.u.)'].max():.4f} p.u.")
    print(f"   Média: {df_tensoes['V (p.u.)'].mean():.4f} p.u.")
    
    print(f"\n⚡ POTÊNCIA:")
    potencia_total = df_correntes['P_total (kW)'].sum()
    print(f"   Potência ativa total: {potencia_total:,.2f} kW")
    print(f"   Potência ativa média: {df_correntes['P_total (kW)'].mean():,.2f} kW")
    
    print(f"\n⚠ PROBLEMAS DE TENSÃO:")
    n_horas_subtensao = (df_problemas['Subtensão (<0.95 pu)'] > 0).sum()
    n_horas_sobretensao = (df_problemas['Sobretensão (>1.05 pu)'] > 0).sum()
    print(f"   Horas com subtensão: {n_horas_subtensao}/24")
    print(f"   Horas com sobretensão: {n_horas_sobretensao}/24")
    
    print("\n" + "=" * 70)


# ============================================================================
# 7. FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Executa o pipeline completo de simulação e análise."""
    
    # Configuração de caminhos
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    arquivo_dss = os.path.join(diretorio_atual, "IEEE34_original_with_loadshapes.dss")
    pasta_resultados = os.path.join(diretorio_atual, "resultados_24h")
    
    # Criar pasta de resultados
    os.makedirs(pasta_resultados, exist_ok=True)
    
    # Inicialização
    inicializar_dss(arquivo_dss)
    
    # Simulação
    df_tensoes, df_correntes = simular_24h()
    
    # Análise
    df_problemas = analisar_problemas_tensao(df_tensoes)
    
    # Visualização
    fig = plotar_problemas_tensao(df_problemas)
    
    # Exportação
    exportar_resultados(df_tensoes, df_correntes, df_problemas, pasta_resultados)
    salvar_graficos(fig, pasta_resultados)
    
    # Relatório
    imprimir_relatorio(df_tensoes, df_correntes, df_problemas)
    
    print(f"\n📁 Resultados salvos em: {pasta_resultados}")
    
    # Mostrar gráficos
    plt.show()


if __name__ == "__main__":
    main()