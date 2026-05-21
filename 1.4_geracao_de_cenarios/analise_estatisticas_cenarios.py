"""
Análise de Estatísticas Descritivas para Cenários Monte Carlo
Projeto de Formatura - Poli USP
Etapa: Validação e síntese estatística dos cenários gerados
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Configurações
NIVEIS_ESPECIAIS = [0, 50, 100, 150]  # % de penetração FV
BARRAS_TRIFASICAS = [860, 840, 844, 848, 890]
PASTA_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_monte_carlo", "realizacoes_sorteadas")

def carregar_dados_nivel(pasta_nivel):
    """Carrega todos os CSVs de um nível de penetração."""
    dados = {}
    
    # CSV 1: Resumo configurações
    try:
        dados['resumo'] = pd.read_csv(
            os.path.join(pasta_nivel, "01_resumo_configuracoes.csv"),
            sep=";", decimal=","
        )
    except FileNotFoundError:
        print(f"⚠️ Arquivo 01_resumo_configuracoes.csv não encontrado em {pasta_nivel}")
        dados['resumo'] = None
    
    # CSV 2: Perfis irradiância
    try:
        dados['irradiancia'] = pd.read_csv(
            os.path.join(pasta_nivel, "02_perfis_irradiancia.csv"),
            sep=";", decimal=","
        )
    except FileNotFoundError:
        print(f"⚠️ Arquivo 02_perfis_irradiancia.csv não encontrado em {pasta_nivel}")
        dados['irradiancia'] = None
    
    # CSV 3: Fatores incerteza carga
    try:
        dados['incerteza_carga'] = pd.read_csv(
            os.path.join(pasta_nivel, "03_fatores_incerteza_carga.csv"),
            sep=";", decimal=","
        )
    except FileNotFoundError:
        print(f"⚠️ Arquivo 03_fatores_incerteza_carga.csv não encontrado em {pasta_nivel}")
        dados['incerteza_carga'] = None
    
    # CSV 4: Unidades PV
    try:
        dados['pv'] = pd.read_csv(
            os.path.join(pasta_nivel, "04_unidades_pv.csv"),
            sep=";", decimal=","
        )
    except FileNotFoundError:
        print(f"⚠️ Arquivo 04_unidades_pv.csv não encontrado em {pasta_nivel}")
        dados['pv'] = None
    
    # CSV 5: Unidades BESS
    try:
        dados['bess'] = pd.read_csv(
            os.path.join(pasta_nivel, "05_unidades_bess.csv"),
            sep=";", decimal=","
        )
    except FileNotFoundError:
        print(f"⚠️ Arquivo 05_unidades_bess.csv não encontrado em {pasta_nivel}")
        dados['bess'] = None
    
    return dados

def analisar_irradiancia(df_irr, df_resumo):
    """
    Analisa irradiância: fator de pico por realização e frequência de tipos de dia.
    Retorna dict com estatísticas.
    """
    stats = {}
    
    if df_irr is None:
        return stats
    
    # Colunas de horas são h00, h01, ..., h23
    colunas_horas = [f"h{h:02d}" for h in range(24)]
    
    # Calcula fator de pico (máximo das 24 horas) para cada realização
    picos = []
    for idx, row in df_irr.iterrows():
        valores_hora = [row[col] for col in colunas_horas]
        pico = max(valores_hora)
        picos.append(pico)
    
    picos = np.array(picos)
    
    stats['Fator de Pico Horário (Irradiância)'] = {
        'Mínimo': picos.min(),
        'Máximo': picos.max(),
        'Média': picos.mean(),
        'Desvio Padrão': picos.std(),
    }
    
    # Frequência de tipos de dia
    if df_resumo is not None:
        tipo_dia_counts = df_resumo['tipo_dia'].value_counts()
        total = len(df_resumo)
        
        for tipo_dia, count in tipo_dia_counts.items():
            freq = count / total
            stats[f'Frequência: {tipo_dia}'] = {
                'Mínimo': '-',
                'Máximo': '-',
                'Média': f'{freq:.1%}',
                'Desvio Padrão': '-',
            }
    
    return stats

def analisar_alocacao_pv(df_pv):
    """
    Analisa alocação PV: potência por barra trifásica.
    Retorna dict com estatísticas para cada barra.
    """
    stats = {}
    
    if df_pv is None:
        return stats
    
    for barra in BARRAS_TRIFASICAS:
        potencias = df_pv[df_pv['barra'] == barra]['potencia_kw'].values
        
        if len(potencias) == 0:
            continue
        
        stats[f'Potência FV na Barra {barra} (kW)'] = {
            'Mínimo': potencias.min(),
            'Máximo': potencias.max(),
            'Média': potencias.mean(),
            'Desvio Padrão': potencias.std(),
        }
    
    return stats

def analisar_bess(df_bess):
    """
    Analisa BESS: potência nominal, capacidade e razão de armazenamento.
    Retorna dict com estatísticas agregadas por realização.
    """
    stats = {}
    
    if df_bess is None:
        return stats
    
    # Agrupa por realização
    por_realizacao = df_bess.groupby('id_realizacao').agg({
        'potencia_kw': 'sum',
        'capacidade_kwh': 'sum',
    }).reset_index()
    
    # Calcula razão de armazenamento (τ = E/P)
    por_realizacao['razao_armazenamento_h'] = (
        por_realizacao['capacidade_kwh'] / por_realizacao['potencia_kw']
    ).replace([np.inf, -np.inf], 0)  # Trata divisão por zero
    
    # Potência BESS
    potencias = por_realizacao['potencia_kw'].values
    stats['Potência BESS Total (kW)'] = {
        'Mínimo': potencias.min(),
        'Máximo': potencias.max(),
        'Média': potencias.mean(),
        'Desvio Padrão': potencias.std(),
    }
    
    # Capacidade BESS
    capacidades = por_realizacao['capacidade_kwh'].values
    stats['Capacidade BESS Total (kWh)'] = {
        'Mínimo': capacidades.min(),
        'Máximo': capacidades.max(),
        'Média': capacidades.mean(),
        'Desvio Padrão': capacidades.std(),
    }
    
    # Razão de armazenamento τ
    razoes = por_realizacao['razao_armazenamento_h'].values
    razoes = razoes[razoes > 0]  # Remove zeros (sem BESS)
    
    if len(razoes) > 0:
        stats['Razão de Armazenamento τ = E/P (horas)'] = {
            'Mínimo': razoes.min(),
            'Máximo': razoes.max(),
            'Média': razoes.mean(),
            'Desvio Padrão': razoes.std(),
        }
    
    return stats

def analisar_incerteza_carga(df_incerteza):
    """
    Analisa fatores de incerteza de carga: média e desvio sobre todas as barras e horas.
    Retorna dict com estatísticas agregadas.
    """
    stats = {}
    
    if df_incerteza is None:
        return stats
    
    # Colunas de horas são h00, h01, ..., h23
    colunas_horas = [f"h{h:02d}" for h in range(24)]
    
    # Extrai todos os valores de fatores (sem id_realizacao e barra)
    todos_fatores = []
    for col in colunas_horas:
        todos_fatores.extend(df_incerteza[col].values)
    
    todos_fatores = np.array(todos_fatores)
    
    stats['Fatores de Incerteza de Carga (agregado)'] = {
        'Mínimo': todos_fatores.min(),
        'Máximo': todos_fatores.max(),
        'Média': todos_fatores.mean(),
        'Desvio Padrão': todos_fatores.std(),
    }
    
    return stats

def compilar_tabelas_por_nivel(nivel_pct):
    """
    Compila todas as estatísticas para um nível de penetração específico.
    Retorna dict com todas as estatísticas organizadas.
    """
    pasta_nivel = os.path.join(PASTA_BASE, f"pen_{nivel_pct:03d}pct")
    
    print(f"\n📊 Processando nível {nivel_pct}%...", end=" ")
    
    if not os.path.exists(pasta_nivel):
        print(f"❌ Pasta não encontrada: {pasta_nivel}")
        return None
    
    # Carrega dados
    dados = carregar_dados_nivel(pasta_nivel)
    
    # Análises
    todas_stats = {}
    
    # Irradiância
    todas_stats.update(analisar_irradiancia(dados['irradiancia'], dados['resumo']))
    
    # Alocação PV
    todas_stats.update(analisar_alocacao_pv(dados['pv']))
    
    # BESS
    todas_stats.update(analisar_bess(dados['bess']))
    
    # Incerteza de carga
    todas_stats.update(analisar_incerteza_carga(dados['incerteza_carga']))
    
    print("✓")
    return todas_stats

def criar_tabela_latex(tabelas_por_nivel):
    """
    Cria uma tabela LaTeX com estatísticas dos 4 níveis especiais.
    Retorna string com código LaTeX.
    """
    latex = []
    latex.append(r"\documentclass[11pt]{article}")
    latex.append(r"\usepackage[utf-8]{inputenc}")
    latex.append(r"\usepackage[brazilian]{babel}")
    latex.append(r"\usepackage{booktabs}")
    latex.append(r"\usepackage{amsmath}")
    latex.append(r"\usepackage{siunitx}")
    latex.append(r"\usepackage{geometry}")
    latex.append(r"\geometry{margin=1cm, landscape}")
    latex.append(r"\begin{document}")
    latex.append("")
    
    # Coleta todas as chaves de variáveis
    todas_variaveis = set()
    for tabela in tabelas_por_nivel.values():
        if tabela:
            todas_variaveis.update(tabela.keys())
    
    todas_variaveis = sorted(todas_variaveis)
    
    # Cria tabela
    latex.append(r"\begin{center}")
    latex.append(r"\footnotesize")
    latex.append(r"\setlength{\tabcolsep}{3pt}")
    latex.append(r"\begin{tabular}{l|rr|rr|rr|rr}")
    latex.append(r"\toprule")
    
    # Cabeçalho
    latex.append(r"Variável & \multicolumn{2}{c|}{0\% PV} & \multicolumn{2}{c|}{50\% PV} & \multicolumn{2}{c|}{100\% PV} & \multicolumn{2}{c}{150\% PV} \\")
    latex.append(r" & Mín/Máx & Med(DP) & Mín/Máx & Med(DP) & Mín/Máx & Med(DP) & Mín/Máx & Med(DP) \\")
    latex.append(r"\midrule")
    
    # Linhas
    for variavel in todas_variaveis:
        linha = [variavel.replace("_", r"\_")]
        
        for nivel in NIVEIS_ESPECIAIS:
            tabela = tabelas_por_nivel.get(nivel)
            if tabela and variavel in tabela:
                stats = tabela[variavel]
                
                # Trata frequências especiais
                if isinstance(stats.get('Média'), str) and '%' in str(stats.get('Média')):
                    # Frequência
                    linha.append(f"& {stats['Média']}")
                else:
                    min_val = stats.get('Mínimo', '-')
                    max_val = stats.get('Máximo', '-')
                    media = stats.get('Média', '-')
                    dp = stats.get('Desvio Padrão', '-')
                    
                    if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                        linha.append(f"& {min_val:.3f}/{max_val:.3f}")
                    else:
                        linha.append(f"& {min_val}/{max_val}")
                    
                    if isinstance(media, (int, float)) and isinstance(dp, (int, float)):
                        linha.append(f" & {media:.3f}({dp:.3f})")
                    else:
                        linha.append(f" & {media}({dp})")
            else:
                linha.append("& - & -")
        
        latex.append(" ".join(linha) + r" \\")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{center}")
    latex.append(r"\end{document}")
    
    return "\n".join(latex)

def criar_tabela_csv(tabelas_por_nivel):
    """
    Cria uma tabela em CSV com estatísticas dos 4 níveis especiais.
    Retorna string com código CSV.
    """
    linhas = []
    
    # Cabeçalho
    cabecalho = ["Variável"]
    for nivel in NIVEIS_ESPECIAIS:
        cabecalho.extend([f"{nivel}% - Mínimo", f"{nivel}% - Máximo", f"{nivel}% - Média", f"{nivel}% - Desvio Padrão"])
    linhas.append(";".join(cabecalho))
    
    # Coleta todas as chaves de variáveis
    todas_variaveis = set()
    for tabela in tabelas_por_nivel.values():
        if tabela:
            todas_variaveis.update(tabela.keys())
    
    todas_variaveis = sorted(todas_variaveis)
    
    # Linhas
    for variavel in todas_variaveis:
        linha = [variavel]
        
        for nivel in NIVEIS_ESPECIAIS:
            tabela = tabelas_por_nivel.get(nivel)
            if tabela and variavel in tabela:
                stats = tabela[variavel]
                
                min_val = stats.get('Mínimo', '-')
                max_val = stats.get('Máximo', '-')
                media = stats.get('Média', '-')
                dp = stats.get('Desvio Padrão', '-')
                
                # Formata valores
                if isinstance(min_val, (int, float)):
                    min_str = f"{min_val:.6f}".rstrip('0').rstrip('.')
                else:
                    min_str = str(min_val)
                
                if isinstance(max_val, (int, float)):
                    max_str = f"{max_val:.6f}".rstrip('0').rstrip('.')
                else:
                    max_str = str(max_val)
                
                if isinstance(media, (int, float)):
                    media_str = f"{media:.6f}".rstrip('0').rstrip('.')
                else:
                    media_str = str(media)
                
                if isinstance(dp, (int, float)):
                    dp_str = f"{dp:.6f}".rstrip('0').rstrip('.')
                else:
                    dp_str = str(dp)
                
                linha.extend([min_str, max_str, media_str, dp_str])
            else:
                linha.extend(["-", "-", "-", "-"])
        
        linhas.append(";".join(linha))
    
    return "\n".join(linhas)

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  ANÁLISE DE ESTATÍSTICAS DESCRITIVAS - CENÁRIOS MONTE CARLO")
    print("=" * 80)
    print(f"  Níveis especiais de análise: {NIVEIS_ESPECIAIS}")
    print(f"  Barras trifásicas: {BARRAS_TRIFASICAS}")
    print("=" * 80)
    
    # Compila estatísticas para cada nível
    tabelas_por_nivel = {}
    for nivel in NIVEIS_ESPECIAIS:
        tabelas_por_nivel[nivel] = compilar_tabelas_por_nivel(nivel)
    
    # Criar arquivos de saída
    pasta_saida = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "resultados_monte_carlo",
        "analise_opendss"
    )
    os.makedirs(pasta_saida, exist_ok=True)
    
    # Salva tabela CSV
    csv_content = criar_tabela_csv(tabelas_por_nivel)
    csv_path = os.path.join(pasta_saida, "tabela_estatisticas_descritivas.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_content)
    print(f"\n✓ Tabela CSV salva: {csv_path}")
    
    # Salva tabela LaTeX
    latex_content = criar_tabela_latex(tabelas_por_nivel)
    latex_path = os.path.join(pasta_saida, "tabela_estatisticas_descritivas.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
    print(f"✓ Tabela LaTeX salva: {latex_path}")
    
    # Cria também uma versão em HTML para preview
    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Análise Estatística - Cenários Monte Carlo</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: right; }}
            th {{ background-color: #4CAF50; color: white; }}
            td:first-child {{ text-align: left; }}
            h1, h2 {{ color: #333; }}
        </style>
    </head>
    <body>
        <h1>Análise Estatística Descritiva - Cenários Monte Carlo</h1>
        <p>Níveis analisados: 0%, 50%, 100%, 150%</p>
        
        <h2>Tabela de Estatísticas</h2>
        <table>
            <thead>
                <tr>
                    <th>Variável</th>
                    <th colspan="4">0% PV</th>
                    <th colspan="4">50% PV</th>
                    <th colspan="4">100% PV</th>
                    <th colspan="4">150% PV</th>
                </tr>
                <tr>
                    <th></th>
                    <th>Mín</th><th>Máx</th><th>Média</th><th>DP</th>
                    <th>Mín</th><th>Máx</th><th>Média</th><th>DP</th>
                    <th>Mín</th><th>Máx</th><th>Média</th><th>DP</th>
                    <th>Mín</th><th>Máx</th><th>Média</th><th>DP</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Coleta todas as variáveis
    todas_variaveis = set()
    for tabela in tabelas_por_nivel.values():
        if tabela:
            todas_variaveis.update(tabela.keys())
    
    for variavel in sorted(todas_variaveis):
        html_content += f"<tr><td><strong>{variavel}</strong></td>"
        for nivel in NIVEIS_ESPECIAIS:
            tabela = tabelas_por_nivel.get(nivel)
            if tabela and variavel in tabela:
                stats = tabela[variavel]
                for col in ['Mínimo', 'Máximo', 'Média', 'Desvio Padrão']:
                    val = stats.get(col, '-')
                    if isinstance(val, (int, float)):
                        html_content += f"<td>{val:.6f}</td>"
                    else:
                        html_content += f"<td>{val}</td>"
            else:
                html_content += "<td>-</td>" * 4
        html_content += "</tr>\n"
    
    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """
    
    html_path = os.path.join(pasta_saida, "tabela_estatisticas_descritivas.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✓ Tabela HTML salva: {html_path}")
    
    print("\n" + "=" * 80)
    print("  ✅ ANÁLISE FINALIZADA COM SUCESSO!")
    print("=" * 80)
    print(f"\nArquivos gerados em: {pasta_saida}")
    print(f"  - tabela_estatisticas_descritivas.csv")
    print(f"  - tabela_estatisticas_descritivas.tex")
    print(f"  - tabela_estatisticas_descritivas.html")
    print("=" * 80 + "\n")
