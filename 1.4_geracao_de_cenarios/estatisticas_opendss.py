import os

import numpy as np
import pandas as pd

from simulacoes_config import BARRAS_TRIFASICAS_ANALISE, NIVEIS_ESPECIAIS_ANALISE


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
    
    for barra in BARRAS_TRIFASICAS_ANALISE:
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
    Analisa BESS: potência nominal e capacidade agregadas por realização.
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

def analisar_defeitos_tensao(pasta_saida_opendss):
    """
    Analisa defeitos de tensão (subtensão e sobretensão) do master_resultados_opendss.csv.
    Retorna dict com estatísticas agregadas sobre todas as faixas horárias.
    """
    stats = {}
    
    # Carrega master_resultados_opendss.csv
    try:
        df_resultados = pd.read_csv(
            os.path.join(pasta_saida_opendss, "master_resultados_opendss.csv"),
            sep=";", decimal=","
        )
    except FileNotFoundError:
        # Se ainda não foi gerado, retorna stats vazio
        return stats
    
    if df_resultados.empty:
        return stats
    
    # Colunas de subtensão
    colunas_subtensao = [col for col in df_resultados.columns if col.startswith('subtensao_')]
    # Colunas de sobretensão
    colunas_sobretensao = [col for col in df_resultados.columns if col.startswith('sobretensao_')]
    
    # Agrega subtensão: soma de todos os defeitos em todas as faixas
    if colunas_subtensao:
        subtensoes = df_resultados[colunas_subtensao].sum(axis=1).values
        stats['Número de Barras com Subtensão'] = {
            'Mínimo': float(subtensoes.min()),
            'Máximo': float(subtensoes.max()),
            'Média': float(subtensoes.mean()),
            'Desvio Padrão': float(subtensoes.std()),
        }
    
    # Agrega sobretensão: soma de todos os defeitos em todas as faixas
    if colunas_sobretensao:
        sobretensoes = df_resultados[colunas_sobretensao].sum(axis=1).values
        stats['Número de Barras com Sobretensão'] = {
            'Mínimo': float(sobretensoes.min()),
            'Máximo': float(sobretensoes.max()),
            'Média': float(sobretensoes.mean()),
            'Desvio Padrão': float(sobretensoes.std()),
        }
    
    return stats

def compilar_tabelas_por_nivel(nivel_pct, pasta_base_monte_carlo, pasta_saida_opendss):
    """
    Compila todas as estatísticas para um nível de penetração específico.
    Retorna dict com todas as estatísticas organizadas.
    """
    pasta_nivel = os.path.join(pasta_base_monte_carlo, "realizacoes_sorteadas", f"pen_{nivel_pct:03d}pct")
    
    if not os.path.exists(pasta_nivel):
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
    
    # Defeitos de tensão (subtensão e sobretensão)
    todas_stats.update(analisar_defeitos_tensao(pasta_saida_opendss))
    
    return todas_stats

def criar_tabela_csv(tabelas_por_nivel):
    """
    Cria uma tabela em CSV com estatísticas dos 4 níveis especiais.
    Retorna string com código CSV.
    """
    linhas = []
    
    # Cabeçalho
    cabecalho = ["Variável"]
    for nivel in NIVEIS_ESPECIAIS_ANALISE:
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
        
        for nivel in NIVEIS_ESPECIAIS_ANALISE:
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

def criar_tabela_html(tabelas_por_nivel, contexto: str = ""):
    """
    Cria uma tabela em HTML com estatísticas dos 4 níveis especiais.
    Retorna string com código HTML.
    """
    sufixo_html = f" — {contexto}" if contexto else ""
    html = []
    html.append(f"""<html>
<head>
    <meta charset="utf-8">
    <title>Análise Estatística - Cenários Monte Carlo{sufixo_html}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; margin: 20px 0; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: right; }}
        th {{ background-color: #4CAF50; color: white; }}
        td:first-child {{ text-align: left; }}
        h1, h2 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>Análise Estatística Descritiva - Cenários Monte Carlo{sufixo_html}</h1>
    <p>Níveis analisados: 0%, 50%, 100%, 150%</p>
    
    <h2>Tabela de Estatísticas</h2>
    <table>
        <thead>
            <tr>
                <th>Variável</th>""")
    
    for nivel in NIVEIS_ESPECIAIS_ANALISE:
        html.append(f'                <th colspan="4">{nivel}% PV</th>')
    
    html.append("""            </tr>
            <tr>
                <th></th>""")
    
    for _ in NIVEIS_ESPECIAIS_ANALISE:
        html.append("""                <th>Mín</th><th>Máx</th><th>Média</th><th>DP</th>""")
    
    html.append("""            </tr>
        </thead>
        <tbody>""")
    
    # Coleta todas as chaves de variáveis
    todas_variaveis = set()
    for tabela in tabelas_por_nivel.values():
        if tabela:
            todas_variaveis.update(tabela.keys())
    
    todas_variaveis = sorted(todas_variaveis)
    
    # Linhas
    for variavel in todas_variaveis:
        html.append(f"            <tr><td>{variavel}</td>")
        
        for nivel in NIVEIS_ESPECIAIS_ANALISE:
            tabela = tabelas_por_nivel.get(nivel)
            if tabela and variavel in tabela:
                stats = tabela[variavel]
                
                min_val = stats.get('Mínimo', '-')
                max_val = stats.get('Máximo', '-')
                media = stats.get('Média', '-')
                dp = stats.get('Desvio Padrão', '-')
                
                # Formata valores
                if isinstance(min_val, (int, float)):
                    html.append(f"<td>{min_val:.4f}</td>")
                else:
                    html.append(f"<td>{min_val}</td>")
                
                if isinstance(max_val, (int, float)):
                    html.append(f"<td>{max_val:.4f}</td>")
                else:
                    html.append(f"<td>{max_val}</td>")
                
                if isinstance(media, (int, float)):
                    html.append(f"<td>{media:.4f}</td>")
                else:
                    html.append(f"<td>{media}</td>")
                
                if isinstance(dp, (int, float)):
                    html.append(f"<td>{dp:.4f}</td>")
                else:
                    html.append(f"<td>{dp}</td>")
            else:
                html.append("<td>-</td><td>-</td><td>-</td><td>-</td>")
        
        html.append("</tr>")
    
    html.append("""        </tbody>
    </table>
</body>
</html>""")
    
    return "\n".join(html)

def gerar_tabelas_estatisticas(pasta_saida, pasta_monte_carlo, pasta_saida_opendss=None, contexto: str = ""):
    """
    Gera tabelas de estatísticas descritivas para os níveis especiais.
    Salva em formatos CSV e HTML.
    """
    print(f"{'=' * 80}")
    print(f"  Níveis especiais de análise: {NIVEIS_ESPECIAIS_ANALISE}")
    print(f"  Barras trifásicas: {BARRAS_TRIFASICAS_ANALISE}")

    if pasta_saida_opendss is None:
        pasta_saida_opendss = pasta_saida
    
    # Compila estatísticas para cada nível
    tabelas_por_nivel = {}
    for nivel in NIVEIS_ESPECIAIS_ANALISE:
        tabelas_por_nivel[nivel] = compilar_tabelas_por_nivel(nivel, pasta_monte_carlo, pasta_saida_opendss)
    
    # Salva tabela CSV
    csv_content = criar_tabela_csv(tabelas_por_nivel)
    csv_path = os.path.join(pasta_saida, "tabela_estatisticas_descritivas.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_content)
    
    # Salva tabela HTML
    html_content = criar_tabela_html(tabelas_por_nivel, contexto=contexto)
    html_path = os.path.join(pasta_saida, "tabela_estatisticas_descritivas.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
