import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulacoes_config import BARRAS_TOPOLOGICAS_IEEE34, TIME_BANDS, V_PU_MAX, V_PU_MIN


def salvar_figura(fig, caminho, dpi=200):
    """
    Salva a figura via arquivo temporario e depois substitui o destino.
    Isso evita falhas do PIL/Windows ao sobrescrever PNGs em pastas sincronizadas.
    """
    pasta = os.path.dirname(caminho)
    os.makedirs(pasta, exist_ok=True)
    nome = os.path.basename(caminho)
    caminho_temporario = os.path.join(pasta, f".tmp_{nome}")

    fig.savefig(caminho_temporario, dpi=dpi, bbox_inches="tight")
    os.replace(caminho_temporario, caminho)


def calcular_limites_y_envelope(global_ymin, global_ymax, margem_minima=0.03):
    """Calcula limites do eixo y com folga para nao cortar envelopes ou limites."""
    valores = np.array([global_ymin, global_ymax, V_PU_MIN, V_PU_MAX], dtype=float)
    valores = valores[np.isfinite(valores)]
    if len(valores) == 0:
        return 0.90, 1.10

    ymin = float(np.min(valores))
    ymax = float(np.max(valores))
    amplitude = max(ymax - ymin, 0.01)
    margem = max(margem_minima, 0.08 * amplitude)
    return ymin - margem, ymax + margem


def plotar_envelope_tensao(pasta_resultados, pasta_saida):
    """
    Gera figuras de envelope de tensão (perfil ao longo das barras do alimentador)
    para cada faixa horária, sobrepondo os diferentes níveis de penetração.

    Para cada combinação (nível de penetração × faixa horária), a função:
      - ordena as barras conforme BARRAS_TOPOLOGICAS_IEEE34;
      - calcula, sobre as 500 realizações, a mediana e os percentis 5 e 95 da
        tensão em cada barra;
      - traça o envelope sombreado (P5–P95) e a linha de mediana;
      - marca com linhas tracejadas os limites regulatórios 0,95 e 1,05 pu.
    """
    faixas = list(TIME_BANDS.keys())

    # Paleta de cores para os níveis de penetração (até 16 níveis: 0–150% em passos de 10%)
    cmap = plt.get_cmap("plasma")
    # Coleta todos os níveis disponíveis examinando as pastas de saída
    niveis_disponiveis = sorted([
        int(d.split("_")[1].replace("pct", ""))
        for d in os.listdir(pasta_resultados)
        if d.startswith("pen_") and os.path.isdir(os.path.join(pasta_resultados, d))
    ])
    if not niveis_disponiveis:
        print("  ⚠ Nenhuma pasta de nível de penetração encontrada para envelope de tensão.")
        return

    n_niveis = len(niveis_disponiveis)
    cores = {nivel: cmap(i / max(n_niveis - 1, 1)) for i, nivel in enumerate(niveis_disponiveis)}

    # Índices das barras na ordem topológica (apenas as que aparecem nos dados)
    barras_topologicas_str = [str(b).lower() for b in BARRAS_TOPOLOGICAS_IEEE34]

    for faixa in faixas:
        fig, ax = plt.subplots(figsize=(16, 6))

        handles_legenda = []
        plotou_algo = False
        global_ymin = V_PU_MIN
        global_ymax = V_PU_MAX

        for nivel in niveis_disponiveis:
            nivel_dir = f"pen_{nivel:03d}pct"
            caminho_csv = os.path.join(pasta_resultados, nivel_dir, f"envelope_tensao_{faixa}.csv")
            if not os.path.isfile(caminho_csv):
                continue

            df_env = pd.read_csv(caminho_csv, sep=";", decimal=",")
            if df_env.empty:
                continue

            # Intersecção entre barras topológicas e barras presentes nos dados
            barras_presentes = [b for b in barras_topologicas_str if b in df_env.columns]
            if not barras_presentes:
                continue

            df_ordenado = df_env[barras_presentes]
            x = list(range(len(barras_presentes)))
            rotulos_x = [str(b).upper() for b in barras_presentes]

            mediana = df_ordenado.median(axis=0).values
            p5 = df_ordenado.quantile(0.05, axis=0).values
            p95 = df_ordenado.quantile(0.95, axis=0).values

            cor = cores[nivel]
            global_ymin = min(global_ymin, float(np.nanmin(p5)))
            global_ymax = max(global_ymax, float(np.nanmax(p95)))
            fill = ax.fill_between(x, p5, p95, alpha=0.20, color=cor)
            line, = ax.plot(x, mediana, color=cor, linewidth=1.8, label=f"{nivel}% PV")
            handles_legenda.append(line)
            plotou_algo = True

        if not plotou_algo:
            plt.close(fig)
            continue

        # Limites regulatórios
        lim_min = ax.axhline(V_PU_MIN, color="red", linestyle="--", linewidth=1.2, label=f"Limite inferior ({V_PU_MIN} pu)")
        lim_max = ax.axhline(V_PU_MAX, color="darkred", linestyle="--", linewidth=1.2, label=f"Limite superior ({V_PU_MAX} pu)")

        ax.set_xticks(x)
        ax.set_xticklabels(rotulos_x, rotation=90, fontsize=7)
        ax.set_xlabel("Barras do alimentador (ordem topológica — subestação → extremidades)", fontsize=11)
        ax.set_ylabel("Módulo de tensão (pu)", fontsize=11)
        ax.set_title(
            f"Tensão ao longo do alimentador IEEE34 — Faixa horária: {faixa.capitalize()}\n"
            f"Mediana e intervalo P5–P95 para cada nível de penetração FV",
            fontsize=12,
        )
        ax.set_ylim(*calcular_limites_y_envelope(global_ymin, global_ymax))
        ax.grid(True, linestyle="--", alpha=0.35)

        ax.legend(
            handles=handles_legenda + [lim_min, lim_max],
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=8,
            ncol=1,
            title="Penetração FV / Limites",
            title_fontsize=8,
        )
      

        fig.tight_layout(rect=(0, 0, 0.82, 1))
        caminho_fig = os.path.join(pasta_saida, f"envelope_tensao_{faixa}.png")
        salvar_figura(fig, caminho_fig)
        plt.close(fig)
        
def plotar_envelope_tensao_por_hora(caminho_tensoes_completas, pasta_saida):
    """
    Gera uma figura de envelope de tensao para cada hora do dia usando o CSV
    completo de tensoes por fase. Para cada barra/hora/realizacao usa a pior
    fase disponivel como valor representativo.
    """
    if not os.path.isfile(caminho_tensoes_completas):
        print(f"  CSV de tensoes completas nao encontrado: {caminho_tensoes_completas}")
        return []

    colunas_fases = ["tensao_fase_1_pu", "tensao_fase_2_pu", "tensao_fase_3_pu"]
    df = pd.read_csv(caminho_tensoes_completas, sep=";", decimal=",")
    if df.empty:
        return []

    df["barra"] = df["barra"].astype(str).str.lower()
    df["tensao_pior_fase_pu"] = df[colunas_fases].max(axis=1, skipna=True)
    niveis = sorted(df["pen_pct"].dropna().unique())
    barras_topologicas_str = [str(b).lower() for b in BARRAS_TOPOLOGICAS_IEEE34]
    cmap = plt.get_cmap("plasma")
    cores = {nivel: cmap(i / max(len(niveis) - 1, 1)) for i, nivel in enumerate(niveis)}
    caminhos = []

    for hora in range(24):
        df_hora = df[df["hora"] == hora]
        if df_hora.empty:
            continue

        fig, ax = plt.subplots(figsize=(16, 6))
        handles_legenda = []
        plotou_algo = False
        global_ymin = V_PU_MIN
        global_ymax = V_PU_MAX

        for nivel in niveis:
            df_nivel = df_hora[df_hora["pen_pct"] == nivel]
            if df_nivel.empty:
                continue

            tabela = df_nivel.pivot_table(
                index="id_realizacao",
                columns="barra",
                values="tensao_pior_fase_pu",
                aggfunc="max",
            )
            barras_presentes = [b for b in barras_topologicas_str if b in tabela.columns]
            if not barras_presentes:
                continue

            tabela = tabela[barras_presentes]
            x = list(range(len(barras_presentes)))
            rotulos_x = [str(b).upper() for b in barras_presentes]
            mediana = tabela.median(axis=0).values
            p5 = tabela.quantile(0.05, axis=0).values
            p95 = tabela.quantile(0.95, axis=0).values
            cor = cores[nivel]
            global_ymin = min(global_ymin, float(np.nanmin(p5)))
            global_ymax = max(global_ymax, float(np.nanmax(p95)))
            ax.fill_between(x, p5, p95, alpha=0.20, color=cor)
            line, = ax.plot(x, mediana, color=cor, linewidth=1.8, label=f"{int(nivel)}% PV")
            handles_legenda.append(line)
            plotou_algo = True

        if not plotou_algo:
            plt.close(fig)
            continue

        lim_min = ax.axhline(V_PU_MIN, color="red", linestyle="--", linewidth=1.2, label=f"Limite inferior ({V_PU_MIN} pu)")
        lim_max = ax.axhline(V_PU_MAX, color="darkred", linestyle="--", linewidth=1.2, label=f"Limite superior ({V_PU_MAX} pu)")
        ax.set_xticks(x)
        ax.set_xticklabels(rotulos_x, rotation=90, fontsize=7)
        ax.set_xlabel("Barras do alimentador (ordem topologica - subestacao -> extremidades)", fontsize=11)
        ax.set_ylabel("Modulo de tensao da pior fase (pu)", fontsize=11)
        ax.set_title(
            f"Envelope de tensao horario no alimentador IEEE34 - Hora {hora:02d}:00\n"
            "Mediana e intervalo P5-P95 para cada nivel de penetracao FV",
            fontsize=12,
        )
        ax.set_ylim(*calcular_limites_y_envelope(global_ymin, global_ymax))
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(
            handles=handles_legenda + [lim_min, lim_max],
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=8,
            ncol=1,
            title="Penetracao FV / Limites",
            title_fontsize=8,
        )

        fig.tight_layout(rect=(0, 0, 0.82, 1))
        caminho_fig = os.path.join(pasta_saida, f"envelope_tensao_hora_{hora:02d}.png")
        salvar_figura(fig, caminho_fig)
        plt.close(fig)
        caminhos.append(caminho_fig)

    return caminhos

def plotar_boxplot_geral(df_master, pasta_saida):
    """Gera boxplots gerais separados para subtensão e sobretensão"""
    niveis = sorted(df_master["pen_pct"].unique())
    faixas = ["manhã", "tarde", "noite"]  # madrugada removida por irrelevância
    
    # Boxplot para Subtensão
    fig, axes = plt.subplots(3, 1, figsize=(16, 18), constrained_layout=True)
    for ax, faixa in zip(axes, faixas):
        data = [
            df_master.loc[df_master["pen_pct"] == nivel, f"subtensao_{faixa}"]
            for nivel in niveis
        ]
        ax.boxplot(data, tick_labels=[str(n) for n in niveis], showfliers=False)
        ax.set_title(f"Número de barras com SUBTENSÃO (V < 0.95 pu) - faixa {faixa.capitalize()}")
        ax.set_xlabel("Penetração PV (%)")
        ax.set_ylabel("Barras com subtensão")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(0, 38)

    fig.suptitle("Subtensão por faixa de horário (agregado de todos os tipos de dia)", fontsize=18)
    caminho_subtensao = os.path.join(pasta_saida, "boxplot_subtensao_geral.png")
    salvar_figura(fig, caminho_subtensao)
    plt.close(fig)
    
    # Boxplot para Sobretensão
    fig, axes = plt.subplots(3, 1, figsize=(16, 18), constrained_layout=True)
    for ax, faixa in zip(axes, faixas):
        data = [
            df_master.loc[df_master["pen_pct"] == nivel, f"sobretensao_{faixa}"]
            for nivel in niveis
        ]
        ax.boxplot(data, tick_labels=[str(n) for n in niveis], showfliers=False)
        ax.set_title(f"Número de barras com SOBRETENSÃO (V > 1.05 pu) - faixa {faixa.capitalize()}")
        ax.set_xlabel("Penetração PV (%)")
        ax.set_ylabel("Barras com sobretensão")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(0, 38)

    fig.suptitle("Sobretensão por faixa de horário (agregado de todos os tipos de dia)", fontsize=18)
    caminho_sobretensao = os.path.join(pasta_saida, "boxplot_sobretensao_geral.png")
    salvar_figura(fig, caminho_sobretensao)
    plt.close(fig)
    
    return caminho_subtensao, caminho_sobretensao


def plotar_boxplot_por_time_bands(df_master, pasta_saida):
    """Gera boxplots separados por time bands, com distinção entre subtensão e sobretensão"""
    niveis = sorted(df_master["pen_pct"].unique())
    faixas = ["manhã", "tarde", "noite"]  # madrugada removida por irrelevância
    
    # Boxplot para Subtensão
    fig, axes = plt.subplots(3, 1, figsize=(16, 18), constrained_layout=True)
    for ax, faixa in zip(axes, faixas):
        data = [
            df_master.loc[df_master["pen_pct"] == nivel, f"subtensao_{faixa}"]
            for nivel in niveis
        ]
        ax.boxplot(data, tick_labels=[str(n) for n in niveis], showfliers=False)
        ax.set_title(f"Número de barras com SUBTENSÃO (V < 0.95 pu) - faixa {faixa.capitalize()}")
        ax.set_xlabel("Penetração PV (%)")
        ax.set_ylabel("Barras com subtensão")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(0, 38)

    fig.suptitle("Subtensão por faixa de horário (agregado de todos os tipos de dia)", fontsize=18)
    caminho_subtensao = os.path.join(pasta_saida, "boxplot_subtensao_por_time_bands.png")
    salvar_figura(fig, caminho_subtensao)
    plt.close(fig)
    
    # Boxplot para Sobretensão
    fig, axes = plt.subplots(3, 1, figsize=(16, 18), constrained_layout=True)
    for ax, faixa in zip(axes, faixas):
        data = [
            df_master.loc[df_master["pen_pct"] == nivel, f"sobretensao_{faixa}"]
            for nivel in niveis
        ]
        ax.boxplot(data, tick_labels=[str(n) for n in niveis], showfliers=False)
        ax.set_title(f"Número de barras com SOBRETENSÃO (V > 1.05 pu) - faixa {faixa.capitalize()}")
        ax.set_xlabel("Penetração PV (%)")
        ax.set_ylabel("Barras com sobretensão")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(0, 38)

    fig.suptitle("Sobretensão por faixa de horário (agregado de todos os tipos de dia)", fontsize=18)
    caminho_sobretensao = os.path.join(pasta_saida, "boxplot_sobretensao_por_time_bands.png")
    salvar_figura(fig, caminho_sobretensao)
    plt.close(fig)
    
    return caminho_subtensao, caminho_sobretensao


def plotar_boxplot(df_master, pasta_saida):
    """Gera boxplots separados por tipo de dia e por tipo de defeito (subtensão/sobretensão)"""
    niveis = sorted(df_master["pen_pct"].unique())
    tipos_dia = sorted(df_master["tipo_dia"].unique())
    faixas = ["manhã", "tarde", "noite"]  # madrugada removida por irrelevância
    
    caminhos_imagens = []
    
    # Gerar um gráfico para cada tipo de dia
    for tipo_dia in tipos_dia:
        df_filtrado = df_master[df_master["tipo_dia"] == tipo_dia]
        
        # Gráfico para Subtensão
        fig, axes = plt.subplots(3, 1, figsize=(16, 18), constrained_layout=True)
        for ax, faixa in zip(axes, faixas):
            data = [
                df_filtrado.loc[df_filtrado["pen_pct"] == nivel, f"subtensao_{faixa}"]
                for nivel in niveis
            ]
            ax.boxplot(data, tick_labels=[str(n) for n in niveis], showfliers=False)
            ax.set_title(f"Número de barras com SUBTENSÃO (V < 0.95 pu) - faixa {faixa.capitalize()}")
            ax.set_xlabel("Penetração PV (%)")
            ax.set_ylabel("Barras com subtensão")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_ylim(0, 38)

        fig.suptitle(f"Subtensão por faixa de horário - Tipo de dia: {tipo_dia}", fontsize=18)
        caminho_imagem = os.path.join(pasta_saida, f"boxplot_subtensao_por_faixa_{tipo_dia}.png")
        salvar_figura(fig, caminho_imagem)
        plt.close(fig)
        caminhos_imagens.append(caminho_imagem)
        
        # Gráfico para Sobretensão
        fig, axes = plt.subplots(3, 1, figsize=(16, 18), constrained_layout=True)
        for ax, faixa in zip(axes, faixas):
            data = [
                df_filtrado.loc[df_filtrado["pen_pct"] == nivel, f"sobretensao_{faixa}"]
                for nivel in niveis
            ]
            ax.boxplot(data, tick_labels=[str(n) for n in niveis], showfliers=False)
            ax.set_title(f"Número de barras com SOBRETENSÃO (V > 1.05 pu) - faixa {faixa.capitalize()}")
            ax.set_xlabel("Penetração PV (%)")
            ax.set_ylabel("Barras com sobretensão")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_ylim(0, 38)

        fig.suptitle(f"Sobretensão por faixa de horário - Tipo de dia: {tipo_dia}", fontsize=18)
        caminho_imagem = os.path.join(pasta_saida, f"boxplot_sobretensao_por_faixa_{tipo_dia}.png")
        salvar_figura(fig, caminho_imagem)
        plt.close(fig)
        caminhos_imagens.append(caminho_imagem)
    
    return caminhos_imagens


def plotar_boxplot_por_faixa_bess(df_master, pasta_saida):
    """Gera boxplots separados por faixa de operação BESS, com distinção entre subtensão/sobretensão"""
    niveis = sorted(df_master["pen_pct"].unique())
    tipos_dia = sorted(df_master["tipo_dia"].unique())
    
    mapeamento_faixas = {
        "subtensao_carga": ("Subtensão", "Carga (10h-14h)"),
        "sobretensao_carga": ("Sobretensão", "Carga (10h-14h)"),
        "subtensao_pós_carga_pré_descarga": ("Subtensão", "Pós-carga/Pré-descarga (15h-17h)"),
        "sobretensao_pós_carga_pré_descarga": ("Sobretensão", "Pós-carga/Pré-descarga (15h-17h)"),
        "subtensao_descarga": ("Subtensão", "Descarga (18h-21h)"),
        "sobretensao_descarga": ("Sobretensão", "Descarga (18h-21h)"),
        "subtensao_fora_de_operacao": ("Subtensão", "Fora de operação (22h-09h)"),
        "sobretensao_fora_de_operacao": ("Sobretensão", "Fora de operação (22h-09h)"),
    }
    
    caminhos_imagens = []
    
    # Gerar um gráfico para cada tipo de dia
    for tipo_dia in tipos_dia:
        df_filtrado = df_master[df_master["tipo_dia"] == tipo_dia]
        
        # Gráfico para Subtensão
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
        axes_flat = axes.flatten()
        
        faixas_subtensao = [f"subtensao_{faixa}" for faixa in ["carga", "pós_carga_pré_descarga", "descarga", "fora_de_operacao"]]
        for ax, coluna in zip(axes_flat, faixas_subtensao):
            data = [
                df_filtrado.loc[df_filtrado["pen_pct"] == nivel, f"subtensao_bess_{coluna.split('_', 1)[1]}"]
                for nivel in niveis
            ]
            ax.boxplot(data, tick_labels=[str(n) for n in niveis], showfliers=False)
            _, titulo = mapeamento_faixas[coluna]
            ax.set_title(f"Número de barras com SUBTENSÃO (V < 0.95 pu) - {titulo}")
            ax.set_xlabel("Penetração PV (%)")
            ax.set_ylabel("Barras com subtensão")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_ylim(0, 38)

        fig.suptitle(f"Subtensão por faixa de operação BESS - Tipo de dia: {tipo_dia}", fontsize=18)
        caminho_imagem = os.path.join(pasta_saida, f"boxplot_subtensao_por_faixa_bess_{tipo_dia}.png")
        salvar_figura(fig, caminho_imagem)
        plt.close(fig)
        caminhos_imagens.append(caminho_imagem)
        
        # Gráfico para Sobretensão
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
        axes_flat = axes.flatten()
        
        faixas_sobretensao = [f"sobretensao_{faixa}" for faixa in ["carga", "pós_carga_pré_descarga", "descarga", "fora_de_operacao"]]
        for ax, coluna in zip(axes_flat, faixas_sobretensao):
            data = [
                df_filtrado.loc[df_filtrado["pen_pct"] == nivel, f"sobretensao_bess_{coluna.split('_', 1)[1]}"]
                for nivel in niveis
            ]
            ax.boxplot(data, tick_labels=[str(n) for n in niveis], showfliers=False)
            _, titulo = mapeamento_faixas[coluna]
            ax.set_title(f"Número de barras com SOBRETENSÃO (V > 1.05 pu) - {titulo}")
            ax.set_xlabel("Penetração PV (%)")
            ax.set_ylabel("Barras com sobretensão")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_ylim(0, 38)

        fig.suptitle(f"Sobretensão por faixa de operação BESS - Tipo de dia: {tipo_dia}", fontsize=18)
        caminho_imagem = os.path.join(pasta_saida, f"boxplot_sobretensao_por_faixa_bess_{tipo_dia}.png")
        salvar_figura(fig, caminho_imagem)
        plt.close(fig)
        caminhos_imagens.append(caminho_imagem)
    
    return caminhos_imagens
