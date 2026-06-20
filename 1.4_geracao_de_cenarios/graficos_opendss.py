import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from simulacoes_config import (
    BARRAS_EXCLUIDAS_ANALISE,
    BARRAS_TOPOLOGICAS_IEEE34,
    V_PU_MAX,
    V_PU_MIN,
    obter_barras_topologicas_de_carga,
)

# ---------------------------------------------------------------------------
# Constantes de estilo
# ---------------------------------------------------------------------------
FONTSIZE_TITULO   = 16
FONTSIZE_EIXO     = 13
FONTSIZE_TICK     = 11
FONTSIZE_LEGENDA  = 11
FONTSIZE_LEG_TITULO = 12

_PALETA_CENARIOS = ["#1f77b4", "#d62728", "#2ca02c"]  # azul, vermelho, verde
_LINESTYLES = ["-", "--", ":"]

# Rótulos legíveis para os tipos de dia (condição de céu)
_ROTULO_TIPO_DIA = {
    "ceu_aberto":          "Céu aberto",
    "parcialmente_nublado": "Parcialmente nublado",
    "nublado":             "Nublado",
}


# ---------------------------------------------------------------------------
# Utilitários internos
# ---------------------------------------------------------------------------

def salvar_figura(fig, caminho, dpi=200):
    """
    Salva a figura via arquivo temporário e depois substitui o destino.
    Evita falhas do PIL/Windows ao sobrescrever PNGs em pastas sincronizadas.
    """
    pasta = os.path.dirname(caminho)
    os.makedirs(pasta, exist_ok=True)
    nome = os.path.basename(caminho)
    caminho_temporario = os.path.join(pasta, f".tmp_{nome}")
    fig.savefig(caminho_temporario, dpi=dpi, bbox_inches="tight")
    os.replace(caminho_temporario, caminho)


def calcular_limites_y_envelope(global_ymin, global_ymax, margem_minima=0.03):
    """Calcula limites do eixo y com folga para não cortar envelopes ou limites."""
    valores = np.array([global_ymin, global_ymax, V_PU_MIN, V_PU_MAX], dtype=float)
    valores = valores[np.isfinite(valores)]
    if len(valores) == 0:
        return 0.90, 1.10
    ymin = float(np.min(valores))
    ymax = float(np.max(valores))
    amplitude = max(ymax - ymin, 0.01)
    margem = max(margem_minima, 0.08 * amplitude)
    return ymin - margem, ymax + margem


def _aplicar_estilo_limpo(ax):
    """Remove spines superiores e direitos e padroniza a grade."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICK)


def _carregar_contagens_horarias(caminho_tensoes_completas, indicador="sobretensao"):
    """
    Lê master_tensoes_opendss_completas.csv e calcula, para cada
    (pen_pct, id_realizacao, hora), o número de barras de carga com violação
    de tensão (sobretensão ou subtensão) na fase crítica.

    Para sobretensão usa a fase de maior tensão; para subtensão, a de menor tensão.

    Retorna DataFrame com colunas: pen_pct, id_realizacao, tipo_dia, hora, contagem.
    """
    df = pd.read_csv(caminho_tensoes_completas, sep=";", decimal=",", low_memory=False)
    if df.empty:
        return pd.DataFrame()

    df["pen_pct"] = pd.to_numeric(df["pen_pct"], errors="coerce")
    barras_carga = {str(b).lower() for b in obter_barras_topologicas_de_carga()}
    df["barra"] = df["barra"].astype(str).str.lower()
    df = df[df["barra"].isin(barras_carga)].copy()

    colunas_fases = ["tensao_fase_1_pu", "tensao_fase_2_pu", "tensao_fase_3_pu"]

    # Sobretensão: qualquer fase acima do limite → pior = fase MÁXIMA
    # Subtensão:   qualquer fase abaixo do limite → pior = fase MÍNIMA
    if indicador == "sobretensao":
        df["tensao_fase_critica_pu"] = df[colunas_fases].max(axis=1, skipna=True)
        df["violacao"] = df["tensao_fase_critica_pu"] > V_PU_MAX
    else:
        df["tensao_fase_critica_pu"] = df[colunas_fases].min(axis=1, skipna=True)
        df["violacao"] = df["tensao_fase_critica_pu"] < V_PU_MIN

    agrupado = (
        df.groupby(["pen_pct", "id_realizacao", "tipo_dia", "hora"])["violacao"]
        .sum()
        .reset_index()
        .rename(columns={"violacao": "contagem"})
    )
    return agrupado


# ---------------------------------------------------------------------------
# Envelopes de tensão
# ---------------------------------------------------------------------------

def plotar_envelope_tensao(pasta_resultados, pasta_saida, contexto: str = ""):
    """
    Gera figuras de envelope de tensão (perfil ao longo das barras do alimentador)
    para cada faixa horária, sobrepondo os diferentes níveis de penetração.
    Plota apenas barras de carga (excluindo BARRAS_EXCLUIDAS_ANALISE).
    """
    from simulacoes_config import TIME_BANDS
    faixas = list(TIME_BANDS.keys())

    cmap = plt.get_cmap("plasma")
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

    barras_carga = obter_barras_topologicas_de_carga()
    barras_carga_str = [str(b).lower() for b in barras_carga]

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

            barras_presentes = [b for b in barras_carga_str if b in df_env.columns]
            if not barras_presentes:
                continue

            df_ordenado = df_env[barras_presentes]
            x = list(range(len(barras_presentes)))
            rotulos_x = [str(b).upper() for b in barras_presentes]

            mediana = df_ordenado.median(axis=0).values
            p5      = df_ordenado.quantile(0.05, axis=0).values
            p95     = df_ordenado.quantile(0.95, axis=0).values

            cor = cores[nivel]
            global_ymin = min(global_ymin, float(np.nanmin(p5)))
            global_ymax = max(global_ymax, float(np.nanmax(p95)))
            ax.fill_between(x, p5, p95, alpha=0.20, color=cor)
            line, = ax.plot(x, mediana, color=cor, linewidth=1.8, label=f"{nivel}% PV")
            handles_legenda.append(line)
            plotou_algo = True

        if not plotou_algo:
            plt.close(fig)
            continue

        lim_min = ax.axhline(V_PU_MIN, color="red",     linestyle="--", linewidth=1.2,
                             label=f"Limite inferior ({V_PU_MIN} pu)")
        lim_max = ax.axhline(V_PU_MAX, color="darkred", linestyle="--", linewidth=1.2,
                             label=f"Limite superior ({V_PU_MAX} pu)")

        ax.set_xticks(x)
        ax.set_xticklabels(rotulos_x, rotation=90, fontsize=FONTSIZE_TICK)
        ax.set_xlabel("Barras do alimentador (ordem topológica — subestação → extremidades)",
                      fontsize=FONTSIZE_EIXO)
        ax.set_ylabel("Módulo de tensão (pu)", fontsize=FONTSIZE_EIXO)
        ax.set_title(
            f"Tensão ao longo do alimentador IEEE34 — Faixa horária: {faixa.capitalize()}\n"
            f"Mediana e intervalo P5–P95 para cada nível de penetração FV"
            + (f"\n{contexto}" if contexto else ""),
            fontsize=FONTSIZE_TITULO,
        )
        ax.set_ylim(*calcular_limites_y_envelope(global_ymin, global_ymax))
        _aplicar_estilo_limpo(ax)

        ax.legend(
            handles=handles_legenda + [lim_min, lim_max],
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=FONTSIZE_LEGENDA,
            ncol=1,
            title="Penetração FV / Limites",
            title_fontsize=FONTSIZE_LEG_TITULO,
        )

        fig.tight_layout(rect=(0, 0, 0.82, 1))
        caminho_fig = os.path.join(pasta_saida, f"envelope_tensao_{faixa}.png")
        salvar_figura(fig, caminho_fig)
        plt.close(fig)


def plotar_envelope_tensao_por_hora(
    caminho_tensoes_completas,
    pasta_saida,
    indicador: str = "sobretensao",
    contexto: str = "",
):
    """
    Gera UMA figura consolidada com 24 subplots (6 colunas × 4 linhas), um por hora,
    mostrando o envelope de tensão (mediana e P5–P95) ao longo das barras de carga.

    Para 'sobretensao' usa a fase de maior tensão; para 'subtensao', a de menor tensão.
    """
    if not os.path.isfile(caminho_tensoes_completas):
        print(f"  CSV de tensões completas não encontrado: {caminho_tensoes_completas}")
        return

    colunas_fases = ["tensao_fase_1_pu", "tensao_fase_2_pu", "tensao_fase_3_pu"]
    df = pd.read_csv(caminho_tensoes_completas, sep=";", decimal=",", low_memory=False)
    if df.empty:
        return

    df["pen_pct"] = pd.to_numeric(df["pen_pct"], errors="coerce")
    barras_carga = obter_barras_topologicas_de_carga()
    barras_carga_str = [str(b).lower() for b in barras_carga]

    df["barra"] = df["barra"].astype(str).str.lower()
    df = df[df["barra"].isin(set(barras_carga_str))].copy()

    # Sobretensão: pior = fase MÁXIMA; subtensão: pior = fase MÍNIMA
    if indicador == "sobretensao":
        df["tensao_fase_critica_pu"] = df[colunas_fases].max(axis=1, skipna=True)
        aggfunc = "max"
        label_fase = "fase máxima"
        lim_cor_inf, lim_cor_sup = "red", "darkred"
    else:
        df["tensao_fase_critica_pu"] = df[colunas_fases].min(axis=1, skipna=True)
        aggfunc = "min"
        label_fase = "fase mínima"
        lim_cor_inf, lim_cor_sup = "navy", "blue"

    niveis = sorted(df["pen_pct"].dropna().unique())
    n_niveis = len(niveis)
    cmap  = plt.get_cmap("plasma")
    cores = {nivel: cmap(i / max(n_niveis - 1, 1)) for i, nivel in enumerate(niveis)}

    ncols, nrows = 6, 4  # 24 subplots exatos
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5, nrows * 3.8),
        sharey=True,
    )
    axes_flat = axes.flatten()

    handles_legenda = []
    global_ymin, global_ymax = V_PU_MIN, V_PU_MAX

    for hora in range(24):
        ax = axes_flat[hora]
        df_hora = df[df["hora"] == hora]

        plotou_algo = False
        for nivel in niveis:
            df_nivel = df_hora[df_hora["pen_pct"] == nivel]
            if df_nivel.empty:
                continue

            tabela = df_nivel.pivot_table(
                index="id_realizacao",
                columns="barra",
                values="tensao_fase_critica_pu",
                aggfunc=aggfunc,
            )
            barras_presentes = [b for b in barras_carga_str if b in tabela.columns]
            if not barras_presentes:
                continue

            tabela = tabela[barras_presentes]
            x         = list(range(len(barras_presentes)))
            rotulos_x = [str(b).upper() for b in barras_presentes]
            mediana   = tabela.median(axis=0).values
            p5        = tabela.quantile(0.05, axis=0).values
            p95       = tabela.quantile(0.95, axis=0).values
            cor = cores[nivel]
            global_ymin = min(global_ymin, float(np.nanmin(p5)))
            global_ymax = max(global_ymax, float(np.nanmax(p95)))
            ax.fill_between(x, p5, p95, alpha=0.20, color=cor)
            line, = ax.plot(x, mediana, color=cor, linewidth=1.0, label=f"{int(nivel)}% PV")
            if hora == 0:
                handles_legenda.append(line)
            plotou_algo = True

        ax.axhline(V_PU_MIN, color=lim_cor_inf, linestyle="--", linewidth=0.7)
        ax.axhline(V_PU_MAX, color=lim_cor_sup, linestyle="--", linewidth=0.7)
        ax.set_title(f"{hora:02d}h", fontsize=FONTSIZE_TICK)
        if plotou_algo:
            ax.set_xticks(x)
            ax.set_xticklabels(rotulos_x, rotation=90, fontsize=5)
        ax.tick_params(axis="y", labelsize=7)
        _aplicar_estilo_limpo(ax)

    # Aplica ylim compartilhado
    ymin_f, ymax_f = calcular_limites_y_envelope(global_ymin, global_ymax)
    axes_flat[0].set_ylim(ymin_f, ymax_f)

    # Legenda e rótulos globais
    from matplotlib.lines import Line2D
    extras = [
        Line2D([0], [0], color=lim_cor_inf, linestyle="--", linewidth=0.9, label=f"Lim. inf. ({V_PU_MIN} pu)"),
        Line2D([0], [0], color=lim_cor_sup, linestyle="--", linewidth=0.9, label=f"Lim. sup. ({V_PU_MAX} pu)"),
    ]
    leg = fig.legend(
        handles=handles_legenda + extras,
        loc="center right",
        fontsize=FONTSIZE_LEGENDA - 2,
        title="Pen. FV / Limites",
        title_fontsize=FONTSIZE_TICK,
        ncol=1,
        borderaxespad=0.5,
    )
    label_ind = "Sobretensão" if indicador == "sobretensao" else "Subtensão"
    fig.text(0.5, 0.01, "Barras de carga (ordem topológica — subestação → extremidades)",
             ha="center", fontsize=FONTSIZE_TICK)
    fig.text(0.01, 0.5, f"Módulo de tensão — {label_fase} (pu)",
             va="center", rotation="vertical", fontsize=FONTSIZE_TICK)
    fig.suptitle(
        f"Envelope de tensão por hora ({label_ind}) — Alimentador IEEE34\n"
        "Mediana e intervalo P5–P95 por nível de penetração FV"
        + (f"\n{contexto}" if contexto else ""),
        fontsize=FONTSIZE_TITULO,
    )
    fig.tight_layout(rect=[0.03, 0.03, 0.87, 0.95])

    nome_fig = f"envelope_tensao_por_hora_consolidado_{indicador}.png"
    caminho_fig = os.path.join(pasta_saida, nome_fig)
    salvar_figura(fig, caminho_fig)
    plt.close(fig)
    return caminho_fig


# ---------------------------------------------------------------------------
# Boxplots agregados por faixa horária (manhã / tarde / noite / madrugada)
# ---------------------------------------------------------------------------

def plotar_boxplot_por_faixa(
    df_master,
    pasta_saida,
    tipo_dia=None,
    indicador="sobretensao",
    contexto: str = "",
):
    """
    Gera boxplots agregados por faixa horária (TIME_BANDS) do número de barras
    de carga com violação. Usa as colunas pré-calculadas do master_resultados.

    Layout: 2 × 2 subplots (um por faixa: manhã / tarde / noite / madrugada).
    Eixo X: nível de penetração FV (%). Eixo Y: barras com violação.

    Parâmetros
    ----------
    df_master : DataFrame
        master_resultados_opendss.csv já carregado.
    pasta_saida : str
        Pasta de saída das figuras.
    tipo_dia : str ou None
        Filtra por tipo de dia. None = todos.
    indicador : str
        "sobretensao" ou "subtensao".
    contexto : str
        Texto adicional no título.
    """
    from simulacoes_config import TIME_BANDS

    df = df_master if tipo_dia is None else df_master[df_master["tipo_dia"] == tipo_dia]
    if df.empty:
        return

    niveis = sorted(df["pen_pct"].unique())
    faixas = [f for f in TIME_BANDS.keys() if f"{indicador}_{f}" in df.columns]
    if not faixas:
        print(f"  ⚠ Nenhuma coluna '{indicador}_<faixa>' encontrada no df_master.")
        return

    # Ylim global baseado no P99 de todas as faixas
    ylim_top = 2
    for faixa in faixas:
        vals = df[f"{indicador}_{faixa}"].values
        if len(vals):
            ylim_top = max(ylim_top, float(np.nanpercentile(vals, 99)))
    ylim_top = int(ylim_top) + 2
    yticks = np.arange(0, ylim_top + 1, 2)

    ncols = 2
    nrows = int(np.ceil(len(faixas) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(9 * ncols, 5 * nrows),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).flatten()

    for i, faixa in enumerate(faixas):
        ax = axes_flat[i]
        col = f"{indicador}_{faixa}"
        data = [df.loc[df["pen_pct"] == n, col].values for n in niveis]
        ax.boxplot(
            data,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor="#a8c8e8", linewidth=1.0),
            medianprops=dict(color="#1f4e79", linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
        )
        ax.set_xticks(range(1, len(niveis) + 1))
        ax.set_xticklabels([str(int(n)) for n in niveis], fontsize=FONTSIZE_TICK - 1)
        ax.set_xlim(0.2, len(niveis) + 0.8)
        ax.set_title(faixa.capitalize(), fontsize=FONTSIZE_EIXO)
        ax.set_xlabel("Penetração FV (%)", fontsize=FONTSIZE_TICK)
        ax.set_ylabel("Barras c/ violação", fontsize=FONTSIZE_TICK)
        ax.set_ylim(0, ylim_top)
        ax.set_yticks(yticks)
        ax.tick_params(axis="y", labelsize=FONTSIZE_TICK - 1)
        _aplicar_estilo_limpo(ax)

    for j in range(len(faixas), len(axes_flat)):
        axes_flat[j].set_visible(False)

    label_ind  = "Sobretensão (V > 1,05 pu)" if indicador == "sobretensao" else "Subtensão (V < 0,95 pu)"
    td_rotulo  = _ROTULO_TIPO_DIA.get(tipo_dia, tipo_dia) if tipo_dia else "Todos os dias"
    td_label   = f" — {td_rotulo}"
    sufixo_ctx = f"\n{contexto}" if contexto else ""
    fig.suptitle(
        f"{label_ind} por faixa horária{td_label}{sufixo_ctx}",
        fontsize=FONTSIZE_TITULO,
    )

    sufixo_td = f"_{tipo_dia}" if tipo_dia else "_geral"
    nome = f"boxplot_faixa_{indicador}{sufixo_td}.png"
    salvar_figura(fig, os.path.join(pasta_saida, nome))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Boxplots horários (substituem as antigas funções segmentadas por faixa)
# ---------------------------------------------------------------------------

def plotar_boxplot_horario(
    caminho_tensoes_completas,
    pasta_saida,
    tipo_dia=None,
    indicador="sobretensao",
    contexto: str = "",
):
    """
    Gera boxplots horários de número de barras de carga com violação de tensão.

    Para cada nível de penetração, um subplot exibe as 24 horas no eixo X.
    A janela do período solar é destacada com fundo colorido para facilitar
    a leitura comparativa entre períodos sem dividir a figura.

    Parâmetros
    ----------
    caminho_tensoes_completas : str
        Caminho para master_tensoes_opendss_completas.csv.
    pasta_saida : str
        Pasta onde as figuras serão salvas.
    tipo_dia : str ou None
        Filtra por tipo de dia ("util", "sabado", "domingo"). None = todos.
    indicador : str
        "sobretensao" ou "subtensao".
    contexto : str
        Texto adicional no título da figura.
    """
    if not os.path.isfile(caminho_tensoes_completas):
        print(f"  ⚠ CSV de tensões não encontrado: {caminho_tensoes_completas}")
        return

    df_cont = _carregar_contagens_horarias(caminho_tensoes_completas, indicador)
    if df_cont.empty:
        return

    if tipo_dia is not None:
        df_cont = df_cont[df_cont["tipo_dia"] == tipo_dia]
        if df_cont.empty:
            return

    niveis   = sorted(df_cont["pen_pct"].unique())
    n_niveis = len(niveis)
    horas    = list(range(24))

    label_ind  = "Sobretensão (V > 1,05 pu)" if indicador == "sobretensao" else "Subtensão (V < 0,95 pu)"
    td_rotulo  = _ROTULO_TIPO_DIA.get(tipo_dia, tipo_dia) if tipo_dia else "Todos os dias"
    sufixo_td  = f"_{tipo_dia}" if tipo_dia else "_geral"
    sufixo_ctx = f" — {contexto}" if contexto else ""

    # Calcula ylim global para manter escala uniforme entre subplots
    ylim_top = 2  # mínimo
    for nivel in niveis:
        df_niv_tmp = df_cont[df_cont["pen_pct"] == nivel]
        for h in horas:
            vals = df_niv_tmp.loc[df_niv_tmp["hora"] == h, "contagem"].values
            if len(vals):
                ylim_top = max(ylim_top, float(np.nanpercentile(vals, 99)))
    ylim_top = int(ylim_top) + 2  # margem acima do P99
    yticks = np.arange(0, ylim_top + 1, 2)

    ncols = 3
    nrows = int(np.ceil(n_niveis / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(8 * ncols, 4.5 * nrows),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).flatten()

    for i, nivel in enumerate(niveis):
        ax = axes_flat[i]
        df_niv = df_cont[df_cont["pen_pct"] == nivel]
        data = [
            df_niv.loc[df_niv["hora"] == h, "contagem"].values
            for h in horas
        ]
        ax.boxplot(
            data,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor="#a8c8e8", linewidth=1.0),
            medianprops=dict(color="#1f4e79", linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
        )
        # Ticks do eixo X definidos manualmente (compatível com todas as versões do matplotlib)
        ax.set_xticks(range(1, len(horas) + 1))
        ax.set_xticklabels([str(h) for h in horas], fontsize=FONTSIZE_TICK - 1)
        ax.set_xlim(0.2, len(horas) + 0.8)
        ax.set_title(f"{int(nivel)}% PV", fontsize=FONTSIZE_TICK + 1)
        ax.set_xlabel("Hora do dia", fontsize=FONTSIZE_TICK)
        ax.set_ylabel("Barras c/ violação", fontsize=FONTSIZE_TICK)
        ax.set_ylim(0, ylim_top)
        ax.set_yticks(yticks)
        ax.tick_params(axis="y", labelsize=FONTSIZE_TICK - 1)
        _aplicar_estilo_limpo(ax)

    # Oculta subplots vazios
    for j in range(n_niveis, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{label_ind} por hora do dia — {td_rotulo}{sufixo_ctx}",
        fontsize=FONTSIZE_TITULO,
    )

    nome_arquivo = f"boxplot_horario_{indicador}{sufixo_td}.png"
    salvar_figura(fig, os.path.join(pasta_saida, nome_arquivo))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Curva mestre comparativa (multi-cenário)
# ---------------------------------------------------------------------------

def plotar_curva_mestre_comparativa(
    cenarios,
    pasta_saida,
    indicador="sobretensao",
    contexto: str = "",
):
    """
    Plota curvas de mediana (com banda P5–P95 para o primeiro cenário) do
    número de barras de carga com violação diária ao longo dos níveis de
    penetração FV.

    Parâmetros
    ----------
    cenarios : list[dict]
        Lista de 1 a 3 dicionários, cada um com:
          - "rotulo"            : str   — rótulo na legenda
          - "caminho_master_csv": str   — caminho para master_resultados_opendss.csv
          - "cor"               : str   — cor da linha (aceita qualquer spec matplotlib)
    pasta_saida : str
        Pasta onde a figura será salva.
    indicador : str
        "sobretensao" ou "subtensao".
    contexto : str
        Texto adicional no subtítulo.
    """
    if not (1 <= len(cenarios) <= 3):
        raise ValueError("cenarios deve ter entre 1 e 3 elementos.")

    prefixo_col = indicador  # "sobretensao" ou "subtensao"

    fig, ax = plt.subplots(figsize=(12, 6))
    handles = []

    for idx, cen in enumerate(cenarios):
        caminho = cen["caminho_master_csv"]
        if not os.path.isfile(caminho):
            print(f"  ⚠ CSV não encontrado: {caminho}")
            continue

        df = pd.read_csv(caminho, sep=";", decimal=",")
        if df.empty:
            continue

        # Total diário por realização: soma de todas as faixas com o indicador
        cols_ind = [c for c in df.columns if c.startswith(prefixo_col + "_")
                    and not c.startswith(prefixo_col + "_bess_")]
        if not cols_ind:
            print(f"  ⚠ Nenhuma coluna '{prefixo_col}_*' em {caminho}")
            continue

        df["total_diario"] = df[cols_ind].max(axis=1)  # pior faixa do dia por realizacao

        stats = (
            df.groupby("pen_pct")["total_diario"]
            .agg(mediana="median", p5=lambda s: s.quantile(0.05), p95=lambda s: s.quantile(0.95))
            .reset_index()
            .sort_values("pen_pct")
        )

        cor = cen.get("cor", _PALETA_CENARIOS[idx % len(_PALETA_CENARIOS)])
        ls  = _LINESTYLES[idx % len(_LINESTYLES)]

        if idx == 0:
            ax.fill_between(
                stats["pen_pct"], stats["p5"], stats["p95"],
                alpha=0.18, color=cor, label="_nolegend_",
            )

        line, = ax.plot(
            stats["pen_pct"], stats["mediana"],
            color=cor, linestyle=ls, linewidth=2.0,
            marker="o", markersize=5,
            label=cen["rotulo"],
        )
        handles.append(line)

    label_ind = "Sobretensão (V > 1,05 pu)" if indicador == "sobretensao" else "Subtensão (V < 0,95 pu)"
    ax.set_xlabel("Penetração fotovoltaica (%)", fontsize=FONTSIZE_EIXO)
    ax.set_ylabel("Barras de carga c/ violação (pior faixa)", fontsize=FONTSIZE_EIXO)
    ax.set_title(
        f"Curva-mestre comparativa — {label_ind}"
        + (f"\n{contexto}" if contexto else ""),
        fontsize=FONTSIZE_TITULO,
    )
    ax.legend(handles=handles, fontsize=FONTSIZE_LEGENDA, title_fontsize=FONTSIZE_LEG_TITULO)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.set_ylim(bottom=0)
    _aplicar_estilo_limpo(ax)
    fig.tight_layout()

    # Nome do arquivo reflete os cenários incluídos
    if len(cenarios) == 1:
        sufixo = _slugify(cenarios[0]["rotulo"])
    elif len(cenarios) == 2:
        sufixo = f"{_slugify(cenarios[0]['rotulo'])}_vs_{_slugify(cenarios[1]['rotulo'])}"
    else:
        sufixo = "sintese_tres_cenarios"

    nome = f"curva_mestre_{indicador}_{sufixo}.png"
    salvar_figura(fig, os.path.join(pasta_saida, nome))
    plt.close(fig)


def _slugify(texto):
    """Converte rótulo em slug para uso em nomes de arquivo."""
    return (
        texto.lower()
        .replace(" ", "_")
        .replace("ã", "a").replace("ç", "c").replace("é", "e")
        .replace("ê", "e").replace("á", "a").replace("í", "i")
        .replace("ó", "o").replace("ú", "u").replace("â", "a")
        .replace("/", "_").replace("\\", "_")
    )


# ---------------------------------------------------------------------------
# Boxplot comparativo hora a hora entre cenários
# ---------------------------------------------------------------------------

def plotar_boxplot_horario_comparativo(
    cenarios,
    caminho_tensoes_por_cenario,
    pasta_saida,
    nivel_pen,
    indicador="sobretensao",
    contexto: str = "",
):
    """
    Para um nível de penetração fixo, plota um boxplot agrupado hora a hora:
    eixo X com as 24 horas do dia; para cada hora, um boxplot por cenário
    lado a lado, colorido por cenário.

    Parâmetros
    ----------
    cenarios : list[dict]
        Lista de 1–3 dicts com "rotulo" e (opcionalmente) "cor".
    caminho_tensoes_por_cenario : list[str]
        Caminhos para master_tensoes_opendss_completas.csv de cada cenário,
        em correspondência com `cenarios`.
    pasta_saida : str
        Pasta de saída.
    nivel_pen : int
        Nível de penetração FV a analisar (ex.: 100).
    indicador : str
        "sobretensao" ou "subtensao".
    contexto : str
        Texto adicional no título.
    """
    n_cen = len(cenarios)
    if n_cen != len(caminho_tensoes_por_cenario):
        raise ValueError("cenarios e caminho_tensoes_por_cenario devem ter o mesmo tamanho.")

    horas = list(range(24))
    cores = [cen.get("cor", _PALETA_CENARIOS[i % len(_PALETA_CENARIOS)]) for i, cen in enumerate(cenarios)]

    # Carrega dados de contagem por hora para cada cenário.
    # Usa comparação por proximidade (<0.5) para tolerar pen_pct float/int misto no CSV.
    dados_por_cenario = []
    for caminho, cen in zip(caminho_tensoes_por_cenario, cenarios):
        if not os.path.isfile(caminho):
            print(f"  ⚠ CSV não encontrado: {caminho}")
            dados_por_cenario.append(None)
            continue
        df_cont = _carregar_contagens_horarias(caminho, indicador)
        if df_cont.empty:
            print(f"  ⚠ Sem dados em: {caminho}")
            dados_por_cenario.append(None)
            continue
        mask = (df_cont["pen_pct"] - float(nivel_pen)).abs() < 0.5
        df_niv = df_cont[mask]
        if df_niv.empty:
            vals_uniq = sorted(df_cont["pen_pct"].dropna().unique())
            print(f"  ⚠ Nenhuma linha com pen_pct≈{nivel_pen} em '{cen['rotulo']}'. "
                  f"Valores encontrados: {vals_uniq[:10]}")
        dados_por_cenario.append(df_niv)

    largura_box = 0.7 / n_cen
    offsets = np.linspace(-(n_cen - 1) * largura_box / 2,
                           (n_cen - 1) * largura_box / 2, n_cen)

    # Computa o topo do eixo Y a partir do P99 de todos os dados carregados
    todos_valores = []
    for df_niv in dados_por_cenario:
        if df_niv is not None and not df_niv.empty:
            todos_valores.extend(df_niv["contagem"].dropna().tolist())
    if todos_valores:
        p99 = float(np.percentile(todos_valores, 99))
        ylim_top = max(int(np.ceil(p99)) + 2, 4)
    else:
        ylim_top = 10

    fig, ax = plt.subplots(figsize=(22, 6))

    for i, (cen, df_niv) in enumerate(zip(cenarios, dados_por_cenario)):
        cor = cores[i]
        for h in horas:
            pos = h + offsets[i]
            if df_niv is None or df_niv.empty:
                valores = []
            else:
                valores = df_niv.loc[df_niv["hora"] == h, "contagem"].dropna().values
            ax.boxplot(
                [valores] if len(valores) > 0 else [[]],
                positions=[pos],
                widths=largura_box * 0.85,
                showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor=cor, alpha=0.65, linewidth=0.8),
                medianprops=dict(color="black", linewidth=1.4),
                whiskerprops=dict(linewidth=0.8, color=cor),
                capprops=dict(linewidth=0.8, color=cor),
            )

    # Ticks centrados em cada hora
    ax.set_xticks(horas)
    ax.set_xticklabels([f"{h}h" for h in horas], fontsize=FONTSIZE_TICK)
    ax.set_xlim(-0.7, 23.7)
    ax.set_xlabel("Hora do dia", fontsize=FONTSIZE_EIXO)
    ax.set_ylabel("Barras de carga c/ violação", fontsize=FONTSIZE_EIXO)
    tick_step = max(1, ylim_top // 10)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(tick_step))
    ax.set_ylim(0, ylim_top)
    _aplicar_estilo_limpo(ax)

    label_ind = "Sobretensão (V > 1,05 pu)" if indicador == "sobretensao" else "Subtensão (V < 0,95 pu)"
    rotulos_cen = " vs. ".join(c["rotulo"] for c in cenarios)
    ax.set_title(
        f"{label_ind} hora a hora — {rotulos_cen} — {nivel_pen}% PV"
        + (f"\n{contexto}" if contexto else ""),
        fontsize=FONTSIZE_TITULO,
    )

    from matplotlib.patches import Patch
    patches = [Patch(facecolor=cores[i], alpha=0.65, label=cen["rotulo"])
               for i, cen in enumerate(cenarios)]
    ax.legend(handles=patches, fontsize=FONTSIZE_LEGENDA,
              title_fontsize=FONTSIZE_LEG_TITULO, loc="upper right")

    fig.tight_layout()
    sufixo_cen = "_vs_".join(_slugify(c["rotulo"]) for c in cenarios)
    nome = f"boxplot_horario_comparativo_{indicador}_{sufixo_cen}_{nivel_pen}pct.png"
    salvar_figura(fig, os.path.join(pasta_saida, nome))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Heatmap topológico-horário de probabilidade de sobretensão
# ---------------------------------------------------------------------------

def plotar_heatmap_topologico_horario(
    caminho_tensoes_completas,
    pasta_saida,
    pen_pct_alvo,
    indicador="sobretensao",
    contexto: str = "",
):
    """
    Heatmap (barras × horas) de probabilidade de violação de tensão para um
    nível de penetração específico, considerando apenas barras de carga.

    Eixo X: barras na ordem topológica (apenas barras de carga).
    Eixo Y: horas 0–23.
    Cor: fração das realizações em que a pior fase violou o limite.
    """
    if not os.path.isfile(caminho_tensoes_completas):
        print(f"  ⚠ CSV de tensões não encontrado: {caminho_tensoes_completas}")
        return

    colunas_fases = ["tensao_fase_1_pu", "tensao_fase_2_pu", "tensao_fase_3_pu"]
    df = pd.read_csv(caminho_tensoes_completas, sep=";", decimal=",", low_memory=False)
    if df.empty:
        return

    df["pen_pct"] = pd.to_numeric(df["pen_pct"], errors="coerce")
    barras_carga = obter_barras_topologicas_de_carga()
    barras_carga_str = [str(b).lower() for b in barras_carga]

    df["barra"] = df["barra"].astype(str).str.lower()
    df = df[(df["pen_pct"] == pen_pct_alvo) & (df["barra"].isin(set(barras_carga_str)))].copy()
    if df.empty:
        print(f"  ⚠ Sem dados para pen_pct={pen_pct_alvo}")
        return

    if indicador == "sobretensao":
        df["tensao_fase_critica_pu"] = df[colunas_fases].max(axis=1, skipna=True)
        df["violacao"] = (df["tensao_fase_critica_pu"] > V_PU_MAX).astype(float)
    else:
        df["tensao_fase_critica_pu"] = df[colunas_fases].min(axis=1, skipna=True)
        df["violacao"] = (df["tensao_fase_critica_pu"] < V_PU_MIN).astype(float)

    prob = (
        df.groupby(["hora", "barra"])["violacao"]
        .mean()
        .reset_index()
        .rename(columns={"violacao": "prob"})
    )

    # Pivô: linhas=hora, colunas=barra (ordem topológica)
    pivot = prob.pivot(index="hora", columns="barra", values="prob")
    barras_presentes = [b for b in barras_carga_str if b in pivot.columns]
    pivot = pivot.reindex(columns=barras_presentes).reindex(index=range(24)).fillna(0.0)

    fig, ax = plt.subplots(figsize=(max(12, len(barras_presentes) * 0.5), 7))
    im = ax.pcolormesh(
        pivot.values,
        cmap="YlOrRd",
        vmin=0, vmax=1,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Probabilidade de violação", fontsize=FONTSIZE_EIXO)
    cbar.ax.tick_params(labelsize=FONTSIZE_TICK)

    ax.set_xticks(np.arange(len(barras_presentes)) + 0.5)
    ax.set_xticklabels([b.upper() for b in barras_presentes], rotation=90, fontsize=FONTSIZE_TICK)
    ax.set_yticks(np.arange(24) + 0.5)
    ax.set_yticklabels([f"{h:02d}h" for h in range(24)], fontsize=FONTSIZE_TICK)
    ax.set_xlabel("Barras de carga (ordem topológica)", fontsize=FONTSIZE_EIXO)
    ax.set_ylabel("Hora do dia", fontsize=FONTSIZE_EIXO)

    label_ind = "Sobretensão (V > 1,05 pu)" if indicador == "sobretensao" else "Subtensão (V < 0,95 pu)"
    ax.set_title(
        f"Probabilidade de {label_ind} — {pen_pct_alvo}% PV"
        + (f"\n{contexto}" if contexto else ""),
        fontsize=FONTSIZE_TITULO,
    )

    fig.tight_layout()
    nome = f"heatmap_topologico_horario_{indicador}_{_slugify(contexto or 'cenario')}_{pen_pct_alvo}pct.png"
    salvar_figura(fig, os.path.join(pasta_saida, nome))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Heatmap de diferença entre dois cenários
# ---------------------------------------------------------------------------

def plotar_heatmap_diferenca(
    caminho_tensoes_a,
    caminho_tensoes_b,
    pasta_saida,
    pen_pct_alvo,
    rotulo_a: str,
    rotulo_b: str,
    indicador="sobretensao",
    contexto: str = "",
):
    """
    Plota a diferença de probabilidade de violação entre dois cenários
    (cenário A − cenário B), célula a célula em (barra × hora).

    Colormap divergente: verde = A reduziu violação, vermelho = B piorou.

    Parâmetros
    ----------
    caminho_tensoes_a, caminho_tensoes_b : str
        Caminhos para master_tensoes_opendss_completas.csv de cada cenário.
    pasta_saida : str
        Pasta de saída.
    pen_pct_alvo : int
        Nível de penetração FV.
    rotulo_a, rotulo_b : str
        Rótulos descritivos de cada cenário (usados no título e no nome do arquivo).
    indicador : str
        "sobretensao" ou "subtensao".
    contexto : str
        Texto adicional no título.
    """
    def _calcular_prob(caminho):
        if not os.path.isfile(caminho):
            return None
        colunas_fases = ["tensao_fase_1_pu", "tensao_fase_2_pu", "tensao_fase_3_pu"]
        df = pd.read_csv(caminho, sep=";", decimal=",", low_memory=False)
        if df.empty:
            return None
        df["pen_pct"] = pd.to_numeric(df["pen_pct"], errors="coerce")
        barras_carga_str = {str(b).lower() for b in obter_barras_topologicas_de_carga()}
        df["barra"] = df["barra"].astype(str).str.lower()
        df = df[(df["pen_pct"] == pen_pct_alvo) & (df["barra"].isin(barras_carga_str))].copy()
        if df.empty:
            return None
        if indicador == "sobretensao":
            df["tensao_fase_critica_pu"] = df[colunas_fases].max(axis=1, skipna=True)
            df["violacao"] = (df["tensao_fase_critica_pu"] > V_PU_MAX).astype(float)
        else:
            df["tensao_fase_critica_pu"] = df[colunas_fases].min(axis=1, skipna=True)
            df["violacao"] = (df["tensao_fase_critica_pu"] < V_PU_MIN).astype(float)
        return (
            df.groupby(["hora", "barra"])["violacao"]
            .mean()
            .reset_index()
            .rename(columns={"violacao": "prob"})
        )

    prob_a = _calcular_prob(caminho_tensoes_a)
    prob_b = _calcular_prob(caminho_tensoes_b)
    if prob_a is None or prob_b is None:
        print("  ⚠ Um dos CSVs de tensões não encontrado ou vazio.")
        return

    barras_carga = obter_barras_topologicas_de_carga()
    barras_carga_str = [str(b).lower() for b in barras_carga]

    pivot_a = prob_a.pivot(index="hora", columns="barra", values="prob")
    pivot_b = prob_b.pivot(index="hora", columns="barra", values="prob")
    barras_comuns = [b for b in barras_carga_str if b in pivot_a.columns and b in pivot_b.columns]
    pivot_a = pivot_a.reindex(columns=barras_comuns).reindex(index=range(24)).fillna(0.0)
    pivot_b = pivot_b.reindex(columns=barras_comuns).reindex(index=range(24)).fillna(0.0)
    diferenca = pivot_a.values - pivot_b.values  # positivo = A tem mais violação

    vmax = max(abs(diferenca).max(), 0.01)
    fig, ax = plt.subplots(figsize=(max(12, len(barras_comuns) * 0.5), 7))
    im = ax.pcolormesh(diferenca, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label(f"Δ probabilidade ({rotulo_a} − {rotulo_b})", fontsize=FONTSIZE_EIXO)
    cbar.ax.tick_params(labelsize=FONTSIZE_TICK)

    ax.set_xticks(np.arange(len(barras_comuns)) + 0.5)
    ax.set_xticklabels([b.upper() for b in barras_comuns], rotation=90, fontsize=FONTSIZE_TICK)
    ax.set_yticks(np.arange(24) + 0.5)
    ax.set_yticklabels([f"{h:02d}h" for h in range(24)], fontsize=FONTSIZE_TICK)
    ax.set_xlabel("Barras de carga (ordem topológica)", fontsize=FONTSIZE_EIXO)
    ax.set_ylabel("Hora do dia", fontsize=FONTSIZE_EIXO)

    label_ind = "Sobretensão" if indicador == "sobretensao" else "Subtensão"
    ax.set_title(
        f"Diferença de probabilidade de {label_ind} ({rotulo_a} − {rotulo_b}) — {pen_pct_alvo}% PV"
        + (f"\n{contexto}" if contexto else ""),
        fontsize=FONTSIZE_TITULO,
    )

    fig.tight_layout()
    nome = (
        f"heatmap_diferenca_{indicador}_"
        f"{_slugify(rotulo_a)}_vs_{_slugify(rotulo_b)}_{pen_pct_alvo}pct.png"
    )
    salvar_figura(fig, os.path.join(pasta_saida, nome))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gráficos 3D de probabilidade de violação (penetração × barra × probabilidade)
# ---------------------------------------------------------------------------

def plotar_3d_probabilidade(
    caminho_tensoes_completas,
    pasta_saida,
    indicador: str = "sobretensao",
    contexto: str = "",
):
    """
    Gera um gráfico 3D de superfície mostrando a probabilidade de violação de
    tensão em função do nível de penetração FV (eixo X) e da barra do
    alimentador (eixo Y), com a probabilidade no eixo Z.

    A probabilidade é calculada como a fração de pares (realização, hora) em
    que a barra apresentou violação, agregando todas as horas do dia.

    Para sobretensão usa a fase de maior tensão; para subtensão, a de menor.

    Parâmetros
    ----------
    caminho_tensoes_completas : str
        Caminho para master_tensoes_opendss_completas.csv.
    pasta_saida : str
        Pasta de destino.
    indicador : str
        "sobretensao" ou "subtensao".
    contexto : str
        Texto adicional no título.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if not os.path.isfile(caminho_tensoes_completas):
        print(f"  ⚠ CSV de tensões não encontrado: {caminho_tensoes_completas}")
        return

    colunas_fases = ["tensao_fase_1_pu", "tensao_fase_2_pu", "tensao_fase_3_pu"]
    df = pd.read_csv(caminho_tensoes_completas, sep=";", decimal=",", low_memory=False)
    if df.empty:
        return

    df["pen_pct"] = pd.to_numeric(df["pen_pct"], errors="coerce")
    barras_carga = obter_barras_topologicas_de_carga()
    barras_carga_str = [str(b).lower() for b in barras_carga]
    df["barra"] = df["barra"].astype(str).str.lower()
    df = df[df["barra"].isin(set(barras_carga_str))].copy()
    if df.empty:
        return

    if indicador == "sobretensao":
        df["v_critica"] = df[colunas_fases].max(axis=1, skipna=True)
        df["violacao"]  = (df["v_critica"] > V_PU_MAX).astype(float)
        cmap_nome  = "YlOrRd"
        label_ind  = "Sobretensão (V > 1,05 pu)"
    else:
        df["v_critica"] = df[colunas_fases].min(axis=1, skipna=True)
        df["violacao"]  = (df["v_critica"] < V_PU_MIN).astype(float)
        cmap_nome  = "Blues"
        label_ind  = "Subtensão (V < 0,95 pu)"

    # Probabilidade por (pen_pct, barra): fração de (realização, hora) com violação
    prob = (
        df.groupby(["pen_pct", "barra"])["violacao"]
        .mean()
        .reset_index()
        .rename(columns={"violacao": "prob"})
    )
    if prob.empty:
        return

    niveis = sorted(prob["pen_pct"].dropna().unique())
    pivot  = prob.pivot(index="barra", columns="pen_pct", values="prob")
    barras_presentes = [b for b in barras_carga_str if b in pivot.index]
    pivot  = pivot.reindex(index=barras_presentes, columns=niveis).fillna(0.0)

    Z = pivot.values           # shape: (n_barras, n_niveis)
    n_barras = len(barras_presentes)
    Y_idx    = np.arange(n_barras, dtype=float)
    X_nivs   = np.array(niveis, dtype=float)
    X, Y     = np.meshgrid(X_nivs, Y_idx)

    fig = plt.figure(figsize=(15, 9))
    ax  = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap=cmap_nome, alpha=0.88, edgecolor="none")
    cbar = fig.colorbar(surf, ax=ax, fraction=0.012, pad=0.12, shrink=0.65)
    cbar.set_label("Probabilidade de violação", fontsize=FONTSIZE_EIXO - 1)
    cbar.ax.tick_params(labelsize=FONTSIZE_TICK - 1)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    ax.set_xlabel("Penetração FV (%)",               fontsize=FONTSIZE_EIXO - 1, labelpad=12)
    ax.set_ylabel("Barra (subestação → extremidades)", fontsize=FONTSIZE_EIXO - 1, labelpad=12)
    ax.set_zlabel("P(violação)",                      fontsize=FONTSIZE_EIXO - 1, labelpad=8)
    ax.set_zlim(0.0, 1.0)

    ax.set_xticks(X_nivs)
    ax.set_xticklabels([f"{int(n)}%" for n in X_nivs], fontsize=FONTSIZE_TICK - 2)

    # Eixo Y com rótulos de barras intercalados (evita superlotação)
    step = max(1, n_barras // 10)
    ytick_idx = list(range(0, n_barras, step))
    ax.set_yticks([float(i) for i in ytick_idx])
    ax.set_yticklabels(
        [barras_presentes[i].upper() for i in ytick_idx],
        fontsize=FONTSIZE_TICK - 2,
    )

    ax.view_init(elev=28, azim=-55)
    ax.set_title(
        f"Probabilidade de {label_ind}\npor barra e nível de penetração FV"
        + (f"\n{contexto}" if contexto else ""),
        fontsize=FONTSIZE_TITULO - 1,
        pad=14,
    )

    fig.tight_layout()
    nome_fig = f"3d_probabilidade_{indicador}.png"
    salvar_figura(fig, os.path.join(pasta_saida, nome_fig))
    plt.close(fig)
    return os.path.join(pasta_saida, nome_fig)
