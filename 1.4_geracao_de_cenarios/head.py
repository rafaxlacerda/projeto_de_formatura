# Main de processamento dos resultados do Monte Carlo via OpenDSS
# Veja parse_args para saber como rodar este script
# Para rodar sem controle, rodar sem argumentos
#
# Exemplos de uso:
#
#   Sem controle (fluxo de potência puro):
#     python head.py
#
#   Só controle primário VoltVar (NBR 16149:2013):
#     python head.py --controle-primario
#
#   Controle secundário por consenso sem primário (modo sempre_ativo obrigatório):
#     python head.py --controle-secundario --modo-secundario sempre_ativo
#
#   Controle secundário com primário, modo sempre ativo:
#     python head.py --controle-primario --controle-secundario --modo-secundario sempre_ativo
#
#   Controle secundário com primário, modo condicional:
#     python head.py --controle-primario --controle-secundario --modo-secundario condicional

import argparse
import os
import sys

import pandas as pd

from estatisticas_opendss import gerar_estatisticas_comparativas, gerar_tabelas_estatisticas
from graficos_opendss import (
    plotar_boxplot_horario,
    plotar_boxplot_horario_comparativo,
    plotar_boxplot_por_faixa,
    plotar_curva_mestre_comparativa,
    plotar_envelope_tensao,
    plotar_envelope_tensao_por_hora,
    plotar_heatmap_diferenca,
    plotar_heatmap_topologico_horario,
    plotar_3d_probabilidade,
)
from controle_primario import processar_nivel_controle_primario
from controle_secundario import processar_nivel_controle_secundario
from simulacao_opendss import processar_nivel
from simulacoes_config import DEFAULT_DSS_FILE

# Cenários com resultado fixo conhecido, em ordem de relevância para o relatório.
# Usado por --somente-graficos para regenerar todos os cenários disponíveis.
_CENARIOS_CONHECIDOS = [
    {
        "subdir":      "analise_opendss",
        "subdir_graf": "graficos_tabelas_descritivas",
        "contexto":    "",
        "descricao":   "Sem controle",
    },
    {
        "subdir":      "analise_controle_primario",
        "subdir_graf": "graficos_tabelas_controle_primario",
        "contexto":    "Com Controle Primário VoltVar (NBR 16149:2013)",
        "descricao":   "Controle primário",
    },
    {
        "subdir":      "analise_secundario_condicional_com_primario",
        "subdir_graf": "graficos_tabelas_secundario_condicional_com_primario",
        "contexto":    (
            "Controle Secundário por Consenso (Condicional) "
            "com Controle Primário VoltVar (NBR 16149:2013)"
        ),
        "descricao":   "Controle duplo condicional",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Simula os cenários de Monte Carlo via OpenDSS e compila defeitos de "
            "tensão por faixa horária. Suporta quatro modos de controle: sem controle, "
            "só controle primário VoltVar, controle secundário por consenso sem primário "
            "(sempre ativo), e controle secundário com primário (sempre ativo ou condicional)."
        )
    )
    parser.add_argument(
        "--nivel",
        type=str,
        default=None,
        help="Nível de penetração FV específico para processar (ex: 050 para 50%%).",
    )
    parser.add_argument(
        "--max-realizacoes",
        type=int,
        default=None,
        help="Número máximo de realizações a processar por nível (útil para testes).",
    )
    parser.add_argument(
        "--dss-file",
        type=str,
        default=DEFAULT_DSS_FILE,
        help="Arquivo DSS a ser usado para a simulação.",
    )
    parser.add_argument(
        "--somente-graficos",
        action="store_true",
        help="Gera apenas gráficos e tabelas a partir dos CSVs existentes, sem rodar o OpenDSS.",
    )
    parser.add_argument(
        "--controle-primario",
        action="store_true",
        help="Ativa o controle primário VoltVar (NBR 16149:2013) durante a simulação.",
    )
    parser.add_argument(
        "--controle-secundario",
        action="store_true",
        help=(
            "Ativa o controle secundário por consenso distribuído sobre os BESS. "
            "Requer --modo-secundario. Quando usado sem --controle-primario, "
            "o modo 'condicional' não é permitido."
        ),
    )
    parser.add_argument(
        "--modo-secundario",
        type=str,
        choices=["sempre_ativo", "condicional"],
        default=None,
        help=(
            "Modo de operação do controle secundário: "
            "'sempre_ativo' aplica o consenso em todas as horas; "
            "'condicional' aplica apenas nas horas com violação remanescente "
            "após o controle primário (requer --controle-primario)."
        ),
    )
    parser.add_argument(
        "--gerar-comparativos",
        action="store_true",
        help=(
            "Gera apenas as figuras comparativas multi-cenário (blocos 1–4 do relatório) "
            "assumindo que os CSVs de todos os cenários relevantes já existem em "
            "resultados_monte_carlo/. Não executa simulações."
        ),
    )
    return parser.parse_args()


def _validar_args(args):
    """Valida as combinações de argumentos e encerra com mensagem clara em caso de erro."""
    if args.controle_secundario:
        if args.modo_secundario is None:
            print(
                "Erro: --controle-secundario requer --modo-secundario "
                "('sempre_ativo' ou 'condicional').",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.modo_secundario == "condicional" and not args.controle_primario:
            print(
                "Erro: o modo 'condicional' do controle secundário requer "
                "--controle-primario ativo, pois avalia as violações remanescentes "
                "após a atuação do primário.",
                file=sys.stderr,
            )
            sys.exit(1)


def _definir_pastas_e_contexto(args, base_dir):
    """
    Define as pastas de saída e a string de contexto para títulos de gráficos
    com base na combinação de controles selecionada.
    """
    pasta_base = os.path.join(base_dir, "resultados_monte_carlo")

    if args.controle_secundario:
        modo = args.modo_secundario
        if args.controle_primario:
            subdir       = f"analise_secundario_{modo}_com_primario"
            subdir_graf  = f"graficos_tabelas_secundario_{modo}_com_primario"
            contexto     = (
                f"Controle Secundário por Consenso ({modo.replace('_', ' ').title()}) "
                f"com Controle Primário VoltVar (NBR 16149:2013)"
            )
        else:
            subdir       = "analise_secundario_sempre_ativo_sem_primario"
            subdir_graf  = "graficos_tabelas_secundario_sempre_ativo_sem_primario"
            contexto     = "Controle Secundário por Consenso (Sempre Ativo) — Sem Controle Primário"
    elif args.controle_primario:
        subdir       = "analise_controle_primario"
        subdir_graf  = "graficos_tabelas_controle_primario"
        contexto     = "Com Controle Primário VoltVar (NBR 16149:2013)"
    else:
        subdir       = "analise_opendss"
        subdir_graf  = "graficos_tabelas_descritivas"
        contexto     = ""

    return (
        os.path.join(pasta_base, subdir),
        os.path.join(pasta_base, subdir_graf),
        contexto,
    )


def _selecionar_funcao_processar(args):
    """Retorna a função de processamento de nível adequada à combinação de controles."""
    if args.controle_secundario:
        com_primario = args.controle_primario
        modo         = args.modo_secundario

        def processar(pasta_nivel, dss_path, pasta_saida_nivel, max_realizacoes=None):
            return processar_nivel_controle_secundario(
                pasta_nivel,
                dss_path,
                pasta_saida_nivel,
                com_primario=com_primario,
                modo=modo,
                max_realizacoes=max_realizacoes,
            )

        return processar

    if args.controle_primario:
        return processar_nivel_controle_primario

    return processar_nivel


def gerar_graficos_e_tabelas(
    df_master, pasta_saida, pasta_saida_descritiva,
    caminho_master_tensoes, base_dir, contexto=""
):
    """Gera todos os gráficos e tabelas de um único cenário de controle."""
    tipos_dia = sorted(df_master["tipo_dia"].unique()) if "tipo_dia" in df_master.columns else []

    for indicador in ("sobretensao", "subtensao"):
        print(f"\nGerando boxplots por faixa horária (geral) — {indicador}...")
        plotar_boxplot_por_faixa(
            df_master, pasta_saida_descritiva,
            tipo_dia=None, indicador=indicador, contexto=contexto,
        )
        for td in tipos_dia:
            print(f"\nGerando boxplots por faixa horária ({td}) — {indicador}...")
            plotar_boxplot_por_faixa(
                df_master, pasta_saida_descritiva,
                tipo_dia=td, indicador=indicador, contexto=contexto,
            )

        print(f"\nGerando boxplots horários (geral) — {indicador}...")
        plotar_boxplot_horario(
            caminho_master_tensoes, pasta_saida_descritiva,
            tipo_dia=None, indicador=indicador, contexto=contexto,
        )
        for td in tipos_dia:
            print(f"\nGerando boxplots horários ({td}) — {indicador}...")
            plotar_boxplot_horario(
                caminho_master_tensoes, pasta_saida_descritiva,
                tipo_dia=td, indicador=indicador, contexto=contexto,
            )

    print("\nGerando envelopes de tensão por faixa horária...")
    plotar_envelope_tensao(pasta_saida, pasta_saida_descritiva, contexto=contexto)

    print("\nGerando envelopes de tensão por hora (fase máxima — sobretensão)...")
    plotar_envelope_tensao_por_hora(
        caminho_master_tensoes, pasta_saida_descritiva,
        indicador="sobretensao", contexto=contexto,
    )
    print("\nGerando envelopes de tensão por hora (fase mínima — subtensão)...")
    plotar_envelope_tensao_por_hora(
        caminho_master_tensoes, pasta_saida_descritiva,
        indicador="subtensao", contexto=contexto,
    )

    print("\nGerando gráficos 3D de probabilidade de sobretensão por barra e penetração...")
    plotar_3d_probabilidade(
        caminho_master_tensoes, pasta_saida_descritiva,
        indicador="sobretensao", contexto=contexto,
    )
    print("\nGerando gráficos 3D de probabilidade de subtensão por barra e penetração...")
    plotar_3d_probabilidade(
        caminho_master_tensoes, pasta_saida_descritiva,
        indicador="subtensao", contexto=contexto,
    )

    print("\nGerando tabelas de estatísticas descritivas...")
    pasta_monte_carlo = os.path.join(base_dir, "resultados_monte_carlo")
    gerar_tabelas_estatisticas(
        pasta_saida_descritiva, pasta_monte_carlo, pasta_saida, contexto=contexto
    )


def gerar_graficos_comparativos(base_dir):
    """
    Gera as figuras comparativas multi-cenário (blocos 1–5 do relatório).
    Assume que os CSVs de todos os cenários relevantes já existem em disco.

    Estrutura de saída:
      resultados_monte_carlo/graficos_tabelas_comparativos/
        bloco1_sem_controle/
        bloco2_sem_controle_vs_primario/
        bloco3_sem_controle_vs_duplo/
        bloco4_sintese_tres_cenarios/
        bloco5_sem_controle_vs_primario_vs_secundario/
    """
    pasta_base = os.path.join(base_dir, "resultados_monte_carlo")

    # Caminhos conhecidos de cada cenário
    _csv_master = {
        "sem_controle":   os.path.join(pasta_base, "analise_opendss",             "master_resultados_opendss.csv"),
        "primario":       os.path.join(pasta_base, "analise_controle_primario",    "master_resultados_opendss.csv"),
        "duplo_cond":     os.path.join(pasta_base, "analise_secundario_condicional_com_primario", "master_resultados_opendss.csv"),
        "secundario_so":  os.path.join(pasta_base, "analise_secundario_sempre_ativo_sem_primario", "master_resultados_opendss.csv"),
    }
    _csv_tensoes = {
        "sem_controle":   os.path.join(pasta_base, "analise_opendss",             "master_tensoes_opendss_completas.csv"),
        "primario":       os.path.join(pasta_base, "analise_controle_primario",    "master_tensoes_opendss_completas.csv"),
        "duplo_cond":     os.path.join(pasta_base, "analise_secundario_condicional_com_primario", "master_tensoes_opendss_completas.csv"),
        "secundario_so":  os.path.join(pasta_base, "analise_secundario_sempre_ativo_sem_primario", "master_tensoes_opendss_completas.csv"),
    }

    # Detecta quais cenários estão disponíveis
    disp = {k: os.path.isfile(v) for k, v in _csv_master.items()}
    print(f"Cenários disponíveis: { {k: v for k, v in disp.items()} }")

    pasta_comp = os.path.join(pasta_base, "graficos_tabelas_comparativos")

    def _pasta_bloco(nome):
        p = os.path.join(pasta_comp, nome)
        os.makedirs(p, exist_ok=True)
        return p

    CEN_SC = {"rotulo": "Sem controle",           "caminho_master_csv": _csv_master["sem_controle"],  "cor": "#1f77b4"}
    CEN_PR = {"rotulo": "Controle primário",      "caminho_master_csv": _csv_master["primario"],       "cor": "#d62728"}
    CEN_DC = {"rotulo": "Controle duplo (cond.)", "caminho_master_csv": _csv_master["duplo_cond"],    "cor": "#2ca02c"}
    CEN_SO = {"rotulo": "Controle secundário",    "caminho_master_csv": _csv_master["secundario_so"], "cor": "#ff7f0e"}

    for indicador in ("sobretensao", "subtensao"):

        # Bloco 1 — apenas sem controle
        if disp["sem_controle"]:
            print(f"\n[Bloco 1] Curva mestre — {indicador}")
            plotar_curva_mestre_comparativa(
                [CEN_SC], _pasta_bloco("bloco1_sem_controle"), indicador=indicador,
            )
            if os.path.isfile(_csv_tensoes["sem_controle"]):
                for pen in [50, 100]:
                    plotar_heatmap_topologico_horario(
                        _csv_tensoes["sem_controle"], _pasta_bloco("bloco1_sem_controle"),
                        pen, indicador=indicador, contexto="Sem controle",
                    )

        # Bloco 2 — sem controle vs. primário
        if disp["sem_controle"] and disp["primario"]:
            print(f"\n[Bloco 2] Comparação sem controle vs. primário — {indicador}")
            plotar_curva_mestre_comparativa(
                [CEN_SC, CEN_PR], _pasta_bloco("bloco2_sem_controle_vs_primario"), indicador=indicador,
            )
            gerar_estatisticas_comparativas(
                [CEN_SC, CEN_PR], _pasta_bloco("bloco2_sem_controle_vs_primario"), indicador=indicador,
            )
            for pen in [50, 100]:
                plotar_boxplot_horario_comparativo(
                    [CEN_SC, CEN_PR],
                    [_csv_tensoes["sem_controle"], _csv_tensoes["primario"]],
                    _pasta_bloco("bloco2_sem_controle_vs_primario"),
                    nivel_pen=pen, indicador=indicador,
                )
                if os.path.isfile(_csv_tensoes["sem_controle"]) and os.path.isfile(_csv_tensoes["primario"]):
                    plotar_heatmap_diferenca(
                        _csv_tensoes["sem_controle"], _csv_tensoes["primario"],
                        _pasta_bloco("bloco2_sem_controle_vs_primario"),
                        pen_pct_alvo=pen,
                        rotulo_a="Sem controle", rotulo_b="Primário",
                        indicador=indicador,
                    )

        # Bloco 3 — sem controle vs. duplo condicional
        if disp["sem_controle"] and disp["duplo_cond"]:
            print(f"\n[Bloco 3] Comparação sem controle vs. duplo — {indicador}")
            plotar_curva_mestre_comparativa(
                [CEN_SC, CEN_DC], _pasta_bloco("bloco3_sem_controle_vs_duplo"), indicador=indicador,
            )
            gerar_estatisticas_comparativas(
                [CEN_SC, CEN_DC], _pasta_bloco("bloco3_sem_controle_vs_duplo"), indicador=indicador,
            )
            for pen in [50, 100]:
                plotar_boxplot_horario_comparativo(
                    [CEN_SC, CEN_DC],
                    [_csv_tensoes["sem_controle"], _csv_tensoes["duplo_cond"]],
                    _pasta_bloco("bloco3_sem_controle_vs_duplo"),
                    nivel_pen=pen, indicador=indicador,
                )
                if os.path.isfile(_csv_tensoes["sem_controle"]) and os.path.isfile(_csv_tensoes["duplo_cond"]):
                    plotar_heatmap_diferenca(
                        _csv_tensoes["sem_controle"], _csv_tensoes["duplo_cond"],
                        _pasta_bloco("bloco3_sem_controle_vs_duplo"),
                        pen_pct_alvo=pen,
                        rotulo_a="Sem controle", rotulo_b="Duplo condicional",
                        indicador=indicador,
                    )

        # Bloco 4 — síntese três cenários
        cens_disponiveis = [c for c, k in [(CEN_SC, "sem_controle"), (CEN_PR, "primario"), (CEN_DC, "duplo_cond")] if disp[k]]
        if len(cens_disponiveis) >= 2:
            print(f"\n[Bloco 4] Síntese — {indicador}")
            plotar_curva_mestre_comparativa(
                cens_disponiveis, _pasta_bloco("bloco4_sintese_tres_cenarios"), indicador=indicador,
            )
            gerar_estatisticas_comparativas(
                cens_disponiveis, _pasta_bloco("bloco4_sintese_tres_cenarios"), indicador=indicador,
            )

        # Bloco 5 — sem controle vs. primário vs. apenas secundário sempre ativo
        if disp["sem_controle"] and disp["primario"] and disp["secundario_so"]:
            print(f"\n[Bloco 5] Comparação sem controle vs. primário vs. secundário — {indicador}")
            _b5 = _pasta_bloco("bloco5_sem_controle_vs_primario_vs_secundario")
            plotar_curva_mestre_comparativa(
                [CEN_SC, CEN_PR, CEN_SO], _b5, indicador=indicador,
            )
            gerar_estatisticas_comparativas(
                [CEN_SC, CEN_PR, CEN_SO], _b5, indicador=indicador,
            )
            for pen in [50, 100]:
                plotar_boxplot_horario_comparativo(
                    [CEN_SC, CEN_PR, CEN_SO],
                    [_csv_tensoes["sem_controle"], _csv_tensoes["primario"], _csv_tensoes["secundario_so"]],
                    _b5,
                    nivel_pen=pen, indicador=indicador,
                )
                if (
                    os.path.isfile(_csv_tensoes["sem_controle"])
                    and os.path.isfile(_csv_tensoes["secundario_so"])
                ):
                    plotar_heatmap_diferenca(
                        _csv_tensoes["sem_controle"], _csv_tensoes["secundario_so"],
                        _b5,
                        pen_pct_alvo=pen,
                        rotulo_a="Sem controle", rotulo_b="Secundário",
                        indicador=indicador,
                    )
                if (
                    os.path.isfile(_csv_tensoes["primario"])
                    and os.path.isfile(_csv_tensoes["secundario_so"])
                ):
                    plotar_heatmap_diferenca(
                        _csv_tensoes["primario"], _csv_tensoes["secundario_so"],
                        _b5,
                        pen_pct_alvo=pen,
                        rotulo_a="Primário", rotulo_b="Secundário",
                        indicador=indicador,
                    )

    print("\nFiguras comparativas geradas em:", pasta_comp)


def main():
    args = parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Modo comparativos: independente das flags de controle
    if args.gerar_comparativos:
        gerar_graficos_comparativos(base_dir)
        return

    _validar_args(args)

    dss_path  = os.path.abspath(os.path.join(base_dir, args.dss_file))
    pasta_montecarlo = os.path.join(
        base_dir, "resultados_monte_carlo", "realizacoes_sorteadas"
    )

    pasta_saida, pasta_saida_descritiva, contexto = _definir_pastas_e_contexto(
        args, base_dir
    )
    os.makedirs(pasta_saida, exist_ok=True)
    os.makedirs(pasta_saida_descritiva, exist_ok=True)

    caminho_master_resultados = os.path.join(
        pasta_saida, "master_resultados_opendss.csv"
    )
    caminho_master_tensoes = os.path.join(
        pasta_saida, "master_tensoes_opendss_completas.csv"
    )

    # ------------------------------------------------------------------
    # Modo somente gráficos
    # ------------------------------------------------------------------
    if args.somente_graficos:
        pasta_base_mc = os.path.join(base_dir, "resultados_monte_carlo")
        sem_flags_controle = not args.controle_primario and not args.controle_secundario

        if sem_flags_controle:
            # Sem flags de controle → regenera TODOS os cenários disponíveis
            print("Modo somente gráficos: regenerando todos os cenários disponíveis.")
            for cen in _CENARIOS_CONHECIDOS:
                pasta_saida_cen  = os.path.join(pasta_base_mc, cen["subdir"])
                pasta_graf_cen   = os.path.join(pasta_base_mc, cen["subdir_graf"])
                csv_master_cen   = os.path.join(pasta_saida_cen, "master_resultados_opendss.csv")
                csv_tensoes_cen  = os.path.join(pasta_saida_cen, "master_tensoes_opendss_completas.csv")
                if not os.path.isfile(csv_master_cen):
                    print(f"  ⚠ CSV não encontrado para '{cen['descricao']}', pulando.")
                    continue
                print(f"\n=== {cen['descricao']} ===")
                df_cen = pd.read_csv(csv_master_cen, sep=";", decimal=",")
                os.makedirs(pasta_graf_cen, exist_ok=True)
                gerar_graficos_e_tabelas(
                    df_cen, pasta_saida_cen, pasta_graf_cen,
                    csv_tensoes_cen, base_dir, contexto=cen["contexto"],
                )
        else:
            # Com flags de controle → regenera apenas o cenário selecionado
            if not os.path.isfile(caminho_master_resultados):
                raise FileNotFoundError(
                    f"CSV de resultados não encontrado: {caminho_master_resultados}"
                )
            print(f"Modo somente gráficos: regenerando cenário '{contexto or 'sem controle'}'.")
            df_master = pd.read_csv(caminho_master_resultados, sep=";", decimal=",")
            if args.nivel:
                nivel_pct = int(args.nivel)
                df_master = df_master[df_master["pen_pct"] == nivel_pct]
                if df_master.empty:
                    raise ValueError(
                        f"Nenhum resultado encontrado para penetração {nivel_pct}%%."
                    )
            gerar_graficos_e_tabelas(
                df_master, pasta_saida, pasta_saida_descritiva,
                caminho_master_tensoes, base_dir, contexto=contexto,
            )

        # Em ambos os casos, regenera também os comparativos
        print("\n=== Gráficos comparativos multi-cenário ===")
        gerar_graficos_comparativos(base_dir)
        print("\nTodos os gráficos atualizados.")
        return

    # ------------------------------------------------------------------
    # Simulação completa
    # ------------------------------------------------------------------
    if not os.path.isfile(dss_path):
        raise FileNotFoundError(f"Arquivo DSS não encontrado: {dss_path}")

    niveis = sorted(
        d for d in os.listdir(pasta_montecarlo)
        if d.startswith("pen_") and os.path.isdir(os.path.join(pasta_montecarlo, d))
    )

    if args.nivel:
        nivel_dir = f"pen_{args.nivel}pct"
        if nivel_dir not in niveis:
            raise FileNotFoundError(
                f"Nível de penetração não encontrado: {nivel_dir}"
            )
        niveis = [nivel_dir]

    processar = _selecionar_funcao_processar(args)

    todos_resultados = []
    if os.path.exists(caminho_master_tensoes):
        os.remove(caminho_master_tensoes)

    total_realizacoes_com_erro = 0
    for nivel_dir in niveis:
        pasta_nivel       = os.path.join(pasta_montecarlo, nivel_dir)
        pasta_saida_nivel = os.path.join(pasta_saida, nivel_dir)
        print(f"Processando nível: {nivel_dir}")

        df_nivel, realizacoes_com_erro, df_tensoes_nivel = processar(
            pasta_nivel,
            dss_path,
            pasta_saida_nivel,
            max_realizacoes=args.max_realizacoes,
        )
        todos_resultados.append(df_nivel)

        # Acumula o CSV de tensões completas em modo append
        header = not os.path.exists(caminho_master_tensoes)
        df_tensoes_nivel.to_csv(
            caminho_master_tensoes,
            index=False, sep=";", decimal=",",
            mode="a", header=header,
        )
        total_realizacoes_com_erro += realizacoes_com_erro

    if todos_resultados:
        df_master = pd.concat(todos_resultados, ignore_index=True)
        df_master.to_csv(caminho_master_resultados, index=False, sep=";", decimal=",")
        gerar_graficos_e_tabelas(
            df_master, pasta_saida, pasta_saida_descritiva,
            caminho_master_tensoes, base_dir, contexto=contexto,
        )
        print("\nAnálise finalizada. Master CSV salvo.")
        print("\n⚠ RESUMO DE ERROS:")
        print(
            f"  Total de realizações com erro 'Max Control Iterations': "
            f"{total_realizacoes_com_erro}"
        )
        if total_realizacoes_com_erro > 0:
            print(
                "  Detalhes por realização disponíveis na coluna "
                "'horas_com_erro_max_control' do CSV de resultados."
            )
    else:
        print("Nenhum nível de penetração encontrado em realizacoes_sorteadas.")


if __name__ == "__main__":
    main()
