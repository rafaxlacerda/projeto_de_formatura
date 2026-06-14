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

from estatisticas_opendss import gerar_tabelas_estatisticas
from graficos_opendss import (
    plotar_boxplot,
    plotar_boxplot_geral,
    plotar_boxplot_por_faixa_bess,
    plotar_boxplot_por_time_bands,
    plotar_envelope_tensao,
    plotar_envelope_tensao_por_hora,
)
from controle_primario import processar_nivel_controle_primario
from controle_secundario import processar_nivel_controle_secundario
from simulacao_opendss import processar_nivel
from simulacoes_config import DEFAULT_DSS_FILE


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
    print("\nGerando boxplot geral (sem segmentação)...")
    plotar_boxplot_geral(df_master, pasta_saida_descritiva, contexto=contexto)

    print("\nGerando boxplots por time bands...")
    plotar_boxplot_por_time_bands(df_master, pasta_saida_descritiva, contexto=contexto)

    print("\nGerando boxplots por tipo de dia...")
    plotar_boxplot(df_master, pasta_saida_descritiva, contexto=contexto)

    print("\nGerando boxplots por faixa de operação BESS...")
    plotar_boxplot_por_faixa_bess(df_master, pasta_saida_descritiva, contexto=contexto)

    print("\nGerando envelopes de tensão por faixa horária...")
    plotar_envelope_tensao(pasta_saida, pasta_saida_descritiva, contexto=contexto)

    print("\nGerando envelopes de tensão para cada hora do dia...")
    plotar_envelope_tensao_por_hora(
        caminho_master_tensoes, pasta_saida_descritiva, contexto=contexto
    )

    print("\nGerando tabelas de estatísticas descritivas...")
    pasta_monte_carlo = os.path.join(base_dir, "resultados_monte_carlo")
    gerar_tabelas_estatisticas(
        pasta_saida_descritiva, pasta_monte_carlo, pasta_saida, contexto=contexto
    )


def main():
    args = parse_args()
    _validar_args(args)

    base_dir  = os.path.dirname(os.path.abspath(__file__))
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
        if not os.path.isfile(caminho_master_resultados):
            raise FileNotFoundError(
                f"CSV de resultados não encontrado: {caminho_master_resultados}"
            )
        print("Modo somente gráficos: usando CSVs existentes, sem rodar simulações.")
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
        print(
            f"\nGráficos e tabelas atualizados em "
            f"{os.path.basename(pasta_saida_descritiva)}."
        )
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
