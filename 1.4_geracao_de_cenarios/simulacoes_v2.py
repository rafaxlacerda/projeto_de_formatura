# Main de processamento dos resultados do Monte Carlo via OpenDSS
import argparse
import os

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
from simulacao_opendss import processar_nivel
from simulacoes_config import DEFAULT_DSS_FILE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simula os cenários de Monte Carlo via OpenDSS e compila defeitos de tensão por faixa horária."
    )
    parser.add_argument(
        "--nivel",
        type=str,
        default=None,
        help="Penetracao especifica para processar"
    )
    parser.add_argument(
        "--max-realizacoes",
        type=int,
        default=None,
        help="Número máximo de realizações a processar por nível."
    )
    parser.add_argument(
        "--dss-file",
        type=str,
        default=DEFAULT_DSS_FILE,
        help="Arquivo DSS a ser usado para a simulação."
    )
    parser.add_argument(
        "--somente-graficos",
        action="store_true",
        help="Gera apenas gráficos e tabelas a partir dos CSVs existentes, sem rodar o OpenDSS."
    )
    return parser.parse_args()


def gerar_graficos_e_tabelas(df_master, pasta_saida, pasta_saida_descritiva, caminho_master_tensoes, base_dir):
    print("\nGerando boxplot geral (sem segmentação)...")
    plotar_boxplot_geral(df_master, pasta_saida_descritiva)

    print("\nGerando boxplots por time bands...")
    plotar_boxplot_por_time_bands(df_master, pasta_saida_descritiva)

    print("\nGerando boxplots por tipo de dia...")
    plotar_boxplot(df_master, pasta_saida_descritiva)

    print("\nGerando boxplots por faixa de operação BESS...")
    plotar_boxplot_por_faixa_bess(df_master, pasta_saida_descritiva)

    print("\nGerando envelopes de tensão por faixa horária...")
    plotar_envelope_tensao(pasta_saida, pasta_saida_descritiva)

    print("\nGerando envelopes de tensão para cada hora do dia...")
    plotar_envelope_tensao_por_hora(caminho_master_tensoes, pasta_saida_descritiva)

    print("\nGerando tabelas de estatísticas descritivas...")
    pasta_monte_carlo = os.path.join(base_dir, "resultados_monte_carlo")
    gerar_tabelas_estatisticas(pasta_saida_descritiva, pasta_monte_carlo, pasta_saida)


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dss_path = os.path.abspath(os.path.join(base_dir, args.dss_file))
    pasta_montecarlo = os.path.join(base_dir, "resultados_monte_carlo", "realizacoes_sorteadas")
    pasta_saida = os.path.join(base_dir, "resultados_monte_carlo", "analise_opendss")
    pasta_saida_descritiva = os.path.join(base_dir, "resultados_monte_carlo", "graficos_tabelas_descritivas")
    os.makedirs(pasta_saida, exist_ok=True)
    os.makedirs(pasta_saida_descritiva, exist_ok=True)
    caminho_master_resultados = os.path.join(pasta_saida, "master_resultados_opendss.csv")
    caminho_master_tensoes = os.path.join(pasta_saida, "master_tensoes_opendss_completas.csv")

    if args.somente_graficos:
        if not os.path.isfile(caminho_master_resultados):
            raise FileNotFoundError(f"CSV de resultados não encontrado: {caminho_master_resultados}")

        print("Modo somente gráficos: usando CSVs existentes, sem rodar simulações.")
        df_master = pd.read_csv(caminho_master_resultados, sep=";", decimal=",")
        if args.nivel:
            nivel_pct = int(args.nivel)
            df_master = df_master[df_master["pen_pct"] == nivel_pct]
            if df_master.empty:
                raise ValueError(f"Nenhum resultado encontrado para penetração {nivel_pct}%.")

        gerar_graficos_e_tabelas(df_master, pasta_saida, pasta_saida_descritiva, caminho_master_tensoes, base_dir)
        print("\nGráficos e tabelas atualizados em graficos_tabelas_descritivas.")
        return

    if not os.path.isfile(dss_path):
        raise FileNotFoundError(f"Arquivo DSS não encontrado: {dss_path}")

    niveis = sorted(
        [d for d in os.listdir(pasta_montecarlo) if d.startswith("pen_") and os.path.isdir(os.path.join(pasta_montecarlo, d))]
    )

    if args.nivel:
        nivel_dir = f"pen_{args.nivel}pct"
        if nivel_dir not in niveis:
            raise FileNotFoundError(f"Nível de penetração não encontrado: {nivel_dir}")
        niveis = [nivel_dir]

    todos_resultados = []
    if os.path.exists(caminho_master_tensoes):
        os.remove(caminho_master_tensoes)

    total_realizacoes_com_erro = 0
    for nivel_dir in niveis:
        pasta_nivel = os.path.join(pasta_montecarlo, nivel_dir)
        pasta_saida_nivel = os.path.join(pasta_saida, nivel_dir)
        print(f"Processando nível: {nivel_dir}")
        df_nivel, realizacoes_com_erro, df_tensoes_nivel = processar_nivel(
            pasta_nivel,
            dss_path,
            pasta_saida_nivel,
            max_realizacoes=args.max_realizacoes,
        )
        todos_resultados.append(df_nivel)
        df_tensoes_nivel.to_csv(
            caminho_master_tensoes,
            index=False,
            sep=";",
            decimal=",",
            mode="a",
            header=not os.path.exists(caminho_master_tensoes),
        )
        total_realizacoes_com_erro += realizacoes_com_erro

    if todos_resultados:
        df_master = pd.concat(todos_resultados, ignore_index=True)
        df_master.to_csv(caminho_master_resultados, index=False, sep=";", decimal=",")
        gerar_graficos_e_tabelas(df_master, pasta_saida, pasta_saida_descritiva, caminho_master_tensoes, base_dir)

        print("\nAnálise finalizada. Master CSV salvo")

        print("\n⚠ RESUMO DE ERROS:")
        print(f"  Total de realizações com erro 'Max Control Iterations': {total_realizacoes_com_erro}")
        if total_realizacoes_com_erro > 0:
            print("  Detalhes por realização disponíveis na coluna 'horas_com_erro_max_control' do CSV de resultados.")
    else:
        print("Nenhum nível de penetração encontrado em realizacoes_sorteadas.")


if __name__ == "__main__":
    main()
