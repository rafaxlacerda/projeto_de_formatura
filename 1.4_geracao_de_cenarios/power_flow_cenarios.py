import argparse
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from opendssdirect import dss
except ImportError as exc:
    raise ImportError(
        "opendssdirect não está instalado. Instale com `pip install opendssdirect` "
        "e verifique se o OpenDSS está disponível no ambiente." 
    ) from exc

# Parâmetros de defeito de tensão
V_PU_MIN = 0.95
V_PU_MAX = 1.05

# Caminho padrão do modelo IEEE34
DEFAULT_DSS_FILE = os.path.join("..", "IEEE34bus", "IEEE34_2.dss")

# Faixas de horário usadas na análise
TIME_BANDS = {
    "dia": list(range(7, 18)),       # 07h-17h
    "tarde": list(range(18, 22)),    # 18h-21h
    "noite": list(range(22, 24)) + list(range(0, 7)),  # 22h-06h
}

# Perfil fixo de BESS (o sinal negativo significa absorção/recarga)
BESS_PERFIL = np.zeros(24)
BESS_PERFIL[10:15] = -1.0
BESS_PERFIL[18:22] = 1.0


def ler_perfis_carga(pasta_nivel):
    caminho = os.path.join(pasta_nivel, "03_perfis_carga_por_barra.csv")
    df = pd.read_csv(caminho, sep=";", decimal=",")
    df = df.sort_values(["id_realizacao", "barra"])
    return df


def ler_perfis_irradiancia(pasta_nivel):
    caminho = os.path.join(pasta_nivel, "02_perfis_irradiancia.csv")
    df = pd.read_csv(caminho, sep=";", decimal=",")
    df = df.sort_values("id_realizacao")
    return df


def ler_resumo_configuracoes(pasta_nivel):
    caminho = os.path.join(pasta_nivel, "01_resumo_configuracoes.csv")
    return pd.read_csv(caminho, sep=";", decimal=",")


def ler_elementos_opendss(pasta_nivel):
    caminho = os.path.join(pasta_nivel, "06_elementos_opendss.csv")
    if not os.path.isfile(caminho):
        return pd.DataFrame(columns=["id_realizacao", "classe", "nome", "barra", "potencia_kw", "capacidade_kwh", "perfil"])
    return pd.read_csv(caminho, sep=";", decimal=",")


def redirecionar_circuito(dss_path):
    dss.Basic.ClearAll()
    comando = f'Redirect "{dss_path.replace("\\", "/")}"'
    dss.Text.Command(comando)


def carregar_cargas_base():
    cargas = {}
    nome = dss.Loads.First()
    while nome:
        dss.Circuit.SetActiveElement(f"Load.{nome}")
        kw = dss.Loads.kW()
        kvar = dss.Loads.kvar()
        bus = dss.CktElement.BusNames()[0].split(".")[0]
        cargas[nome] = {
            "barra": int(bus),
            "kW": kw,
            "kvar": kvar,
        }
        nome = dss.Loads.Next()
    return cargas


def _parse_bus_number(nome_load):
    # Função mantida para compatibilidade, mas não usada mais
    try:
        return int(nome_load.lstrip("B").split()[0])
    except ValueError:
        return None


def criar_elementos_simulacao(pv_df, bess_df, id_realizacao):
    for _, linha in pv_df.iterrows():
        nome = f"PV_{id_realizacao}_barra{linha['barra']}"
        barra = int(linha["barra"])
        dss.Text.Command(
            f"New Load.{nome} Bus1={barra} Phases=3 Conn=Wye kW=0 kvar=0"
        )
    for _, linha in bess_df.iterrows():
        nome = f"BESS_{id_realizacao}_barra{linha['barra']}"
        barra = int(linha["barra"])
        dss.Text.Command(
            f"New Load.{nome} Bus1={barra} Phases=3 Conn=Wye kW=0 kvar=0"
        )


def editar_load(nome, kw, kvar=0.0):
    dss.Text.Command(f"Edit Load.{nome} kW={kw:.6f} kvar={kvar:.6f}")


def extrair_tensoes_por_barra():
    dados = {}
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        pu_vals = dss.Bus.puVmagAngle()
        mags = pu_vals[0::2]
        if not mags:
            continue
        dados[bus.lower()] = float(np.mean(mags))
    return dados


def agrupar_horas_por_faixa(horas):
    resultado = {}
    for faixa, horas_faixa in TIME_BANDS.items():
        resultado[faixa] = [h for h in horas_faixa if h in horas]
    return resultado


def calcular_defeitos_por_faixa(tensoes_por_hora):
    defeitos = {}
    for faixa, horas in TIME_BANDS.items():
        medias_por_barra = {}
        for barra in next(iter(tensoes_por_hora.values())).keys():
            valores = [tensoes_por_hora[h][barra] for h in horas]
            medias_por_barra[barra] = np.mean(valores)
        defeitos[faixa] = sum(
            1 for media in medias_por_barra.values()
            if media < V_PU_MIN or media > V_PU_MAX
        )
    return defeitos


def simular_realizacao(realizacao_id, resumo, pv_df, bess_df, perfis_irr, perfis_carga, cargas_base):
    radiancias = perfis_irr.loc[perfis_irr["id_realizacao"] == realizacao_id].iloc[0]
    fatores_irradiancia = [float(radiancias[f"h{h:02d}"]) for h in range(24)]

    carga_real = perfis_carga[perfis_carga["id_realizacao"] == realizacao_id]
    fatores_carga = {
        int(linha["barra"]): [float(linha[f"h{h:02d}"]) for h in range(24)]
        for _, linha in carga_real.iterrows()
    }

    pv_real = pv_df[pv_df["id_realizacao"] == realizacao_id]
    bess_real = bess_df[bess_df["id_realizacao"] == realizacao_id]

    # Criar elementos adicionais para PV e BESS
    criar_elementos_simulacao(pv_real, bess_real, id_realizacao)

    tensoes_por_hora = {}
    for hora in range(24):
        for nome_load, carga in cargas_base.items():
            barra = carga["barra"]
            fator = fatores_carga.get(barra, [1.0] * 24)[hora]
            kw = carga["kW"] * fator
            kvar = carga["kvar"] * fator
            editar_load(nome_load, kw, kvar)

        for _, linha in pv_real.iterrows():
            nome = f"PV_{id_realizacao}_barra{linha['barra']}"
            potencia_kw = float(linha["potencia_kw"])
            kw = -potencia_kw * fatores_irradiancia[hora]
            editar_load(nome, kw, 0.0)

        for _, linha in bess_real.iterrows():
            nome = f"BESS_{id_realizacao}_barra{linha['barra']}"
            potencia_kw = float(linha["potencia_kw"])
            kw = -potencia_kw * BESS_PERFIL[hora]
            editar_load(nome, kw, 0.0)

        dss.Solution.Solve()
        tensoes_por_hora[hora] = extrair_tensoes_por_barra()

    defeitos = calcular_defeitos_por_faixa(tensoes_por_hora)

    return {
        "id_realizacao": realizacao_id,
        "tipo_dia": resumo.loc[resumo["id_realizacao"] == realizacao_id, "tipo_dia"].iloc[0],
        "pv_unidades": int(resumo.loc[resumo["id_realizacao"] == realizacao_id, "pv_unidades"].iloc[0]),
        "pv_potencia_total_kw": float(resumo.loc[resumo["id_realizacao"] == realizacao_id, "pv_potencia_total_kw"].iloc[0]),
        "bess_unidades": int(resumo.loc[resumo["id_realizacao"] == realizacao_id, "bess_unidades"].iloc[0]),
        "bess_potencia_total_kw": float(bess_real["potencia_kw"].sum()) if not bess_real.empty else 0.0,
        "defeitos_dia": defeitos["dia"],
        "defeitos_tarde": defeitos["tarde"],
        "defeitos_noite": defeitos["noite"],
    }


def processar_nivel(pasta_nivel, dss_path, pasta_saida_nivel, max_realizacoes=None):
    resumo = ler_resumo_configuracoes(pasta_nivel)
    perfis_irr = ler_perfis_irradiancia(pasta_nivel)
    perfis_carga = ler_perfis_carga(pasta_nivel)
    elementos = ler_elementos_opendss(pasta_nivel)

    if resumo.empty:
        raise ValueError(f"Resumo de configurações não encontrado em {pasta_nivel}")

    resultados = []
    for _, row in resumo.head(max_realizacoes).iterrows():
        id_realizacao = int(row["id_realizacao"])

        redirecionar_circuito(dss_path)
        cargas_base = carregar_cargas_base()

        resultado = simular_realizacao(
            id_realizacao,
            resumo,
            elementos[elementos["id_realizacao"] == id_realizacao],
            elementos[(elementos["id_realizacao"] == id_realizacao) & (elementos["classe"] == "Storage")],
            perfis_irr,
            perfis_carga,
            cargas_base,
        )
        resultado["pen_pct"] = int(os.path.basename(pasta_nivel).split("_")[1].replace("pct", ""))
        resultados.append(resultado)
        print(f"  ✓ Realização {id_realizacao} - defeitos: dia={resultado['defeitos_dia']} tarde={resultado['defeitos_tarde']} noite={resultado['defeitos_noite']}")

    df_resultados = pd.DataFrame(resultados)
    os.makedirs(pasta_saida_nivel, exist_ok=True)
    df_resultados.to_csv(os.path.join(pasta_saida_nivel, "resultados_opendss_por_realizacao.csv"), index=False, sep=";", decimal=",")
    return df_resultados


def plotar_boxplot(df_master, pasta_saida):
    grupos = [group for _, group in df_master.groupby("pen_pct")]
    niveis = sorted(df_master["pen_pct"].unique())

    fig, axes = plt.subplots(3, 1, figsize=(16, 18), constrained_layout=True)
    for ax, faixa in zip(axes, ["dia", "tarde", "noite"]):
        data = [
            df_master.loc[df_master["pen_pct"] == nivel, f"defeitos_{faixa}"]
            for nivel in niveis
        ]
        ax.boxplot(data, labels=[str(n) for n in niveis], showfliers=False)
        ax.set_title(f"Número de barras com defeito de tensão - faixa {faixa}")
        ax.set_xlabel("Penetração PV (%)")
        ax.set_ylabel("Barras defectivas")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(0, max(df_master[f"defeitos_{faixa}"].max() + 2, 5))

    fig.suptitle("Defeitos de tensão por faixa de horário e nível de penetração", fontsize=18)
    caminho_imagem = os.path.join(pasta_saida, "boxplot_defeitos_por_faixa.png")
    fig.savefig(caminho_imagem, dpi=200)
    plt.close(fig)
    return caminho_imagem


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
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dss_path = os.path.abspath(os.path.join(base_dir, args.dss_file))
    pasta_montecarlo = os.path.join(base_dir, "resultados_monte_carlo_v2")
    pasta_saida = os.path.join(pasta_montecarlo, "analise_opendss")
    os.makedirs(pasta_saida, exist_ok=True)

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
    for nivel_dir in niveis:
        pasta_nivel = os.path.join(pasta_montecarlo, nivel_dir)
        pasta_saida_nivel = os.path.join(pasta_saida, nivel_dir)
        print(f"Processando nível: {nivel_dir}")
        df_nivel = processar_nivel(pasta_nivel, dss_path, pasta_saida_nivel, max_realizacoes=args.max_realizacoes)
        todos_resultados.append(df_nivel)

    if todos_resultados:
        df_master = pd.concat(todos_resultados, ignore_index=True)
        df_master.to_csv(os.path.join(pasta_saida, "master_resultados_opendss.csv"), index=False, sep=";", decimal=",")
        path_plot = plotar_boxplot(df_master, pasta_saida)
        print(f"\nAnálise finalizada. Master CSV salvo em: {os.path.join(pasta_saida, 'master_resultados_opendss.csv')}")
        print(f"Boxplot salvo em: {path_plot}")
    else:
        print("Nenhum nível de penetração encontrado em resultados_monte_carlo_v2.")


if __name__ == "__main__":
    main()
