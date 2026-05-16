import argparse
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from opendssdirect import dss
from dss._cffi_api_util import DSSException

# Parâmetros de defeito de tensão
V_PU_MIN = 0.95
V_PU_MAX = 1.05

# Caminho padrão do modelo IEEE34 - Usa arquivo original com loadshapes.
DEFAULT_DSS_FILE = os.path.join("..", "IEEE34bus", "IEEE34_original_with_loadshapes.dss")

# Faixas de horário usadas na análise
TIME_BANDS = {
    "manhã": list(range(6, 12)),       # 06h-11h
    "tarde": list(range(12, 18)),       # 12h-17h
    "noite": list(range(18, 24)),       # 18h-23h
    "madrugada": list(range(0, 6)),     # 00h-05h
}

# Faixas de operação da bateria (BESS)
BESS_BANDS = {
    "carga": list(range(10, 15)),                           # 10h-14h (absorção/recarga)
    "pós_carga_pré_descarga": list(range(15, 18)),          # 15h-17h (standby entre operações)
    "descarga": list(range(18, 22)),                        # 18h-21h (injeção)
    "fora_de_operacao": list(range(22, 24)) + list(range(0, 10))  # 22h-09h (inativo)
}

# Perfil fixo de BESS (o sinal negativo significa absorção/recarga)
# TODO: definir de acordo com perfil de cargas da rede -> que gera o melhor perfil de tensão ao longo das 24h ( menos barras com problema)
BESS_PERFIL = np.zeros(24)
BESS_PERFIL[10:15] = -1.0
BESS_PERFIL[18:22] = 1.0
# controle intermediario: ativa ou desativa carregamento - etapa 2

def ler_fatores_incerteza_carga(pasta_nivel):
    caminho = os.path.join(pasta_nivel, "03_fatores_incerteza_carga.csv")
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
    dss.Text.Command(f'Redirect "{dss_path.replace("\\", "/")}"')
    dss.Text.Command("Set MaxControlIter=100")


def carregar_cargas_base():
    cargas = {}
    nome = dss.Loads.First()
    while nome:
        dss.Circuit.SetActiveElement(f"Load.{nome}")
        kw = dss.Loads.kW()
        kvar = dss.Loads.kvar()
        bus = dss.CktElement.BusNames()[0].split(".")[0]
        shape_nome = dss.Loads.Daily()
        cargas[nome] = {
            "barra": int(bus),
            "kW": kw,
            "kvar": kvar,
            "shape": shape_nome
        }
        nome = dss.Loads.Next()
    return cargas


def obter_multiplicadores_shapes():
    """Obtém os multiplicadores (PMult) de todos os LoadShapes definidos no DSS."""
    shapes = {}
    idx = dss.LoadShape.First()
    while idx > 0:
        # Name() retorna o nome do LoadShape ativo
        nome = dss.LoadShape.Name()
        # PMult retorna a lista de multiplicadores (os valores de 'mult' no seu DSS)
        shapes[nome.lower()] = dss.LoadShape.PMult()
        idx = dss.LoadShape.Next()
    return shapes


def _parse_bus_number(nome_load):
    # Função mantida para compatibilidade, mas não usada mais
    try:
        return int(nome_load.lstrip("B").split()[0])
    except ValueError:
        return None

# Alternativa ao PVsystem e Storage nativos 
def criar_elementos_simulacao(pv_df, bess_df, id_realizacao):
    for idx, (_, linha) in enumerate(pv_df.iterrows()):
        nome = f"PV_{id_realizacao}_{idx}_barra{linha['barra']}"
        barra = int(linha["barra"])
        dss.Text.Command(
            f"New Load.{nome} Bus1={barra} Phases=3 Conn=Wye kW=0 kvar=0"
        )
    for idx, (_, linha) in enumerate(bess_df.iterrows()):
        nome = f"BESS_{id_realizacao}_{idx}_barra{linha['barra']}"
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
        # Filtrar apenas horas que foram simuladas com sucesso
        horas_disponiveis = [h for h in horas if h in tensoes_por_hora]
        
        if not horas_disponiveis:
            # Se nenhuma hora da faixa foi simulada, registrar 0 defeitos
            defeitos[faixa] = 0
            continue
            
        for barra in next(iter(tensoes_por_hora.values())).keys():
            valores = [tensoes_por_hora[h][barra] for h in horas_disponiveis]
            medias_por_barra[barra] = np.mean(valores)
        defeitos[faixa] = sum(
            1 for media in medias_por_barra.values()
            if media < V_PU_MIN or media > V_PU_MAX
        )
    return defeitos


def calcular_defeitos_por_faixa_bess(tensoes_por_hora):
    """Calcula defeitos de tensão para as faixas de operação da bateria"""
    defeitos = {}
    for faixa, horas in BESS_BANDS.items():
        medias_por_barra = {}
        # Filtrar apenas horas que foram simuladas com sucesso
        horas_disponiveis = [h for h in horas if h in tensoes_por_hora]
        
        if not horas_disponiveis:
            # Se nenhuma hora da faixa foi simulada, registrar 0 defeitos
            defeitos[faixa] = 0
            continue
            
        for barra in next(iter(tensoes_por_hora.values())).keys():
            valores = [tensoes_por_hora[h][barra] for h in horas_disponiveis]
            medias_por_barra[barra] = np.mean(valores)
        defeitos[faixa] = sum(
            1 for media in medias_por_barra.values()
            if media < V_PU_MIN or media > V_PU_MAX
        )
    return defeitos


def simular_realizacao(realizacao_id, resumo, pv_df, bess_df, perfis_irr, fatores_incerteza, cargas_base):
    radiancias = perfis_irr.loc[perfis_irr["id_realizacao"] == realizacao_id].iloc[0]
    fatores_irradiancia = [float(radiancias[f"h{h:02d}"]) for h in range(24)]

    carga_real = fatores_incerteza[fatores_incerteza["id_realizacao"] == realizacao_id]
    fatores_carga = {
        int(linha["barra"]): [float(linha[f"h{h:02d}"]) for h in range(24)]
        for _, linha in carga_real.iterrows()
    }

    pv_real = pv_df[pv_df["id_realizacao"] == realizacao_id]
    bess_real = bess_df[bess_df["id_realizacao"] == realizacao_id]

    # Criar elementos adicionais para PV e BESS
    criar_elementos_simulacao(pv_real, bess_real, realizacao_id)
    
    multiplicadores_perfil_carga = obter_multiplicadores_shapes()

    tensoes_por_hora = {}
    horas_com_erro = 0
    for hora in range(24):
        # Perfil carga
        for nome_load, carga in cargas_base.items():
            barra = carga["barra"]
            base_kw = carga["kW"]
            base_kvar = carga["kvar"]
            nome_shape = carga["shape"].lower()

            # Fator 1: Incerteza (multiplicador hora a hora, default 1.0 se não definido)
            fator_incerteza = fatores_carga.get(barra, [1.0] * 24)[hora]
            
            # Fator 2: LoadShape (comercial, industrial ou residencial)
            fator_perfil = multiplicadores_perfil_carga.get(nome_shape, [1.0] * 24)[hora]

            kw_final = base_kw * fator_perfil * fator_incerteza
            kvar_final = base_kvar * fator_perfil * fator_incerteza
        
            editar_load(nome_load, kw_final, kvar_final)

        # Perfil PV (o - define que é gerador, injeção de potência)
        for idx, (_, linha) in enumerate(pv_real.iterrows()):
            nome = f"PV_{realizacao_id}_{idx}_barra{linha['barra']}"
            potencia_kw = float(linha["potencia_kw"])
            kw = -potencia_kw * fatores_irradiancia[hora]
            editar_load(nome, kw, 0.0)

        # Perfil BESS
        for idx, (_, linha) in enumerate(bess_real.iterrows()):
            nome = f"BESS_{realizacao_id}_{idx}_barra{linha['barra']}"
            potencia_kw = float(linha["potencia_kw"])
            # Aplica perfil fixo de carga/descarga (negativo = absorção, positivo = injeção)
            kw = -potencia_kw * BESS_PERFIL[hora]
            editar_load(nome, kw, 0.0)

        try:
            dss.Solution.Solve()
            tensoes_por_hora[hora] = extrair_tensoes_por_barra()
        except DSSException as e:
            # Ignorar erro de Max Control Iterations (pode ocorrer devido a não convergência do controle dos reguladores)
            if e.args[0] == 485 or "Max Control Iterations" in str(e):
                horas_com_erro += 1
                # Pular esta hora e continuar com a próxima
                continue
            else:
                raise

    defeitos = calcular_defeitos_por_faixa(tensoes_por_hora)
    defeitos_bess = calcular_defeitos_por_faixa_bess(tensoes_por_hora)

    return {
        "id_realizacao": realizacao_id,
        "tipo_dia": resumo.loc[resumo["id_realizacao"] == realizacao_id, "tipo_dia"].iloc[0],
        "pv_unidades": int(resumo.loc[resumo["id_realizacao"] == realizacao_id, "pv_unidades"].iloc[0]),
        "pv_potencia_total_kw": float(resumo.loc[resumo["id_realizacao"] == realizacao_id, "pv_potencia_total_kw"].iloc[0]),
        "bess_unidades": int(resumo.loc[resumo["id_realizacao"] == realizacao_id, "bess_unidades"].iloc[0]),
        "bess_potencia_total_kw": float(bess_real["potencia_kw"].sum()) if not bess_real.empty else 0.0,
        "defeitos_manhã": defeitos["manhã"],
        "defeitos_tarde": defeitos["tarde"],
        "defeitos_noite": defeitos["noite"],
        "defeitos_madrugada": defeitos["madrugada"],
        "defeitos_bess_carga": defeitos_bess["carga"],
        "defeitos_bess_pós_carga_pré_descarga": defeitos_bess["pós_carga_pré_descarga"],
        "defeitos_bess_descarga": defeitos_bess["descarga"],
        "defeitos_bess_fora_de_operacao": defeitos_bess["fora_de_operacao"],
        "horas_com_erro_max_control": horas_com_erro,
    }


def processar_nivel(pasta_nivel, dss_path, pasta_saida_nivel, max_realizacoes=None):
    resumo = ler_resumo_configuracoes(pasta_nivel)
    perfis_irr = ler_perfis_irradiancia(pasta_nivel)
    fatores_incerteza = ler_fatores_incerteza_carga(pasta_nivel)
    elementos = ler_elementos_opendss(pasta_nivel)

    if resumo.empty:
        raise ValueError(f"Resumo de configurações não encontrado em {pasta_nivel}")

    resultados = []
    realizacoes_com_erro = 0
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
            fatores_incerteza,
            cargas_base,
        )
        resultado["pen_pct"] = int(os.path.basename(pasta_nivel).split("_")[1].replace("pct", ""))
        resultados.append(resultado)
        
        if resultado["horas_com_erro_max_control"] > 0:
            realizacoes_com_erro += 1
            print(f"  ✓ Realização {id_realizacao} - defeitos: manhã={resultado['defeitos_manhã']} tarde={resultado['defeitos_tarde']} noite={resultado['defeitos_noite']} madrugada={resultado['defeitos_madrugada']} ⚠ {resultado['horas_com_erro_max_control']} hora(s) com erro Max Control Iter")
        else:
            print(f"  ✓ Realização {id_realizacao} - defeitos: manhã={resultado['defeitos_manhã']} tarde={resultado['defeitos_tarde']} noite={resultado['defeitos_noite']} madrugada={resultado['defeitos_madrugada']}")

    df_resultados = pd.DataFrame(resultados)
    os.makedirs(pasta_saida_nivel, exist_ok=True)
    df_resultados.to_csv(os.path.join(pasta_saida_nivel, "resultados_opendss_por_realizacao.csv"), index=False, sep=";", decimal=",")
    return df_resultados, realizacoes_com_erro


def plotar_boxplot_geral(df_master, pasta_saida):
    """Gera boxplot geral sem segmentação por tipo de dia"""
    niveis = sorted(df_master["pen_pct"].unique())
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), constrained_layout=True)
    for ax, faixa in zip(axes, ["manhã", "tarde", "noite", "madrugada"]):
        data = [
            df_master.loc[df_master["pen_pct"] == nivel, f"defeitos_{faixa}"]
            for nivel in niveis
        ]
        ax.boxplot(data, labels=[str(n) for n in niveis], showfliers=False)
        ax.set_title(f"Número de barras com defeito de tensão - faixa {faixa.capitalize()}")
        ax.set_xlabel("Penetração PV (%)")
        ax.set_ylabel("Barras defectivas")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(0, 35)

    fig.suptitle("Defeitos de tensão por faixa de horário (agregado de todos os tipos de dia)", fontsize=18)
    caminho_imagem = os.path.join(pasta_saida, "boxplot_defeitos_geral.png")
    fig.savefig(caminho_imagem, dpi=200)
    plt.close(fig)
    print(f"  Boxplot geral salvo em: {caminho_imagem}")
    return caminho_imagem


def plotar_boxplot_por_time_bands(df_master, pasta_saida):
    """Gera boxplots segmentados apenas por time bands (sem separar por tipo de dia)"""
    niveis = sorted(df_master["pen_pct"].unique())
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), constrained_layout=True)
    for ax, faixa in zip(axes, ["manhã", "tarde", "noite", "madrugada"]):
        data = [
            df_master.loc[df_master["pen_pct"] == nivel, f"defeitos_{faixa}"]
            for nivel in niveis
        ]
        ax.boxplot(data, labels=[str(n) for n in niveis], showfliers=False)
        ax.set_title(f"Número de barras com defeito de tensão - faixa {faixa.capitalize()}")
        ax.set_xlabel("Penetração PV (%)")
        ax.set_ylabel("Barras defectivas")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(0, 35)

    fig.suptitle("Defeitos de tensão por faixa de horário (agregado de todos os tipos de dia)", fontsize=18)
    caminho_imagem = os.path.join(pasta_saida, "boxplot_defeitos_por_time_bands.png")
    fig.savefig(caminho_imagem, dpi=200)
    plt.close(fig)
    print(f"  Boxplot por time bands salvo em: {caminho_imagem}")
    return caminho_imagem


def plotar_boxplot(df_master, pasta_saida):
    """Gera boxplots separados por tipo de dia de irradiância"""
    niveis = sorted(df_master["pen_pct"].unique())
    tipos_dia = sorted(df_master["tipo_dia"].unique())
    
    caminhos_imagens = []
    
    # Gerar um gráfico para cada tipo de dia
    for tipo_dia in tipos_dia:
        df_filtrado = df_master[df_master["tipo_dia"] == tipo_dia]
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 20), constrained_layout=True)
        for ax, faixa in zip(axes, ["manhã", "tarde", "noite", "madrugada"]):
            data = [
                df_filtrado.loc[df_filtrado["pen_pct"] == nivel, f"defeitos_{faixa}"]
                for nivel in niveis
            ]
            ax.boxplot(data, labels=[str(n) for n in niveis], showfliers=False)
            ax.set_title(f"Número de barras com defeito de tensão - faixa {faixa.capitalize()}")
            ax.set_xlabel("Penetração PV (%)")
            ax.set_ylabel("Barras defectivas")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_ylim(0, 35)

        fig.suptitle(f"Defeitos de tensão por faixa de horário - Tipo de dia: {tipo_dia}", fontsize=18)
        caminho_imagem = os.path.join(pasta_saida, f"boxplot_defeitos_por_faixa_{tipo_dia}.png")
        fig.savefig(caminho_imagem, dpi=200)
        plt.close(fig)
        caminhos_imagens.append(caminho_imagem)
        print(f"  Boxplot para tipo de dia '{tipo_dia}' salvo em: {caminho_imagem}")
    
    return caminhos_imagens


def plotar_boxplot_por_faixa_bess(df_master, pasta_saida):
    """Gera boxplots separados por faixa de operação da bateria (BESS)"""
    niveis = sorted(df_master["pen_pct"].unique())
    tipos_dia = sorted(df_master["tipo_dia"].unique())
    
    caminhos_imagens = []
    
    # Mapeamento de nomes das colunas e títulos para melhor legibilidade
    faixas_bess = {
        "defeitos_bess_carga": "Carga (10h-14h)",
        "defeitos_bess_pós_carga_pré_descarga": "Pós-carga/Pré-descarga (15h-17h)",
        "defeitos_bess_descarga": "Descarga (18h-21h)",
        "defeitos_bess_fora_de_operacao": "Fora de operação (22h-09h)"
    }
    
    # Gerar um gráfico para cada tipo de dia
    for tipo_dia in tipos_dia:
        df_filtrado = df_master[df_master["tipo_dia"] == tipo_dia]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
        axes_flat = axes.flatten()
        
        for ax, (coluna, titulo) in zip(axes_flat, faixas_bess.items()):
            data = [
                df_filtrado.loc[df_filtrado["pen_pct"] == nivel, coluna]
                for nivel in niveis
            ]
            ax.boxplot(data, labels=[str(n) for n in niveis], showfliers=False)
            ax.set_title(f"Número de barras com defeito de tensão - {titulo}")
            ax.set_xlabel("Penetração PV (%)")
            ax.set_ylabel("Barras defectivas")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_ylim(0, 35)

        fig.suptitle(f"Defeitos de tensão por faixa de operação BESS - Tipo de dia: {tipo_dia}", fontsize=18)
        caminho_imagem = os.path.join(pasta_saida, f"boxplot_defeitos_por_faixa_bess_{tipo_dia}.png")
        fig.savefig(caminho_imagem, dpi=200)
        plt.close(fig)
        caminhos_imagens.append(caminho_imagem)
        print(f"  Boxplot BESS para tipo de dia '{tipo_dia}' salvo em: {caminho_imagem}")
    
    return caminhos_imagens


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
    pasta_montecarlo = os.path.join(base_dir, "resultados_monte_carlo", "realizacoes_sorteadas")
    pasta_saida = os.path.join(base_dir, "resultados_monte_carlo", "analise_opendss")
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
    total_realizacoes_com_erro = 0
    for nivel_dir in niveis:
        pasta_nivel = os.path.join(pasta_montecarlo, nivel_dir)
        pasta_saida_nivel = os.path.join(pasta_saida, nivel_dir)
        print(f"Processando nível: {nivel_dir}")
        df_nivel, realizacoes_com_erro = processar_nivel(pasta_nivel, dss_path, pasta_saida_nivel, max_realizacoes=args.max_realizacoes)
        todos_resultados.append(df_nivel)
        total_realizacoes_com_erro += realizacoes_com_erro

    if todos_resultados:
        df_master = pd.concat(todos_resultados, ignore_index=True)
        df_master.to_csv(os.path.join(pasta_saida, "master_resultados_opendss.csv"), index=False, sep=";", decimal=",")
        
        # Gerar boxplot geral (sem segmentação)
        print(f"\nGerando boxplot geral (sem segmentação)...")
        caminho_plot_geral = plotar_boxplot_geral(df_master, pasta_saida)
        
        # Gerar boxplots por time bands apenas
        print(f"\nGerando boxplots por time bands...")
        caminho_plot_time_bands = plotar_boxplot_por_time_bands(df_master, pasta_saida)
        
        # Gerar boxplots por tipo de dia
        print(f"\nGerando boxplots por tipo de dia...")
        caminhos_plots = plotar_boxplot(df_master, pasta_saida)
        
        # Gerar boxplots por faixa de operação BESS
        print(f"\nGerando boxplots por faixa de operação BESS...")
        caminhos_plots_bess = plotar_boxplot_por_faixa_bess(df_master, pasta_saida)
        
        print(f"\nAnálise finalizada. Master CSV salvo em: {os.path.join(pasta_saida, 'master_resultados_opendss.csv')}")
        print(f"\nBoxplot geral salvo em:")
        print(f"  - {caminho_plot_geral}")
        print(f"\nBoxplot por time bands salvo em:")
        print(f"  - {caminho_plot_time_bands}")
        print(f"\nBoxplots por tipo de dia salvos em:")
        for caminho in caminhos_plots:
            print(f"  - {caminho}")
        print(f"\nBoxplots por faixa de operação BESS salvos em:")
        for caminho in caminhos_plots_bess:
            print(f"  - {caminho}")
        print(f"\n⚠ RESUMO DE ERROS:")
        print(f"  Total de realizações com erro 'Max Control Iterations': {total_realizacoes_com_erro}")
        if total_realizacoes_com_erro > 0:
            print(f"  Detalhes por realização disponíveis na coluna 'horas_com_erro_max_control' do CSV de resultados.")
    else:
        print("Nenhum nível de penetração encontrado em realizacoes_sorteadas.")


if __name__ == "__main__":
    main()
