import argparse
import os
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

# Ordem topológica das barras do alimentador IEEE34, da subestação até as extremidades.
# Corresponde a uma travessia em largura (BFS) a partir da barra de fonte 800,
# mantendo a sequência física do alimentador conforme IEEE PES Distribution Test Feeders.
BARRAS_TOPOLOGICAS_IEEE34 = [
    800, 802, 806, 808, 810, 812, 814, 850,
    816, 818, 820, 822, 824, 826, 828, 830,
    854, 832, 858, 834, 860, 836, 862, 838,
    842, 844, 846, 848, 852, 856, 888, 890,
]

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

# Constantes para análise de estatísticas descritivas
NIVEIS_ESPECIAIS_ANALISE = [0, 50, 100, 150]  # % de penetração FV
BARRAS_TRIFASICAS_ANALISE = [860, 840, 844, 848, 890]

# controle intermediario: ativa ou desativa carregamento - etapa 2

def ler_fatores_incerteza_carga(pasta_nivel):
    caminho = os.path.join(pasta_nivel, "03_fatores_incerteza_carga.csv")
    df = pd.read_csv(caminho, sep=";", decimal=",")
    df = df.sort_values(["id_realizacao", "barra"])
    return df


def ler_perfis_irradiancia(pasta_nivel):
    caminho = os.path.join(pasta_nivel, "02_perfis_irradiancia.csv")
    df = pd.read_csv(caminho, sep=";", decimal=",")
    df["id_realizacao"] = pd.to_numeric(df["id_realizacao"], errors="coerce").astype("Int64")
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
        dados[bus.lower()] = list(mags)  # Retorna lista das tensões das fases
    return dados


def agrupar_horas_por_faixa(horas):
    resultado = {}
    for faixa, horas_faixa in TIME_BANDS.items():
        resultado[faixa] = [h for h in horas_faixa if h in horas]
    return resultado


def calcular_defeitos_por_faixa(tensoes_por_hora):
    """
    Calcula subtensão e sobretensão por faixa horária.
    Subtensão: alguma fase < 0.95 pu
    Sobretensão: alguma fase > 1.05 pu
    Retorna dict com keys: subtensao_{faixa}, sobretensao_{faixa}
    """
    defeitos = {}
    for faixa, horas in TIME_BANDS.items():
        horas_disponiveis = [h for h in horas if h in tensoes_por_hora]
        
        if not horas_disponiveis:
            defeitos[f"subtensao_{faixa}"] = 0
            defeitos[f"sobretensao_{faixa}"] = 0
            continue
        
        # Coleta todas as barras com subtensão ou sobretensão nesta faixa
        barras_com_subtensao = set()
        barras_com_sobretensao = set()
        
        for barra in next(iter(tensoes_por_hora.values())).keys():
            tensoes_fases = []
            for h in horas_disponiveis:
                tensoes_fases.extend(tensoes_por_hora[h][barra])
            
            # Se alguma fase está em subtensão
            if any(t < V_PU_MIN for t in tensoes_fases):
                barras_com_subtensao.add(barra)
            
            # Se alguma fase está em sobretensão
            if any(t > V_PU_MAX for t in tensoes_fases):
                barras_com_sobretensao.add(barra)
        
        defeitos[f"subtensao_{faixa}"] = len(barras_com_subtensao)
        defeitos[f"sobretensao_{faixa}"] = len(barras_com_sobretensao)
    
    return defeitos


def calcular_defeitos_por_faixa_bess(tensoes_por_hora):
    """
    Calcula subtensão e sobretensão para as faixas de operação da BESS.
    Subtensão: alguma fase < 0.95 pu
    Sobretensão: alguma fase > 1.05 pu
    Retorna dict com keys: subtensao_{faixa}, sobretensao_{faixa}
    """
    defeitos = {}
    for faixa, horas in BESS_BANDS.items():
        horas_disponiveis = [h for h in horas if h in tensoes_por_hora]
        
        if not horas_disponiveis:
            defeitos[f"subtensao_{faixa}"] = 0
            defeitos[f"sobretensao_{faixa}"] = 0
            continue
        
        # Coleta todas as barras com subtensão ou sobretensão nesta faixa
        barras_com_subtensao = set()
        barras_com_sobretensao = set()
        
        for barra in next(iter(tensoes_por_hora.values())).keys():
            tensoes_fases = []
            for h in horas_disponiveis:
                tensoes_fases.extend(tensoes_por_hora[h][barra])
            
            # Se alguma fase está em subtensão
            if any(t < V_PU_MIN for t in tensoes_fases):
                barras_com_subtensao.add(barra)
            
            # Se alguma fase está em sobretensão
            if any(t > V_PU_MAX for t in tensoes_fases):
                barras_com_sobretensao.add(barra)
        
        defeitos[f"subtensao_{faixa}"] = len(barras_com_subtensao)
        defeitos[f"sobretensao_{faixa}"] = len(barras_com_sobretensao)
    
    return defeitos

def simular_realizacao(realizacao_id, row_info, pv_df, bess_df, perfis_irr, fatores_incerteza, cargas_base):
    """
    Simula uma realização específica do Monte Carlo.
    
    Parameters:
    -----------
    realizacao_id : int
        ID da realização
    row_info : pd.Series
        Série com informações da realização (tipo_dia, pv_unidades, etc.)
    pv_df : pd.DataFrame
        DataFrame com dados de PV filtrado para esta realização
    bess_df : pd.DataFrame
        DataFrame com dados de BESS filtrado para esta realização
    perfis_irr : pd.DataFrame
        DataFrame com perfis de irradiância
    fatores_incerteza : pd.DataFrame
        DataFrame com fatores de incerteza de carga
    cargas_base : dict
        Dicionário com cargas base da rede
    """
    # Buscar perfis de irradiância da realização
    perfis_irr_realizacao = perfis_irr.loc[perfis_irr["id_realizacao"] == realizacao_id]
    if perfis_irr_realizacao.empty:
        raise ValueError(
            f"Perfis de irradiância não encontrados para id_realizacao={realizacao_id}. "
            f"IDs disponíveis: {sorted(perfis_irr['id_realizacao'].unique())}"
        )
    radiancias = perfis_irr_realizacao.iloc[0]
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
        "_tensoes_por_hora": tensoes_por_hora,  # dados brutos para envelope de tensão
        "tipo_dia": row_info["tipo_dia"],
        "pv_unidades": int(row_info["pv_unidades"]),
        "pv_potencia_total_kw": float(row_info["pv_potencia_total_kw"]),
        "bess_unidades": int(row_info["bess_unidades"]),
        "bess_potencia_total_kw": float(bess_real["potencia_kw"].sum()) if not bess_real.empty else 0.0,
        "subtensao_manhã": defeitos["subtensao_manhã"],
        "sobretensao_manhã": defeitos["sobretensao_manhã"],
        "subtensao_tarde": defeitos["subtensao_tarde"],
        "sobretensao_tarde": defeitos["sobretensao_tarde"],
        "subtensao_noite": defeitos["subtensao_noite"],
        "sobretensao_noite": defeitos["sobretensao_noite"],
        "subtensao_madrugada": defeitos["subtensao_madrugada"],
        "sobretensao_madrugada": defeitos["sobretensao_madrugada"],
        "subtensao_bess_carga": defeitos_bess["subtensao_carga"],
        "sobretensao_bess_carga": defeitos_bess["sobretensao_carga"],
        "subtensao_bess_pós_carga_pré_descarga": defeitos_bess["subtensao_pós_carga_pré_descarga"],
        "sobretensao_bess_pós_carga_pré_descarga": defeitos_bess["sobretensao_pós_carga_pré_descarga"],
        "subtensao_bess_descarga": defeitos_bess["subtensao_descarga"],
        "sobretensao_bess_descarga": defeitos_bess["sobretensao_descarga"],
        "subtensao_bess_fora_de_operacao": defeitos_bess["subtensao_fora_de_operacao"],
        "sobretensao_bess_fora_de_operacao": defeitos_bess["sobretensao_fora_de_operacao"],
        "horas_com_erro_max_control": horas_com_erro,
    }


def _acumular_tensoes_envelope(tensoes_envelope, tensoes_por_hora):
    """
    Acumula, para cada faixa horária e cada barra, a tensão média (sobre as fases)
    de cada realização no dicionário tensoes_envelope.

    A estrutura de tensoes_envelope é:
        {faixa: {nome_barra: [media_fase_realizacao_1, media_fase_realizacao_2, ...]}}
    """
    for faixa, horas_faixa in TIME_BANDS.items():
        horas_disponiveis = [h for h in horas_faixa if h in tensoes_por_hora]
        if not horas_disponiveis:
            continue

        # Conjunto de barras disponíveis nesta realização
        barras = set()
        for h in horas_disponiveis:
            barras.update(tensoes_por_hora[h].keys())

        for barra in barras:
            # Coleta todas as magnitudes de fase dessa barra nas horas da faixa
            valores = []
            for h in horas_disponiveis:
                if barra in tensoes_por_hora[h]:
                    valores.extend(tensoes_por_hora[h][barra])

            if not valores:
                continue

            media_realizacao = float(np.mean(valores))

            if barra not in tensoes_envelope[faixa]:
                tensoes_envelope[faixa][barra] = []
            tensoes_envelope[faixa][barra].append(media_realizacao)


def _salvar_envelope_csv(tensoes_envelope, pasta_saida_nivel):
    """
    Persiste os dados de envelope de tensão em um CSV por faixa horária.
    Cada linha corresponde a uma realização; colunas são as barras.
    Arquivo salvo como: envelope_tensao_{faixa}.csv
    """
    for faixa, dados_barras in tensoes_envelope.items():
        if not dados_barras:
            continue
        df_env = pd.DataFrame(dados_barras)
        caminho = os.path.join(pasta_saida_nivel, f"envelope_tensao_{faixa}.csv")
        df_env.to_csv(caminho, index=False, sep=";", decimal=",")


def plotar_envelope_tensao(pasta_saida, pasta_montecarlo):
    """
    Gera figuras de envelope de tensão (perfil ao longo das barras do alimentador)
    para cada faixa horária, sobrepondo os diferentes níveis de penetração.

    Para cada combinação (nível de penetração × faixa horária), a função:
      - ordena as barras conforme BARRAS_TOPOLOGICAS_IEEE34;
      - calcula, sobre as 500 realizações, a mediana e os percentis 5 e 95 da
        tensão em cada barra;
      - traça o envelope sombreado (P5–P95) e a linha de mediana;
      - marca com linhas tracejadas os limites regulatórios 0,95 e 1,05 pu.

    Uma figura separada é gerada por faixa horária; cada nível de penetração
    corresponde a uma cor distinta, permitindo comparação visual direta.
    """
    faixas = list(TIME_BANDS.keys())

    # Paleta de cores para os níveis de penetração (até 16 níveis: 0–150% em passos de 10%)
    cmap = plt.get_cmap("plasma")
    # Coleta todos os níveis disponíveis examinando as pastas de saída
    niveis_disponiveis = sorted([
        int(d.split("_")[1].replace("pct", ""))
        for d in os.listdir(pasta_saida)
        if d.startswith("pen_") and os.path.isdir(os.path.join(pasta_saida, d))
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

        for nivel in niveis_disponiveis:
            nivel_dir = f"pen_{nivel:03d}pct"
            caminho_csv = os.path.join(pasta_saida, nivel_dir, f"envelope_tensao_{faixa}.csv")
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
            f"Envelope de tensão ao longo do alimentador IEEE34 — Faixa horária: {faixa.capitalize()}\n"
            f"Mediana e intervalo P5–P95 sobre {n_niveis} níveis de penetração FV",
            fontsize=12,
        )
        ax.set_ylim(0.88, 1.10)
        ax.grid(True, linestyle="--", alpha=0.35)

        # Legenda: níveis de penetração + limites regulatórios
        legenda_niveis = ax.legend(
            handles=handles_legenda + [lim_min, lim_max],
            loc="lower left",
            fontsize=8,
            ncol=max(1, n_niveis // 4 + 1),
            title="Penetração FV / Limites",
            title_fontsize=8,
        )
        ax.add_artist(legenda_niveis)

        fig.tight_layout()
        caminho_fig = os.path.join(pasta_saida, f"envelope_tensao_{faixa}.png")
        fig.savefig(caminho_fig, dpi=200)
        plt.close(fig)
        print(f"  ✓ Envelope de tensão salvo: {caminho_fig}")


def processar_nivel(pasta_nivel, dss_path, pasta_saida_nivel, max_realizacoes=None):
    resumo = ler_resumo_configuracoes(pasta_nivel)
    perfis_irr = ler_perfis_irradiancia(pasta_nivel)
    fatores_incerteza = ler_fatores_incerteza_carga(pasta_nivel)
    elementos = ler_elementos_opendss(pasta_nivel)

    if resumo.empty:
        raise ValueError(f"Resumo de configurações não encontrado em {pasta_nivel}")

    resultados = []
    realizacoes_com_erro = 0
    # Estrutura para envelope de tensão: {faixa: {barra: [tensoes_medias_por_realizacao]}}
    tensoes_envelope = {faixa: {} for faixa in TIME_BANDS}

    for _, row in resumo.head(max_realizacoes).iterrows():
        id_realizacao = int(row["id_realizacao"])

        redirecionar_circuito(dss_path)
        cargas_base = carregar_cargas_base()

        resultado = simular_realizacao(
            id_realizacao,
            row,
            elementos[elementos["id_realizacao"] == id_realizacao],
            elementos[(elementos["id_realizacao"] == id_realizacao) & (elementos["classe"] == "Storage")],
            perfis_irr,
            fatores_incerteza,
            cargas_base,
        )
        resultado["pen_pct"] = int(os.path.basename(pasta_nivel).split("_")[1].replace("pct", ""))

        # Extrai tensoes_por_hora para o envelope e remove do dict de resultado antes de salvar
        tensoes_por_hora_real = resultado.pop("_tensoes_por_hora")
        _acumular_tensoes_envelope(tensoes_envelope, tensoes_por_hora_real)

        resultados.append(resultado)
        
        if resultado["horas_com_erro_max_control"] > 0:
            realizacoes_com_erro += 1
            print(f"  ✓ Realização {id_realizacao} - Subtensão: manhã={resultado['subtensao_manhã']} tarde={resultado['subtensao_tarde']} noite={resultado['subtensao_noite']} madrugada={resultado['subtensao_madrugada']} | Sobretensão: manhã={resultado['sobretensao_manhã']} tarde={resultado['sobretensao_tarde']} noite={resultado['sobretensao_noite']} madrugada={resultado['sobretensao_madrugada']} ⚠ {resultado['horas_com_erro_max_control']} hora(s) com erro Max Control Iter")
        else:
            print(f"  ✓ Realização {id_realizacao} - Subtensão: manhã={resultado['subtensao_manhã']} tarde={resultado['subtensao_tarde']} noite={resultado['subtensao_noite']} madrugada={resultado['subtensao_madrugada']} | Sobretensão: manhã={resultado['sobretensao_manhã']} tarde={resultado['sobretensao_tarde']} noite={resultado['sobretensao_noite']} madrugada={resultado['sobretensao_madrugada']}")

    df_resultados = pd.DataFrame(resultados)
    os.makedirs(pasta_saida_nivel, exist_ok=True)
    df_resultados.to_csv(os.path.join(pasta_saida_nivel, "resultados_opendss_por_realizacao.csv"), index=False, sep=";", decimal=",")

    # Salva os dados do envelope de tensão em CSV para uso posterior na plotagem
    _salvar_envelope_csv(tensoes_envelope, pasta_saida_nivel)

    return df_resultados, realizacoes_com_erro


def plotar_boxplot_geral(df_master, pasta_saida):
    """Gera boxplots gerais separados para subtensão e sobretensão"""
    niveis = sorted(df_master["pen_pct"].unique())
    faixas = ["manhã", "tarde", "noite"] #tirei madrugada pois irrelevante
    
    # Boxplot para Subtensão
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), constrained_layout=True)
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
        ax.set_ylim(0, 35)

    fig.suptitle("Subtensão por faixa de horário (agregado de todos os tipos de dia)", fontsize=18)
    caminho_subtensao = os.path.join(pasta_saida, "boxplot_subtensao_geral.png")
    fig.savefig(caminho_subtensao, dpi=200)
    plt.close(fig)
    
    # Boxplot para Sobretensão
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), constrained_layout=True)
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
        ax.set_ylim(0, 35)

    fig.suptitle("Sobretensão por faixa de horário (agregado de todos os tipos de dia)", fontsize=18)
    caminho_sobretensao = os.path.join(pasta_saida, "boxplot_sobretensao_geral.png")
    fig.savefig(caminho_sobretensao, dpi=200)
    plt.close(fig)
    
    return caminho_subtensao, caminho_sobretensao


def plotar_boxplot_por_time_bands(df_master, pasta_saida):
    """Gera boxplots separados por time bands, com distinção entre subtensão e sobretensão"""
    niveis = sorted(df_master["pen_pct"].unique())
    faixas = ["manhã", "tarde", "noite"] #tirei madrugada pois irrelevante
    
    # Boxplot para Subtensão
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), constrained_layout=True)
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
        ax.set_ylim(0, 35)

    fig.suptitle("Subtensão por faixa de horário (agregado de todos os tipos de dia)", fontsize=18)
    caminho_subtensao = os.path.join(pasta_saida, "boxplot_subtensao_por_time_bands.png")
    fig.savefig(caminho_subtensao, dpi=200)
    plt.close(fig)
    
    # Boxplot para Sobretensão
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), constrained_layout=True)
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
        ax.set_ylim(0, 35)

    fig.suptitle("Sobretensão por faixa de horário (agregado de todos os tipos de dia)", fontsize=18)
    caminho_sobretensao = os.path.join(pasta_saida, "boxplot_sobretensao_por_time_bands.png")
    fig.savefig(caminho_sobretensao, dpi=200)
    plt.close(fig)
    
    return caminho_subtensao, caminho_sobretensao


def plotar_boxplot(df_master, pasta_saida):
    """Gera boxplots separados por tipo de dia e por tipo de defeito (subtensão/sobretensão)"""
    niveis = sorted(df_master["pen_pct"].unique())
    tipos_dia = sorted(df_master["tipo_dia"].unique())
    faixas = ["manhã", "tarde", "noite"] #tirei madrugada pois irrelevante
    
    caminhos_imagens = []
    
    # Gerar um gráfico para cada tipo de dia
    for tipo_dia in tipos_dia:
        df_filtrado = df_master[df_master["tipo_dia"] == tipo_dia]
        
        # Gráfico para Subtensão
        fig, axes = plt.subplots(4, 1, figsize=(16, 20), constrained_layout=True)
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
            ax.set_ylim(0, 35)

        fig.suptitle(f"Subtensão por faixa de horário - Tipo de dia: {tipo_dia}", fontsize=18)
        caminho_imagem = os.path.join(pasta_saida, f"boxplot_subtensao_por_faixa_{tipo_dia}.png")
        fig.savefig(caminho_imagem, dpi=200)
        plt.close(fig)
        caminhos_imagens.append(caminho_imagem)
        
        # Gráfico para Sobretensão
        fig, axes = plt.subplots(4, 1, figsize=(16, 20), constrained_layout=True)
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
            ax.set_ylim(0, 35)

        fig.suptitle(f"Sobretensão por faixa de horário - Tipo de dia: {tipo_dia}", fontsize=18)
        caminho_imagem = os.path.join(pasta_saida, f"boxplot_sobretensao_por_faixa_{tipo_dia}.png")
        fig.savefig(caminho_imagem, dpi=200)
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
            ax.set_ylim(0, 35)

        fig.suptitle(f"Subtensão por faixa de operação BESS - Tipo de dia: {tipo_dia}", fontsize=18)
        caminho_imagem = os.path.join(pasta_saida, f"boxplot_subtensao_por_faixa_bess_{tipo_dia}.png")
        fig.savefig(caminho_imagem, dpi=200)
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
            ax.set_ylim(0, 35)

        fig.suptitle(f"Sobretensão por faixa de operação BESS - Tipo de dia: {tipo_dia}", fontsize=18)
        caminho_imagem = os.path.join(pasta_saida, f"boxplot_sobretensao_por_faixa_bess_{tipo_dia}.png")
        fig.savefig(caminho_imagem, dpi=200)
        plt.close(fig)
        caminhos_imagens.append(caminho_imagem)
    
    return caminhos_imagens


# =============================================================================
# FUNÇÕES DE ANÁLISE DE ESTATÍSTICAS DESCRITIVAS
# =============================================================================

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

def criar_tabela_html(tabelas_por_nivel):
    """
    Cria uma tabela em HTML com estatísticas dos 4 níveis especiais.
    Retorna string com código HTML.
    """
    html = []
    html.append("""<html>
<head>
    <meta charset="utf-8">
    <title>Análise Estatística - Cenários Monte Carlo</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; margin: 20px 0; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: right; }
        th { background-color: #4CAF50; color: white; }
        td:first-child { text-align: left; }
        h1, h2 { color: #333; }
    </style>
</head>
<body>
    <h1>Análise Estatística Descritiva - Cenários Monte Carlo</h1>
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

def gerar_tabelas_estatisticas(pasta_saida, pasta_monte_carlo):
    """
    Gera tabelas de estatísticas descritivas para os níveis especiais.
    Salva em formatos CSV e HTML.
    """
    print(f"\n{'=' * 80}")
    print("  GERANDO TABELAS DE ESTATÍSTICAS DESCRITIVAS")
    print(f"{'=' * 80}")
    print(f"  Níveis especiais de análise: {NIVEIS_ESPECIAIS_ANALISE}")
    print(f"  Barras trifásicas: {BARRAS_TRIFASICAS_ANALISE}")
    print(f"{'=' * 80}")
    
    # Compila estatísticas para cada nível
    tabelas_por_nivel = {}
    for nivel in NIVEIS_ESPECIAIS_ANALISE:
        print(f"\n📊 Processando nível {nivel}%...", end=" ")
        tabelas_por_nivel[nivel] = compilar_tabelas_por_nivel(nivel, pasta_monte_carlo, pasta_saida)
        if tabelas_por_nivel[nivel]:
            print("✓")
        else:
            print("❌ (pasta não encontrada)")
    
    # Salva tabela CSV
    csv_content = criar_tabela_csv(tabelas_por_nivel)
    csv_path = os.path.join(pasta_saida, "tabela_estatisticas_descritivas.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_content)
    print(f"\n✓ Tabela CSV salva: {csv_path}")
    
    # Salva tabela HTML
    html_content = criar_tabela_html(tabelas_por_nivel)
    html_path = os.path.join(pasta_saida, "tabela_estatisticas_descritivas.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✓ Tabela HTML salva: {html_path}")
    
    print(f"\n{'=' * 80}")


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
        print("\nGerando boxplot geral (sem segmentação)...")
        plotar_boxplot_geral(df_master, pasta_saida)
        
        # Gerar boxplots por time bands apenas
        print("\nGerando boxplots por time bands...")
        plotar_boxplot_por_time_bands(df_master, pasta_saida)
        
        # Gerar boxplots por tipo de dia
        print("\nGerando boxplots por tipo de dia...")
        plotar_boxplot(df_master, pasta_saida)
        
        # Gerar boxplots por faixa de operação BESS
        print("\nGerando boxplots por faixa de operação BESS...")
        plotar_boxplot_por_faixa_bess(df_master, pasta_saida)
        
        # Gerar envelope de tensão ao longo das barras do alimentador
        print("\nGerando envelopes de tensão por faixa horária...")
        plotar_envelope_tensao(pasta_saida, pasta_montecarlo)

        # Gerar tabelas de estatísticas descritivas
        pasta_monte_carlo = os.path.join(base_dir, "resultados_monte_carlo")
        gerar_tabelas_estatisticas(pasta_saida, pasta_monte_carlo)
        
        print(f"\nAnálise finalizada. Master CSV salvo em: {os.path.join(pasta_saida, 'master_resultados_opendss.csv')}")
        print("\nBoxplots salvos")
        
        print("\n⚠ RESUMO DE ERROS:")
        print(f"  Total de realizações com erro 'Max Control Iterations': {total_realizacoes_com_erro}")
        if total_realizacoes_com_erro > 0:
            print("  Detalhes por realização disponíveis na coluna 'horas_com_erro_max_control' do CSV de resultados.")
    else:
        print("Nenhum nível de penetração encontrado em realizacoes_sorteadas.")


if __name__ == "__main__":
    main()