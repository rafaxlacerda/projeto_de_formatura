import os

import numpy as np
import pandas as pd
from dss._cffi_api_util import DSSException
from opendssdirect import dss

from simulacoes_config import BESS_BANDS, BESS_PERFIL, TIME_BANDS, V_PU_MAX, V_PU_MIN


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

            pior_fase_realizacao = float(np.max(valores))

            if barra not in tensoes_envelope[faixa]:
                tensoes_envelope[faixa][barra] = []
            tensoes_envelope[faixa][barra].append(pior_fase_realizacao)


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


def _linhas_tensoes_completas(pen_pct, resultado, tensoes_por_hora):
    """
    Expande as tensoes brutas em uma tabela longa:
    uma linha por nivel, realizacao, hora e barra, com ate tres fases em colunas.
    """
    linhas = []
    for hora, tensoes_barras in sorted(tensoes_por_hora.items()):
        for barra, tensoes_fases in sorted(tensoes_barras.items()):
            linha = {
                "pen_pct": pen_pct,
                "id_realizacao": resultado["id_realizacao"],
                "tipo_dia": resultado["tipo_dia"],
                "hora": hora,
                "barra": barra,
                "n_fases": len(tensoes_fases),
                "tensao_fase_1_pu": np.nan,
                "tensao_fase_2_pu": np.nan,
                "tensao_fase_3_pu": np.nan,
            }
            for idx, tensao in enumerate(tensoes_fases[:3], start=1):
                linha[f"tensao_fase_{idx}_pu"] = float(tensao)
            linhas.append(linha)
    return linhas


def processar_nivel(pasta_nivel, dss_path, pasta_saida_nivel, max_realizacoes=None):
    resumo = ler_resumo_configuracoes(pasta_nivel)
    perfis_irr = ler_perfis_irradiancia(pasta_nivel)
    fatores_incerteza = ler_fatores_incerteza_carga(pasta_nivel)
    elementos = ler_elementos_opendss(pasta_nivel)

    if resumo.empty:
        raise ValueError(f"Resumo de configurações não encontrado em {pasta_nivel}")

    resultados = []
    tensoes_completas = []
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
        pen_pct = int(os.path.basename(pasta_nivel).split("_")[1].replace("pct", ""))
        resultado["pen_pct"] = pen_pct

        # Extrai tensoes_por_hora para o envelope e remove do dict de resultado antes de salvar
        tensoes_por_hora_real = resultado.pop("_tensoes_por_hora")
        _acumular_tensoes_envelope(tensoes_envelope, tensoes_por_hora_real)
        tensoes_completas.extend(_linhas_tensoes_completas(pen_pct, resultado, tensoes_por_hora_real))

        resultados.append(resultado)
        
        if resultado["horas_com_erro_max_control"] > 0:
            realizacoes_com_erro += 1
            print(f"  ✓ Realização {id_realizacao} - Subtensão: manhã={resultado['subtensao_manhã']} tarde={resultado['subtensao_tarde']} noite={resultado['subtensao_noite']} madrugada={resultado['subtensao_madrugada']} | Sobretensão: manhã={resultado['sobretensao_manhã']} tarde={resultado['sobretensao_tarde']} noite={resultado['sobretensao_noite']} madrugada={resultado['sobretensao_madrugada']} ⚠ {resultado['horas_com_erro_max_control']} hora(s) com erro Max Control Iter")
        else:
            print(f"  ✓ Realização {id_realizacao} - Subtensão: manhã={resultado['subtensao_manhã']} tarde={resultado['subtensao_tarde']} noite={resultado['subtensao_noite']} madrugada={resultado['subtensao_madrugada']} | Sobretensão: manhã={resultado['sobretensao_manhã']} tarde={resultado['sobretensao_tarde']} noite={resultado['sobretensao_noite']} madrugada={resultado['sobretensao_madrugada']}")

    df_resultados = pd.DataFrame(resultados)
    df_tensoes_completas = pd.DataFrame(tensoes_completas)
    os.makedirs(pasta_saida_nivel, exist_ok=True)
    df_resultados.to_csv(os.path.join(pasta_saida_nivel, "resultados_opendss_por_realizacao.csv"), index=False, sep=";", decimal=",")
    df_tensoes_completas.to_csv(
        os.path.join(pasta_saida_nivel, "tensoes_opendss_completas.csv"),
        index=False,
        sep=";",
        decimal=",",
    )

    # Salva os dados do envelope de tensão em CSV para uso posterior na plotagem
    _salvar_envelope_csv(tensoes_envelope, pasta_saida_nivel)

    return df_resultados, realizacoes_com_erro, df_tensoes_completas
