"""
Controle Secundário Distribuído por Consenso para BESS
=======================================================
Implementa um algoritmo de consenso distribuído sobre os sistemas de
armazenamento de energia em baterias (BESS) do alimentador IEEE 34 bus,
operando como camada secundária sobre o controle primário VoltVar
(ABNT NBR 16149:2013) já implementado em controle_primario.py.

Arquitetura de controle
-----------------------
O controle primário atua localmente em cada DER (PV e BESS) por meio do
ajuste de potência reativa Q em função da tensão na barra (curva VoltVar).
O controle secundário atua exclusivamente sobre a potência ativa P dos BESS,
adicionando um sinal de correção ΔP calculado pelo algoritmo de consenso.
Como os dois controles operam em variáveis ortogonais do inversor (Q e P,
respectivamente), não há conflito de atuação direta entre as camadas.

Para manter o fator de potência do controle primário, o sinal de correção
ΔP do secundário é acompanhado de um ΔQ proporcional, calculado de forma
a preservar o ângulo de potência aparente vigente em cada BESS após o
primário. Quando o primário não está ativo, o BESS opera em fator de
potência unitário e o secundário também não impõe reativo adicional.

Topologia de comunicação
------------------------
A rede de comunicação entre os BESS é modelada como um grafo esparso não
dirigido G = (V, E), em que cada nó representa um BESS e cada aresta
representa um link de comunicação entre dois BESS vizinhos. A vizinhança é
definida pela proximidade elétrica no alimentador IEEE 34: dois BESS se
comunicam caso não haja outro BESS intermediário no caminho elétrico entre
eles. A matriz Laplaciana L = D - A, onde D é a matriz de grau e A é a
matriz de adjacência ponderada, governa a dinâmica do algoritmo.

Algoritmo de consenso
---------------------
A variável de consenso adotada é a potência ativa normalizada de cada BESS
em relação à sua potência nominal:

    x_i(t) = P_i(t) / P_nom_i

O sinal de correção de potência ativa para o BESS i é dado por:

    ΔP_i = -α * P_nom_i * Σ_{j ∈ N_i} w_ij * (x_i - x_j)

onde α é o ganho do consenso, w_ij é o peso do link (ij) e N_i é o
conjunto de vizinhos de i no grafo de comunicação. Esse sinal tende a
igualar as potências normalizadas de todos os BESS, distribuindo a carga
de forma proporcional às suas capacidades nominais.

Modos de operação
-----------------
Dois modos são disponibilizados, selecionáveis via argumento --modo-secundario:

  sempre_ativo   — O consenso é aplicado a cada hora do dia,
                   independentemente de haver violações de tensão. O sinal
                   ΔP é somado à referência de potência do BESS antes do
                   fluxo de potência ser resolvido no OpenDSS.

  condicional    — O consenso é ativado apenas nas horas em que o controle
                   primário (ou o fluxo de potência base) deixou barras com
                   subtensão ou sobretensão. Nas horas sem violação, os BESS
                   operam exatamente com o perfil de referência.
                   Este modo requer que o controle primário esteja ativo.

Referências
-----------
Parada Contzen, R. et al. A distributed double-layer control algorithm for
medium-voltage regulation. IEEE Transactions on Power Systems, 2024.

Zhang, Y. et al. Distributed step-by-step finite-time consensus design for
battery energy storage devices with droop control. CSEE Journal of Power
and Energy Systems, v. 9, n. 5, p. 1893-1903, 2023.
"""

import os

import numpy as np
import pandas as pd
from dss._cffi_api_util import DSSException
from opendssdirect import dss

from simulacoes_config import BESS_BANDS, BESS_PERFIL, TIME_BANDS, V_PU_MAX, V_PU_MIN
from simulacao_opendss import (
    ler_fatores_incerteza_carga,
    ler_perfis_irradiancia,
    ler_resumo_configuracoes,
    ler_elementos_opendss,
    redirecionar_circuito,
    carregar_cargas_base,
    criar_elementos_simulacao,
    editar_load,
    extrair_tensoes_por_barra,
    obter_multiplicadores_shapes,
    calcular_defeitos_por_faixa,
    calcular_defeitos_por_faixa_bess,
    _acumular_tensoes_envelope,
    _linhas_tensoes_completas,
    _salvar_envelope_csv,
)
from controle_primario import (
    _calcular_q_refs,
    volt_var_curve,
    MAX_VOLTVAR_ITER,
    Q_MAX_PU,
)

# ---------------------------------------------------------------------------
# Parâmetros do algoritmo de consenso
# ---------------------------------------------------------------------------

# Ganho do consenso α. Controla a velocidade de redistribuição da potência
# entre os BESS. Valores excessivamente altos podem tornar a dinâmica
# instável; valores muito baixos resultam em convergência lenta.
# Calibrado empiricamente para o IEEE 34 bus com 5 nós BESS.
CONSENSO_ALPHA = 0.15

# Número máximo de iterações do algoritmo de consenso por hora simulada.
# Como a simulação é quasi-estática (resolução horária), cada "iteração"
# representa uma troca de informação entre vizinhos seguida de um novo
# fluxo de potência no OpenDSS.
CONSENSO_MAX_ITER = 5

# Tolerância de convergência: o consenso é encerrado antecipadamente se o
# desvio máximo entre potências normalizadas dos BESS for inferior a este valor.
CONSENSO_TOL = 0.01

# Barras trifásicas do IEEE 34 bus onde os BESS podem ser alocados.
# Correspondem às barras com cargas trifásicas no modelo original.
BESS_BARRAS_TRIFASICAS = [860, 840, 844, 848, 890]

# ---------------------------------------------------------------------------
# Grafo de comunicação entre BESS (adjacência elétrica no IEEE 34 bus)
# ---------------------------------------------------------------------------
# A topologia é definida com base na ordem topológica do alimentador
# (BARRAS_TOPOLOGICAS_IEEE34 em simulacoes_config.py). Dois BESS são
# considerados vizinhos se não há outro BESS no caminho elétrico entre eles.
# Para as barras [860, 840, 844, 848, 890], a adjacência elétrica é:
#
#   860 — 840  (caminho: 860→836→840)
#   840 — 844  (caminho: 840→842→844)
#   844 — 848  (caminho: 844→846→848)
#   840 — 848  (caminho: 840→852→856→... via outro ramal — não adjacente direto)
#   848 — 890  (caminho: 848→888→890)
#
# O grafo resultante é uma cadeia linear: 860-840-844-848-890, que é conexo
# e portanto garante convergência do algoritmo de consenso (λ₂(L) > 0).
#
# Pesos unitários são adotados (w_ij = 1) por simplicidade; pesos
# proporcionais ao inverso da impedância da linha poderiam ser usados
# para refletir a influência elétrica relativa de cada vizinho.

def construir_grafo_comunicacao(barras_bess: list) -> np.ndarray:
    """
    Constrói a matriz Laplaciana do grafo de comunicação entre os BESS
    presentes em uma realização específica.

    A topologia de comunicação é determinada pela adjacência elétrica no
    alimentador IEEE 34 bus: dois BESS são vizinhos se nenhum outro BESS
    se interpõe no caminho elétrico entre eles. A ordem topológica das
    barras BESS é [860, 840, 844, 848, 890], formando uma cadeia linear.

    Parameters
    ----------
    barras_bess : list of int
        Barras onde há BESS alocados na realização atual. Pode ser um
        subconjunto das 5 barras trifásicas possíveis.

    Returns
    -------
    L : np.ndarray, shape (n, n)
        Matriz Laplaciana do grafo de comunicação, onde n = len(barras_bess).
        L = D - A, com A sendo a matriz de adjacência e D a matriz de grau.
    barras_ordenadas : list of int
        Barras BESS ordenadas conforme a ordem topológica do alimentador,
        correspondendo às linhas/colunas de L.
    """
    # Ordem topológica das barras BESS no alimentador IEEE 34
    ordem_topologica = [b for b in BESS_BARRAS_TRIFASICAS if b in barras_bess]
    n = len(ordem_topologica)

    if n < 2:
        # Com menos de 2 BESS, não há grafo de comunicação
        return np.zeros((n, n)), ordem_topologica

    # Monta matriz de adjacência: cadeia linear (cada nó conectado ao próximo)
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0

    # Matriz Laplaciana: L = D - A
    D = np.diag(A.sum(axis=1))
    L = D - A

    return L, ordem_topologica


def calcular_delta_p_consenso(
    kw_bess_atual: dict,
    barras_bess: list,
    potencias_nominais: dict,
    L: np.ndarray,
    barras_ordenadas: list,
) -> dict:
    """
    Calcula o sinal de correção de potência ativa ΔP para cada BESS com
    base no algoritmo de consenso distribuído.

    A variável de consenso é a potência ativa normalizada pela potência
    nominal de cada BESS: x_i = P_i / P_nom_i. O sinal de correção é:

        ΔP_i = -α * P_nom_i * [L * x]_i

    onde [L * x]_i é a i-ésima componente do produto Laplaciano-vetor,
    correspondente à diferença ponderada entre x_i e as potências
    normalizadas de seus vizinhos.

    Parameters
    ----------
    kw_bess_atual : dict
        {nome_load: (kw_atual, barra_int)} — potências ativas vigentes dos BESS.
    barras_bess : list of int
        Barras onde há BESS alocados.
    potencias_nominais : dict
        {barra_int: potencia_nominal_kw} — potência nominal de cada BESS.
    L : np.ndarray
        Matriz Laplaciana do grafo de comunicação.
    barras_ordenadas : list of int
        Ordem das barras correspondendo às linhas/colunas de L.

    Returns
    -------
    delta_p : dict
        {nome_load: delta_p_kw} — correção de potência ativa para cada BESS.
    """
    n = len(barras_ordenadas)
    if n < 2:
        return {nome: 0.0 for nome in kw_bess_atual}

    # Mapeia barra → nome_load para recuperar o kw atual de cada nó
    barra_para_nome = {barra: nome for nome, (_, barra) in kw_bess_atual.items()}

    # Vetor de potências normalizadas x_i = P_i / P_nom_i
    x = np.zeros(n)
    for i, barra in enumerate(barras_ordenadas):
        nome = barra_para_nome.get(barra)
        if nome is None:
            continue
        kw_atual, _ = kw_bess_atual[nome]
        p_nom = potencias_nominais.get(barra, 1.0)
        # Evita divisão por zero quando o BESS está inativo (BESS_PERFIL = 0)
        x[i] = kw_atual / p_nom if abs(p_nom) > 1e-6 else 0.0

    # Produto Laplaciano: erro de consenso para cada nó
    Lx = L @ x

    # Sinal de correção para cada nome_load
    delta_p = {}
    for nome, (kw_atual, barra) in kw_bess_atual.items():
        if barra not in barras_ordenadas:
            delta_p[nome] = 0.0
            continue
        i = barras_ordenadas.index(barra)
        p_nom = potencias_nominais.get(barra, 1.0)
        delta_p[nome] = -CONSENSO_ALPHA * p_nom * Lx[i]

    return delta_p


def _calcular_q_para_delta_p(
    delta_p: dict,
    kw_bess_apos_primario: dict,
    q_refs_bess_primario: dict,
) -> dict:
    """
    Calcula a variação de potência reativa ΔQ associada ao sinal de
    correção ΔP do consenso, de forma a preservar o fator de potência
    resultante do controle primário.

    Para cada BESS i, o fator de potência após o primário é:

        FP_i = P_i / √(P_i² + Q_i²)

    Com a correção ΔP do secundário, a nova potência ativa é P_i + ΔP_i.
    O ΔQ é calculado para manter o mesmo ângulo φ_i = arctan(Q_i / P_i):

        ΔQ_i = tan(φ_i) * ΔP_i  = (Q_i / P_i) * ΔP_i

    Quando o primário não está ativo, Q_i = 0 e, portanto, ΔQ_i = 0,
    mantendo o fator de potência unitário do BESS.

    Parameters
    ----------
    delta_p : dict
        {nome_load: delta_p_kw}
    kw_bess_apos_primario : dict
        {nome_load: (kw, barra)} — potências ativas após o primário.
    q_refs_bess_primario : dict
        {nome_load: q_ref_kvar} — potências reativas impostas pelo primário.
        Passa-se um dict vazio {} quando o primário não está ativo.

    Returns
    -------
    delta_q : dict
        {nome_load: delta_q_kvar}
    """
    delta_q = {}
    for nome, dp in delta_p.items():
        kw, _ = kw_bess_apos_primario[nome]
        q_ref = q_refs_bess_primario.get(nome, 0.0)
        if abs(kw) > 1e-6:
            tan_phi = q_ref / kw
        else:
            tan_phi = 0.0
        delta_q[nome] = tan_phi * dp
    return delta_q


def _barra_tem_violacao(tensoes: dict, barra_str: str) -> bool:
    """Retorna True se alguma fase da barra está fora da faixa [V_PU_MIN, V_PU_MAX]."""
    fases = tensoes.get(barra_str, [])
    return any(v < V_PU_MIN or v > V_PU_MAX for v in fases)


def _sistema_tem_violacao(tensoes: dict) -> bool:
    """Retorna True se há pelo menos uma barra com violação de tensão."""
    return any(_barra_tem_violacao(tensoes, b) for b in tensoes)


# ---------------------------------------------------------------------------
# Núcleo da simulação por realização
# ---------------------------------------------------------------------------

def simular_realizacao_controle_secundario(
    realizacao_id,
    row_info,
    pv_df,
    bess_df,
    perfis_irr,
    fatores_incerteza,
    cargas_base,
    com_primario: bool,
    modo: str,
):
    """
    Simula uma realização do Monte Carlo com controle secundário por consenso.

    Parameters
    ----------
    realizacao_id : int
        Identificador da realização.
    row_info : pd.Series
        Metadados da realização (tipo_dia, pv_unidades, etc.).
    pv_df : pd.DataFrame
        Dados dos sistemas fotovoltaicos filtrados para esta realização.
    bess_df : pd.DataFrame
        Dados dos BESS filtrados para esta realização.
    perfis_irr : pd.DataFrame
        Perfis de irradiância horária por realização.
    fatores_incerteza : pd.DataFrame
        Fatores multiplicativos de incerteza de carga por barra e hora.
    cargas_base : dict
        Cargas base da rede, conforme retornado por carregar_cargas_base().
    com_primario : bool
        Se True, o controle primário VoltVar é aplicado antes do consenso.
    modo : str
        'sempre_ativo' — consenso aplicado em todas as horas.
        'condicional'  — consenso aplicado apenas nas horas com violação
                         remanescente após o primário (requer com_primario=True).

    Returns
    -------
    dict com resultados da realização, incluindo defeitos de tensão e
    métricas de ativação dos controles.
    """
    # ------------------------------------------------------------------
    # Carrega perfis e fatores da realização
    # ------------------------------------------------------------------
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

    pv_real   = pv_df[pv_df["id_realizacao"] == realizacao_id]
    bess_real = bess_df[bess_df["id_realizacao"] == realizacao_id]

    criar_elementos_simulacao(pv_real, bess_real, realizacao_id)
    multiplicadores_perfil_carga = obter_multiplicadores_shapes()

    # ------------------------------------------------------------------
    # Constrói grafo de comunicação e potências nominais dos BESS
    # ------------------------------------------------------------------
    barras_bess_realizacao = list(bess_real["barra"].astype(int).unique())
    L, barras_ordenadas = construir_grafo_comunicacao(barras_bess_realizacao)

    # Potência nominal de cada BESS (somada por barra, caso haja mais de uma unidade)
    potencias_nominais = (
        bess_real.groupby("barra")["potencia_kw"].sum().to_dict()
        if not bess_real.empty else {}
    )
    potencias_nominais = {int(k): float(v) for k, v in potencias_nominais.items()}

    # ------------------------------------------------------------------
    # Loop horário
    # ------------------------------------------------------------------
    tensoes_por_hora    = {}
    horas_com_erro      = 0
    horas_com_voltvar   = 0
    horas_com_consenso  = 0

    for hora in range(24):

        # ---- Cargas base ----
        for nome_load, carga in cargas_base.items():
            barra      = carga["barra"]
            base_kw    = carga["kW"]
            base_kvar  = carga["kvar"]
            nome_shape = carga["shape"].lower()
            fator_incerteza = fatores_carga.get(barra, [1.0] * 24)[hora]
            fator_perfil    = multiplicadores_perfil_carga.get(nome_shape, [1.0] * 24)[hora]
            editar_load(nome_load,
                        base_kw   * fator_perfil * fator_incerteza,
                        base_kvar * fator_perfil * fator_incerteza)

        # ---- PV ----
        kw_pv = {}
        for idx, (_, linha) in enumerate(pv_real.iterrows()):
            nome = f"PV_{realizacao_id}_{idx}_barra{linha['barra']}"
            kw   = -float(linha["potencia_kw"]) * fatores_irradiancia[hora]
            kw_pv[nome] = (kw, int(linha["barra"]))
            editar_load(nome, kw, 0.0)

        # ---- BESS (referência base: perfil fixo, sem consenso ainda) ----
        kw_bess = {}
        for idx, (_, linha) in enumerate(bess_real.iterrows()):
            nome = f"BESS_{realizacao_id}_{idx}_barra{linha['barra']}"
            kw   = -float(linha["potencia_kw"]) * BESS_PERFIL[hora]
            kw_bess[nome] = (kw, int(linha["barra"]))
            editar_load(nome, kw, 0.0)

        # ---- Solve base (reguladores livres) ----
        try:
            dss.Text.Command("Set ControlMode=STATIC")
            dss.Solution.Solve()
        except DSSException as e:
            if e.args[0] == 485 or "Max Control Iterations" in str(e):
                horas_com_erro += 1
                continue
            raise

        tensoes_atuais = extrair_tensoes_por_barra()
        tensoes_por_hora[hora] = tensoes_atuais

        # ==============================================================
        # CONTROLE PRIMÁRIO (VoltVar sobre Q de PV e BESS)
        # ==============================================================
        q_refs_pv   = {}
        q_refs_bess = {}

        if com_primario:
            q_refs_pv   = _calcular_q_refs(kw_pv,   tensoes_atuais)
            q_refs_bess = _calcular_q_refs(kw_bess, tensoes_atuais)
            voltvar_ativo = any(
                abs(q) > 1e-6
                for q in {**q_refs_pv, **q_refs_bess}.values()
            )

            if voltvar_ativo:
                horas_com_voltvar += 1
                dss.Text.Command("Set ControlMode=OFF")

                for _ in range(MAX_VOLTVAR_ITER):
                    for nome, (kw, _) in kw_pv.items():
                        editar_load(nome, kw, -q_refs_pv[nome])
                    for nome, (kw, _) in kw_bess.items():
                        editar_load(nome, kw, -q_refs_bess[nome])

                    try:
                        dss.Solution.Solve()
                    except DSSException as e:
                        if e.args[0] == 485 or "Max Control Iterations" in str(e):
                            break
                        raise

                    tensoes_atuais = extrair_tensoes_por_barra()
                    tensoes_por_hora[hora] = tensoes_atuais

                    q_refs_pv   = _calcular_q_refs(kw_pv,   tensoes_atuais)
                    q_refs_bess = _calcular_q_refs(kw_bess, tensoes_atuais)

                    if not any(
                        abs(q) > 1e-6
                        for q in {**q_refs_pv, **q_refs_bess}.values()
                    ):
                        break

                dss.Text.Command("Set ControlMode=STATIC")

        # ==============================================================
        # CONTROLE SECUNDÁRIO (Consenso sobre P dos BESS)
        # ==============================================================
        # Verifica se o consenso deve ser ativado nesta hora
        if modo == "condicional":
            ativar_consenso = _sistema_tem_violacao(tensoes_atuais)
        else:  # sempre_ativo
            ativar_consenso = True

        # O consenso só tem efeito quando há pelo menos 2 BESS e o
        # BESS_PERFIL != 0 (ou seja, o BESS está efetivamente operando)
        bess_operando = abs(BESS_PERFIL[hora]) > 1e-6
        ativar_consenso = ativar_consenso and len(barras_ordenadas) >= 2 and bess_operando

        if ativar_consenso:
            horas_com_consenso += 1
            dss.Text.Command("Set ControlMode=OFF")

            for _ in range(CONSENSO_MAX_ITER):
                # Calcula ΔP pelo algoritmo de consenso
                delta_p = calcular_delta_p_consenso(
                    kw_bess, barras_bess_realizacao, potencias_nominais,
                    L, barras_ordenadas,
                )

                # Calcula ΔQ para preservar o fator de potência do primário
                delta_q = _calcular_q_para_delta_p(
                    delta_p, kw_bess, q_refs_bess,
                )

                # Verifica convergência antes de aplicar
                x_norm = []
                for nome, (kw, barra) in kw_bess.items():
                    p_nom = potencias_nominais.get(barra, 1.0)
                    if abs(p_nom) > 1e-6:
                        x_norm.append(kw / p_nom)
                if x_norm and (max(x_norm) - min(x_norm)) < CONSENSO_TOL:
                    break

                # Aplica correção: P_novo = P_base + ΔP; Q_novo = Q_primário + ΔQ
                for nome, (kw, barra) in kw_bess.items():
                    kw_novo  = kw + delta_p[nome]
                    q_base   = -q_refs_bess.get(nome, 0.0)  # já negado (conv. carga)
                    q_novo   = q_base + delta_q[nome]
                    kw_bess[nome] = (kw_novo, barra)
                    editar_load(nome, kw_novo, q_novo)

                try:
                    dss.Solution.Solve()
                except DSSException as e:
                    if e.args[0] == 485 or "Max Control Iterations" in str(e):
                        break
                    raise

                tensoes_atuais = extrair_tensoes_por_barra()
                tensoes_por_hora[hora] = tensoes_atuais

                # Atualiza Q_refs do primário com as tensões pós-consenso,
                # para que o ΔQ da próxima iteração seja consistente
                if com_primario:
                    q_refs_bess = _calcular_q_refs(kw_bess, tensoes_atuais)

            dss.Text.Command("Set ControlMode=STATIC")

    # ------------------------------------------------------------------
    # Compilação dos resultados
    # ------------------------------------------------------------------
    defeitos      = calcular_defeitos_por_faixa(tensoes_por_hora)
    defeitos_bess = calcular_defeitos_por_faixa_bess(tensoes_por_hora)

    return {
        "id_realizacao":                             realizacao_id,
        "_tensoes_por_hora":                         tensoes_por_hora,
        "tipo_dia":                                  row_info["tipo_dia"],
        "pv_unidades":                               int(row_info["pv_unidades"]),
        "pv_potencia_total_kw":                      float(row_info["pv_potencia_total_kw"]),
        "bess_unidades":                             int(row_info["bess_unidades"]),
        "bess_potencia_total_kw":                    float(bess_real["potencia_kw"].sum()) if not bess_real.empty else 0.0,
        "subtensao_manhã":                           defeitos["subtensao_manhã"],
        "sobretensao_manhã":                         defeitos["sobretensao_manhã"],
        "subtensao_tarde":                           defeitos["subtensao_tarde"],
        "sobretensao_tarde":                         defeitos["sobretensao_tarde"],
        "subtensao_noite":                           defeitos["subtensao_noite"],
        "sobretensao_noite":                         defeitos["sobretensao_noite"],
        "subtensao_madrugada":                       defeitos["subtensao_madrugada"],
        "sobretensao_madrugada":                     defeitos["sobretensao_madrugada"],
        "subtensao_bess_carga":                      defeitos_bess["subtensao_carga"],
        "sobretensao_bess_carga":                    defeitos_bess["sobretensao_carga"],
        "subtensao_bess_pós_carga_pré_descarga":     defeitos_bess["subtensao_pós_carga_pré_descarga"],
        "sobretensao_bess_pós_carga_pré_descarga":   defeitos_bess["sobretensao_pós_carga_pré_descarga"],
        "subtensao_bess_descarga":                   defeitos_bess["subtensao_descarga"],
        "sobretensao_bess_descarga":                 defeitos_bess["sobretensao_descarga"],
        "subtensao_bess_fora_de_operacao":           defeitos_bess["subtensao_fora_de_operacao"],
        "sobretensao_bess_fora_de_operacao":         defeitos_bess["sobretensao_fora_de_operacao"],
        "horas_com_erro_max_control":                horas_com_erro,
        "horas_com_voltvar_ativo":                   horas_com_voltvar,
        "horas_com_consenso_ativo":                  horas_com_consenso,
    }


# ---------------------------------------------------------------------------
# Processamento de um nível de penetração completo
# ---------------------------------------------------------------------------

def processar_nivel_controle_secundario(
    pasta_nivel,
    dss_path,
    pasta_saida_nivel,
    com_primario: bool,
    modo: str,
    max_realizacoes=None,
):
    """
    Processa todas as realizações de um nível de penetração com controle
    secundário por consenso.

    Parameters
    ----------
    pasta_nivel : str
        Caminho para a pasta do nível de penetração (contém os CSVs do
        Monte Carlo gerados por sortear_monte_carlo.py).
    dss_path : str
        Caminho para o arquivo .dss do modelo IEEE 34 bus.
    pasta_saida_nivel : str
        Pasta de destino para os CSVs e envelopes de tensão desta rodada.
    com_primario : bool
        Indica se o controle primário VoltVar está ativo.
    modo : str
        Modo do consenso: 'sempre_ativo' ou 'condicional'.
    max_realizacoes : int or None
        Limita o número de realizações processadas (útil para testes).

    Returns
    -------
    df_resultados : pd.DataFrame
    realizacoes_com_erro : int
    df_tensoes_completas : pd.DataFrame
    """
    resumo            = ler_resumo_configuracoes(pasta_nivel)
    perfis_irr        = ler_perfis_irradiancia(pasta_nivel)
    fatores_incerteza = ler_fatores_incerteza_carga(pasta_nivel)
    elementos         = ler_elementos_opendss(pasta_nivel)

    if resumo.empty:
        raise ValueError(f"Resumo de configurações não encontrado em {pasta_nivel}")

    resultados           = []
    tensoes_completas    = []
    realizacoes_com_erro = 0
    tensoes_envelope     = {faixa: {} for faixa in TIME_BANDS}

    for _, row in resumo.head(max_realizacoes).iterrows():
        id_realizacao = int(row["id_realizacao"])

        redirecionar_circuito(dss_path)
        cargas_base = carregar_cargas_base()

        resultado = simular_realizacao_controle_secundario(
            id_realizacao,
            row,
            elementos[elementos["id_realizacao"] == id_realizacao],
            elementos[
                (elementos["id_realizacao"] == id_realizacao)
                & (elementos["classe"] == "Storage")
            ],
            perfis_irr,
            fatores_incerteza,
            cargas_base,
            com_primario=com_primario,
            modo=modo,
        )

        pen_pct = int(os.path.basename(pasta_nivel).split("_")[1].replace("pct", ""))
        resultado["pen_pct"] = pen_pct

        tensoes_por_hora_real = resultado.pop("_tensoes_por_hora")
        _acumular_tensoes_envelope(tensoes_envelope, tensoes_por_hora_real)
        tensoes_completas.extend(
            _linhas_tensoes_completas(pen_pct, resultado, tensoes_por_hora_real)
        )
        resultados.append(resultado)

        if resultado["horas_com_erro_max_control"] > 0:
            realizacoes_com_erro += 1

        erro_str = (
            f" ⚠ {resultado['horas_com_erro_max_control']} hora(s) com erro Max Control Iter"
            if resultado["horas_com_erro_max_control"] > 0 else ""
        )
        primario_str = f"VoltVar: {resultado['horas_com_voltvar_ativo']}h | " if com_primario else ""
        print(
            f"  ✓ Realização {id_realizacao} - "
            f"Subtensão: manhã={resultado['subtensao_manhã']} tarde={resultado['subtensao_tarde']} "
            f"noite={resultado['subtensao_noite']} madrugada={resultado['subtensao_madrugada']} | "
            f"Sobretensão: manhã={resultado['sobretensao_manhã']} tarde={resultado['sobretensao_tarde']} "
            f"noite={resultado['sobretensao_noite']} madrugada={resultado['sobretensao_madrugada']} | "
            f"{primario_str}"
            f"Consenso: {resultado['horas_com_consenso_ativo']}h"
            f"{erro_str}"
        )

    df_resultados        = pd.DataFrame(resultados)
    df_tensoes_completas = pd.DataFrame(tensoes_completas)
    os.makedirs(pasta_saida_nivel, exist_ok=True)
    df_resultados.to_csv(
        os.path.join(pasta_saida_nivel, "resultados_opendss_por_realizacao.csv"),
        index=False, sep=";", decimal=",",
    )
    df_tensoes_completas.to_csv(
        os.path.join(pasta_saida_nivel, "tensoes_opendss_completas.csv"),
        index=False, sep=";", decimal=",",
    )
    _salvar_envelope_csv(tensoes_envelope, pasta_saida_nivel)

    return df_resultados, realizacoes_com_erro, df_tensoes_completas
