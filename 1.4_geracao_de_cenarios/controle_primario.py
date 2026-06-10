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

# ---------------------------------------------------------------------------
# Constantes VoltVar — ABNT NBR 16149:2013, Seção 4.7.3 / Figura 2
# ---------------------------------------------------------------------------
V_DEADBAND_LOW   = 0.97
V_DEADBAND_HIGH  = 1.03
V_SAT_LOW        = 0.90
V_SAT_HIGH       = 1.10
Q_MAX_PU         = 0.4358   # tan(arccos(0.90)) ≈ 43,58 % de P_inst
MAX_VOLTVAR_ITER = 1


def volt_var_curve(v_pu: float, p_inst_kw: float) -> float:
    """
    Retorna Q_ref (kvar) para um dado v_pu na barra do DER.

    Convenção de GERADOR (positivo = injeção de reativo na barra):
      Q > 0 → capacitivo (injeta reativo, eleva tensão — para subtensão)
      Q < 0 → indutivo   (absorve reativo, reduz tensão — para sobretensão)

    ATENÇÃO: Load elements do OpenDSS usam convenção de CARGA (oposta):
      kvar > 0 → consome reativo (reduz tensão)
      kvar < 0 → injeta reativo  (eleva tensão)
    Ao aplicar este Q num Load element, use -volt_var_curve(v, p).

    Q_max escala com P_inst para garantir FP ≥ 0,90 (NBR 16149:2013, Figura 2).
    """
    q_max = Q_MAX_PU * p_inst_kw

    if v_pu <= V_SAT_LOW:
        return +q_max
    elif v_pu < V_DEADBAND_LOW:
        slope = -q_max / (V_DEADBAND_LOW - V_SAT_LOW)
        return q_max + slope * (v_pu - V_SAT_LOW)
    elif v_pu <= V_DEADBAND_HIGH:
        return 0.0
    elif v_pu < V_SAT_HIGH:
        slope = -q_max / (V_SAT_HIGH - V_DEADBAND_HIGH)
        return slope * (v_pu - V_DEADBAND_HIGH)
    else:
        return -q_max


def _get_q_ref_barra(tensoes_fases: list, p_inst_kw: float) -> float:
    """
    Avalia VoltVar para fase mínima (subtensão) e fase máxima (sobretensão)
    e retorna o Q_ref de maior magnitude absoluta.
    """
    q_vmin = volt_var_curve(min(tensoes_fases), p_inst_kw)
    q_vmax = volt_var_curve(max(tensoes_fases), p_inst_kw)
    return q_vmin if abs(q_vmin) >= abs(q_vmax) else q_vmax


def _calcular_q_refs(der_kw: dict, tensoes: dict) -> dict:
    """
    Calcula Q_ref para cada DER load.

    der_kw: {nome: (kw_hora, barra_int)}
    tensoes: resultado de extrair_tensoes_por_barra()
    Retorna {nome: q_ref_kvar}
    """
    q_refs = {}
    for nome, (kw, barra) in der_kw.items():
        p_inst    = abs(kw)
        barra_str = str(barra)
        q_refs[nome] = (
            _get_q_ref_barra(tensoes[barra_str], p_inst)
            if p_inst > 1e-6 and barra_str in tensoes else 0.0
        )
    return q_refs


def simular_realizacao_controle_primario(
    realizacao_id, row_info, pv_df, bess_df, perfis_irr, fatores_incerteza, cargas_base
):
    perfis_irr_realizacao = perfis_irr.loc[perfis_irr["id_realizacao"] == realizacao_id]
    if perfis_irr_realizacao.empty:
        raise ValueError(
            f"Perfis de irradiância não encontrados para id_realizacao={realizacao_id}. "
            f"IDs disponíveis: {sorted(perfis_irr['id_realizacao'].unique())}"
        )
    radiancias        = perfis_irr_realizacao.iloc[0]
    fatores_irradiancia = [float(radiancias[f"h{h:02d}"]) for h in range(24)]

    carga_real   = fatores_incerteza[fatores_incerteza["id_realizacao"] == realizacao_id]
    fatores_carga = {
        int(linha["barra"]): [float(linha[f"h{h:02d}"]) for h in range(24)]
        for _, linha in carga_real.iterrows()
    }

    pv_real   = pv_df[pv_df["id_realizacao"] == realizacao_id]
    bess_real = bess_df[bess_df["id_realizacao"] == realizacao_id]

    criar_elementos_simulacao(pv_real, bess_real, realizacao_id)
    multiplicadores_perfil_carga = obter_multiplicadores_shapes()

    tensoes_por_hora  = {}
    horas_com_erro    = 0
    horas_com_voltvar = 0

    for hora in range(24):
        # --- Cargas base ---
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

        # --- PV loads: (kw, barra) por nome ---
        kw_pv = {}
        for idx, (_, linha) in enumerate(pv_real.iterrows()):
            nome = f"PV_{realizacao_id}_{idx}_barra{linha['barra']}"
            kw   = -float(linha["potencia_kw"]) * fatores_irradiancia[hora]
            kw_pv[nome] = (kw, int(linha["barra"]))
            editar_load(nome, kw, 0.0)

        # --- BESS loads: (kw, barra) por nome ---
        kw_bess = {}
        for idx, (_, linha) in enumerate(bess_real.iterrows()):
            nome = f"BESS_{realizacao_id}_{idx}_barra{linha['barra']}"
            kw   = -float(linha["potencia_kw"]) * BESS_PERFIL[hora]
            kw_bess[nome] = (kw, int(linha["barra"]))
            editar_load(nome, kw, 0.0)

        # --- Solve principal com reguladores livres ---
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

        # --- Calcula Q_ref inicial para todos os DERs ---
        q_refs_pv   = _calcular_q_refs(kw_pv,   tensoes_atuais)
        q_refs_bess = _calcular_q_refs(kw_bess, tensoes_atuais)

        voltvar_ativo = any(abs(q) > 1e-6 for q in {**q_refs_pv, **q_refs_bess}.values())

        # --- Loop VoltVar com taps congelados na posição do solve principal ---
        if voltvar_ativo:
            horas_com_voltvar += 1
            dss.Text.Command("Set ControlMode=OFF")

            for _ in range(MAX_VOLTVAR_ITER):
                for nome, (kw, _) in kw_pv.items():
                    editar_load(nome, kw, -q_refs_pv[nome])   # negação: Load usa conv. de carga
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

                if not any(abs(q) > 1e-6 for q in {**q_refs_pv, **q_refs_bess}.values()):
                    break

            dss.Text.Command("Set ControlMode=STATIC")

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
    }


def processar_nivel_controle_primario(pasta_nivel, dss_path, pasta_saida_nivel, max_realizacoes=None):
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

        resultado = simular_realizacao_controle_primario(
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

        tensoes_por_hora_real = resultado.pop("_tensoes_por_hora")
        _acumular_tensoes_envelope(tensoes_envelope, tensoes_por_hora_real)
        tensoes_completas.extend(_linhas_tensoes_completas(pen_pct, resultado, tensoes_por_hora_real))
        resultados.append(resultado)

        if resultado["horas_com_erro_max_control"] > 0:
            realizacoes_com_erro += 1

        erro_str = (
            f" ⚠ {resultado['horas_com_erro_max_control']} hora(s) com erro Max Control Iter"
            if resultado["horas_com_erro_max_control"] > 0 else ""
        )
        print(
            f"  ✓ Realização {id_realizacao} - "
            f"Subtensão: manhã={resultado['subtensao_manhã']} tarde={resultado['subtensao_tarde']} "
            f"noite={resultado['subtensao_noite']} madrugada={resultado['subtensao_madrugada']} | "
            f"Sobretensão: manhã={resultado['sobretensao_manhã']} tarde={resultado['sobretensao_tarde']} "
            f"noite={resultado['sobretensao_noite']} madrugada={resultado['sobretensao_madrugada']} | "
            f"VoltVar ativo: {resultado['horas_com_voltvar_ativo']}h"
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
