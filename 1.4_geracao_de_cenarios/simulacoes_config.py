import os

import numpy as np

# Parâmetros de defeito de tensão
V_PU_MIN = 0.95
V_PU_MAX = 1.05

# Barras removidas das analises de tensao por nao representarem barras de carga.
BARRAS_EXCLUIDAS_ANALISE = {
    "sourcebus",
    "800",
    "812",
    "814",
    "814r",
    "850",
    "852",
    "852r",
    "888",
}

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
# Corresponde a uma travessia em largura (BFS) a partir da barra de fonte 800
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
