import matplotlib.pyplot as plt
import numpy as np

# Perfil fixo de BESS (negativo = absorção/recarga, positivo = injeção)
BESS_PERFIL = np.zeros(24)
BESS_PERFIL[10:15] = -1.0
BESS_PERFIL[18:22] = 1.0

hours = np.arange(24)

plt.figure(figsize=(10, 5))
plt.step(hours, BESS_PERFIL, where='mid', marker='o')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel('Hora do dia')
plt.ylabel('Potência BESS (pu)')
plt.title('Perfil de Operação do BESS')
plt.xticks(hours)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('perfil_bess.png', dpi=200)
plt.show()