import matplotlib.pyplot as plt
import re

# Caminho para o arquivo .dss
dss_file_path = r"c:\Users\barana13\OneDrive - ANDRITZ AG\Documents\projeto_de_formatura\IEEE34bus\IEEE34_original_with_loadshapes.dss"

# Função para extrair loadshapes do arquivo .dss
def extract_loadshapes(file_path):
    loadshapes = {}
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Regex para encontrar definições de LoadShape
    pattern = r'New LoadShape\.(\w+)\s+npts=\d+\s+interval=\d+\s+~ mult=\[([^\]]+)\]'
    matches = re.findall(pattern, content, re.IGNORECASE)
    
    for name, mult_str in matches:
        # Converter a string de multiplicadores em lista de floats
        mult_values = [float(x.strip()) for x in mult_str.split()]
        loadshapes[name] = mult_values
    
    return loadshapes

# Extrair loadshapes
loadshapes = extract_loadshapes(dss_file_path)

# Plotar as curvas
plt.figure(figsize=(10, 6))
hours = list(range(24))  # Horas de 0 a 23

for name, values in loadshapes.items():
    plt.plot(hours, values, label=name, marker='o')

plt.xlabel('Hora do Dia')
plt.ylabel('Fator de Carga (pu)')
plt.title('Curvas de Carga')
plt.legend()
plt.grid(True)
plt.xticks(hours)
plt.tight_layout()

# Salvar o gráfico
plt.savefig('curvas_loadshapes.png')
plt.show()