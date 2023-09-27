import numpy as np
import csv

matriz = -np.ones((26, 26))
np.fill_diagonal(matriz, 1)

# Especifique o nome do arquivo CSV
nome_arquivo_csv = "./letras/matriz.csv"

# Escreva a matriz em um arquivo CSV
with open(nome_arquivo_csv, "w", newline="") as arquivo_csv:
    writer = csv.writer(arquivo_csv)
    writer.writerows(matriz)

print(f"A matriz foi exportada para {nome_arquivo_csv}.")
