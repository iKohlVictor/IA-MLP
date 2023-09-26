import string
import os

alfabeto = list(string.ascii_uppercase)


nome_arquivo = "letter-recognition.txt"

matriz = []

with open(nome_arquivo, "r") as arquivo:
    for linha in arquivo:
        linha = linha.strip()
        
        matriz.append(linha)


caminhoDaPasta='./letras'


for indexLetra, letra in enumerate(alfabeto):
    countLinha = 0
    for indexLinha, linha in enumerate(matriz):
        if linha[0] == letra:
            arrayLinha = linha.split(',')
            arrayLinha = arrayLinha[1:]
            arrayLinha = ' '.join(arrayLinha)

            while countLinha <= 1:
                nome_arquivo = letra + '_' + str(countLinha) + ".txt"
                caminho_completo = os.path.join(caminhoDaPasta, nome_arquivo)
                with open(caminho_completo, "w") as arquivo:
                    arquivo.write(arrayLinha)
                countLinha += 1

        

