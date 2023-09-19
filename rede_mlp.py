## inicializando as variavéis

import numpy as np
import random as rd
import matplotlib.pyplot as plt
import os
current_directory = os.getcwd()
completodiretorio=current_directory+'/digitos';

os.chdir(completodiretorio)

ampdigitos=50                                       ## digitos
vsai=10                                             ## saidas
amostras=ampdigitos*vsai                            ## digitos * saidas
entradas=256         ## entradas -> 256 pixels representação dos pixels dos arquivo [Vetor salsichão]
neur=200                                            ## quantidade de neuronios
limiar=0.0                                          ## limiar
alfa=0.005                                          ## taxa de aprendizagem
errotolerado=0.2                                   ## até onde o treinamento ira continuar
listaciclo=[]                                       ## armazenar os ciclos
listaerro=[]                                        ## armazenar o erro em função dos ciclos

# MONTANDO O ARQUIVO DE AMOSTRAS DE TREINAMENTO
x = np.zeros((amostras, entradas))  ## matriz de zeros para armazenar as entradas
k2 = '_'  ## caracteres para concat e geração do nome do arqv
k4 = '.txt'  ## caracter para nomear arquivo
cont = 0  ## contagem para orientação da linha para preencher a matriz X
ordem = np.zeros(amostras)  ## apontar qual o vetor alvo que a amostra é correspondente

for m in range(vsai):  ## laço para percorrer vetores de saidas
    k1 = str(m)  ## transforma de int para str

    for n in range(ampdigitos):  ## laço para percorrer a qtd de amostras por digitos
        k3a = n + 1  ## somando 1 para começar a amostra no 1
        k3 = str(k3a)  ## transforma de int para str
        nome = k1 + k2 + k3 + k4  ## concat nome do arq
        ## k1 -> digito
        ## k2 -> underline
        ## k3 -> ordem da amostra do digito
        ## k4 -> .txt

        entrada = np.loadtxt(
            nome)  ## entrada != entradas, responsavel por fazer a leitura do arq e conversão para vetor

        x[cont, :] = entrada[:]  ## linha de ordem cont, e percorrer tods as colunas
        ordem[cont] = m  ## ordem[cont] recebe o digito correspondente
        cont = cont + 1  ## soma + 1 no contador
        ## vai executar o for e pegar todas os 0 até os 9

ordem = ordem.astype('int')  ## conversão do vetor ordem para inteiro

# LENDO O ARQUIVO DE SAÍDAS ESPERADAS (TARGET)
t = np.loadtxt('respostas.csv', delimiter=',', skiprows=0)  ## pegando o arquivo de targets

# GERAR MATRIZ DE PESOS SINÁPTICOS ALEATORIAMENTE
vanterior = np.zeros((entradas, neur))  ## matrix zeros para os pesos, recebe entradas e neuronios para dimesão

aleatorio = 0.2  ## range de valores aleatorios

for i in range(entradas):  ## laço q percorre entradas
    for j in range(neur):  ## laço q percorre neuronios

        vanterior[i][j] = rd.uniform(-aleatorio, aleatorio)  ## gera um valor aleatorio +0.2 e -0.2 para matriz

v0anterior = np.zeros((1, neur))  ## baias da camadas intermediaria, recebe 1 valor por neuro

for j in range(neur):  ## laço q percorre neuro
    v0anterior[0][j] = rd.uniform(-aleatorio, aleatorio)  ## gera um valor aleatorio +0.2 e -0.2 para matriz

## mesmos processo acima para camada intermidiara e camada de saida
wanterior = np.zeros((neur, vsai))
aleatorio = 0.2

for i in range(neur):
    for j in range(vsai):
        wanterior[i][j] = rd.uniform(-aleatorio, aleatorio)
w0anterior = np.zeros((1, vsai))

for j in range(vsai):
    w0anterior[0][j] = rd.uniform(-aleatorio, aleatorio)

# MATRIZES DE ATUALIZAÇÃO DE PESOS E VALORES DE SAÍDA DA REDE (todas as matrizes foram iniciadas com zeros)

vnovo = np.zeros((entradas, neur))
v0novo = np.zeros((1, neur))
wnovo = np.zeros((neur, vsai))
w0novo = np.zeros((1, vsai))
zin = np.zeros((1, neur))  ## valores da camada intermediaria
z = np.zeros((1, neur))  ## valores da camada de saida
deltinhak = np.zeros((vsai, 1))  ## atualiza os pesos
deltaw0 = np.zeros((vsai, 1))
deltinha = np.zeros((1, neur))
xaux = np.zeros((1, entradas))
h = np.zeros((vsai, 1))
target = np.zeros((vsai, 1))
deltinha2 = np.zeros((neur, 1))
ciclo = 0
errototal = 100000

while errotolerado < errototal:  ## roda enquanto o erro tolerado for menor que o error total

    errototal = 0  ## zerando o error total pois sera calculador ciclo a ciclo

    for padrao in range(amostras):  ## laço para percorrer amostras
        for j in range(neur):  ## laço q percorre os neuronios
            zin[0][j] = np.dot(x[padrao, :], vanterior[:, j]) + v0anterior[0][
                j]  ## multiplicando peso com entradas e somando com baias

        z = np.tanh(zin)  ## função de ativação [Tangente hiperbolica]
        yin = np.dot(z, wanterior) + w0anterior  ## camadas de saida [vetor de valores que chegam na camada de saida]
        y = np.tanh(yin)  ## função de ativação [Tangente hiperbolica]

        ## fazendo a transposição
        for m in range(vsai):
            h[m][0] = y[0][m]
        for m in range(vsai):
            target[m][0] = t[m][ordem[padrao]]

        errototal = errototal + np.sum(0.5 * ((target - h) ** 2))  ## calculando o erro -> acumulativo

        # OBTER MATRIZES PARA ATUALIZAÇÕES DOS PESOS
        deltinhak = (target - h) * (1 + h) * (1 - h)
        deltaw = alfa * (np.dot(deltinhak, z))
        deltaw0 = alfa * deltinhak
        deltinhain = np.dot(np.transpose(deltinhak), np.transpose(wanterior))
        deltinha = deltinhain * (1 + z) * (1 - z)

        ## transposta do deltinha para o deltinha 2
        for m in range(neur):
            deltinha2[m][0] = deltinha[0][m]
        for k in range(entradas):
            xaux[0][k] = x[padrao][k]

        deltav = alfa * np.dot(deltinha2, xaux)
        deltav0 = alfa * deltinha

        # REALIZANDO AS ATUALIZAÇÕES DE PESOS
        vnovo = vanterior + np.transpose(deltav)
        v0novo = v0anterior + np.transpose(deltav0)
        wnovo = wanterior + np.transpose(deltaw)
        w0novo = w0anterior + np.transpose(deltaw0)
        vanterior = vnovo
        v0anterior = v0novo
        wanterior = wnovo
        w0anterior = w0novo

    ciclo = ciclo + 1
    listaciclo.append(ciclo)
    listaerro.append(errototal)
    print('Ciclo\t Erro')
    print(ciclo, '\t', errototal)

#protando grafico
plt.plot(listaciclo,listaerro)
plt.xlabel('Ciclo')
plt.ylabel('Erro')
plt.show()

print("vnovo", vnovo)
print("v0novo", v0novo)
print("wnovo", wnovo)
print("w0novo", w0novo)

np.savetxt("vnovo.csv", vnovo, delimiter=';')
np.savetxt("v0novo.csv", v0novo, delimiter=';')
np.savetxt("wnovo.csv", wnovo, delimiter=';')
np.savetxt("w0novo.csv", w0novo, delimiter=';')

