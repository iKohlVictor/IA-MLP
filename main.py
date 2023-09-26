import numpy as np
import random as rd
import matplotlib.pyplot as plt
import os
import string

current_directory = os.getcwd()
completodiretorio=current_directory+'/letras';

os.chdir(completodiretorio)

ampdigitos=2
vsai=26
amostras=ampdigitos*vsai
alfabeto = list(string.ascii_uppercase)
entradas=16
neur=200
limiar=0.0
alfa=0.005
errotolerado=2
listaciclo=[]
listaerro=[]

#MONTANDO O ARQUIVO DE AMOSTRAS DE TREINAMENTO
x=np.zeros((amostras, entradas))
k2='_'
k4='.txt'
cont=0
ordem=np.zeros(amostras)


for m in range(vsai):
    k1=alfabeto[m]

    for n in range(ampdigitos):
        k3a=n
        k3=str(k3a)
        nome=k1+k2+k3+k4
        entrada=np.loadtxt(nome)
        x[cont,:]=entrada[:]
        ordem[cont]=m
        cont=cont+1
ordem=ordem.astype('int')

#LENDO O ARQUIVO DE SAÍDAS ESPERADAS (TARGET)
t = np.loadtxt('matriz.csv', delimiter=',', skiprows=0)
# t=np.loadtxt('results.txt')

#GERAR MATRIZ DE PESOS SINÁPTICOS ALEATORIAMENTE
vanterior=np.zeros((entradas, neur))
aleatorio=0.2

for i in range(entradas):
    for j in range(neur):
        vanterior[i][j]=rd.uniform(-aleatorio,aleatorio)
v0anterior=np.zeros((1,neur))

for j in range(neur):
    v0anterior[0][j]=rd.uniform(-aleatorio,aleatorio)

wanterior=np.zeros((neur, vsai))
aleatorio=0.2

for i in range(neur):
    for j in range(vsai):
        wanterior[i][j]=rd.uniform(-aleatorio,aleatorio)
w0anterior=np.zeros((1,vsai))

for j in range(vsai):
    w0anterior[0][j]=rd.uniform(-aleatorio,aleatorio)

#MATRIZES DE ATUALIZAÇÃO DE PESOS E VALORES DE SAÍDA DA REDE
vnovo=np.zeros((entradas,neur))
v0novo=np.zeros((1,neur))
wnovo=np.zeros((neur,vsai))
w0novo=np.zeros((1,vsai))
zin=np.zeros((1,neur))
z=np.zeros((1,neur))
deltinhak=np.zeros((vsai, 1))
deltaw0=np.zeros((vsai, 1))
deltinha=np.zeros((1 ,neur))
xaux=np.zeros((1, entradas))
h=np.zeros((vsai, 1))
target=np.zeros((vsai, 1))
deltinha2=np.zeros((neur, 1))
ciclo=0
errototal=100000

while errotolerado < errototal:

    errototal = 0

    for padrao in range(amostras):
        for j in range(neur):
            zin[0][j] = np.dot(x[padrao, :], vanterior[:, j]) + v0anterior[0][j]
        z = np.tanh(zin)
        yin = np.dot(z, wanterior) + w0anterior
        y = np.tanh(yin)

        for m in range(vsai):
            h[m][0] = y[0][m]
        for m in range(vsai):
            target[m][0] = t[m][ordem[padrao]]

        errototal = errototal + np.sum(0.5 * ((target - h) ** 2))

        # OBTER MATRIZES PARA ATUALIZAÇÕES DOS PESOS
        deltinhak = (target - h) * (1 + h) * (1 - h)
        deltaw = alfa * (np.dot(deltinhak, z))
        deltaw0 = alfa * deltinhak
        deltinhain = np.dot(np.transpose(deltinhak), np.transpose(wanterior))
        deltinha = deltinhain * (1 + z) * (1 - z)

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

print("vnovo", vnovo)
print("v0novo", v0novo)
print("wnovo", wnovo)
print("w0novo", w0novo)

np.savetxt("vnovo.csv", vnovo, delimiter=';')
np.savetxt("v0novo.csv", v0novo, delimiter=';')
np.savetxt("wnovo.csv", wnovo, delimiter=';')
np.savetxt("w0novo.csv", w0novo, delimiter=';')

plt.plot(listaciclo,listaerro)
plt.xlabel('Ciclo')
plt.ylabel('Erro')
plt.show()



# #os digitos vão de 0 a 135 pra cada padrão
# #temos a amostra tal pra frente
# aminicial=1
# #quantidade de amostras pra cada digito
# amtestedigitos=89
# #tantas linhas da rede pra uma coluna
# yteste=np.zeros((vsai,1))
#
# #quantidade total
# cont=0
# #quantidade quando a rede acertar
# contcerto=0
#
# ordem=np.zeros(amostras)
# for m in range(vsai):
#     k1=str(m)
#     for n in range(amtestedigitos):
#         k3a=n+aminicial
#         k3=str(k3a)
#         nome=k1+k2+k3+k4
#         xteste=np.loadtxt(nome)
#         for m2 in range(vsai):
#             for n2 in range(neur):
#                 zin[0][n2]=np.dot(xteste,vanterior[:,n2]+v0anterior[0][n2])
#                 np.tanh(zin)
#                 yin=np.dot(z,wanterior)+w0anterior
#                 y=np.tanh(yin)
#             for j in range(vsai):
#                 if y[0][j]>=limiar:
#                     y[0][j]=1.0
#                 else:
#                     y[0][j]=-1.0
#             for j in range(vsai):
#                 yteste[j][0]=y[0][j]
#
#             for j in range(vsai):
#                 target[j][0]=y[0][j]
#             soma=np.sum(y-target)
#
#             if soma==0:
#                 contcerto=contcerto+1
#             cont=cont+1
# taxa=contcerto/cont
# print("taxa",taxa)

#0.7888888888888889