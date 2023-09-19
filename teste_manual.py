import os
import numpy as np

t = np.loadtxt('respostas.csv', delimiter=',', skiprows=0)

vanterior = np.loadtxt('vnovo.csv', delimiter=';', skiprows=0)
v0anterior = np.loadtxt('v0novo.csv', delimiter=';', skiprows=0)
wanterior = np.loadtxt('wnovo.csv', delimiter=';', skiprows=0)
w0anterior = np.loadtxt('w0novo.csv', delimiter=';', skiprows=0)

(vent, neur) = np.shape(vanterior)
(vsai, numclasses) = np.shape(t)
limiar = 0
zin = np.zeros((1, neur))
target = np.zeros((vsai, 1))

## teste manual

xteste=np.loadtxt('2_72.txt')
for m2 in range(vsai):
    for n2 in range(neur):
        zin[0][n2]=np.dot(xteste, vanterior[:,n2] + v0anterior[0][n2])

    z=np.tanh(zin)
    yin=np.dot(z, wanterior) + (w0anterior)
    y=np.tanh(yin)
print(yin[0:1])
for j in range(vsai):
    if(yin[0][j] >= limiar):
        y[0][j]=1.0
    else:
        y[0][j]=-1.0

print(y[0],j)