# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 21:28:08 2021

@author: Vitor Fontenele
"""

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
#import scipy.sparse.linalg
#from shapely.geometry import Point, Polygon

"""
Caracteristicas basicas da malha
"""

fileName = "malhaliddriven.msh"
#elementSizeFactor = 0.03

import malhaModulo

malha = malhaModulo.Malha(fileName)

X = malha.X
Y = malha.Y
Lx = malha.Lx
Ly = malha.Ly
nx = malha.nx
ny = malha.ny
IEN = malha.IEN
IENBound = malha.IENBound
cc = malha.cc
ne = malha.ne
npoints = malha.npoints

"""
Pontos de contorno e criacao da bval
"""

c1 = 0.0
c2 = 0.0
dc = c2 - c1

bval = np.zeros(npoints,dtype = 'float')
vx = np.zeros(npoints,dtype = 'float')
vy = np.zeros(npoints,dtype = 'float')

#cantos da malha
bval[0] = c1
bval[1] = c2
bval[2] = c2
bval[3] = c1

start = 4
#ignore = []

for k1 in range (ny-2):
    #contorno direito
    bval[k1 + start] = c1
k1 += 1

for k2 in range (nx-2):
    #contorno superior
    bval[k2 + k1 + start] = c2
k2 += 1

for k3 in range (ny-2):
    #contorno esquerdo
    bval[k3 + k2 + k1 + start] = c2
k3 += 1
    
for k4 in range (nx-2):
    #contorno inferior
    bval[k4 + k3 + k2 + k1 + start] = c1

"""
Matrizes Globais (K, M, GX e GY)
"""

matrizesGlobais = malhaModulo.MatrizesGlobais(fileName)
K = matrizesGlobais.K
M = matrizesGlobais.M
GX = matrizesGlobais.GX
GY = matrizesGlobais.GY

"""
Parametros utilizados
"""

dt = 0.01
Re = 100   
iteracoes = 5

"""
Solucao utilizando eq de transporte e de funcao corrente
"""

#matriz identidade
ident = np.identity(npoints)

for g in range (iteracoes):
    #Condicao de Contorno de Vorticidade
    B1 = np.dot(GX,vy) - np.dot(GY,vx)
    wz = np.linalg.solve(M,B1)
    
    #Matriz de estabilizacao
    matrizesGlobais.construirMatrizKest(vx,vy,dt)
    Kest = matrizesGlobais.Kest
                
    #Solucao da equacao de transporte
    vx_id = ident*vx
    vy_id = ident*vy
    A2 = M/dt  
    B2 = np.dot((M/dt - np.dot(vx_id,GX) - np.dot(vy_id,GY) - K/Re - Kest), wz)
    for i in cc:
        A2[i,:] = 0.0
        B2[i] = wz[i]
        A2[i,i] = 1.0  
    wz = np.linalg.solve(A2,B2)

    #Reinicializacao da matriz de estabilizacao
    matrizesGlobais.Kest = np.zeros( (npoints,npoints), dtype='double')
    
    #Solucao da equacao de funcao corrente
    A3 = K
    B3 = np.dot(M,wz) 
    for i in cc:
        #nao = [4,5,6,7,8,9,10,11,12]
        A3[i,:] = 0.0
        B3[i] = bval[i]
        A3[i,i] = 1.0   
                
    Psi = np.linalg.solve(A3,B3)
    
    #Atualizacao de vx e vy
    #vx
    A4 = M
    B4 = np.dot(GY,Psi)
    vx = np.linalg.solve(A4,B4)
    #vy
    A5 = M
    B5 = np.dot(-GX,Psi)
    vy = np.linalg.solve(A5,B5)
    
    #Imposicao das cc de vx e vy
    vx[0] = 0
    vx[1] = 1
    vx[2] = 1
    vx[3] = 0

    vy[0] = 0
    vy[1] = 0
    vy[2] = 0
    vy[3] = 0
    
    start = 4
      
    for k1 in range (ny-2):
        #contorno direito
        vy[k1 + start] = 0
        vx[k1 + start] = 0
    k1 += 1  
    
    for k2 in range (nx-2):
        #contorno superior
        vx[k2 + k1 + start] = 1
        vy[k2 + k1 + start] = 0
    k2 += 1

    for k3 in range (ny-2):
        #contorno esquerdo
        vx[k3 + k2 + k1 + start] = 0 
        vy[k3 + k2 + k1 + start] = 0
    k3 += 1
    
    for k4 in range (nx-2):
        #contorno inferior
        vx[k4 + k3 + k2 + k1 + start] = 0
        vy[k4 + k3 + k2 + k1 + start] = 0   

#Setup do plot
#plt.rc('text', usetex=True)
triang = mtri.Triangulation(X,Y,IEN)
ax = plt.axes()
ax.set_aspect("equal")

"""O que voce quer plotar?"""
#--Funcao corrente --> Psi
#--Vorticidade --> wz
#--Velocidade em x --> vx
#--Velocidade em y --> vy
ax.tricontourf(triang,Psi,levels=100,cmap = 'jet')
plt.title("Função corrente")

ax.set_xlabel("x")
ax.set_ylabel("y")

plt.show()
#plt.savefig("liddriven.png",dpi=300)


