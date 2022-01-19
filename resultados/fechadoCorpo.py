# -*- coding: utf-8 -*-
"""
Created on Thu May 20 22:08:00 2021

@author: Vitor Fontenele
"""

import numpy as np 
#import scipy.sparse.linalg
from shapely.geometry import Point, Polygon

"""
Caracteristicas basicas da malha
"""

fileName = "fechadoCorpo.msh"
#elementSizeFactor = 0.05

import malhaModulo

malha = malhaModulo.Malha(fileName)

X = malha.X
Y = malha.Y
Lx = malha.Lx
Ly = malha.Ly
#nx = malha.nx
#ny = malha.ny
IEN = malha.IEN
IENBound = malha.IENBound
cc = malha.cc
ne = malha.ne
npoints = malha.npoints

"""
Pontos de contorno e criação da bval
"""

bval = np.zeros(npoints,dtype = 'float')
vx = np.zeros(npoints,dtype = 'float')
vy = np.zeros(npoints,dtype = 'float')

#boca
pontos_boca = 10
pboca = 18
yo = Y[pboca]
phi_o = 0
bval[pboca] = phi_o
uboca = 0.25
boca = []
boca.append(pboca)

for i in range (pontos_boca-1):
    bval[pboca-1] = phi_o + uboca*(Y[pboca-1]-yo)
    vx[pboca-1] = uboca
    boca.append(pboca-1)
    pboca-=1

#cantos da malha  
bval[0] = 0
bval[1] = 0
bval[2] = bval[boca[-1]]
bval[3] = bval[boca[-1]]

#contorno direito
ignore = []
ndir_inic = 62
ndir_fin = 97
i = ndir_inic
while i <= ndir_fin:
    ignore.append(i)
    i+=1

#contorno inferior
ninf_inic = 35
ninf_fin = 61
i = ninf_inic
inferior=[]
while i <= ninf_fin:
    bval[i] = 0
    inferior.append(i)
    i += 1
    
#contorno superior
nsup_inic = 98
nsup_fin = 126
i = nsup_inic
superior=[]
while i <= nsup_fin:
    bval[i] = bval[boca[-1]]
    superior.append(i)
    i += 1
    
#nos excedentes
exc = [[4,34],[127,157]]
exc_lista=[]
for i in range (len(exc)):
    ninic = exc[i][0]
    nfin = exc[i][1]
    j = ninic
    while j <= nfin:
        if j not in boca:
            bval[j] = 0
            exc_lista.append(j)
        j += 1

#nos acima da boca e abaixo da parte superior
nentre_inic= 4
nentre_fin = 10
i = nentre_inic
while i <= nentre_fin:
    bval[i] = bval[boca[-1]]
    i += 1
bval[127]= bval[boca[-1]]
bval[128]= bval[boca[-1]]

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

#geral
u_real = 50
U = u_real/uboca
x_real = 1.83
L = x_real/Ly
vo = 1.66*10**(-5)
rho_ar = 1.14
mi_ar = 1.90*10**(-5)
D_ar = 110*10**(-6)
rho_w = 993.51
tau_v = (rho_w*D_ar**2)/(18*mi_ar)
g = 9.81
dt = 0.01
Re =(U*L)/vo
iteracoes = 10

#partida da goticula
xg = 0.07205
yg = 1.6291
vxg = u_real
vyg = 0
dt_real = dt*L/U

"""
Solucao utilizando eq de transporte e de funcao corrente
"""

#matriz identidade
ident = np.identity(npoints)

xg_lista = []
yg_lista = []

for q in range (iteracoes):
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
    B2 = np.dot((M/dt - np.dot(vx_id,GX) - np.dot(vy_id,GY) - K/Re),wz) - np.dot(Kest,wz)
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
        if i not in ignore:
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
    vx[1] = 0
    vx[2] = 0
    vx[3] = 0
    
    vy[0] = 0
    vy[1] = 0
    vy[2] = 0
    vy[3] = 0
    
    for i in ignore:
        #contorno direito
        vy[i] = 0
        #vx[i] = ...
      
    for i in superior:
        #contorno superior
        vx[i] = 0
        vy[i] = 0

    for i in inferior:
        #contorno inferior
        vx[i] = 0 
        vy[i] = 0
    
    for i in boca:
        #contorno da boca
        vx[i] = uboca
        vy[i] = 0 

    for i in exc_lista:
        #nos excedentes
        vx[i] = 0
        vy[i] = 0
    
    #primeiro e ultimo no da boca
    #vx[boca[0]] = 0
    #vx[boca[-1]] = 0
    
    #Goticula
    p = Point(xg,yg)
    for e in range(0,ne):
        v1 = IEN[e,0]
        v2 = IEN[e,1]
        v3 = IEN[e,2]
            
        coords = [(X[v1],Y[v1]),(X[v2],Y[v2]),(X[v3],Y[v3])]
        poly = Polygon(coords)
        if p.within(poly):
            a1 = 1/(xg - X[v1])**2
            a2 = 1/(xg - X[v2])**2
            a3 = 1/(xg - X[v3])**2
                
            b1 = 1/(yg - Y[v1])**2
            b2 = 1/(yg - Y[v2])**2
            b3 = 1/(yg - Y[v3])**2
            
            vx_ar = U*(vx[v1]*a1 + vx[v2]*a2 + vx[v3]*a3)/(a1+a2+a3)
            vy_ar = U*(vy[v1]*b1 + vy[v2]*b2 + vy[v3]*b3)/(b1+b2+b3)
            
            v_ar = np.sqrt(vx_ar**2 + vy_ar**2)
            v_gota = np.sqrt(vxg**2 + vyg**2)
            
            Re_r = (rho_ar*abs(v_ar - v_gota)*D_ar)/mi_ar
            
            if Re_r < 1000:
                f = 1 + (Re_r**(2/3))/6
            else:
                f = 0.0183*Re_r
            
            vxg += (f*dt_real)/tau_v * (vx_ar -  vxg)
            vyg += (f*dt_real)/tau_v * (vy_ar - vyg) - g*dt_real
            
            xg += vxg*dt_real
            yg += vyg*dt_real
            
            xg_lista.append(xg)
            yg_lista.append(yg)
                       
            break

"""O que voce quer plotar?"""
#--Funcao corrente --> Psi
#--Vorticidade --> wz
#--Velocidade em x --> U*vx
#--Velocidade em y --> U*vy
objetoPlot = Psi
tituloPlot = "Função corrente"
salvarPlot = False
arquivoPlot = "fechadoCorpo.png"
malha.plotar(objetoPlot,tituloPlot)

"""Plot da goticula"""
tituloPlot = "Trajetória da gotícula (D="+str(np.round(D_ar*10**6,2))+" mícrons), t=0 até t="+str(np.round(len(xg_lista)*dt_real,4))+"s"
salvarPlot = False
arquivoPlot = "fechadoCorpoGoticula.png"
#malha.plotarGoticula(tituloPlot,xg_lista,yg_lista)




