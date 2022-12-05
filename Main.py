import numpy as np 
import matplotlib.pyplot as plt
import pyvista as pv
from NonlinearBar import *

###########################################################################################################################
#################################################### Manejo de Datos ######################################################
###########################################################################################################################

Conectividad = np.loadtxt('conectividad2.txt')  ## [numero elemento, nodo inicio, nodo final]

conectividad = []

for i in  Conectividad:
    aux = [int(i[0]),int(i[1]),int(i[2])]
    conectividad.append(aux)


nodos = np.loadtxt('nodos2.txt')  ## [Coordenada]

cargas = np.loadtxt('carga2.txt') ##  [nodo, Carga en x, Carga en y]

Fix = np.loadtxt('fix2.txt')  ##[Nodo ,  COOX_value , COOY_value] ; value == 0 libre (default) ; value == 1 fijo 

fix = []

for i in Fix:
    auxX = (2*i[0]+1)*i[1]
    auxY = (2*i[0]+2)*i[2]
    if auxX != 0:
        fix.append(int(auxX-1))
    if auxY != 0:
        fix.append(int(auxY-1))
 
## MAIN code ###

#input

#conectividad = [[0, 0 , 1 ]]

#nodos = np.array([[ 0. ,  0. ],  #nodo 1: 0 ; 1
#                  [ 1. ,  0. ]]) #nodo 2: 2 ; 3

#fix = [0,1,3]
var = [i for i in range(len(nodos)*len(nodos[0]))]
dof = var.copy()

for i in var:
    if i in fix:
        dof.remove(i)

mu = 29.5e4/2 #10
landa = 29.5e4/4 #10 
A =  1


barras = []
for elem in conectividad:
    
    ni = elem[1] #numero de nodos
    nj = elem[2]
    
    Ni = nodos[ni] #COO del nodo
    Nj = nodos[nj] #COO del nodo
    
    barra = Barra_nolineal_rot(Ni,Nj,A,mu,landa)
    barras.append(barra)
#barra.T_int(Ni,Nj).T[0]



X = nodos  #Se definen las coordenadas materiales 

NF = 100 #numero de substeps de carga 

#n_fs = np.array([1, 10 , 0])  #Lista con la informacion de como se aplica la fuerza [nodo, fx, fy]
#n_f = n_fs[0]       #nodo donde aplico la fuerza

#F_ext  = np.linspace(0,n_fs[1], NF) #Aumento de cargas

tol = 1e-6  #Tolerancia
iter_max = 10 #Iteraciones maximas
F , R = np.zeros([len(nodos)*len(nodos[0])]) , np.zeros([len(nodos)*len(nodos[0])]) #Fuerza y reciduo inicial

x = X.copy() #Coordenadas espaciales iniciales
for i in range(NF):
    
    for j in cargas:
        n_f = int(j[0])
        n_fs = j[1:]
        F[2*n_f: 2*n_f+2] = F[2*n_f: 2*n_f+2] + n_fs/NF
        R[2*n_f: 2*n_f+2] = R[2*n_f: 2*n_f+2] - n_fs/NF
    
    #print(f'################################################ Carga = {F}#######################################')
    
    iterador = 0
    error = 100  #Error inicial
    while error > tol and iterador < iter_max:
        
        
        K = np.zeros((len(var),len(var)))
        for elem in conectividad:
        
            ni , nj , ne  = elem[1], elem[2] ,elem[0]  # numero de nodo i numero de nodo j y numero de elemento
            Ni = x[ni] #Coordenadas de nodo i
            Nj = x[nj] #Coordenadas de nodo j
            barra = barras[ne]
            k = barra.K(Ni,Nj)
            
            K[2*ni:2*ni+2,2*ni:2*ni+2] +=  k[:2 ,:2]
            K[2*nj:2*nj+2,2*nj:2*nj+2] +=  k[2: ,2:]
            K[2*ni:2*ni+2,2*nj:2*nj+2] +=  k[:2 ,2:]
            K[2*nj:2*nj+2,2*ni:2*ni+2] +=  k[2: ,:2]
            
            
        #print(K)
        
        aux = K[dof,:]
        K_red = aux[:,dof]
        
        u = np.linalg.solve(K_red,-R[dof].T)
        #print(u)
        
        x = x.reshape((-1))
        x[dof] = x[dof] + u
        #print(x)
        x = x.reshape((-1,2))
        
        T = np.zeros([len(nodos)*len(nodos[0])])
        for elem in conectividad:
            ni , nj , ne  = elem[1], elem[2] ,elem[0]
            Ni = x[ni] 
            Nj = x[nj]
            barra = barras[ne]
            t = barra.T_int(Ni,Nj).reshape((-1))
            T[2*ni:2*ni+2] += t[:2]
            T[2*nj:2*nj+2] += t[2:]
    
        R = T - F
        error = np.linalg.norm(R)
        
        iterador = iterador +1

        
       
###########################################################################################################################
#################################################### Grafico ##############################################################
###########################################################################################################################
barra = Conectividad
nodos = X
conectividad = 2*np.ones((len(barra),3))
Nodos = np.zeros((len(nodos),3))

Nodos[:,:2] = nodos

c  = 0
for  i in conectividad:
    i[1:] = barra[c][1:]
    c = c+1
    

Conectividad = []

for i in conectividad:
    aux = []
    for j in i:
        aux.append(int(j))
        
    Conectividad.append(aux)

poly = pv.PolyData()
poly.points = Nodos
poly.lines = Conectividad

poly["scalars"] = np.arange(poly.n_points)

tube = poly.tube(radius = 0.3)

tube.plot(background='k',text="Estructura de Barras Original", show_scalar_bar=False,color = 'g' , cpos = 'xy')


###########################################################################################################################
#################################################### Grafico ##############################################################
###########################################################################################################################
barra = Conectividad
nodos = x
conectividad = 2*np.ones((len(barra),3))
Nodos = np.zeros((len(nodos),3))

Nodos[:,:2] = nodos

c  = 0
for  i in conectividad:
    i[1:] = barra[c][1:]
    c = c+1
    

Conectividad = []

for i in conectividad:
    aux = []
    for j in i:
        aux.append(int(j))
        
    Conectividad.append(aux)

poly = pv.PolyData()
poly.points = Nodos
poly.lines = Conectividad

poly["scalars"] = np.arange(poly.n_points)

tube = poly.tube(radius = 0.3)

tube.plot(background='k',text="Estructura de Barras Deformada", show_scalar_bar=False,color = 'g' , cpos = 'xy')
