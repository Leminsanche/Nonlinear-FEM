import numpy as np
import matplotlib.pyplot as plt
from NonlinearBar import *

## MAIN code ###

#input

conectividad = [[0,1]]
nodos = np.array([[0.], [2.]])

mu = 10
landa = 10
A = 1


barras = []
for i in conectividad:
    n_i = i[0]
    n_j = i[1]
    
    Xi = nodos[n_i]
    Xj = nodos[n_j]
    
    barra = Barra_nolineal(Xi,Xj,A,mu,landa)
    barras.append(barra)

X = nodos 

dF = 0.01
carga = 10
F_ext  = np.arange(0,carga+dF, dF)

tol = 1e-6
iter_max = 100
error = 1000
F , R = 0 , 0
x = X.copy()

for f in F_ext:
    F = f
    #print('Cargaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', f)
    R = R-dF
    iterador = 0
    error = 100
    
    while error > tol and iterador < iter_max:
        
        barra = barras[0]
        x1 , x2 = x[0] , x[1]
        k = barra.K_c(x1,x2)
           
        K = k[1,1]
        
        
        u = -R/K
        x[1] = x[1] + u
        T = barra.T(x1,x2) 
        T = T[1]
        
        R = T-F
        
        error = abs(R)
        iterador = iterador +1
