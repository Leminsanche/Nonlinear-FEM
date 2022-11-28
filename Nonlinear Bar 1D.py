import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

#### Codigo para estructura de barras no lineal considerando un grado de libertad el
#### Los vectores se escriben como matriz columna [[x1],[x2]]

class Barra_nolineal:
    #X1 y X2 coordenada material, estado inicial
    def __init__(self,X1,X2,A,mu,landa):
        
        self.area = A 
        self.mu = mu
        self.landa = landa
        self.nodos = np.array([[X1],[X2]])
        self.largo = abs(X2)-abs(X1)
        self.D = self.landa + 2* self.mu
        
    def fun_N(self, xhi):
        N1 = (1-xhi) / 2
        N2 = (1+xhi) / 2
        N = np.array([N1,N2])
        return N
        
    def der_N_xhi(self):
        dN1_xhi = -1/2 
        dN2_xhi = 1/2
        dN_xhi = np.array([-1/2,1/2])
        return dN_xhi
    
    def der_X_xhi(self): #Estado deformado
        l = self.largo
        der = np.matmul(self.nodos.T,self.der_N_xhi().T) 
        # deberia ser igual a l/2
        return der[0]
    
    def der_N_X(self): #En X halamos de la coordenada material 
        dXdxhi = self.der_X_xhi()
        out = (dXdxhi**-1) * self.der_N_xhi()
        return out
    
    
    def der_x_xhi(self,x1,x2):
        nodos_x = np.array([[x1],[x2]])
        l = self.largo
        der = np.matmul(nodos_x.T,self.der_N_xhi().T) 
        # deberia ser igual a l/2
        return der[0]
    
    def der_N_x(self,x1,x2): #En x halamos de la coordenada deformada 
        dxdxhi = self.der_x_xhi(x1,x2)
        out = (dxdxhi**-1) * self.der_N_xhi()
        return out
    

    def F(self,x1,x2):
        suma = 0 #Numero de coordenadas y numero de nodos del elemento
        nodos_x = np.array([[x1],[x2]])
        it = 0
        for x in nodos_x:
            suma = suma + x*self.der_N_X()[it]
            it = it+1
        return suma[0]
        
    def sigma(self,x1,x2):
        #Material neo-hookeano
        
        mu = self.mu
        landa = self.landa
        
        F = self.F(x1,x2)
        b = F**2
        
        s = mu*(b - 1)
        
        return s
    
    def K_c(self,x1,x2):
        
        l = abs(x2 - x1)
        #Puntos de Gauss
        x = [-(1/3)**0.5 , (1/3)**0.5 ]
        #Pesos
        w = [1 , 1]
        
        
        B = self.der_N_x(x1,x2)
        #print(B)
        D = self.D
        A = self.area
        f = lambda x: np.outer(B.T,B) * D * A * l/2
        k  = np.zeros((2,2))
        
        it = 0
        for xi in x:
            k = k + f(xi) * w[it]
            it = it +1
            
        return k
        
   
