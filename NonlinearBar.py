import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

#### Codigo para estructura de barras no lineal considerando un grado de libertad
#### Con elementos isoparametricos
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
        
    def T(self, x1 ,x2): ##Fuerza interna
        
        nodos_x = np.array([[x1],[x2]])
        l = abs(x2 - x1)
        
        dNdx = self.der_N_x(x1,x2)
        s = self.sigma(x1,x2) #Una constante en este caso
        A = self.area
        
        #Puntos de Gauss
        x_gauss = [-(1/3)**0.5 , (1/3)**0.5 ]

        #Pesos
        w = [1 , 1]
        
        T = []
        
        it = 0
        for x in nodos_x:
            integral = 0
            dNadx = dNdx[it]
            jt = 0
            for xi in x_gauss:
                wi = w[jt]
                integral = integral + A*(l/2)*s*wi*dNadx
                
                jt = jt +  1
                
            T.append(integral)
                
            it = it + 1
            
        return T
    
    
## PAra elementos rotados
class Barra_nolinal_rot(Barra_nolineal):
    
    def __init__(self,X1,Y1,X2,Y2,A = 1,mu = 0.25,landa = 0.5):
        
        self.largo = ( (X2-X1)**2 + (Y2-Y1)**2 )**0.5
        self.cos = (X2-X1)/self.largo
        self.sin = (Y2-Y1)/self.largo
        self.L = np.array([self.cos, self.sin])
        self.R = np.outer(self.L.T,self.L)
        
        self.X1 = np.matmul(self.L.T, np.array([[X1],[Y1]]) )
        self.X2 = np.matmul(self.L.T, np.array([[X2],[Y2]]) )
        
        self.area = A 
        self.mu = mu
        self.landa = landa
        self.nodos = np.array([[self.X1[0]],[self.X2[0]]])
        self.D = self.landa + 2* self.mu
    
    def K(self, x1_n,y1_n,x2_n,y2_n):
        R = self.R
        L = self.L
        x1 = np.matmul(self.L.T, np.array([[x1_n],[y1_n]]) )
        x2 = np.matmul(self.L.T, np.array([[x2_n],[y2_n]]) )
        
        out = np.matmul(R,self.K_c(x1,x2))
        
        return out
    
    
    def T_int(self, x1_n,y1_n,x2_n,y2_n):
        R = self.R
        L = self.L
        x1 = np.matmul(self.L.T, np.array([[x1_n],[y1_n]]) )
        x2 = np.matmul(self.L.T, np.array([[x2_n],[y2_n]]) )
        
        out = np.outer(self.T(x1,x2).T,L)
        
        return out
    
