"""
@author : Nicolás Sánchez
"""


import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

#### Codigo para estructura de barras no lineal considerando un grado de libertad
#### Con elementos isoparametricos
#### Los vectores se escriben como matriz columna [[x1],[x2]]
class Barra_nolineal:
    #X1 y X2 coordenada material, estado inicial
    def __init__(self,X1,X2,A = 1,mu = 0.25,landa = 0.5):
        
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
            
        return np.array(T)

    
    
class Barra_nolineal_rot(Barra_nolineal):
    
    def __init__(self,n1,n2,A = 1,mu = 0.25,landa = 0.5):
        
        X1 = n1[0]
        Y1 = n1[1]
        X2 = n2[0]
        Y2 = n2[1]
        
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
    
    def K(self,n1,n2):
        
        x1_n = n1[0]
        y1_n = n1[1]
        x2_n = n2[0]
        y2_n = n2[1]
        
        R = self.R
        L = self.L
        x1 = np.matmul(self.L.T, np.array([[x1_n],[y1_n]]) )
        x2 = np.matmul(self.L.T, np.array([[x2_n],[y2_n]]) )
        
        Ka = self.K_c(x1,x2) #Matriz de rigidez en el sistema local
        Kout = np.zeros([4,4]) #Matriz de rigidez sistea global

        Kout[:2,:2] = R * Ka[0,0]
        Kout[2:,2:] = R * Ka[1,1]
        Kout[:2,2:] = R * Ka[1,0]
        Kout[2:,:2] = R * Ka[0,1]
        
        return Kout
    
    
    def T_int(self, n1,n2):
        x1_n = n1[0]
        y1_n = n1[1]
        x2_n = n2[0]
        y2_n = n2[1]
        
        R = self.R
        L = self.L
        x1 = np.matmul(self.L.T, np.array([[x1_n],[y1_n]]) )
        x2 = np.matmul(self.L.T, np.array([[x2_n],[y2_n]]) )
        
        out = np.outer(self.T(x1,x2).T,L)
        
        return out.reshape((-1,1))  


### As i did a bar code for nonlinear strctures, i created a general code tu create truss structurs like this one:
# 
# ---------------- --------> F
#            --- l
#          ---   l
#        ---     l
#      ---       l
#    ---         l
#  ---           l
# ---------------l --------> F


def Estructura(l , n , ang , mag):
    #Largo de las barras 0.1
    n = l
    l  = n

    nodos = np.zeros((2*n+2,2))

    # 0 0  0
    # 1 0.1  0
    # 2 0.1  0.1
    # 3 0  0.1

    it = 0

    # Nodos
    nodos   = np.zeros((2*n+2,2))

    it = 0
    for ind,arr in enumerate(nodos):
        nodos[ind] = [ind*l,0]
        it = it +1 
        if it == n+1:
            break

    nodos[n+1: ,1] = l 
    nodos[n+1: ,0] = [i for i in reversed(list(nodos[:n+1 ,0]))]

    # Conectividad

    # 0 0 1
    # 1 1 2
    # 2 2 3
    # 3 0 2


    Conectividad = []

    for i in range(n):
        c = 0

        if c == 0:
            aux = [i , i , i+1]
            Conectividad.append(aux)
            c = c+1

        if c ==1:
            aux = [i+1 , 2*n - i , i+1]
            Conectividad.append(aux)
            c= c+1
        if c ==2:
            aux = [i+1 , i , 2*n -i]

            Conectividad.append(aux)
            c= c+1
        if c ==3:
            aux = [i+1  , 2*n+1 -i , 2*n - i]
            Conectividad.append(aux)
            c= c+1


    angulo  = ang  #angulo = float(input('angulo respecto la horizontal: '))
    magnitud = mag #magnitud = float(input('Maginutd de la fuerza: '))
    nodo = n+1
    carga = np.array([[nodo   , magnitud * np.cos(angulo*(np.pi/180)) , magnitud * np.sin(angulo*(np.pi/180)) ],
                      [nodo-1 , magnitud * np.cos(angulo*(np.pi/180)) , magnitud * np.sin(angulo*(np.pi/180))]])
    #magnitud * np.cos(angulo*(np.pi/180)) , magnitud * np.sin(angulo*(np.pi/180))
    fix = np.array([[0 , 1 , 1],
                    [2*n+1 , 1 , 1]])

    Conect = []

    for ind , i in  enumerate(Conectividad):
        aux = [ind , i[1] , i[2]]
        Conect.append(aux)





    np.savetxt('data/carga3.txt',carga)
    np.savetxt('data/nodos3.txt',nodos)
    np.savetxt('data/conectividad3.txt',Conect)
    np.savetxt('data/fix3.txt',fix)
    return Conectividad, nodos


######################################### Codigo Elemento Hexaedrico #######################################################################

class Neohooke_comp:
    iden = np.eye(3)
    def __init__(self, mu, clam):
        self.mu = mu
        self.clam = clam
    
    def __call__(self, F):
        b = np.matmul(F,F.T)
        j = np.linalg.det(F)
        return (self.mu/j)*(b - self.iden) + (self.clam/j)*np.log(j)*self.iden

    def tan(self, F):
        iden = self.iden
        iden4 = np.einsum('ij,kl', iden, iden)
        iden4Sym = (np.einsum('ik,jl', iden, iden) + np.einsum('il,jk', iden, iden))/2
        j = np.linalg.det(F)
        return self.clam/j*iden4 + 2/j*(self.mu - self.clam*np.log(j))*iden4Sym



class Hex:
    def __init__(self, material, nodes, conn):
        self.conn = conn
        self.mat = material
        self.nodes = nodes[conn]
        self.nnodes = 8

    def _get_nodes(self, x):
        return x[self.conn,:]


    def N_func(self, xi):
        xi0 = xi[0]
        xi1 = xi[1]
        xi2 = xi[2]
        N1 = (1.0 - xi0)*(1.0 - xi1)*(1.0 - xi2)/8.0
        N2 = (1.0 + xi0)*(1.0 - xi1)*(1.0 - xi2)/8.0
        N3 = (1.0 + xi0)*(1.0 + xi1)*(1.0 - xi2)/8.0
        N4 = (1.0 - xi0)*(1.0 + xi1)*(1.0 - xi2)/8.0
        N5 = (1.0 - xi0)*(1.0 - xi1)*(1.0 + xi2)/8.0
        N6 = (1.0 + xi0)*(1.0 - xi1)*(1.0 + xi2)/8.0
        N7 = (1.0 + xi0)*(1.0 + xi1)*(1.0 + xi2)/8.0
        N8 = (1.0 - xi0)*(1.0 + xi1)*(1.0 + xi2)/8.0
        return np.array([N1, N2, N3, N4, N5, N6, N7, N8])

    def der_N_fun(self, xi):
        xi0 = xi[0]
        xi1 = xi[1]
        xi2 = xi[2]
        return np.array([[  -(1.0 - xi1)*(1.0 - xi2)/8.0, -(1.0 - xi0)*(1.0 - xi2)/8.0, -(1.0 - xi0)*(1.0 - xi1)/8.0],
                         [   (1.0 - xi1)*(1.0 - xi2)/8.0, -(1.0 + xi0)*(1.0 - xi2)/8.0, -(1.0 + xi0)*(1.0 - xi1)/8.0],
                         [   (1.0 + xi1)*(1.0 - xi2)/8.0,  (1.0 + xi0)*(1.0 - xi2)/8.0, -(1.0 + xi0)*(1.0 + xi1)/8.0],
                         [  -(1.0 + xi1)*(1.0 - xi2)/8.0,  (1.0 - xi0)*(1.0 - xi2)/8.0, -(1.0 - xi0)*(1.0 + xi1)/8.0],
                         [  -(1.0 - xi1)*(1.0 + xi2)/8.0, -(1.0 - xi0)*(1.0 + xi2)/8.0,  (1.0 - xi0)*(1.0 - xi1)/8.0],
                         [   (1.0 - xi1)*(1.0 + xi2)/8.0, -(1.0 + xi0)*(1.0 + xi2)/8.0,  (1.0 + xi0)*(1.0 - xi1)/8.0],
                         [   (1.0 + xi1)*(1.0 + xi2)/8.0,  (1.0 + xi0)*(1.0 + xi2)/8.0,  (1.0 + xi0)*(1.0 + xi1)/8.0],
                         [  -(1.0 + xi1)*(1.0 + xi2)/8.0,  (1.0 - xi0)*(1.0 + xi2)/8.0,  (1.0 - xi0)*(1.0 + xi1)/8.0],
                         ]) 

    def der_X_xi(self, xi):  # 7.6b
        return np.einsum('ai,aj', self.nodes, self.der_N_fun(xi))

    def der_N_X(self, xi):  # 7.6b
        inv_der_X_xi = np.linalg.inv(self.der_X_xi(xi).T)
        return np.matmul(inv_der_X_xi,self.der_N_fun(xi).T).T

    def der_x_xi(self, x, xi):  # 7.11a
        return np.einsum('ai,aj', x, self.der_N_fun(xi))

    def der_N_x(self, x, xi):  # 7.11b
        inv_der_x_xi = np.linalg.inv(self.der_x_xi(x, xi).T)
        
        return np.matmul(inv_der_x_xi,self.der_N_fun(xi).T).T

    def f(self, x, xi):  # gradiente de deformacion -- 7.5
        return np.einsum('ai,aj->ij', x, self.der_N_X(xi))

    def stress(self, x, xi): # from mat
        F = self.f(x, xi)
        return self.mat(F)

    def t_int(self, x_complete): # 7.15b
        x = self._get_nodes(x_complete)
        gauss_points = np.array([[-1/3**0.5, -1/3**0.5, -1/3**0.5],
                                 [ 1/3**0.5, -1/3**0.5, -1/3**0.5],
                                 [ 1/3**0.5,  1/3**0.5, -1/3**0.5],
                                 [-1/3**0.5,  1/3**0.5, -1/3**0.5],
                                 [-1/3**0.5, -1/3**0.5,  1/3**0.5],
                                 [ 1/3**0.5, -1/3**0.5,  1/3**0.5],
                                 [ 1/3**0.5,  1/3**0.5,  1/3**0.5],
                                 [-1/3**0.5,  1/3**0.5,  1/3**0.5],
                                 ])

        t_int = np.zeros((self.nnodes, 3))
        for gp in gauss_points:
            F = self.f(x, gp)
            sig = self.mat(F)
            temp = np.einsum('ij,aj->ai', sig, self.der_N_x(x, gp))
            temp *= np.linalg.det(self.der_x_xi(x, gp))
            t_int += temp

        return t_int
    
    def k_sig(self, x):
        gauss_points = np.array([[-1/3**0.5, -1/3**0.5, -1/3**0.5],
                                 [ 1/3**0.5, -1/3**0.5, -1/3**0.5],
                                 [ 1/3**0.5,  1/3**0.5, -1/3**0.5],
                                 [-1/3**0.5,  1/3**0.5, -1/3**0.5],
                                 [-1/3**0.5, -1/3**0.5,  1/3**0.5],
                                 [ 1/3**0.5, -1/3**0.5,  1/3**0.5],
                                 [ 1/3**0.5,  1/3**0.5,  1/3**0.5],
                                 [-1/3**0.5,  1/3**0.5,  1/3**0.5],
                                 ])
        iden = np.eye(3)

        k_sig = np.zeros((8, 3, 8, 3))
        for gp in gauss_points:
            F = self.f(x, gp)
            temp = np.einsum('ak, kl, bl, ij -> aibj', self.der_N_x(x, gp), self.mat(F), self.der_N_x(x, gp), iden)
            temp *= np.linalg.det(self.der_x_xi(x, gp))
            k_sig += temp

        return k_sig


    def k_const(self, x): # 7.35
        gauss_points = np.array([[-1/3**0.5, -1/3**0.5, -1/3**0.5],
                                 [ 1/3**0.5, -1/3**0.5, -1/3**0.5],
                                 [ 1/3**0.5,  1/3**0.5, -1/3**0.5],
                                 [-1/3**0.5,  1/3**0.5, -1/3**0.5],
                                 [-1/3**0.5, -1/3**0.5,  1/3**0.5],
                                 [ 1/3**0.5, -1/3**0.5,  1/3**0.5],
                                 [ 1/3**0.5,  1/3**0.5,  1/3**0.5],
                                 [-1/3**0.5,  1/3**0.5,  1/3**0.5],
                                 ])

        k_int = np.zeros((8, 3, 8, 3))
        for gp in gauss_points:
            F = self.f(x, gp)
            temp = np.einsum('ak, ikjl, bl -> aibj', self.der_N_x(x, gp), self.mat.tan(F), self.der_N_x(x, gp))
            temp *= np.linalg.det(self.der_x_xi(x, gp))
            k_int += temp
        return k_int

    def k(self, x):
        return self.k_const(x) + self.k_sig(x)

    def k(self, x_complete): # 7.35
        x = self._get_nodes(x_complete)
        return self.k_const(x) + self.k_sig(x)

mat = Neohooke_comp(mu=150, clam=1000)

nodes_init = np.array([[0.0, 0.0, 0.0],  # 1
                       [1.0, 0.0, 0.0],  # 2
                       [1.0, 1.0, 0.0],  # 3
                       [0.0, 1.0, 0.0],  # 4
                       [0.0, 0.0, 1.0],  # 5
                       [1.0, 0.0, 1.0],  # 6
                       [1.0, 1.0, 1.0],  # 7
                       [0.0, 1.0, 1.0],  # 8
                       [2.0, 0.0, 1.0],  # 9
                       [2.0, 1.0, 1.0],  # 10
                       [2.0, 0.0, 0.0],  # 11
                       [2.0, 1.0, 0.0],  # 12
                       ])

def make_2d_indices(dof):
    iresult, jresult = [], []
    for i in dof:
        for j in dof:
            iresult.append(i)
            jresult.append(j)
    return iresult, jresult 
