import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from Nonlinear import *
from pyvista import CellType
import time

def make_2d_indices(dof):
    iresult, jresult = [], []
    for i in dof:
        for j in dof:
            iresult.append(i)
            jresult.append(j)
    return iresult, jresult

################################################# Main code ############################################
largo_inicial = 2                   ############ LArgo incial #################
mat = Neohooke_comp(mu=15e10, clam=10e10)

nodos_init = np.loadtxt('malla3D/nodos.txt')
elementos = np.loadtxt('malla3D/elementos.txt')

Conectividad = []

for i in elementos:
    aux = []
    for j in i:
        aux.append(int(j-1))        
    Conectividad.append(aux)
    
conectividad = [i[1:] for i in Conectividad]
        
mesh = []    
for i in conectividad:
    hexahedro = Hex(material=mat, nodes=nodos_init, conn=i)
    mesh.append(hexahedro)

ind_movil  = []
ind_fijo  = []

for ind , i in enumerate(nodos_init):
    if i[1] == largo_inicial:
        #print(i , ind)           #################### Condicion de borde desplazamiento 
        ind_movil.append(ind)
    if i[1] == 0:                 ###################### COndicion de borde fix en el borde
        #print(i , ind)
        ind_fijo.append(ind)
        
ind_fix = ind_fijo +  ind_movil

var = [i for i in range(3*len(nodos_init))]

fix = []
for ind, i in enumerate(ind_fix):
    aux0 = ind*3
    aux1 = ind*3+1
    aux2 = ind*3+2
    #print(aux0, aux1,aux2)
    fix.append(aux0)
    fix.append(aux1)
    fix.append(aux2)
    
dof = []   
for i in var:
    if i not in fix:
        dof.append(i)

        
        
nnodes = len(nodos_init)

idof2d, jdof2d = make_2d_indices(dof)

# desplazmaiento impuesto 
traccion = 4.55  ############################ Traccion 

steps = 3
dt = traccion / steps

desp_imp =  largo_inicial  

x = nodos_init.copy()
iter_max = 10
tol = 10e-4

inicio = time.time()
for i in range(steps):
    print(f'######################### Paso {i+1}/{steps} ################################################')

    desp_imp = desp_imp + dt

    print('Desplazamiento impuesto', desp_imp - largo_inicial, )
    x_f = x
    for ind, i in enumerate(nodos_init):
        if ind in ind_movil:
            x_f[ind][1] = desp_imp


    x = x_f

    #print('equis\n',x)
    #print('equis\n',x_f)
    iteraciones = 0
    error = 1000
    while error > tol and iteraciones <= iter_max:
        t_new = np.zeros((nnodes, 3))
        tan = np.zeros((nnodes, 3, nnodes, 3))

        for ielem, elem in enumerate(mesh):
            t_new[elem.conn, :] += elem.t_int(x)
            iconn, jconn = make_2d_indices(elem.conn)
            count = [i for i in range(len(elem.conn))]
            iiconn, jjconn = make_2d_indices(count)
            tan[iconn, :, jconn, :] += elem.k(x)[iiconn, :, jjconn, :]

        t_new = t_new.flatten()[dof]
        tan = tan.reshape((nnodes*3, nnodes*3))[idof2d, jdof2d].reshape((len(dof), len(dof)))
        d_u = np.linalg.solve(tan, -t_new)
        error = np.linalg.norm(d_u)
        #print(error)
        x.reshape((-1,))[dof] += d_u

        iteraciones = iteraciones + 1 

        #if error <= tol:
        #    print('Criterio de convergencia')
    #print('du:\n', d_u)
        #print('x:\n', x)

    xf = x
#print(nodos_init)
#print(elementos)
#print('nodos init\n',nodos_init)
fin = time.time()
print(fin-inicio)

######################################### Graficos ###########################################################
cells = np.array(Conectividad)
points = nodos_init
points2 = x
grid = pv.UnstructuredGrid({CellType.HEXAHEDRON: cells.reshape([-1, 9])[:, 1:]}, points)
grid2 = pv.UnstructuredGrid({CellType.HEXAHEDRON: cells.reshape([-1, 9])[:, 1:]}, points2)
#grid2['desplazamientos'] = [np.linalg.norm(i) for i in x]
grid2['Desplazamientos Y FOM'] = [i[1] for i in x]
#grid2.add_field_data([np.linalg.norm(i) for i in x],'Desplazamientos')
#grid2.set_active_scalars('Desplazamientos Totales')
grid.plot(show_edges=True)
grid2.plot(show_edges=True, cmap = 'turbo')
