#!/usr/bin/python
# -*- coding: utf-8 -*-
import shlex
import copy
import argparse



#*************************************************************************************************************************
#********************************   INTER   ******************************************************************************
#*************************************************************************************************************************

actions={
         'ElemsMalla': 1,
         'Restricciones': 2,
         'NodosElemFisicos': 3,
         'Cargas': 4
        } 

def findPhysicalId(physicalsNames,name,iline):
  try:
    return int(name), None, None
  except ValueError as verr:
    namePhysical = name
    founded = False
    for physicalName in physicalsNames:
      if namePhysical == physicalName.name:
        physical = physicalName.id
        dimension =  physicalName.dimension
        founded = True
    if not founded:
      raise ValueError ('El nombre "'+namePhysical+'" no existe, error en la línea '+str(iline))
    return physical,name,dimension

class ElemMallaDat:
  def __init__(self,iline,line,physicalsNames):
    data = shlex.split(line)
#    print line
    if len(data) != 2:
      raise ValueError ("La linea de elementos debe ser de dos valores 'set Fisico' y 'set de vulcan' y se pasa "+str(len(data))+" argumentos, Error en la linea "+str(iline)) 
    self.physical, self.namePhysical, self.dimension = findPhysicalId(physicalsNames, data[0], iline)
    self.setVulcan = int(data[1]) 
    self.__readed(iline)
  def __str__(self):
    toReturn = "['"+self.namePhysical+"', "+str(self.physical)+", "+str(self.setVulcan)+"]"
    return toReturn
  def __readed(self,iline):
    print "$Elem_Malla, Nombre fisico: '", self.namePhysical, "' Id físico: ", self.physical, " Set de Vulcan: ", self.setVulcan, " Línea: ",iline 
       

class ElemsMallaDat:
  def __init__(self):
    self.setVulcan = None
    self.problemDimension = None
    self.isReadedDimension = False
    self.data = []
    self.iItemIterator = 0
  def read(self,iline,line,action,physicalsNames):
    if (action == None) and line.strip().split()[0].lower().startswith("$elem_malla"):  # Como la accion es None, no está en ningun bloque particular, por lo que busca la carta de lectura de elementos
      action = actions['ElemsMalla']
      print "\n$Elem_Malla, Línea :",iline
    elif (action == actions['ElemsMalla']):                                      # Como la accion es ElemsMalla, está en el bloque de elemsMalla
      if line.strip().split()[0].lower().startswith("$fin_elem_malla"):
        action = None
        print "$Fin_Elem_Malla, Línea :",iline
      elif not self.isReadedDimension:
        data = line.strip()
        if data == '3D' or data == '3d':
          self.problemDimension = 3
          self.isReadedDimension = True
        elif data == '2D' or data == '2d':
          self.problemDimension = 2
          self.isReadedDimension = True
        else:
          raise ValueError("No se acepta el tipo "+data+", error en la linea "+str(iline))
        print "$Elem_Malla, Dimensión del problema: ", self.problemDimension, " Línea: ", iline
      else:                                                                 #Fuera de la definición de si es 3D o 2D          
        self.data.append(ElemMallaDat(iline,line,physicalsNames))     
    return action    
  def __iter__(self):
    if len(self.data) == 0:
      raise StopIteration
    else:
      self.iItemIterator = -1
      return self
    return
  def next(self):
    self.iItemIterator += 1
    if len(self.data) > self.iItemIterator :
      return self.data[self.iItemIterator]
    else:
      raise StopIteration
    
    

class Constraint:
  def __init__(self,iline,line,physicalsNames,problemDimension):
    self.physical = None
    self.additive = None # para ver si es exacta o aditiva 
    self.constraintxVulcan = None
    self.constraintyVulcan = None
    self.constraintzVulcan = None
    self.funcionVulcan = None
    self.valxVulcan = None
    self.valyVulcan = None
    self.valzVulcan = None
    data = shlex.split(line)
    if not ((len(data) == 7 and problemDimension == 3) or (len(data) == 6 and problemDimension == 2)):
      raise ValueError ("La linea de restricciones debe tener siete(3D) o seis(2D) valores: 'Nombre Fisico (GMSH)', 'Exacta o Aditiva', 'Grados Impuestos', 'FuncionVulcan' y 'Valores x y z de la función' y solo se dieron "+str(len(data))+" argumentos")

    self.physical, self.namePhysical, self.dimension = findPhysicalId(physicalsNames, data[0], iline)

    if data[1] == "Exacto":
      self.additive = False
    elif data[1] == "Aditivo":
      self.additive = True
    else:
      raise ValueError ("El segundo argumento de las restricciones tiene que ser 'Exacto' o 'Aditivo', no puede ser '"+data[1]+ "', Error en la linea " + str(iline))
    
    gradosImpuestos = data[2]
    self.setDegree(gradosImpuestos,problemDimension)
    
    self.functionVulcan = int(data[3])

    self.valxVulcan = float(data[4])
    self.valyVulcan = float(data[5])
    if problemDimension == 3:
      self.valzVulcan = float(data[6])
    else:
      self.valzVulcan = None
    self.__readed(iline)

  def setDegree(self,gradosImpuestos,problemDimension):
    if (len(gradosImpuestos) != problemDimension):
      raise ValueError ('La cantidad de grados impuestos deben ser igual a la cantidad de dimensiones del problema')
    
    if int(gradosImpuestos[0]) == 0:
      self.constraintxVulcan = False
    elif int(gradosImpuestos[0]) == 1:
      self.constraintxVulcan = True
    else:
      raise ValueError ('Los grados impuestos solo pueden ser 1 o 0 no pueden ser ' + gradosImpuestos + ", Error en la linea " + iline)
    
    if int(gradosImpuestos[1]) == 0:
      self.constraintyVulcan = False
    elif int(gradosImpuestos[1]) == 1:
      self.constraintyVulcan = True
    else:
      raise ValueError ('Los grados impuestos solo pueden ser 1 o 0 no pueden ser ' + gradosImpuestos + ", Error en la linea " + iline)
    if (problemDimension == 3):
      if int(gradosImpuestos[2]) == 0:
        self.constraintzVulcan = False
      elif int(gradosImpuestos[2]) == 1:
        self.constraintzVulcan = True
      else:
        raise ValueError ('Los grados impuestos solo pueden ser 1 o 0 no pueden ser ' + gradosImpuestos + ", Error en la linea " + iline)
  def __readed(self,iline):
    if self.constraintzVulcan != None:
      print "$Restricciones, Nombre fisico: '", self.namePhysical, "' Id físico: ", self.physical, "Aditivo: ", self.additive ," Restricciones Impuestas: ", int(self.constraintxVulcan), int(self.constraintyVulcan), int(self.constraintzVulcan), " Función: ", self.functionVulcan, "Valores: ", self.valxVulcan, self.valyVulcan, self.valzVulcan, " Línea: ",iline 
    else:
      print "$Restricciones, Nombre fisico: '", self.namePhysical, "' Id físico: ", self.physical, "Aditivo: ", self.additive ," Restricciones Impuestas: ", int(self.constraintxVulcan), int(self.constraintyVulcan), " Función: ", self.functionVulcan, "Valores: ", self.valxVulcan, self.valyVulcan, " Línea: ",iline

 
class Constraints:
  def __init__(self):
    self.data = []
  def read(self,iline,line,action,physicalsNames,problemDimension):
    if (action == None) and line.strip().split()[0].lower().startswith("$restricciones"):  # Como la accion es None, no está en ningun bloque particular, por lo que busca la carta de lectura de elementos
      action = actions['Restricciones']
      self.data.append([])
      print "\n$Restricciones, Línea: ",iline
    elif (action == actions['Restricciones']):                                     # Como la accion es ElemsMalla, está en el bloque de elemsMalla
      if line.strip().split()[0].lower().startswith("$fin_restricciones"):
        action = None
        print "$Fin_Restricciones, Línea: ",iline
      else:                                                                 #Fuera de la definición de si es 3D o 2D          
        self.data[-1].append(Constraint(iline,line,physicalsNames,problemDimension))     
    return action    
  def __iter__(self):
    if len(self.data) == 0:
      raise StopIteration
    else:
      self.iItemIterator = -1
      return self
    return
  def next(self):
    self.iItemIterator += 1
    if len(self.data) > self.iItemIterator :
      return self.data[self.iItemIterator]
    else:
      raise StopIteration
  def __len__(self):
    return len(self.data)

  def __getitem__(self,idata):
    return self.data[idata]

class Load:
  def __init__(self,iline,line,physicalsNames,problemDimension):
    self.physical = None
    self.functionVulcan = None
    self.valxVulcan = None
    self.valyVulcan = None
    self.valzVulcan = None
    self.type = None
    data = shlex.split(line)
    if not (
             ( data[0] == "GRAVITY"  or data[0] == "FACE_LOCAL" or data[0] == "FACE_GLOBAL" or data[0] == "POINT_LOAD")  and  
             ((len(data) == 6 and problemDimension == 3) or (len(data) == 5 and problemDimension == 2)) 
           ):
      raise ValueError ("La linea de carga debe tener como primer argumento 'GRAVITY', 'FACE_LOCAL', 'FACE_GLOBAL' o 'POINT_LOAD' y la cantidad total de argumentos deben ser seis(3D) o cinco(2D) valores: 'Tipo de Carga', 'Elementos Fisicos', 'FuncionVulcan' y 'Valores x y z de la función' y solo se dieron "+str(len(data))+" argumentos, en la linea "+str(iline)+"\n"+ data[0])

    self.type = data[0]
    if self.type != "GRAVITY":
      self.physical, self.namePhysical, self.dimension = findPhysicalId(physicalsNames, data[1], iline)
    else:
      self.gravity = float(data[1])
    self.functionVulcan = int(data[2])
    self.valxVulcan = float(data[3])
    self.valyVulcan = float(data[4])
    if problemDimension == 3:
      self.valzVulcan = float(data[5])
    else:
      self.valzVulcan = None   
    self.__readed(iline)
  def __readed(self,iline):
    if self.type != "GRAVITY":
      if self.valzVulcan != None:
        print "$Cargas, Tipo: ", self.type, " Nombre fisico: '", self.namePhysical, "' Id físico: ", self.physical, " Función: ", self.functionVulcan, "Valores: ", self.valxVulcan, self.valyVulcan, self.valzVulcan, " Línea: ",iline   
      else:
        print "$Cargas, Tipo: ", self.type, " Nombre fisico: '", self.namePhysical, "' Id físico: ", self.physical, " Función: ", self.functionVulcan, "Valores: ", self.valxVulcan, self.valyVulcan, " Línea: ",iline   
    else:
      print "$Cargas, Tipo: ", self.type,  "Valor Gravedad: ", self.gravity, " Función: ", self.functionVulcan, "Valores: ", self.valxVulcan, self.valyVulcan, self.valzVulcan, " Línea: ",iline
     

class Loads:
  def __init__(self):
    self.data = []
  def read(self,iline,line,action,physicalsNames,problemDimension):
    if (action == None) and line.strip().split()[0].lower().startswith("$cargas"):  # Como la accion es None, no está en ningun bloque particular, por lo que busca la carta de lectura de elementos
      action = actions['Cargas']
      self.data.append([])
      print "\n$Cargas, Línea: ",iline
    elif (action == actions['Cargas']):                                     # Como la accion es ElemsMalla, está en el bloque de elemsMalla
      if line.strip().split()[0].lower().startswith("$fin_cargas"):
        action = None
        print "$Fin_Cargas, Línea: ",iline
      else:                                                                 #Fuera de la definición de si es 3D o 2D          
        self.data[-1].append(Load(iline,line,physicalsNames,problemDimension))     
    return action    
  def __iter__(self):
    if len(self.data) == 0:
      raise StopIteration
    else:
      self.iItemIterator = -1
      return self
    return
  def next(self):
    self.iItemIterator += 1
    if len(self.data) > self.iItemIterator :
      return self.data[self.iItemIterator]
    else:
      raise StopIteration
  def __len__(self):
    return len(self.data)
  def __getitem__(self,idata):
    return self.data[idata]


class ElementoAImprimir:
  def __init__(self,iline,line,physicalsNames):
    self.physical = None
    self.direction = None
    self.file = None
    data = shlex.split(line)
    if (len(data) != 3 ):
      raise ValueError ("La linea de Nodos de elementos debe tener 3 valores: 'Nombre Fisico (GMSH)', 'direccion (x:1, y:2, z:3)', 'Nombre de archivo de salida' y solo se dieron "+str(len(data))+" argumentos, error en la linea "+str(iline))

    self.physical, self.namePhysical, dummy = findPhysicalId(physicalsNames, data[0], iline)

    self.direction=int(data[1])
    self.file=data[2]
    self.__readed(iline)
  def __readed(self,iline):
    print "$Nodos_Elem_Fisicos, Nombre fisico: '", self.namePhysical, "' Id físico: ", self.physical, " direccion: ", self.direction, " Archivo: ", self.file, " Línea: ",iline   


class ElementosAImprimir:
  def __init__(self):
    self.data = []
  def read(self,iline,line,action,physicalsNames):
    if (action == None) and line.strip().split()[0].lower().startswith("$nodos_elem_fisicos"):  # Como la accion es None, no está en ningun bloque particular, por lo que busca la carta de lectura de elementos
      action = actions['NodosElemFisicos']
      print "\n$Nodos_Elem_Fisicos, Línea: ",iline
    elif (action == actions['NodosElemFisicos']):                                     # Como la accion es ElemsMalla, está en el bloque de elemsMalla
      if line.strip().split()[0].lower().startswith("$fin_nodos_elem_fisicos"):
        action = None
        print "$Fin_Nodos_Elem_Fisicos, Línea: ",iline
      else:                                                                 #Fuera de la definición de si es 3D o 2D          
        self.data.append(ElementoAImprimir(iline,line,physicalsNames))     
    return action    
  def __iter__(self):
    if len(self.data) == 0:
      raise StopIteration
    else:
      self.iItemIterator = -1
      return self
    return
  def next(self):
    self.iItemIterator += 1
    if len(self.data) > self.iItemIterator :
      return self.data[self.iItemIterator]
    else:
      raise StopIteration
  def write(self,datosgmsh,orderOldToNew):
    for lineToPrint in self.data:
      nodes, dummy = datosgmsh.getPhysicalNodes([lineToPrint.physical])
      for inode,node in enumerate(nodes): #Pasando de nodos antiguos a nodos nuevos
        nodes[inode] = orderOldToNew[node]
      if len(nodes) > 0:
        with open(lineToPrint.file,'w') as f:
          f.write(str(len(nodes)) + "\n")
          for number in nodes:
            f.write("  " + str(number) + " " + str(lineToPrint.direction) + "\n") 


class ElementInter:
  def __init__(self, number, nodes, setElement, physicalId):
    self.number = number
    self.set = setElement
    self.nodes = nodes
    self.nnodes = len(nodes)
    self.physicalId = physicalId
  def __iter__(self):
    if len(self.nodes) == 0:
      raise StopIteration
    else:
      self.iItemIterator = -1
      return self
    return
  def next(self):
    self.iItemIterator += 1
    if len(self.nodes) > self.iItemIterator :
      return self.nodes[self.iItemIterator]
    else:
      raise StopIteration


class NodeInter:
  def __init__(self, number, positions):
    self.number = number
    self.positions = positions

class interGmshVulcan:
  def __init__(self,filename,datosgmsh):
    self.nodes = []
    self.elements = []
    self.myelemsDat = ElemsMallaDat()
    self.constraints = Constraints()
    self.myElementosAImprimir = ElementosAImprimir()
    self.loads = Loads()
    action = None
    self.problemDimension = None
    f = filename
    self.printRenumber = False
    for iline,line in enumerate(f,start=1):
      line=line.split('/')[0].strip()
      if len(line)<=0:
        continue
      action = self.myelemsDat.read(iline,line,action,datosgmsh.physicalsNames)
      action = self.constraints.read(iline,line,action,datosgmsh.physicalsNames,self.problemDimension)
      action = self.myElementosAImprimir.read(iline,line,action,datosgmsh.physicalsNames)
      action = self.loads.read(iline,line,action,datosgmsh.physicalsNames,self.problemDimension)
      if self.myelemsDat.problemDimension != None:
        self.problemDimension = self.myelemsDat.problemDimension

  def __makeNewElements(self,datosgmsh,oldNumbersElements, newElements):
    inverseOldNumbersElements = dict()
    for newNumber,oldNumber in enumerate(oldNumbersElements,1):
      inverseOldNumbersElements[oldNumber] = newNumber
    self.inverseOldNumbersElements = inverseOldNumbersElements
    if self.printRenumber:
      with open("elemRenumber.txt",'w') as f:
        f.write("# gmshNumber      vulcanNumber\n")
        for key in sorted(self.inverseOldNumbersElements):
          f.write(str(key) + " " + str(self.inverseOldNumbersElements[key]) + "\n")

    for number,element in enumerate(newElements,1):
      physicalId = datosgmsh.getPhysicalIdFromElement(oldNumbersElements[number-1])
      founded = False
      for elemDat in self.myelemsDat:
        if physicalId == elemDat.physical:
          if elemDat.setVulcan <0:
            self.elements.append(ElementInter(number,element[::-1],-elemDat.setVulcan,physicalId))
          else:
            self.elements.append(ElementInter(number,element,elemDat.setVulcan,physicalId))
          founded = True
      if not founded:
        raise ValueError("ERROR HABLAR CON MATIAS")

  def __makeNewNodesUpdateElements(self, oldNumbersNodes, newNodes):
    for number,positions in enumerate(newNodes):
      self.nodes.append(NodeInter(number, positions))

    inverseOldNumbersNodes = dict()                                   #Crea un diccionario inverso, donde la llave es el numero viejo y el valor el numero nuevo
    for newNumber,oldNumber in enumerate(oldNumbersNodes,1):
      inverseOldNumbersNodes[oldNumber] = newNumber
    self.inverseOldNumbersNodes = inverseOldNumbersNodes 

    if self.printRenumber:
      with open("nodeRenumber.txt",'w') as f:
        f.write("# gmshNumber      vulcanNumber\n")
        for key in sorted(self.inverseOldNumbersNodes):
          f.write(str(key) + " " + str(self.inverseOldNumbersNodes[key]) + "\n")
 
    for elem in self.elements:
      for number,node in enumerate(elem):
        elem.nodes[number] = inverseOldNumbersNodes[node]

  def makeNewGeo(self,datosgmsh):
    listOfPhysicalsId = []
    for elemDat in self.myelemsDat:
      listOfPhysicalsId.append(elemDat.physical)
    oldNumbersElements, newElements = datosgmsh.getPhysicalsElements(listOfPhysicalsId)
    self.__makeNewElements(datosgmsh,oldNumbersElements, newElements)
    oldNumbersNodes, newNodes = datosgmsh.getPhysicalNodes(listOfPhysicalsId)
    self.__makeNewNodesUpdateElements(oldNumbersNodes, [node[0:self.problemDimension] for node in newNodes])

  def getElements(self):
    toReturn = []
    for element in self.elements:
      toReturn.append(element.nodes)
    return copy.deepcopy(toReturn)

  def getSetElements(self):
    toReturn = []
    for element in self.elements:
      toReturn.append(element.set)
    return copy.deepcopy(toReturn)

  def getNodes(self):
    toReturn = []
    for node in self.nodes:
      toReturn.append(node.positions)
    return copy.deepcopy(toReturn)

  def lenConstraints(self):
    return len(self.constraints)

  def getConstraints(self,datosgmsh,iConstraint):
    conditions = []
    tempConstraints = self.constraints[iConstraint]
    for constraint in tempConstraints:
      nodes, dummy = datosgmsh.getPhysicalNodes([constraint.physical])
      for inode,node in enumerate(nodes): #Pasando de nodos antiguos a nodos nuevos
        nodes[inode] = self.inverseOldNumbersNodes[node]
      conditions.append({
                          "additive": constraint.additive,
                          "constraintX": constraint.constraintxVulcan,
                          "constraintY": constraint.constraintyVulcan,
                          "constraintZ": constraint.constraintzVulcan,
                          "function": constraint.functionVulcan,
                          "valX"    : constraint.valxVulcan,
                          "valY"    : constraint.valyVulcan,
                          "valZ"    : constraint.valzVulcan,
                          "nodes"   : copy.deepcopy(nodes)})
    return conditions

  def writeElementsToIntergplo(self,datosgmsh):
    orderOldToNew = self.inverseOldNumbersNodes
    self.myElementosAImprimir.write(datosgmsh,orderOldToNew)

  def lenLoads(self):
    return len(self.constraints)

  def __updateNodesOfElements(self,oldElements,elements):
    for number,element in enumerate(elements):
    #  oldElements[number] = self.inverseOldNumberElements[oldElements[number]]
      for inode,node in enumerate(element):
        element[inode] = self.inverseOldNumbersNodes[node]
    return elements

  def __findElementByNodes(self,nodes):
    for element in self.elements:
      if all(nodeToSearch in element.nodes  for nodeToSearch in nodes):
        return element.number
    raise ValueError('No se encontro ningun elemento con los nodos '+str(nodes))

  def getLoads(self,datosgmsh,iLoad):
    loads = []
    tempLoads = self.loads[iLoad]
    for load in tempLoads:
      if load.type == "FACE_LOCAL" or load.type == "FACE_GLOBAL":
        oldElements, elements = datosgmsh.getPhysicalsElements([load.physical])
        elements = self.__updateNodesOfElements(oldElements, elements)
        elemToPass = dict()  
        for nodes in elements:
          elemNumber = self.__findElementByNodes(nodes)
          elemToPass[elemNumber] = nodes
        loads.append({
                          "type"    : load.type,
                          "function": load.functionVulcan,
                          "valX"    : load.valxVulcan,
                          "valY"    : load.valyVulcan,
                          "valZ"    : load.valzVulcan,
                          "elements": copy.deepcopy(elemToPass)})
      elif load.type == "POINT_LOAD":
        nodes, dummy = datosgmsh.getPhysicalNodes([load.physical])
        for inode,node in enumerate(nodes): #Pasando de nodos antiguos a nodos nuevos
          nodes[inode] = self.inverseOldNumbersNodes[node]
        loads.append({
                          "type"    : load.type,
                          "function": load.functionVulcan,
                          "valX"    : load.valxVulcan,
                          "valY"    : load.valyVulcan,
                          "valZ"    : load.valzVulcan,
                          "nodes": copy.deepcopy(nodes)})
      else:
        loads.append({
                          "type"    : load.type,
                          "gravity" : load.gravity,
                          "function": load.functionVulcan,
                          "valX"    : load.valxVulcan,
                          "valY"    : load.valyVulcan,
                          "valZ"    : load.valzVulcan})
    return loads
  
      

#*************************************************************************************************************************
#********************************   FIN INTER    *************************************************************************
#*************************************************************************************************************************

#*************************************************************************************************************************
#********************************   GMSH    ******************************************************************************
#*************************************************************************************************************************


class PhysicalName:
  def __init__(self,line):
    line = shlex.split(line)
    self.dimension = int(line[0])
    self.id        = int(line[1])
    self.name      = line[2]


class Nodes:
  def __init__(self,line):
    self.toPrint = False
    data = line.strip().split()
    self.number = int(data[0])
    self.coord = data[1:]
    self.coord = [float(num) for num in self.coord]


class Elements:
  def __init__(self,line):
    self.toPrint = False
    data = line.strip().split()
    self.number = int(data[0])
#    self.dimension = int(data[1])
    elType = int(data[1])
    self.type = elType
#    if elType == 1 :
#      self.nnodes = 2
#    elif 
    self.physicalId = int(data[3])
    self.elementaryId = int(data[4])
    self.nodes = [int(node) for node in data[5:]]
  def __iter__(self):
    if len(self.nodes) == 0:
      raise StopIteration
    else:
      self.iItemIterator = -1
      return self
    return
  def next(self):
    self.iItemIterator += 1
    if len(self.nodes) > self.iItemIterator :
      return self.nodes[self.iItemIterator]
    else:
      raise StopIteration
  def __str__(self):
    toReturn = str(self.nodes)
    return toReturn


class ReaderGmsh:
  def __init__(self, file):
    self.physicalsNames = []
    self.nodes = []
    self.elements = []
    f = file
    f.readline()                                           # $MeshFormat
    f.readline()                                           # 2.2 0 8
    f.readline()                                           # $EndMeshFormat

    line = f.readline()                                    # $PhysicalNames or $Nodes
    if line.lower().startswith("$physicalnames"):
      line = f.readline().strip()
      numOfPhysicalEntities = int(line)
      for iline in range(numOfPhysicalEntities):
        line = f.readline()
        self.physicalsNames.append(PhysicalName(line))
      f.readline()                                         # $EndPhysicalNames
      f.readline()                                         # $Nodes

    line = f.readline()
    numOfNodes = int(line) # Numero de nodos
    for iline in range(numOfNodes): #Nodos parten de 1
      line = f.readline().strip()
      self.nodes.append(Nodes(line))
    line = f.readline()                                           # $EndNodes

    line = f.readline()                                           # $Elements
    numOfElements = int(f.readline()) # Numero de elementos

    for iline in range(numOfElements):
       line = f.readline()
       self.elements.append(Elements(line))


  def getPhysicalsElements(self,listOfPhysicalsId):
    toReturnElements = []
    toReturniElements = []
    for elem in self.elements:
      for physicalId in listOfPhysicalsId:
        if elem.physicalId == physicalId:
          toReturnElements.append(elem.nodes)
          toReturniElements.append(elem.number)
    return copy.deepcopy(toReturniElements), copy.deepcopy(toReturnElements)

  def getPhysicalIdFromElement(self,ielem):
    return self.elements[ielem-1].physicalId

  def getPhysicalNodes(self,listOfPhysicalsId):
    toReturnNodes = []
    toReturniNodes = []
    for node in self.nodes:
      node.toPrint = False
    for elem in self.elements:
      for physicalId in listOfPhysicalsId:
        if elem.physicalId == physicalId:
          for node in elem:
            self.nodes[node-1].toPrint = True
    for node in self.nodes:
      if node.toPrint == True:
        toReturnNodes.append(node.coord)
        toReturniNodes.append(node.number)
    return copy.deepcopy(toReturniNodes), copy.deepcopy(toReturnNodes)



#*************************************************************************************************************************
#********************************  FIN GMSH    ***************************************************************************
#*************************************************************************************************************************


#*************************************************************************************************************************
#********************************   VULCAN  ******************************************************************************
#*************************************************************************************************************************

class ElementVulcan:
  def __init__(self, number, nodes, setElement):
    self.number = number
    self.set = setElement
    self.nodes = nodes
    self.nnodes = len(nodes)
  def __str__(self):
    return " {:7d}".format(self.number) + ' ' + "{:2d}".format(self.set) + ' '.join(["{:7d}".format(node) for node in self.nodes])+"\n"

class NodeVulcan:
  def __init__(self, number, positions):
    self.number = number
    self.positions = positions
  def __str__(self):
    return " {:7d}".format(self.number) + ' ' + ' '.join(["{:14.8G}".format(position) for position in self.positions])+"\n" 


class ConstraintNodeVulcan(dict):
  def __init__(self,constraint,number,dimension):
    self["number"     ] = number
    self["dimension"  ] = dimension 
    self["constraintX"] = constraint["constraintX"]
    self["constraintY"] = constraint["constraintY"]
    self["constraintZ"] = constraint["constraintZ"]
    self["function"   ] = constraint["function"]
    self["valX"       ] = constraint["valX"]
    self["valY"       ] = constraint["valY"]
    self.dimension=dimension
    if dimension == 3:
      self["valZ"       ] = constraint["valZ"]

  def set(self,constraint):
    if constraint["additive"]:
      self["constraintX"] = self["constraintX"] or constraint["constraintX"]
      self["constraintY"] = self["constraintY"] or constraint["constraintY"]
      self["constraintZ"] = self["constraintZ"] or constraint["constraintZ"]
      self["function"   ] = constraint["function"]
      self["valX"       ] = self["valX"]+constraint["valX"]
      self["valY"       ] = self["valY"]+constraint["valY"]
      if self.dimension == 3:
        self["valZ"       ] = self["valZ"]+constraint["valZ"]
    else:
      self["constraintX"] = constraint["constraintX"]
      self["constraintY"] = constraint["constraintY"]
      self["constraintZ"] = constraint["constraintZ"]
      self["function"   ] = constraint["function"]
      self["valX"       ] = constraint["valX"]
      self["valY"       ] = constraint["valY"]
      if self.dimension == 3:
        self["valZ"       ] = constraint["valZ"]

  def __str__(self):
    toReturn = "{:7d}".format(self["number"])+" "
    if self["dimension"] == 2:
      toReturn += str(int(self["constraintX"]))+str(int(self["constraintY"]))
    else:
      toReturn += str(int(self["constraintX"]))+str(int(self["constraintY"]))+str(int(self["constraintZ"]))

    toReturn += " "+str(self["function"])
    if self["dimension"] == 2:
      toReturn += " {:14.8E}".format(self["valX"])
      toReturn += " {:14.8E}".format(self["valY"])
    elif self["dimension"] == 3:
      toReturn += " {:14.8E}".format(self["valX"])
      toReturn += " {:14.8E}".format(self["valY"])
      toReturn += " {:14.8E}".format(self["valZ"])
    else:
      raise ValueError("HABLAR CON MATIAS 2")
    toReturn += "\n"
    return toReturn
    

class ConstraintsNodesVulcan:
  def __init__(self):
    self.data = dict()
  def set(self,constraint,dimension):
  #  print constraint["nodes"]
    for node in constraint["nodes"]:
    #  print node
      if not (node in self.data): #INITIALIZE
        self.data[node] = ConstraintNodeVulcan(constraint,node,dimension)
      else:
        self.data[node].set(constraint)



class Gravity(dict):
  def __init__(self,load,dimension):
    self["dimension"] = dimension
    self["gravity"] = load["gravity"]
    self["function"] = load["function"]
    self["valX"] = load["valX"]
    self["valY"] = load["valY"]
    self["valZ"] = load["valZ"]
  def __str__(self):
    toReturn = "GRAVITY \n"
    toReturn += str(self["gravity"])
    toReturn += " " + str(self["function"])
    toReturn += " " + str(self["valX"])
    toReturn += " " + str(self["valY"])
    if self["dimension"] == 3:
      toReturn += " " + str(self["valZ"])
    toReturn += "\n"
    return toReturn



class FaceLoad(dict):
  def __init__(self,load,number,dimension,nodes):
    self["number"] = number
    self["dimension"] = dimension
    self["function"] = load["function"]
    self["valX"] = load["valX"]
    self["valY"] = load["valY"]
    self["valZ"] = load["valZ"]
    self["nodes"] = nodes
  def set(self,load,nodes):
    self["function"] = load["function"]
    self["valX"] = load["valX"]
    self["valY"] = load["valY"]
    self["valZ"] = load["valZ"]
    self["nodes"] = nodes
  def __str__(self):
    toReturn = "  " + str(self["number"])
    toReturn += " " + str(self["function"]) + "\n"
    toReturn += " " + ' '.join(["{:7d}".format(node) for node in self["nodes"]]) + "\n"
    if self["dimension"] == 2:
      toReturn += "LOCAL: " + ' '.join(["{:8.2E} {:8.2E}".format(self["valX"],self["valY"]) for node in self["nodes"]]) #Euge agrege "LOCAL: " +
    else:
      toReturn += "LOCAL: " + ' '.join(["{:8.2E} {:8.2E} {:8.2E}".format(self["valX"],self["valY"],self["valZ"]) for node in self["nodes"]]) #Euge agrege "LOCAL: " +
    toReturn += "\n"  
    return toReturn

class PointLoad(dict):
  def __init__(self,load,number,dimension):
    self["number"] = number
    self["dimension"] = dimension
    self["function"] = load["function"]
    self["valX"] = load["valX"]
    self["valY"] = load["valY"]
    self["valZ"] = load["valZ"]
  def set(self,load):
    self["function"] = load["function"]
    self["valX"] = load["valX"]
    self["valY"] = load["valY"]
    self["valZ"] = load["valZ"]
  def __str__(self):
    toReturn = " " + str(self["number"]) + " " + str(self["function"])
    if self["dimension"] == 2:
      toReturn += " " + str(self["valX"]) + " " + str(self["valY"])
    else:
      toReturn += " " + str(self["valX"]) + " " + str(self["valY"]) + " " + str(self["valZ"])
    toReturn += "\n"
    return toReturn


class LoadsVulcan:
  def __init__(self):
    self.faceLocal = []
    self.faceGlobal = []
    self.pointLoad = dict()
  def __len__(self):
    toReturn =  len(self.faceLocal) 
    toReturn += len(self.faceGlobal)
    toReturn += len(self.pointLoad)
    if hasattr(self,'gravity'):
      toReturn += 1
    return toReturn
  def __setPointLoad(self,load,dimension):
    for node in load["nodes"]:
      if not (node in self.pointLoad):
        self.pointLoad[node] = PointLoad(load,node,dimension)
      else:
        self.pointLoad[node].set(load)

  def __setFaceLocal(self,load,dimension):
    for elem,nodes in load["elements"].iteritems():                      #elements es un diccionario con key del numero de elemento y valores de la lista de nodos
#      if not (elem in self.faceLocal):
#        self.faceLocal[elem] = FaceLoad(load,elem,dimension,nodes)
#      else:
#        self.faceLocal[elem].set(load,nodes)
      self.faceLocal.append([elem, FaceLoad(load,elem,dimension,nodes)])

  def __setFaceGlobal(self,load,dimension):
    for elem,nodes in load["elements"].iteritems():                      #elements es un diccionario con key del numero de elemento y valores de la lista de nodos
#      if not (elem in self.faceGlobal):
#        self.faceGlobal[elem] = FaceLoad(load,elem,dimension,nodes)
#      else:
#        self.faceGlobal[elem].set(load,nodes)
      self.faceGlobal.append([elem, FaceLoad(load,elem,dimension,nodes)])
       

  def set(self,load,dimension):
    if load["type"] == "POINT_LOAD":
      self.__setPointLoad(load,dimension)
    elif load["type"] == "FACE_LOCAL":
      self.__setFaceLocal(load,dimension)
    elif load["type"] == "FACE_GLOBAL":
      self.__setFaceGlobal(load,dimension)
    elif load["type"] == "GRAVITY":
      self.gravity = Gravity(load,dimension)
    else:
      raise ValueError( "No se encontró la carga adecuada, LLAMAR A MATIAS 3")

  def __writePointLoad(self,fileOpened):
    numberOfPoints = len(self.pointLoad)
    if numberOfPoints > 0:
      fileOpened.write("POINT_LOAD,"+str(numberOfPoints)+"\n")
      for node in sorted(self.pointLoad.iterkeys()):
        fileOpened.write(str(self.pointLoad[node]))

  def __writeFaceLocal(self,fileOpened):
    from operator import itemgetter
    numberOfElements = len(self.faceLocal)
    if numberOfElements > 0:
      fileOpened.write("FACE_LOAD, LOCAL, "+str(numberOfElements)+",DEFORMATION_DEPENDENT_FACE_LOAD\n") #Euge agrege DEFORMATION_DEPENDENT_FACE_LOAD
#      for element in sorted(self.faceLocal.iterkeys()):
      for ielem, load in sorted(self.faceLocal, key=itemgetter(0)):
#        fileOpened.write(str(self.faceLocal[element]))
        fileOpened.write(str(load))

  def __writeFaceGlobal(self,fileOpened):
    from operator import itemgetter
    numberOfElements = len(self.faceGlobal)
    if numberOfElements > 0:
      fileOpened.write("FACE_LOAD, GLOBA, "+str(numberOfElements)+"\n")
#      for element in sorted(self.faceGlobal.iterkeys()):
#        fileOpened.write(str(self.faceGlobal[element]))
      for ielem, load in sorted(self.faceGlobal,key=itemgetter(0)):
        fileOpened.write(str(load))

  def __writeGravity(self,fileOpened):
    if hasattr(self,'gravity'):
      fileOpened.write(str(self.gravity))

  def write(self,fileOpened):
    self.__writeGravity(fileOpened)
    self.__writePointLoad(fileOpened)
    self.__writeFaceLocal(fileOpened)
    self.__writeFaceGlobal(fileOpened)
#class PointLoad(dict):

class DataVulcan:
  def __init__(self, elements, setElements, nodes):
    self.elements = []
    self.nodes = []
    self.constraints = ConstraintsNodesVulcan()
    self.loads = LoadsVulcan()
    self.dimension = 3
    for ielem, (element, setElement) in enumerate(zip(elements, setElements), 1):
      self.elements.append(ElementVulcan(ielem, element, setElement))
    for inode, positions in enumerate(nodes,1):
      self.nodes.append(NodeVulcan(inode, positions))
    self.setConstraint({  "additive": True,                            #Impose last node to print him
                          "constraintX": False,
                          "constraintY": False,
                          "constraintZ": False,
                          "function": 1,
                          "valX"    : 0.0,
                          "valY"    : 0.0,
                          "valZ"    : 0.0,
                          "nodes"   : [len(self.nodes)]})

  def writeGeo(self,filename):
    f = filename
    for element in self.elements:
      f.write(str(element))
    for node in self.nodes:
      f.write(str(node))

  def setConstraint(self, constraint):
    self.constraints.set(constraint,self.dimension)


  def writeConstraint(self, filename):
    f = filename
    for node in sorted(self.constraints.data.iterkeys()):
      f.write(str(self.constraints.data[node]))

  def setLoad(self,load):
    self.loads.set(load,self.dimension)

  def writeLoads(self, filename):
    f = filename
    self.loads.write(f)

  def getNNodes(self):
    return len(self.nodes)
    
  def getNElements(self):
    return len(self.elements)

#*************************************************************************************************************************
#********************************  FIN VULCAN  ***************************************************************************
#*************************************************************************************************************************


parser = argparse.ArgumentParser(description='Traspasa la malla de tipo GMSH a Tipo Vulcan, los comando de inicio son: "$Elem_Malla", "$Restricciones", "$Cargas" y "$Nodos_Elem_Fisicos"')
parser.add_argument('-p', '--printRenumber',help="Imprime la renumeración de nodos y elements en nodeRenumber.txt y elemRenumber.txt", action='store_true')
parser.add_argument('inter',help="Archivo de intercambio",type=argparse.FileType('r'))
parser.add_argument('gmsh',help="Archivo de malla de gmsh",type=argparse.FileType('r'))
parser.add_argument('fileOut',help="Archivos de Salida, sin extensión")
args = parser.parse_args()


datosgmsh = ReaderGmsh(args.gmsh)
args.gmsh.close()


inter = interGmshVulcan(args.inter, datosgmsh)
inter.printRenumber = args.printRenumber
args.inter.close()
inter.makeNewGeo(datosgmsh)

elements = inter.getElements()
setElements = inter.getSetElements()
nodes = inter.getNodes()
mydataVulcan = DataVulcan(elements, setElements, nodes)

fileGeo = open(args.fileOut+".geo", 'w')
mydataVulcan.writeGeo(fileGeo)

if inter.lenConstraints() > 0:
  fileFix = open(args.fileOut+".fix", 'w')
  for iConstraints in range(inter.lenConstraints()):
    constraints = inter.getConstraints(datosgmsh,iConstraints)
    for constraint in constraints:
      mydataVulcan.setConstraint(constraint)
    mydataVulcan.writeConstraint(fileFix)

if len(inter.loads) > 0:
  fileLoa = open(args.fileOut+".loa", 'w')
  for iLoad in range(inter.lenLoads()):
    loads = inter.getLoads(datosgmsh,iLoad)
    for load in loads:
      mydataVulcan.setLoad(load)
  mydataVulcan.writeLoads(fileLoa)

inter.writeElementsToIntergplo(datosgmsh)

print "\n\nElementos: ", mydataVulcan.getNElements(), " Nodos: ", mydataVulcan.getNNodes()
