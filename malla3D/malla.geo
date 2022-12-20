

// mallado en mm

// Parametros

ancho = 1;
largo = 1;
Alto = 2;

//Numero de Elementos
elX = 20 ;
elZ = 1 ;
elY = 20 ;

//  Creacion de puntos
p1 = newp; Point(p1) = {0,0,0};
p2 = newp; Point(p2) = {largo,0,0};
p3 = newp; Point(p3) = {0,0,ancho};
p4 = newp; Point(p4) = {largo,0,ancho};

l1 = newl; Line(l1) = {p1, p2};
l2 = newl; Line(l2) = {p2, p4};
l3 = newl; Line(l3) = {p4, p3};
l4 = newl; Line(l4) = {p3, p1};

ll = newll; Line Loop(ll) = {l1,l2,l3,l4};


ll = newll; Line Loop(ll) = {l1,l2,l3,l4};

s = news; Surface(s) = {ll};

Transfinite Line {l1,-l3} = elX+1;
Transfinite Line {l2,-l4} = elZ+1 ;

Transfinite Surface "*";
Recombine Surface "*";

Extrude {0,Alto,0} { 
  Surface{s}; Layers{elY}; Recombine;
}

