import sys
sys.path.append('C://Users//YuvalKon//Desktop//Niv//Test');
sys.path.append('C://Users//yuval//Desktop//Python_Implementation//Test');
import Vector_OLE
from Vector_OLE import VOLE


#k=182;
#w=10000;

k=240;
w=20000;

mu=0.25;
d_max=10;
bits=128;
u_factor=1.4;
vole=VOLE(k,w,mu,d_max,bits,u_factor)

#M=Vector_OLE.random_d_sparse_matrix(m,k,dmax);
M=vole.random_d_sparse_matrix(vole.m,k,d_max);
print("Matrix has been created");

file_name="Matrix{0}_{1}".format(k,bits);
vole.write_matrix_to_file(M,file_name);
print("Matrix has been written to file");

