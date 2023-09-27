import sys
sys.path.append('..');
import Vector_OLE
from Vector_OLE import VOLE


k=182;
w=10000;

#k=240;
#w=20000;

bits=128;
#mu=0.25;
#d=10;
#u_factor=1.4;
vole=VOLE(k,w,bits);#,mu,d,u_factor)

M=vole.random_d_sparse_matrix(vole.m,k,vole.d);
print("Matrix has been created");

file_name="Matrix{0}_{1}".format(k,bits);
vole.write_matrix_to_file(M,file_name);
print("Matrix has been written to file");

