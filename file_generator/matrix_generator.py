import sys
sys.path.append('..');
import Vector_OLE
from Vector_OLE import VOLE


k_list=[182,240];
w_list=[10000,20000];
bits_list=[16,32,64,128];
#mu=0.25;
#d=10;
#u_factor=1.4;

for k,w in zip(k_list,w_list):
    for bits in bits_list:
        vole=VOLE(k,w,bits);#,mu,d,u_factor) 
        M=vole.random_d_sparse_matrix(vole.m,k,vole.d);
        print("Matrix for k={0} and {1} bits has been created".format(k,bits));
        file_name="Matrix{0}_{1}".format(k,bits);
        vole.write_matrix_to_file(M,file_name);
        print("Matrix for k={0} and {1} bits has been written to file".format(k,bits));
        print();
