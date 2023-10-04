import sys
sys.path.append('..');
import Vector_OLE
from Vector_OLE import VOLE



k_list=[182,240];
w_list=[10000,20000];

bits=128;

for k,w in zip(k_list,w_list):
    vole=VOLE(k,w,bits);
    Ecc=vole.luby_encoder(vole.v,w);
    print("Ecc for (k,w)=({0},{1}) has been created".format(k,w));
    file_name="Ecc{0}_{1}".format(k,w);
    vole.write_ecc_to_file(Ecc,file_name);    
    print("Ecc for (k,w)=({0},{1}) has been written to file".format(k,w));
    print();

