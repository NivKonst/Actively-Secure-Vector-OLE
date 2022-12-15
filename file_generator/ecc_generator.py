import sys
sys.path.append('C://Users//YuvalKon//Desktop//Test');
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


Ecc=vole.luby_encoder(vole.v,w);
print("Ecc has been created");


file_name="Ecc{0}_{1}".format(k,w);
vole.write_ecc_to_file(Ecc,file_name);    
print("Ecc has been written to file");

