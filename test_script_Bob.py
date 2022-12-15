import Bob
import Vector_OLE
from Vector_OLE import VOLE
import time

def sum_dict_into_dict(dest_dic,dic):
    for key in dic:
        if not(key in dest_dic):
            dest_dic[key]=0;
        dest_dic[key]+=dic[key];

def avg_dic(dic,devisor):
    for key in dic:
        dic[key]=dic[key]/devisor;



def main():
    repetitions=300;
    times_dic={};
    adds_dic={};
    mults_dic={};
    
    #k=182;
    #w=10000;

    k=240;
    w=20000;
    
    mu=0.25;
    d_max=10;
    bits=16;
    u_factor=1.4;


    vole=VOLE(k,w,mu,d_max,bits,u_factor);
    if bits<=64:
        file_name=".\\Matrices\\Matrix{0}_{1}.npz".format(k,bits);
    else:
        file_name=".\\Matrices\\Matrix{0}_{1}.txt".format(k,bits);
    vole.read_matrix_from_file(file_name);
    file_name=".\\Ecc\\Ecc{0}_{1}.npz".format(k,w);
    vole.read_ecc_from_file(file_name.format(k,w));

    for i in range(1,repetitions+1):
        print("Test {0}".format(i));
        (current_times,current_adds,current_mults)=Bob.run(vole);
        sum_dict_into_dict(times_dic,current_times);
        sum_dict_into_dict(adds_dic,current_adds);
        sum_dict_into_dict(mults_dic,current_mults);
        print();
        time.sleep(7);
    avg_dic(times_dic,repetitions);
    avg_dic(adds_dic,repetitions);
    avg_dic(mults_dic,repetitions);
    print("Avarage Times:");
    print(times_dic);
    print("Avarage additions number:");
    print(adds_dic);
    print("Avarage multiplications number:");
    print(mults_dic);

if __name__ == "__main__":
    main();



