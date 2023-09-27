import Vector_OLE
from Vector_OLE import VOLE
import time
import socket
import random
import numpy as np

def run(vole):
    times={};

    #OFFLINE PHASE
    print("Alice: Offline Phase");
    r_tag=vole.random_vector(vole.k);
    b_tag=vole.random_vector(vole.w);
    x_tag=random.randint(0,vole.fn-1);

    #start_ns=time.time_ns();
    #end_ns=time.time_ns();
    start=time.time();
    M_rtag=vole.M_mult_vector(r_tag);
    end=time.time();
    vole.print_message_with_time("Alice computed Mr'",start,end);
    times["Mr' Computation"]=(end-start)*1000;

    start=time.time();
    ec_code=vole.ecc(b_tag);
    end=time.time();
    vole.print_message_with_time("Alice computed Ecc(b')",start,end);
    times["Ecc(b') Computation"]=(end-start)*1000;




    padded_ec_code=vole.pad_vector_with_leading_zeros(ec_code,vole.u);
    start=time.time();
    d_tag=vole.add_vectors(padded_ec_code,M_rtag);
    end=time.time();
    times["d' computation"]=(end-start)*1000;



    h=vole.random_vector(vole.m);

    start = time.time();
    gamma=vole.vector_mult_T(h);
    end = time.time();
    vole.print_message_with_time("Alice computed γ=h*T",start,end);
    times['γ Computation']=(end-start)*1000;


    #ONLINE PHASE
    #Bob_ip_addr='';
    Bob_ip_addr=socket.gethostbyname(socket.gethostname());
    Bob_port=500;
    ot_port=2000;
    ot_ip_addr=socket.gethostbyname(socket.gethostname());

    print("Alice: Online Phase");
    print("Alice is trying to connect to Bob",end="");
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
    #sock.settimeout(10);
    connected = False;
    while connected != True:
        try:
            sock.connect((Bob_ip_addr, Bob_port));
            connected = True;
        except:
            print(".",end="");
            time.sleep(3);
    print("");
    print("Alice is connected to Bob");


    #Inputs
    x=random.randint(0,vole.fn-1);

    start=time.time();
    c=vole.recv_vector(sock);
    end=time.time();
    if c is None:
        print("Socket connection is broken");
        sock.close();
        exit(0);
    vole.print_message_with_time("Alice got c from Bob",start,end);

    start=time.time();
    d=vole.scalar_mult_and_add_vector(x_tag,c,d_tag);
    end=time.time(); 
    vole.print_message_with_time("Alice computed d=x'*c+Er'(b')",start,end);
    times['d computation']=(end-start)*1000;


    start = time.time();
    h_c=vole.dot_product(h,c);
    cds_secret=x-x_tag;
    alpha=(h_c+cds_secret)%vole.fn;
    end = time.time();
    vole.print_message_with_time("Alice computed α=h*c+(x-x')={0}".format(alpha),start,end);
    times['α Computation']=(end-start)*1000;
    print("The CDS secret is x-x'={0}".format(cds_secret%vole.fn));

    alpha_gamma=vole.concat_scalar_vector(alpha,gamma);

    start=time.time();
    if vole.send_vector(sock,alpha_gamma)==False:
        print("Socket connection is broken");
        sock.close();
        exit(0);
    end = time.time();
    vole.print_message_with_time("Alice sent α and γ to Bob",start,end);

    vole.sender_oblivious_transfer(vole.m,h,d,vole.bits,ot_ip_addr,ot_port,times);

    vole.send_scalar(sock,x);
    vole.send_vector(sock,b_tag);


    start=time.time();
    shifted_info=vole.recv_vector(sock);
    end=time.time();
    if shifted_info is None:
        print("Socket connection is broken");
        sock.close();
        exit(0);
    vole.print_message_with_time("Alice got ax+b+b' from Bob",start,end);
    sock.close();
    
    start=time.time();
    result_info=vole.sub_vectors(shifted_info,b_tag);
    end=time.time(); 
    vole.print_message_with_time("Alice computed ax+b",start,end);
    times["Result shifting by b'"]=(end-start)*1000;


    print("Times:");
    print(times);
    return times;



def main():
    k=182;
    w=10000;
    
    #k=240;
    #w=20000;
    
    bits=128;
    vole=VOLE(k,w,bits);
    if bits<=64:
        file_name=".\\Matrices\\Matrix{0}_{1}.npz".format(k,bits);
    else:
        file_name=".\\Matrices\\Matrix{0}_{1}.txt".format(k,bits);
    vole.read_matrix_from_file(file_name);
    file_name=".\\Ecc\\Ecc{0}_{1}.npz".format(k,w);
    vole.read_ecc_from_file(file_name.format(k,w));
    run(vole);

if __name__ == "__main__":
    main();




