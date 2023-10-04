import Vector_OLE
from Vector_OLE import VOLE
import time
import socket
import random
import numpy as np

def run(vole):
    times={};



    #OFFLINE PHASE
    print("Bob: Offline Phase");
    e=vole.random_mu_vector(vole.m,vole.mu);
    I=vole.anti_support(e);

    M_I_top_neighbors=vole.M_I_top_neighbors(I);
  
    start1 =time.time();
    PLUQ_tuple=vole.sparse_PLUQ_decomposition(M_I_top_neighbors,vole.k,None,None,True);
    end1 = time.time();


    start2=time.time();
    Ecc_solutions=vole.offline_luby_decoding(vole.Ecc_neighbors,I[vole.u:vole.m]);
    end2=time.time();

    while PLUQ_tuple==None or Ecc_solutions==None:
        if(PLUQ_tuple==None):
            print("M[I] isn't full rank, trying another I");
        if (Ecc_solutions==None):
            print("Ecc[I] cannot be solved, trying another I");
        e=vole.random_mu_vector(vole.m,vole.mu);
        I=vole.anti_support(e);
        M_I_top_neighbors=vole.M_I_top_neighbors(I);
        time.sleep(3);

        start1 =time.time();
        PLUQ_tuple=vole.sparse_PLUQ_decomposition(M_I_top_neighbors,vole.k,None,None,True);#,add_dic,mult_dic);
        end1 = time.time();

        start2=time.time();
        Ecc_solutions=vole.offline_luby_decoding(vole.Ecc_neighbors,I[vole.u:vole.m]);
        end2=time.time();

    vole.print_message_with_time("Bob performed an offline PLUQ decomposition",start1,end1);
    times['PLUQ Decomposition']=(end1-start1)*1000;

    vole.print_message_with_time("Bob performed an offline Luby decoding",start2,end2);
    times['Offline Luby Decoding']=(end2-start2)*1000;

    e_support=[i for i in range(0,vole.m) if e[i]!=0];
    r=vole.random_vector(vole.k);

    start=time.time();
    pseudorandom_vector=vole.pseudorandom_vector(r,e,e_support);
    end=time.time();
    vole.print_message_with_time("Bob computed Mr+e",start,end);
    times['Mr+e Computation']=(end-start)*1000;

    #ONLINE PHASE
    port=500;
    ip_addr=socket.gethostbyname(socket.gethostname());
    ot_port=2000;
    server_socket=socket.socket(socket.AF_INET, socket.SOCK_STREAM);
    server_socket.bind((ip_addr, port));
    server_socket.listen(1);
    print("Bob: Online Phase");
    print("Bob is waiting for Alice");
    (client_socket, client_ip_port) = server_socket.accept();
    server_socket.close();
    print("Bob is connected to Alice");
    ot_ip_addr=client_ip_port[0];

    a=vole.random_vector(vole.w);
    b=vole.random_vector(vole.w);

    start=time.time();
    ec_code=vole.ecc(a);
    end=time.time();
    vole.print_message_with_time("Bob computed Ecc(a)",start,end);
    times['Ecc(a) Computation']=(end-start)*1000;


    v=vole.concat_vectors(r,a);
    padded_ec_code=vole.pad_vector_with_leading_zeros(ec_code,vole.u);
    

    start=time.time();
    c=vole.add_vectors(pseudorandom_vector,padded_ec_code);
    end=time.time();
    times["c computation"]=(end-start)*1000;


    
    start = time.time();
    if vole.send_vector(client_socket,c)==False:
        print("socket connection is broken");
        client_socket.close();
        exit(0);
    end = time.time();
    vole.print_message_with_time("Bob sent c to Alice",start,end);

    start=time.time();
    alpha_gamma=vole.recv_vector(client_socket);
    end=time.time();
    if alpha_gamma is None:
        print("socket connection is broken");
        client_socket.close();
        exit(0);
    alpha=int(alpha_gamma[0]);
    gamma=alpha_gamma[1:];
    vole.print_message_with_time("Bob got α={0} and γ from Alice".format(alpha),start,end);

    ot_outputs_vector=vole.receiver_oblivious_transfer(vole.m,I,vole.bits,ot_ip_addr,ot_port,times);

    d_I=vole.zero_vector(vole.m);
    h_I_not=vole.zero_vector(vole.m);
    h_I_not_support=[];
    for i in range(0,vole.m):
        if I[i]==1:
            d_I[i]=ot_outputs_vector[i];
        else:
            h_I_not[i]=ot_outputs_vector[i];
            h_I_not_support.append(i);
    start=time.time();
    delta_x=vole.cds_decode(c,gamma,alpha,h_I_not,h_I_not_support,v);
    end=time.time();
    vole.print_message_with_time("Bob decoded the CDS, x-x'={0}".format(delta_x),start,end);
    times['CDS decoding']=(end-start)*1000;

    result=vole.E_decode_by_PLUQ(PLUQ_tuple,Ecc_solutions,d_I,I,times);
    if result is None:
        return;
    s=result[0];
    shifted_info=result[1];

    start=time.time();
    decoded_info=vole.scalar_mult_and_add_vector(delta_x,a,shifted_info);        
    end=time.time();
    vole.print_message_with_time("Bob computed ax+b'",start,end);
    times['Result shifting to x']=(end-start)*1000;


    #Get Alice's inputs for validation
    x=vole.recv_scalar(client_socket);
    b_tag=vole.recv_vector(client_socket);

    info_from_Alice=vole.scalar_mult_and_add_vector(x,a,b_tag);

        
    if vole.compare_vectors(decoded_info,info_from_Alice):
        print("Decoding seccseeded");
    else:
        print("Decoding didn't seccseed");

    start=time.time();        
    shifted_info_by_b=vole.add_vectors(decoded_info,b);
    end=time.time();
    vole.print_message_with_time("Bob computed ax+b+b'",start,end);
    times['Result shifting by b']=(end-start)*1000;

    start = time.time();
    if vole.send_vector(client_socket,shifted_info_by_b)==False:
        print("socket connection broken");
        client_socket.close();
        exit(0);
    end = time.time();
    vole.print_message_with_time("Bob sent ax+b+b' to Alice",start,end);
    client_socket.close();    

    print("Times:");
    print(times);
    return times;




def main():
    #k=182;
    #w=10000;
    
    k=240;
    w=20000;

    bits=128;    
    try:
        vole=VOLE(k,w,bits);
        if bits<=64:
            file_name=".\\Matrices\\Matrix{0}_{1}.npz".format(k,bits);
        else:
            file_name=".\\Matrices\\Matrix{0}_{1}.txt".format(k,bits);
        vole.read_matrix_from_file(file_name);
        file_name=".\\Ecc\\Ecc{0}_{1}.npz".format(k,w);
        vole.read_ecc_from_file(file_name);
    except Exception as e:
        print(e);
    else:
        run(vole);

if __name__ == "__main__":
    main();
