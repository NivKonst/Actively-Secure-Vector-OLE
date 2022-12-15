import random
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import load_npz
from scipy.sparse import save_npz
import gmpy2 as gp2
import socket
import pickle
import math
import itertools
import queue
from OT import OT
import threading



fn_128=340282366920938463463374607431768211297;    
fn_64=18446744073709551253;
fn_32=4294967291;
fn_16=65521;
packet_size=4;

class VOLE:
    def __init__(self,k,w,mu,dmax,bits,u_factor):
        self.k=k;
        self.w=w;
        self.u=(int)(u_factor*k);
        self.v=(int)(k**2);
        self.m=self.u+self.v;
        self.mu=mu;
        self.dmax=dmax;
        self.n=k+w;
        self.M_tuple=None;
        self.M_coo=None;
        self.M_csr=None;
        self.M_neighbors=None;
        self.M_bottom_neighbors=None;
        self.M_HL_tuple=None;
        self.transposed_M_HL_tuple=None;
        self.Ecc_tuple=None;
        self.Ecc_coo=None;
        self.Ecc_csr=None;
        self.Ecc_neighbors=None;
        self.bits=bits;
        if self.bits==64:
            self.fn=fn_64;
        elif self.bits==32:
            self.fn=fn_32;
        elif self.bits==16:
            self.fn=fn_16;
        elif self.bits==128:
            self.fn=fn_128;
        self.bytes=self.bits//8;
        self.factor_H=bits//2;
        self.factor_L=2**(self.factor_H)-1;
        self.factor_HL=self.factor_L+1;
        self.factor_HL_squared=self.factor_HL**2;
        #self.F=galois.GF(self.fn);
        self.print_parameters();

    #Prints the parameters of the current VOLE instance
    def print_parameters(self):
        print("Vector OLE parameters:");
        print("k="+str(self.k));
        print("u="+str(self.u));
        print("v="+str(self.v));
        print("m="+str(self.m));
        print("w="+str(self.w));
        print("GF({0}), {1}bits".format(self.fn,self.bits));





    #Generates a random sparse matrix of dimentions rows X cols
    #with at most dmax non-zero elements in a row
    def random_dmax_sparse_matrix(self,rows,cols,dmax):
        if self.bits<=64:
            M=np.zeros((rows,cols),dtype=np.ulonglong);
        else:
            M=[[0]*cols for i in range(0,rows)];
        indices=list(range(0,cols));
        #d_vec=np.random.choice(range(0,dmax+1),rows);
        d_vec=random.sample(range(0,dmax+1),rows);
        for i in range(0,rows):
            d=d_vec[i];
            if(d>0):
                sampled_indices=random.sample(indices,d);
                for index in sampled_indices:
                    #M[i][index]=random.randint(0,self.fn-1);
                    M[i][j]=random.randint(1,self.fn-1);
        return M;





    #Generates a random d-sparse matrix of dimentions rows X cols
    def random_d_sparse_matrix(self,rows,cols,dmax):
        if self.bits<=64:
            M=np.zeros((rows,cols),dtype=np.ulonglong);
        else:
            M=[[0]*cols for i in range(0,rows)];
        indices=list(range(0,cols));
        counter=0;
        for i in range(0,rows):
            sampled_indices=random.sample(indices,dmax);
            if i in sampled_indices:
                counter+=1;
            for j in sampled_indices:
                #M[i][j]=random.randint(0,self.fn-1);
                M[i][j]=random.randint(1,self.fn-1);
        return M;




    #Generates a Luby encoder represented as a matrix of ones, with dimensions code_len X info_len
    def luby_encoder(self,code_len,info_len):
        indices=list(range(0,info_len));
        #Ecc=lil_matrix((code_len,info_len),dtype=np.uint8);
        Ecc=np.zeros((code_len,info_len),dtype=np.uint8);
        dist=self.robust_soliton_distribution(info_len);
        for i in range(0,code_len):
            d=self.sample_by_distribution(dist);
            sampled_indices=random.sample(indices,d);
            for index in sampled_indices:
                Ecc[i,index]=1;
        return Ecc;


    #Samples a distribution
    def sample_by_distribution(self,dist):
        rang=len(dist);
        frac=random.uniform(0, 1);
        i=0;
        while dist[i]<=frac:
            i+=1;
        return i+1;

    #Generates the robust soliton distribution from 1 to rang
    def robust_soliton_distribution(self,rang):
        if rang==10000:
            c=1.17224;
        elif rang==20000:
            c=1.23075;
        else:
            c=1;
        delta=0.01;
        beta=0;
        dist=[0]*rang;
        R=c*(math.log(rang/delta))*math.sqrt(rang);
        print("R={0}".format(R));
        th=math.floor(rang/R);
        for i in range(0,rang):
            d=i+1;
            if(d>1):
                dist[i]=(1.0)/(d*(d-1));
                if(d<th):
                    dist[i]+=R/(d*rang);
                elif(d==th):
                    dist[i]+=(R*math.log(R/delta))/rang;
            else:
                dist[0]=(1.0+R)/rang;
            beta+=dist[i];
        for i in range(0,rang):
            dist[i]*=(1/beta);
        #β=beta;
        #print("β={0}".format(β));
        print("β={0}".format(beta));
        """expected=0;
        for i in range(0,rang):
            expected+=dist[i]*(i+1);
        print(expected);
        """
        cdf=np.cumsum(dist);
        cdf[-1]=1;
        return cdf;
    

    def decompose_vector_high_low(self,v):
        v_list=v.tolist();
        v_len=len(v_list);
        v_H=np.zeros(v_len,dtype=np.ulonglong);
        v_L=np.zeros(v_len,dtype=np.ulonglong);
        for i in range(0,v_len):
            x=v_list[i];
            v_H[i]=x>>self.factor_H;
            v_L[i]=x&self.factor_L;
        return (v_H,v_L);




    def M_mult_vector(self,r):
        if self.bits>=64:
            M_r=self.matrix_neighbors_mult_vector(self.M_neighbors,r);#,add_dic,mult_dic);
            #M_r=self.M_mult_vector_threading(r);
        elif self.bits==32:
            M_r=self.matrix_HL_mult_vector(self.M_HL_tuple,r);
        elif self.bits==16:
            #M_csr=csr_matrix(M_coo);
            #M_csr=M_coo.tocsr();
            #M_csc=M_coo.tocsc();
            M_r=(self.M_csr.dot(r))%self.fn;
            #M_r=(M_csc.dot(r))%self.fn;
            #M_r=(M_coo.dot(r))%self.fn;            
        return M_r;








    def matrix_mult_matrix_list(self,A_list,B_list):
        adds_counter=0;
        mults_counter=0;
        A_rows=len(A_list);
        A_cols=len(A_list[0]);
        B_rows=len(B_list);
        B_cols=len(B_list[0]);
        if A_cols!=B_rows:
            print("Matrices of wrong dimensions");
            return None;
        result_list=[[0]*B_cols for i in range(0,A_rows)];
        for i in range(0,A_rows):
            for j in range(0,B_cols):
                current=0;
                for t in range(0,A_cols):
                    a=A_list[i][t];
                    b=B_list[t][j];
                    if a!=0 and b!=0:
                        current+=a*b;
                        adds_counter+=1;
                        mults_counter+=1;
                result_list[i][j]=current%self.fn;
        print(adds_counter);
        print(mults_counter);
        return result_list;





    def matrix_mult_vector(self,matrix_tuple,r):
        (matrix_coo,matrix_csr,matrix_neighbors,matrix_HL_tuple)=matrix_tuple;
        if self.bits>=64:
            M_r=self.matrix_neighbors_mult_vector(matrix_neighbors,r);#,add_dic,mult_dic);
        elif self.bits==32:
            M_r=self.matrix_HL_mult_vector(matrix_HL_tuple,r);
        else:
            M_r=(matrix_csr.dot(r))%self.fn;
        return M_r;









    def M_I_mult_vector(self,r,I):
        result_len=len(self.M_neighbors[0]);
        if self.bits>=64:
            M_I_r=self.matrix_neighbors_mult_vector_I(self.M_neighbors,r,I);#,add_dic,mult_dic);
        elif self.bits==32:
            M_r=self.matrix_HL_mult_vector(self.M_HL_tuple,r);
            M_I_r_list=[M_r[i] if I[i]==1 else 0 for i in range(0,result_len)];
            M_I_r=np.array(M_I_r_list,dtype=np.ulonglong);
        else:
            M_r=self.M_csr.dot(r)%self.fn;
            M_I_r_list=[M_r[i] if I[i]==1 else 0 for i in range(0,result_len)];
            M_I_r=np.array(M_I_r_list,dtype=np.ulonglong);
        return M_I_r;





    def M_bottom_I_mult_vector(self,r,I):
        if self.bits>=64:
            M_bottom_I_r=self.matrix_neighbors_mult_vector_I(self.M_bottom_neighbors,r,I);#,add_dic,mult_dic);
            return M_bottom_I_r;
        if self.bits==32:
            M_r=self.matrix_HL_mult_vector(self.M_HL_tuple,r);
        else:
            M_r=self.M_csr.dot(r)%self.fn;
        M_bottom_I_r_list=[M_r[i] if I[i-self.u]==1 else 0 for i in range(self.u,self.m)];
        M_bottom_I_r=np.array(M_bottom_I_r_list,dtype=np.ulonglong);
        return M_bottom_I_r;

    
    def matrix_mult_vector_I(self,matrix_tuple,r,I):
        (matrix_coo,matrix_csr,matrix_neighbors,matrix_HL_tuple)=matrix_tuple;
        result_len=len(matrix_neighbors);
        if self.bits>=64:
            M_I_r=self.matrix_neighbors_mult_vector_I(matrix_neighbors,r,I);#,add_dic,mult_dic);
        elif self.bits==32:
            M_r=self.matrix_HL_mult_vector(matrix_HL_tuple,r);
            M_I_r_list=[M_r[i] if I[i]==1 else 0 for i in range(0,result_len)];
            M_I_r=np.array(M_I_r_list,dtype=np.ulonglong);
        else:
            M_r=self.matrix_csr.dot(r);
            #M_I_r_list=[(M_r[i]*I[i])%self.fn for i in range(0,result_len)];
            M_I_r_list=[M_r[i]%self.fn if I[i]==1 else 0 for i in range(0,result_len)];
            M_I_r=np.array(M_I_r_list,dtype=np.ulonglong);
        return M_I_r;


    def matrix_HL_mult_vector(self,M_HL_tuple,r):
        (M_H,M_L)=M_HL_tuple;
        (r_H,r_L)=self.decompose_vector_high_low(r);
        result_len=M_H.shape[0];
        vHH=(M_H.dot(r_H));
        vHL=(M_H.dot(r_L));
        vLH=(M_L.dot(r_H));
        vLL=(M_L.dot(r_L));
        vHL_LH=vHL+vLH;
        vHH_list=vHH.tolist();
        vLL_list=vLL.tolist();
        vHL_LH_list=vHL_LH.tolist();

        result_list=[(self.factor_HL_squared*vHH_list[i]+self.factor_HL*vHL_LH_list[i]+vLL_list[i])%self.fn for i in range(0,result_len)];
        result=np.array(result_list,dtype=np.ulonglong);
        #result=np.zeros(result_len,dtype=np.ulonglong);
        #for i in range(0,result_len):
        #    result[i]=(self.factor_HL_squared*vHH_list[i]+factor_HL*vHL_LH_list[i]+vLL_list[i])%self.fn;  
        return result;





    #update: write self's matrix
    def write_matrix_to_file(self,M,file_name):
        if self.bits<=64:
            save_npz(file_name, csr_matrix(M));
        else:
            rows=len(M);
            cols=len(M[0]);
            str_to_write='';
            for i in range(0,rows):
                current_row_str='';
                for j in range(0,cols):
                    x=M[i][j];
                    if x!=0:
                        current_row_str+=(str(j)+'_'+str(x)+'_');
                current_row_str=current_row_str[0:-1]+'\n';
                str_to_write+=current_row_str;
            file=open(file_name+'.txt','w');
            file.write(str_to_write);
            file.close();


    def write_ecc_to_file(self,Ecc,file_name):
        save_npz(file_name, csr_matrix(Ecc));

    def read_matrix_from_file(self,file_name):
        if self.bits<=64:
            self.M_csr=load_npz(file_name);
            self.M_coo=coo_matrix(self.M_csr);
            self.M_neighbors=self.coo_matrix_to_matrix_neighbors(self.M_coo);
            self.M_bottom_neighbors=(self.M_neighbors[0][self.u:self.m],self.M_neighbors[1][self.u:self.m]);
            self.M_HL_tuple=self.decompose_coo_matrix_to_HL(self.M_coo);
            self.transposed_M_HL_tuple=(np.transpose(self.M_HL_tuple[0]),np.transpose(self.M_HL_tuple[1]));
            self.M_tuple=(self.M_coo,self.M_csr,self.M_neighbors,self.M_HL_tuple);
            #return self.M_tuple;
        else:
            self.M_csr=None;
            self.M_coo=None;
            self.M_HL_tuple=None;
            self.M_tuple=None;
            rows_neighbors=[];
            data_neighbors=[];
            file=open(file_name,'r');
            file_content=file.read();
            file.close();
            file_rows=file_content.split('\n');
            file_rows=file_rows[0:-1];
            counter=0;
            for row_str in file_rows:
                current_row=[];
                current_data=[];
                row_content=row_str.split('_');
                row_content_len=len(row_content);
                j=0;
                while j<row_content_len:
                    current_row.append(int(row_content[j]));
                    current_data.append(int(row_content[j+1]));
                    j+=2;
                rows_neighbors.append(current_row);
                data_neighbors.append(current_data);
                counter+=1;
            self.M_neighbors=(rows_neighbors,data_neighbors);
            self.M_bottom_neighbors=(self.M_neighbors[0][self.u:self.m],self.M_neighbors[1][self.u:self.m]);

        
        
        
            

    #maybe also update m,k,v and u
    def set_matrix_by_neighbors(self,M_neighbors):
        self.M_csr=None;
        self.M_coo=None;
        self.M_HL_tuple=None;
        self.M_tuple=None;
        rows_neighbors=M_neighbors[0];
        data_neighbors=M_neighbors[1];
        rows_neighbors_copy=[x[:] for x in rows_neighbors];
        data_neighbors_copy=[x[:] for x in data_neighbors];
        self.M_neighbors=(rows_neighbors_copy,data_neighbors_copy);


    def matrix_list_to_neighbors(self,M_list):
        rows_neighbors=[];
        data_neighbors=[];
        rows=len(M_list);
        cols=len(M_list[0]);
        for i in range(0,rows):
            current_row_list=[];
            current_data_list=[];
            for j in range(0,cols):
                x=M_list[i][j];
                if x!=0:
                    current_row_list.append(j);
                    current_data_list.append(x);
            rows_neighbors.append(current_row_list);
            data_neighbors.append(current_data_list);
        return (rows_neighbors,data_neighbors);





    def decompose_coo_matrix_to_HL(self,M_coo):
        rows=M_coo.shape[0];
        cols=M_coo.shape[1];
        M_H=np.zeros((rows,cols),dtype=np.ulonglong);
        M_L=np.zeros((rows,cols),dtype=np.ulonglong);
        for i,j,data in zip(M_coo.row,M_coo.col,M_coo.data):
            M_H[i,j]=int(data)>>self.factor_H;
            M_L[i,j]=int(data)&self.factor_L;
        M_HL_tuple=(csr_matrix(M_H),csr_matrix(M_L));
        #M_HL_tuple=(coo_matrix(M_H),coo_matrix(M_L));
        return M_HL_tuple;




    def read_ecc_from_file(self,file_name):
        self.Ecc_csr=load_npz(file_name);
        self.Ecc_coo=coo_matrix(self.Ecc_csr);
        t=self.ecc_coo_matrix_to_neighbors(self.Ecc_coo);
        #t=(Ecc_neighbors,neighbors_degs,symbols_locations,deg_one_queue);
        self.Ecc_neighbors=t[0];
        self.Ecc_decoding_tuple=(t[1],t[2],t[3]);
        self.Ecc_tuple=(self.Ecc_coo,self.Ecc_csr,self.Ecc_neighbors,self.Ecc_decoding_tuple);
        return self.Ecc_tuple;





    def ecc_coo_matrix_to_neighbors(self,Ecc_coo):
        rows=Ecc_coo.shape[0];
        cols=Ecc_coo.shape[1];
        Ecc_neighbors=[];
        symbols_locations=[[] for i in range(0,cols)];
        #symbols_dict = {x: []for x in range(Ecc_coo.shape[1])};
        neighbors_degs=[0]*rows;
        #neighbors_degs_dict={x:0 for x in range(rows)};
        deg_one_queue=queue.Queue();
        current_row_index=0;
        current_row_list=[];
        for i,j in zip(Ecc_coo.row,Ecc_coo.col):
            if current_row_index!=i:
                Ecc_neighbors.append(current_row_list);
                current_row_len=len(current_row_list);
                neighbors_degs[current_row_index]=current_row_len;
                #neighbors_degs_dict[current_row_index]=current_row_len;
                if current_row_len==1:
                    deg_one_queue.put(current_row_index);
                #for h in range(current_row_index+1,i):
                #    Ecc_neighbors.append([]); #WHY DEG CAN BE 0?
                current_row_index=i;
                current_row_list=[];        
            current_row_list.append(j);
            symbols_locations[j].append(current_row_index);
        Ecc_neighbors.append(current_row_list);
        current_row_len=len(current_row_list);
        #neighbors_degs_dict[current_row_index]=current_row_len;
        neighbors_degs[current_row_index]=current_row_len;
        if current_row_len==1:
            deg_one_queue.put(current_row_index);
        #sorted_neighbors_degs_dict=dict(sorted(neighbors_degs_dict.items(),key=lambda t: t[1]));
        return (Ecc_neighbors,neighbors_degs,symbols_locations,deg_one_queue);
        #return Ecc_neighbors;
        

    def coo_matrix_to_matrix_neighbors(self,M_coo):
        rows_neighbors=[];
        data_neighbors=[];
        current_row_index=0;
        current_row_list=[];
        current_data_list=[];
        for i,j,data in zip(M_coo.row,M_coo.col,M_coo.data):
            if current_row_index!=i:
                rows_neighbors.append(current_row_list);
                data_neighbors.append(current_data_list);
                for h in range(current_row_index+1,i):
                    rows_neighbors.append([]);
                    data_neighbors.append([]);
                current_row_list=[];
                current_data_list=[];
                current_row_index=i;
            current_row_list.append(j);
            current_data_list.append(int(data));
        rows_neighbors.append(current_row_list);
        data_neighbors.append(current_data_list);
        for h in range(current_row_index+1,M_coo.shape[0]):
            rows_neighbors.append([]);
            data_neighbors.append([]);
        return (rows_neighbors,data_neighbors);




    def anti_support(self,e):
        e_len=len(e);
        I=[0]*e_len;
        for i in range(0,e_len):
            if e[i]==0:
                I[i]=1;
        return I;
    
        
    def random_vector(self,size):
        result_list=[random.randint(0,self.fn-1) for i in range(0,size)];
        if(self.bits<=64):
            result=np.array(result_list,dtype=np.ulonglong);
        else:
            result=result_list;
        return result;

    def random_mu_vector(self,size,mu):
        if(self.bits<=64):
            result=np.zeros(size,dtype=np.ulonglong);
        else:
            result=[0]*size;
        for i in range(size):
            th=random.uniform(0, 1);
            if th<mu:
                result[i]=random.randint(1,self.fn-1);
        return result;



    def pseudorandom_vector(self,r,e,e_support):
        M_r=self.M_mult_vector(r);
        if self.bits<64:
            return (M_r+e)%self.fn;  
        if self.bits==64:
            M_r_list=M_r.tolist();
            e_list=e.tolist();
        else:
            M_r_list=M_r;
            e_list=e;
        for i in e_support:
            M_r[i]=(M_r_list[i]+e_list[i])%self.fn;
        return M_r;

    def ecc(self,a):
        if self.bits==64:
            #Ecc_a=luby_encode(Ecc_neighbors,a,add_dic);
            #Ecc_a=self.Ecc_matrix_vector_HL(Ecc_coo,a);
            Ecc_a=self.Ecc_matrix_mult_vector_HL(self.Ecc_csr,a);
            return Ecc_a;
        if self.bits<64:
            Ecc_a=(self.Ecc_csr.dot(a))%self.fn;
            return Ecc_a;    
        Ecc_a=self.luby_encode(self.Ecc_neighbors,a);
        return Ecc_a;



    def Ecc_matrix_mult_vector_HL(self,Ecc,a):
        #Ecc_csr=csr_matrix(Ecc_coo);

        (a_H,a_L)=self.decompose_vector_high_low(a);
        result_len=Ecc.shape[0];

        vH=(Ecc.dot(a_H));
        vL=(Ecc.dot(a_L));
        vH_list=vH.tolist();
        vL_list=vL.tolist();

        #result_list=[(self.factor_HL*vH_list[i]+vL_list[i])%self.fn for i in range(0,result_len)];
        #result=np.array(result_list,dtype=np.ulonglong);
        result=np.zeros(result_len,dtype=np.ulonglong);
        for i in range(0,result_len):
            result[i]=(self.factor_HL*vH_list[i]+vL_list[i])%self.fn;
        return result;




    def matrix_neighbors_mult_vector(self,M_neighbors,r):
        rows_neighbors=M_neighbors[0];
        data_neighbors=M_neighbors[1];
        result_len=len(rows_neighbors);
        if self.bits<=64:
            r_list=r.tolist();
            result=np.zeros(result_len,dtype=np.ulonglong);
        else:
            r_list=r;
            result=[0]*result_len;

        for i in range(0,result_len):
            current_row=rows_neighbors[i];
            current_data=data_neighbors[i]
            current_result=0;
            #counter=0;
            current_result=sum(data*r_list[col] for col,data in zip(current_row,current_data) if r_list[col]!=0);
            #for col,data in zip(current_row,current_data):
            #    if r_list[col]!=0:
            #        current_result+=data*r_list[col];
            #        additions_counter+=1;
            #        multiplications_counter+=1
            result[i]=current_result%self.fn;
        return result;

    def matrix_neighbors_mult_vector_I(self,M_neighbors,r,I):
        func_name='matrix_neighbors_mult_vector_I';
        rows_neighbors=M_neighbors[0];
        data_neighbors=M_neighbors[1];
        result_len=len(rows_neighbors);
        if self.bits<=64:
            r_list=r.tolist();
            result=np.zeros(result_len,dtype=np.ulonglong);
        else:
            r_list=r;
            result=[0]*result_len;
        for i in range(0,result_len):
            if(I[i]==1):
                current_row=rows_neighbors[i];
                current_data=data_neighbors[i]
                current_result=0;
                for col,data in zip(current_row,current_data):
                    current_result+=data*r_list[col];
                result[i]=current_result%self.fn;
        return result;


    def coo_matrix_mult_vector(self,M_coo,r):
        r_list=r.tolist();
        rows=M_coo.shape[0];
        result=np.zeros(rows,dtype=np.ulonglong);
        prev_i=0;
        current_result=0;
        for i,j,data in zip(M_coo.row,M_coo.col,M_coo.data):
            if i!=prev_i:
                result[prev_i]=current_result%self.fn;
                current_result=0
                prev_i=i;
            r_j=r_list[j];
            if r_j!=0:
                current_result=current_result+int(data)*r_j;
        result[i]=current_result%self.fn;    
        return result;

    def coo_matrix_mult_vector_I(self,M_coo,r,I):
        r_list=r.tolist();
        rows=M_coo.shape[0];
        result=np.zeros(rows,dtype=np.ulonglong);
        prev_i=0;
        current_result=0;
        for i,j,data in zip(M_coo.row,M_coo.col,M_coo.data):
            if i!=prev_i:
                result[prev_i]=current_result%self.fn;
                current_result=0
                prev_i=i;
            if I[i]==1:
                current_result=current_result+int(data)*r_list[j];
        result[i]=current_result%self.fn;    
        return result;


    def luby_encode(self,Ecc_neighbors,a):
        func_name='luby_encode';
        code_len=len(Ecc_neighbors);
        if self.bits<=64:
            a_list=a.tolist();
            code=np.zeros(code_len,dtype=np.ulonglong);
        else:
            a_list=a;
            code=[0]*code_len;
        additions_counter=0;
        for i in range(0,code_len):
            current_row=Ecc_neighbors[i];
            current_code=0;
            for index in current_row:
                current_code+=a_list[index];
            code[i]=current_code%self.fn;
        return code;


    #not updated
    def E_decode(self,d,I):

        M_top_I=self.M_I_top(I);        
        d_top=d[0:u];

        start = time.time();
        s=self.gaussian_elimination(M_top_I,d_top);
        end = time.time();
        if s is None:
            print("No solution for d[I]=M_top[I]*s");
            return None;
        print_message_with_time("Gaussian elimination",start,end);

        #M_I_csr=csr_matrix(M_I);
        #M_I_s1=M_I_csr.dot(s);


        d_list=d.tolist();
        M_I_s=self.coo_matrix_mult_vector_I(self.M_coo,s,I);
        M_I_s_list=M_I_s.tolist();
        
        ec_code_list=[(x-y)%self.fn for x,y in zip(d_list,M_I_s_list)];
        ec_code=np.array(ec_code_list,dtype=np.ulonglong);
        start = time.time();
        info=self.luby_decoding(self.Ecc_neighbors,ec_code,I);
        end = time.time();
        if info is None:
            return None;
        print_message_with_time("Luby decoding",start,end);
        return (s,info);




    def gaussian_elimination(self,M,d):
        rows=M.shape[0];
        cols=M.shape[1];
        #vec=[0]*rows;
        #vec[:]=d[:];
        if self.bits<=64:
            d_list=d.tolist();
            M_list=M.tolist();
        else:
            d_list=d;
            M_list=M;
            
        for j in range(0,cols):
            gen=(i for i in range(j,rows) if M_list[i][j]!=0);
            pivot=next(gen,None);
            if pivot==None:
                return None;
            pivot_inv=int(gp2.invert(gp2.mpz(M_list[pivot][j]),gp2.mpz(self.fn)));
            #pivot_inv=np.reciprocal(F(mat[pivot][j]));
            prev_pivot=j;
            if pivot!=j:
                temp=M_list[j][j:cols];
                M_list[j][j:cols]=[(pivot_inv*M_list[pivot][t])%self.fn for t in range(j,cols)];
                M_list[pivot][j:cols]=temp;

                temp=d_list[j];
                d_list[j]=(pivot_inv*d_list[pivot]);
                d_list[pivot]=temp;
                prev_pivot=pivot;
                pivot=j;
            elif M_list[pivot][j]!=1:
                M_list[pivot][j:cols]=[(pivot_inv*M_list[pivot][t])%self.fn for t in range(j,cols)];
                d_list[pivot]=(pivot_inv*d_list[pivot]);
            #Here pivot=j and M[pivot][j]=1
            range_rows=list(range(0,pivot))+list(range(prev_pivot+1,rows));
            for i in range_rows:
                if M_list[i][j]%self.fn!=0:
                    factor=M_list[i][j];
                    M_list[i][j:cols]=[(M_list[i][t]-M_list[pivot][t]*factor)%self.fn for t in range(j,cols)];
                    d_list[i]=(d_list[i]-factor*d_list[pivot]);
        result_list=[d_list[i]%self.fn for i in range(0,cols)];

        if self.bits<=64:
            result=np.array(result_list,dtype=np.ulonglong);
        else:
            result=result_list;
        return result;


    def offline_gaussian_elimination(self,M):
        rows=M.shape[0];
        cols=M.shape[1];

        if self.bits<=64:
            M_list=M.tolist();
        else:
            M_list=M;
        instructions=[];
        for j in range(0,cols):
            gen=(i for i in range(j,rows) if M_list[i][j]!=0);
            pivot=next(gen,None);
            if pivot==None:
                return None;
            
            prev_pivot=j;
            pivot_value=M_list[pivot][j];
            if pivot_value!=1:
                pivot_inv=int(gp2.invert(gp2.mpz(pivot_value),gp2.mpz(self.fn)));
            else:
                pivot_inv=1;
            if pivot!=j:
                temp=M_list[j][j:cols];
                M_list[j][j:cols]=[(pivot_inv*M_list[pivot][t])%self.fn for t in range(j,cols)];
                M_list[pivot][j:cols]=temp;
                #temp=d_list[j];
                #d_list[j]=(pivot_inv*d_list[pivot]);
                #d_list[pivot]=temp;
                instructions.append(["r",j,pivot,pivot_inv]);
                prev_pivot=pivot;
                pivot=j;
            elif M_list[pivot][j]!=1:
                M_list[pivot][j:cols]=[(pivot_inv*M_list[pivot][t])%self.fn for t in range(j,cols)];
                #d_list[pivot]=(pivot_inv*d_list[pivot]);
                instructions.append(["m",pivot,pivot_inv]);
            #Here pivot=j and M[pivot][j]=1
            if j<cols-1:
                range_rows=list(range(0,pivot))+list(range(prev_pivot+1,rows));
            else:
                range_rows=list(range(0,pivot));
            for i in range_rows:
                if M_list[i][j]!=0:
                    factor=M_list[i][j];
                    M_list[i][j:cols]=[(M_list[i][t]-M_list[pivot][t]*factor)%self.fn for t in range(j,cols)];
                    #d_list[i]=(d_list[i]-factor*d_list[pivot]);
                    instructions.append(["s",i,pivot,factor]);
        return instructions;



    def offline_gaussian_elimination_LU(self,M):
        func_name='offline_gaussian_elimination_LU';
        if self.bits<=64:
            rows=M.shape[0];
            cols=M.shape[1];
            M_list=M.tolist();
        else:
            rows=len(M);
            cols=len(M[0]);
            M_list=M;
            

        permutation_vector=list(range(0,rows));

        for j in range(0,cols):
            #pivot is the first non zero in the column
            gen=(i for i in range(j,rows) if M_list[i][j]!=0);
            pivot=next(gen,None);
            if pivot==None:
                return None;

            #pivot is the max non zero in the column
            #cur_max=0;
            #cur_max_index=-1;
            #for i in range(j,rows):
            #    if M_list[i][j]>cur_max:
            #        cur_max=M_list[i][j];
            #        cur_max_index=i;
            #if cur_max_index==-1:
            #    return None;
            #pivot=cur_max_index;
                    

            #prev_pivot=j;
            if pivot!=j:
                #temp=M_list[j][j:cols];
                #M_list[j][j:cols]=M_list[pivot][j:cols];
                #M_list[pivot][j:cols]=temp;
                temp=M_list[j];
                M_list[j]=M_list[pivot];
                M_list[pivot]=temp;
        

                temp=permutation_vector[j];
                permutation_vector[j]=permutation_vector[pivot];
                permutation_vector[pivot]=temp;

                #prev_pivot=pivot;
                pivot=j;
            #Here pivot=j
            if j<cols-1:
                pivot_value=M[pivot][j];
                if pivot_value!=1:
                    pivot_inv=int(gp2.invert(gp2.mpz(M_list[pivot][j]),gp2.mpz(self.fn)));
                else:
                    pivot_inv=1;
                for i in range(j+1,rows):
                    if M_list[i][j]!=0:
                        factor=(pivot_inv*M_list[i][j])%self.fn;
                        M_list[i][j]=factor;
                        M_list[i][j+1:cols]=[(M_list[i][t]-M_list[pivot][t]*factor)%self.fn for t in range(j+1,cols)];
        if self.bits<=64:
            LU_matrix=np.array(M_list,dtype=np.ulonglong);
        else:
            LU_matrix=M_list;
        p=np.array(permutation_vector);
        return (LU_matrix,p);




    
    def sparse_PLUQ_decomposition(self,M_neighbors,matrix_cols,initial_rows_permutation,initial_cols_permutation,diag):

        rows_neighbors=M_neighbors[0];
        data_neighbors=M_neighbors[1];

        rows=len(rows_neighbors);
        cols=matrix_cols;

        L_rows=rows;
        L_cols=cols;
        U_rows=L_cols;
        U_cols=U_rows;

        LU_matrix_list=[[0]*U_cols for i in range(0,U_rows)];
        #LU_rows_neighbors=[[] for i in range(0,U_rows)];


        if diag:
            #(M_neighbors,rows_permutation)=self.fill_diagonal_neighbors(M_neighbors,matrix_cols);
            (M_neighbors,rows_permutation)=self.improved_fill_diagonal_neighbors(M_neighbors,matrix_cols);
        elif initial_rows_permutation!=None:
            rows_permutation=initial_rows_permutation;
        else:
            rows_permutation=list(range(0,rows));

        if initial_cols_permutation!=None:
            cols_permutation=initial_cols_permutation;
            cols_permutation_transpose=[cols_permutation.index(j) for j in range(0,cols)];
        else:
            cols_permutation=list(range(0,cols));
            cols_permutation_transpose=list(range(0,cols));

        U_cols_neighbors=([[] for i in range(0,U_cols)],[[] for i in range(0,U_cols)])
        U_diagonal_inverses=[0]*U_cols;

        non_empty_rows_indices=[i for i in range(0,rows) if len(rows_neighbors[i])>0];
        non_empty_rows=len(non_empty_rows_indices);
        if non_empty_rows<U_rows:
            return None;
        bad_rows_counter=0;
        very_bad_row=False;
        very_bad_rows_counter=0;
        maximum_bad_rows=non_empty_rows-U_rows;
        did_cols_perm=False;

        
        i=0;
        #while i<U_rows
            #gen=(t for t in range(i,rows-bad_rows_counter) if len(rows_neighbors[t])>0);
            #next_row=next(gen,None);
            #if next_row==None: #no more empty rows - not full rank
            #    return None;
        current_non_empty_row_index=0;
        while i<U_rows and current_non_empty_row_index<non_empty_rows:
            #print("i={0}".format(i));
            next_row=non_empty_rows_indices[current_non_empty_row_index];
            current_non_empty_row_index+=1;
            
            if next_row!=i: #the current row is empty, we need to premute other row
                #if very_bad_row:
                #    print("HERE");
                #   print(i);
                #   print(next_row);

                #bad_rows_counter+=1;
                #if bad_rows_counter>rows-U_rows: #check if too many bad_rows
                #    return None;
                #while rows_neighbors[-bad_rows_counter]==[]:
                #    bad_rows_counter+=1;
                #    if bad_rows_counter>rows-U_rows: #check if too many bad_rows
                #        return None;
                
                #temp=rows_neighbors[i];
                rows_neighbors[i]=rows_neighbors[next_row];
                #rows_neighbors[next_row]=rows_neighbors[-bad_rows_counter];
                #rows_neighbors[-bad_rows_counter]=[];#temp;
                rows_neighbors[next_row]=[];
                
                data_neighbors[i]=data_neighbors[next_row];
                #data_neighbors[next_row]=data_neighbors[-bad_rows_counter];
                #data_neighbors[-bad_rows_counter]=[];
                data_neighbors[next_row]=[];
                
                temp=rows_permutation[i];
                rows_permutation[i]=rows_permutation[next_row];
                #rows_permutation[next_row]=rows_permutation[-bad_rows_counter];
                #rows_permutation[-bad_rows_counter]=temp;
                rows_permutation[next_row]=temp;
                

            #Here row i in M is not empty
            current_row=rows_neighbors[i];
            current_data=data_neighbors[i];    
            current_row_len=len(current_row);
            if did_cols_perm: #Need to fix the row by the columns permutation
                (current_row,current_data)=self.fix_row_by_cols_permutation(current_row,current_data,cols_permutation_transpose);
                rows_neighbors[i]=current_row;
                data_neighbors[i]=current_data;

            #Here row i in M is acorrding to the columns permutation
            col_index_in_row=0;
            current_col=current_row[0];
            current_col_data=current_data[0];
            row_has_ended=False;
            found_non_zero_u=False;

            LU_non_zero_tuple=self.get_non_zero_indices_of_LU(LU_matrix_list,current_row,i);
            (fill,start,end)=LU_non_zero_tuple;
            #Loop for the computation of LU_i_j            
            L_row_neighbors=([],[]);
            for j in range(start,end):
                if fill[j]:
                    if (not row_has_ended) and current_col==j:
                        z=current_col_data;
                        col_index_in_row+=1;
                        if col_index_in_row<current_row_len:
                            current_col=current_row[col_index_in_row];
                            current_col_data=current_data[col_index_in_row];
                        else:
                            row_has_ended=True;
                    else:
                        z=0;


                    #s=self.row_neighbors_mult_matrix_column(L_row_neighbors,LU_matrix_list,j);
                      
                    U_col_neighbors=(U_cols_neighbors[0][j],U_cols_neighbors[1][j]);
                    s=self.dot_product_in_matrix_neighbors(LU_matrix_list,L_row_neighbors,U_col_neighbors,i,j);#,add_dic,mult_dic,func_name);                        
                    if j<i: #L computation
                        inv_factor=U_diagonal_inverses[j];
                        #s=self.dot_product_in_LU_matrix(LU_matrix_list,i,j,start,j);
                        LU_i_j=((z-s)*inv_factor)%self.fn;
                        if LU_i_j!=0:
                            L_row_neighbors[0].append(j);
                            L_row_neighbors[1].append(LU_i_j);
                    else: #U computation
                        #s=self.dot_product_in_LU_matrix(LU_matrix_list,i,j,start,i);
                        LU_i_j=(z-s)%self.fn;
                        #adds_counter+=1;
                        if LU_i_j!=0:
                            U_col_neighbors[0].append(i);
                            U_col_neighbors[1].append(LU_i_j)
                            if not found_non_zero_u:
                                found_non_zero_u=True;
                                first_non_zero_u=j;
                    LU_matrix_list[i][j]=LU_i_j;


            if found_non_zero_u:
                if first_non_zero_u>i: #The first non-zero u is from the right to the ith column.
                    for t in range(0,i+1): #swap columns
                        temp=LU_matrix_list[t][i];
                        LU_matrix_list[t][i]=LU_matrix_list[t][first_non_zero_u];
                        LU_matrix_list[t][first_non_zero_u]=temp;                        
                    temp=cols_permutation[i];
                    cols_permutation[i]=cols_permutation[first_non_zero_u];
                    cols_permutation[first_non_zero_u]=temp;

                    cols_permutation_transpose[temp]=first_non_zero_u;
                    cols_permutation_transpose[cols_permutation[i]]=i;                    

                    temp=U_cols_neighbors[0][i];
                    U_cols_neighbors[0][i]=U_cols_neighbors[0][first_non_zero_u]
                    U_cols_neighbors[0][first_non_zero_u]=temp;

                    temp=U_cols_neighbors[1][i];
                    U_cols_neighbors[1][i]=U_cols_neighbors[1][first_non_zero_u]
                    U_cols_neighbors[1][first_non_zero_u]=temp;
                    did_cols_perm=True;
                U_diagonal_inverses[i]=int(gp2.invert(gp2.mpz(LU_matrix_list[i][i]),gp2.mpz(self.fn)));
                i+=1;
            else:
                print("Bad Row, U in this row is zero");
                print(i);
                very_bad_rows_counter+=1;
                if very_bad_rows_counter>maximum_bad_rows:
                    return None;
                very_bad_row=True;
                rows_neighbors[i]=[];
                data_neighbors[i]=[];
        
        if self.bits<=64:
            LU_matrix=np.array(LU_matrix_list,dtype=np.ulonglong);
        else:
            LU_matrix=LU_matrix_list;
        P=np.array(rows_permutation);
        Q=np.array(cols_permutation);
        Q_T=np.array(cols_permutation_transpose);
        return (LU_matrix,P,Q,Q_T);



    def flops(self,LU_matrix):
        dim=len(LU_matrix);
        L=[[0]*dim for i in range(0,dim)];
        U=[[0]*dim for i in range(0,dim)];
        for i in range(0,dim):
            L[i][i]=1;
            for j in range(0,dim):
                if j<i:
                    L[i][j]=LU_matrix[i][j];
                else:
                    U[i][j]=LU_matrix[i][j];
        return self.matrix_mult_matrix_list(L,U);






    def dot_product_in_LU_matrix(self,LU_matrix,row,col,size):
        result=0;
        for t in range(0,size):
            if t<row:      
                result+=LU_matrix[row][t]*LU_matrix[t][col];
                if t==col:
                    break;
            else:
                result+=1*LU_matrix[t][col];
                break;
        return result%self.fn;

    def dot_product_in_LU_matrix(self,LU_matrix,row,col,start,end):
        result=0;
        for t in range(start,end):
            if t<row:      
                result+=LU_matrix[row][t]*LU_matrix[t][col];
                if t==col:
                    break;
            else:
                result+=1*LU_matrix[t][col];
                break;
        return result%self.fn;

    def row_neighbors_mult_matrix_column(self,row_neighbors,M,col):
        (row_cols,row_data)=row_neighbors;
        result=0;
        for (j,data) in zip(row_cols,row_data):
            result+=data*M[j][col];
        return result%self.fn;

    def dot_product_in_matrix_neighbors(self,M,row_neighbors,col_neighbors,row,col):
        (row_cols,row_data)=row_neighbors;
        (col_rows,col_data)=col_neighbors;
        
        result=0;
        if len(row_cols)<=len(col_rows):
            for (j,data) in zip(row_cols,row_data):
                x=M[j][col];
                if x!=0:
                    result+=data*x;
        else:
            for (i,data) in zip(col_rows,col_data):
                x=M[row][i];
                if x!=0:
                    result+=x*data;
        return result%self.fn;


    def get_non_zero_indices_of_LU(self,LU_matrix,sparse_matrix_row,LU_row_index):
        row_len=len(sparse_matrix_row);
        #if LU_row_index<0 or row_len==0:
        #    return None;
        LU_cols=len(LU_matrix[0]);
        if row_len==LU_cols:
            return ([True]*LU_cols,0,LU_cols);

        fill=[False]*LU_cols;
        fill_counter=0;
        q=queue.Queue();
        first_fill=LU_cols;
        last_fill=0;
        for j in sparse_matrix_row:
            fill[j]=True;
            fill_counter+=1;
            if fill_counter==1:
                first_fill=j;
            last_fill=j;
            if j<LU_row_index:
                q.put(j);
        while not q.empty():
            i=q.get();
            for j in range(i+1,LU_cols):
                if LU_matrix[i][j]!=0 and not fill[j]:
                    fill[j]=True;
                    fill_counter+=1;
                    if fill_counter==LU_cols:
                        return (fill,first_fill,LU_cols);
                    if last_fill<j:
                        last_fill=j;
                    if j<LU_row_index:
                        q.put(j);
        return (fill,first_fill,last_fill+1);



    def get_non_zero_indices_of_L(self,LU_matrix,sparse_matrix_row,LU_row_index):
        row_len=len(sparse_matrix_row);
        if LU_row_index<=0 or row_len==0:
            return None;
        q=queue.Queue();
        fill=[False]*LU_row_index;
        fill_counter=0;
        first_fill=LU_row_index;
        last_fill=0;
        for j in sparse_matrix_row:
            if j>=LU_row_index:
                break;
            elif not fill[j]:
                fill[j]=True;
                fill_counter+=1;
                if fill_counter==1:
                    first_fill=j;
                last_fill=j;
                q.put(j);
        if fill_counter==LU_row_index:
            return (fill,first_fill,LU_row_index);
        while not q.empty():
            i=q.get();
            for j in range(i,LU_row_index):
                if LU_matrix[i][j]!=0 and not fill[j]:    
                    fill[j]=True;
                    fill_counter+=1;
                    #if fill_counter==1:
                    #    first_fill=j;
                    if fill_counter==LU_row_index:
                        return (fill,first_fill,LU_row_index);
                    if last_fill<j:
                        last_fill=j;
                    q.put(j);
        return (fill,first_fill,last_fill+1);



    

            
    def fix_row_by_cols_permutation(self,row,data,cols_permutation_transpose):
        row_len=len(row);
        #if row_len==0:
        #    return (current_row,current_data);
        permutated_row=[cols_permutation_transpose[x] for x in row];
        if permutated_row==row:
            return (row,data);
        row_data_tuple_list=[(x,y) for x,y in zip(permutated_row,data)];
        row_data_tuple_list.sort(key=lambda x: x[0]);
        sorted_permutated_row=[x[0] for x in row_data_tuple_list];
        sorted_permutated_data=[x[1] for x in row_data_tuple_list];
        return (sorted_permutated_row,sorted_permutated_data);









    def fill_diagonal(self,M):
        if self.bits<=64:
            rows=M.shape[0];
            cols=M.shape[1];
            M_list=M.tolist();
        else:
            rows=len(M);
            cols=len(M[0]);
            M_list=M;

        rows_permutation=list(range(0,rows));
        for j in range(0,cols):
            if M_list[j][j] == 0:
                found=False;
                for i in range(j+1,rows):
                    if M_list[i][j]!=0:
                        temp=M_list[j];
                        M_list[j]=M_list[i];
                        M_list[i]=temp

                        temp=rows_permutation[j];
                        rows_permutation[j]=rows_permutation[i];
                        rows_permutation[i]=temp;                        
                        found=True;
                        break;
                if not found:
                    break;
                    #return None;
        if self.bits<=64:
            result=np.array(M_list,dtype=np.ulonglong);
        else:
            result=M_list;
        return (result,rows_permutation);




    #changing the input M_neighbors 
    def fill_diagonal_neighbors(self,M_neighbors,matrix_cols):
        (rows_neighbors,data_neighbors)=M_neighbors;
        rows=len(rows_neighbors);
        rows_permutation=list(range(0,rows));
        current_row_index=0;
        for j in range(0,matrix_cols):
            found=False;
            for i in range(current_row_index,rows):
                current_row=rows_neighbors[i];
                current_row_len=len(current_row);
                found=current_row_len>0 and current_row[0]==j;
                if found:
                    break;
            if found:
                if current_row_index!=i:
                    temp=rows_neighbors[current_row_index];
                    rows_neighbors[current_row_index]=rows_neighbors[i];
                    rows_neighbors[i]=temp;

                    temp=data_neighbors[current_row_index];
                    data_neighbors[current_row_index]=data_neighbors[i];
                    data_neighbors[i]=temp;

                    temp=rows_permutation[current_row_index];
                    rows_permutation[current_row_index]=rows_permutation[i];
                    rows_permutation[i]=temp;
                current_row_index+=1;
        return (M_neighbors,rows_permutation);



    #changing the input M_neighbors 
    def improved_fill_diagonal_neighbors(self,M_neighbors,matrix_cols):
        (rows_neighbors,data_neighbors)=M_neighbors;
        rows=len(rows_neighbors);
        rows_permutation=list(range(0,rows));
        rows_permutation_transpose=list(range(0,rows));
        pivots_rows=[-1]*matrix_cols;
        pivots_counter=0;
        for i in range(0,rows):
            current_row=rows_neighbors[i];
            current_row_len=len(current_row);
            if current_row_len>0:
                j=current_row[0];
                if pivots_rows[j]==-1:
                    pivots_rows[j]=i;
                    pivots_counter+=1;
                    #if pivots_counter==matrix_cols:
                    #    break;
        #print("{0} rows were selected before PLUQ.format(pivots_counter));
        current_row_index=0;        
        for j in range(0,matrix_cols):
            i=pivots_rows[j];
            if i!=-1:
                correct_i=rows_permutation_transpose[i];
                if correct_i!=current_row_index:
                    temp=rows_neighbors[current_row_index];
                    rows_neighbors[current_row_index]=rows_neighbors[correct_i];
                    rows_neighbors[correct_i]=temp;

                    temp=data_neighbors[current_row_index];
                    data_neighbors[current_row_index]=data_neighbors[correct_i];
                    data_neighbors[correct_i]=temp;

                    temp=rows_permutation[current_row_index];
                    rows_permutation[current_row_index]=rows_permutation[correct_i];
                    rows_permutation[correct_i]=temp;


                    rows_permutation_transpose[temp]=correct_i;
                    rows_permutation_transpose[rows_permutation[current_row_index]]=current_row_index;                    
                current_row_index+=1;
                if current_row_index==pivots_counter:
                    break;
        return (M_neighbors,rows_permutation);








    def check_tuple(self,t):
        (t0,t1,t2,t3)=t;
        if len(t0)>0:
            return t0[0];
        else:
            return t3;


    def sort_by_left_pivot(self,M_neighbors,matrix_cols):
        (rows_neighbors,data_neighbors)=M_neighbors;
        rows=len(rows_neighbors);
        rows_permutation=list(range(0,rows));
        tuple_list=[(x,y,z,matrix_cols) for x,y,z in zip(rows_neighbors,data_neighbors,rows_permutation)];
        tuple_list.sort(key=self.check_tuple);
        rows_neighbors=[t[0] for t in tuple_list];
        data_neighbors=[t[1] for t in tuple_list]; 
        rows_permutation=[t[2] for t in tuple_list]; 
        M_neighbors=(rows_neighbors,data_neighbors);
        return (M_neighbors,rows_permutation);


















    #not updated
    def E_decode_by_instructions(self,M_instructions,d_I,I,times):
        d_I_top=d_I[0:u];
        start = time.time();
        s=self.get_solution_by_instructions(M_instructions,d_I_top,k);
        end = time.time();
        #if s is None:
        #    print("No solution for d[I]=M_top[I]*s");
        #    return None;
        self.print_message_with_time("Solution by instructions",start,end);
        times['Solution by instructions']=(end-start)*1000;

        M_I_s=self.M_I_mult_vector(s,I);
        ec_code=sub_vectors(d_I[self.u:self.m],M_I_s[self.u:self.m]);


        start = time.time();
        info=self.luby_decoding(self.Ecc_neighbors,ec_code,I[self.u:self.m]);
        end = time.time();
        times['Luby decoding']=(end-start)*1000;

        if info is None:
            print("No solution to Ecc");
            return None;
        self.print_message_with_time("Luby decoding",start,end);
        return (s,info);



    def E_decode_by_LU(self,LU_tuple,Ecc_solutions,d_I,I,times):
        d_I_top=d_I[0:self.u];
        start = time.time();
        s=self.get_solution_by_LU(LU_tuple,d_I_top);
        end = time.time();
        self.print_message_with_time("Solution by LU",start,end);
        times['Solution by LU']=(end-start)*1000;

        M_I_s=self.M_I_mult_vector(s,I,add_dic,mult_dic);
        ecc_code=self.sub_vectors(d_I[self.u:self.m],M_I_s[self.u:self.m]);#,add_dic,"E_decode_by_UL");


        start = time.time();
        #info=self.luby_decoding(self.Ecc_neighbors,ec_code,I[self.u:self.m],add_dic);
        info=self.luby_decoding_by_solutions(Ecc_solutions,ecc_code);#,add_dic);
        end = time.time();
        times['Luby decoding']=(end-start)*1000;

        #if info is None:
        #    print("No solution to Ecc");
        #    return None;
        self.print_message_with_time("Luby decoding",start,end);
        return (s,info);




    def get_solution_by_LU(self,LU_tuple,d):
        (LU_matrix,p)=LU_tuple;

        if self.bits<=64:
            LU_list=LU_matrix.tolist();
            d_list=d.tolist();
            L_cols=LU_matrix.shape[1];
        else:
            LU_list=LU_matrix;
            d_list=d;
            L_cols=len(LU_matrix[0]);
        U_cols=L_cols;
        U_rows=U_cols;

        #L solution
        y=[0]*L_cols;
        for i in range(0,L_cols):
            y[i]=d_list[p[i]];
            for j in range(0,i):
                y[i]-=(LU_list[i][j]*y[j]);

        #U solution
        x=[0]*U_cols;
        for i in range(0,U_cols):
            index=U_cols-1-i;
            x[index]=y[index];
            for j in range(index+1,U_cols):
                x[index]-=LU_list[index][j]*x[j];
            factor=int(gp2.invert(gp2.mpz(LU_list[index][index]),gp2.mpz(self.fn)));
            x[index]=(x[index]*factor)%self.fn;
        if self.bits<=64:
            result=np.array(x,dtype=np.ulonglong);
        else:
            result=x;
        return result;




    def E_decode_by_PLUQ(self,PLUQ_tuple,Ecc_solutions,d_I,I,times):
        d_I_top=d_I[0:self.u];
        I_bottom=I[self.u:self.m];
        start = time.time();
        s=self.get_solution_by_PLUQ(PLUQ_tuple,d_I_top);
        end = time.time();
        self.print_message_with_time("Solution by PLUQ",start,end);
        times['Solution by PLUQ']=(end-start)*1000;

        start = time.time();
        M_I_bottom_s=self.M_bottom_I_mult_vector(s,I_bottom);
        end = time.time();
        times['M_I_bottom mult s']=(end-start)*1000;

        start = time.time();
        ecc_code=self.sub_vectors(d_I[self.u:self.m],M_I_bottom_s);
        end = time.time();
        times['Sub vectors d_bot[I]-M_bot[I]s']=(end-start)*1000;


        start = time.time();
        #info=self.luby_decoding(self.Ecc_neighbors,ec_code,I[self.u:self.m],add_dic);
        info=self.luby_decoding_by_solutions(Ecc_solutions,ecc_code);#,add_dic);
        end = time.time();
        times['Luby decoding']=(end-start)*1000;

        #if info is None:
        #    print("No solution to Ecc");
        #    return None;
        self.print_message_with_time("Luby decoding",start,end);
        return (s,info);





    def get_solution_by_PLUQ(self,PLUQ_tuple,d):
        (LU_matrix,P,Q,Q_T)=PLUQ_tuple;

        if self.bits<=64:
            LU_list=LU_matrix.tolist();
            d_list=d.tolist();
            L_cols=LU_matrix.shape[1];
        else:
            LU_list=LU_matrix;
            d_list=d;
            L_cols=len(LU_matrix[0]);
        U_cols=L_cols;
        U_rows=U_cols;

        #L solution
        y=[0]*L_cols;
        for i in range(0,L_cols):
            y[i]=d_list[P[i]];
            for j in range(0,i):
                y[i]-=(LU_list[i][j]*y[j]);

        #U solution
        x=[0]*U_cols;
        for i in range(0,U_cols):
            index=U_cols-1-i;
            x[index]=y[index];
            for j in range(index+1,U_cols):
                x[index]-=LU_list[index][j]*x[j];
            factor=int(gp2.invert(gp2.mpz(LU_list[index][index]),gp2.mpz(self.fn)));
            x[index]=(x[index]*factor)%self.fn;

        s=[x[Q_T[i]] for i in range(0,U_cols)];
        if self.bits<=64:
            result=np.array(s,dtype=np.ulonglong);
        else:
            result=s;
        return result;



    def get_solution_by_instructions(self,instructions,d,solution_len):
        if self.bits<=64:
            d_list=d.tolist();
        else:
            d_list=d;
        for current in instructions:
            inst_type=current[0];
            if inst_type=="r":
                index1=current[1];
                index2=current[2];
                factor=current[3];
                temp=d_list[index1];
                d_list[index1]=(factor*d_list[index2])%self.fn;
                d_list[index2]=temp;
            elif inst_type=="m":
                index=current[1];
                factor=current[2];
                d_list[index]=(factor*d_list[index])%self.fn;
            if inst_type=="s":
                index1=current[1];
                index2=current[2];
                factor=current[3];
                d_list[index1]=(d_list[index1]-d_list[index2]*factor)%self.fn;
        result_list=d_list[0:solution_len];

        if self.bits<=64:
            result=np.array(result_list,dtype=np.ulonglong);
        else:
            result=result_list;
        return result;



    def luby_decoding(self,Ecc_neighbors,code,I):
        info_len=self.w;
        info_list=[0]*info_len;
        code_len=len(code);
        if self.bits<=64:
            code_list=code.tolist();
        else:
            code_list=code;
        solved_counter=0;
        #iterations_counter=0;
        solved_symbols=[False]*info_len;
        active_rows=[I[i]==1 for i in range(0,code_len)];
        change=True;
        while(solved_counter<info_len and change):
            change=False;
            #iterations_counter+=1;
            for i in range(0,code_len):
                if active_rows[i]:
                    current_row=Ecc_neighbors[i];
                    #unsolved_i=[x for x in row_i if not solved_symbols[x]];
                    #if len(unsolved_i)==1:
                    current_unsolved_counter=0;
                    for x in current_row:
                        if not solved_symbols[x]:
                            current_unsolved_counter+=1;
                            if current_unsolved_counter==2:
                                break;
                            if current_unsolved_counter==1:
                                first_unsolved=x;
                    if current_unsolved_counter==0:
                        #Ecc_neighbors[i]=[];
                        active_rows[i]=False;
                    elif current_unsolved_counter==1:
                        symbol=code_list[i];
                        for index in current_row:
                            symbol-=info_list[index];
                        #index=unsolved_i[0];
                        info_list[first_unsolved]=symbol%self.fn;
                        #Ecc_neighbors[i]=[];
                        active_rows[i]=False;
                        change=True;
                        solved_counter+=1;
                        solved_symbols[first_unsolved]=True;
                        if solved_counter==info_len:
                            break;                   
        if solved_counter<info_len:
            return None;
        if self.bits<=64:
            info=np.array(info_list,dtype=np.ulonglong);
        else:
            info=info_list;
        return info;                      

        
    def offline_luby_decoding(self,Ecc_neighbors,I):
        func_name='offline_luby_decoding';
        #add_dic[func_name]=0;
        info_len=self.w;
        code_len=len(Ecc_neighbors);
        solved_counter=0;
        #iterations_counter=0;
        solved_symbols=[False]*info_len;
        active_rows=[I[i]==1 for i in range(0,code_len)];
        symbols_solutions=[];#[None]*info_len;
        change=True;
        while(solved_counter<info_len and change):
            change=False;
            #iterations_counter+=1;
            for i in range(0,code_len):
                if active_rows[i]:
                    current_row=Ecc_neighbors[i];
                    current_row_len=len(current_row);
                    current_unsolved_counter=0;
                    for j in range(0,current_row_len):
                        x=current_row[j];
                        if not solved_symbols[x]:
                            current_unsolved_counter+=1;
                            if current_unsolved_counter==2:
                                break;
                            if current_unsolved_counter==1:
                                first_unsolved_symbol=x;
                                first_unsolved_index=j;
                    if current_unsolved_counter==0:
                        active_rows[i]=False;
                    elif current_unsolved_counter==1:
                        #current_symbol_solution=[0]*(current_row_len+1);
                        #current_symbol_solution[0]=i;
                        #current_symbol_solution[1:]=current_row[:];
                        current_symbol_solution=[i];
                        current_symbol_solution.extend(current_row);

                        if(first_unsolved_index!=0):
                            temp=current_symbol_solution[1];
                            current_symbol_solution[1]=first_unsolved_symbol;
                            current_symbol_solution[1+first_unsolved_index]=temp;
                        
                        #symbols_solutions[solved_counter]=current_symbol_solution;
                        symbols_solutions.append(current_symbol_solution);
                        solved_counter+=1;
                        active_rows[i]=False;
                        change=True;
                        solved_symbols[first_unsolved_symbol]=True;
                        if solved_counter==info_len:
                            break;                   
        if solved_counter<info_len:
            return None;
        return symbols_solutions;






    def improved_offline_luby_decoding(self,Ecc_neighbors,Ecc_decoding_tuple,I):
        info_len=self.w;
        code_len=len(Ecc_neighbors);
        #if len(I)!=code_len:
        #    print("Error in the length of I");

        (neighbors_degs,symbols_locations,deg_one_queue)=Ecc_decoding_tuple;
        
        solved_counter=0;
        #iterations_counter=0;
        solved_symbols=[False]*info_len;
        symbols_solutions=[];


        while(not deg_one_queue.empty() and solved_counter<info_len):
            i=deg_one_queue.get();
            if I[i]==1 and neighbors_degs[i]==1:
                current_row=Ecc_neighbors[i];
                current_row_len=len(current_row);
                for j in range(0,current_row_len):
                    if not solved_symbols[current_row[j]]:
                        current_unsolved_symbol=current_row[j];
                        current_unsolved_symbol_index=j;
                        break;
                
                current_symbol_solution=[i];
                current_symbol_solution.extend(current_row);
                if(current_unsolved_symbol_index!=0):
                    temp=current_symbol_solution[1];
                    current_symbol_solution[1]=current_unsolved_symbol;
                    current_symbol_solution[1+current_unsolved_symbol_index]=temp;
                symbols_solutions.append(current_symbol_solution);
                solved_counter+=1;
                solved_symbols[current_unsolved_symbol]=True;
                neighbors_degs[i]=0;
                if solved_counter==info_len:
                        break;

                current_symbol_locations=symbols_locations[current_unsolved_symbol];
                for i in current_symbol_locations:
                    if I[i]==1:
                        neighbors_degs[i]-=1;
                        if neighbors_degs[i]==1:
                            deg_one_queue.put(i);
            #elif I[i]==1 and neighbors_degs[i]!=1:
            #    print(i);
        if solved_counter<info_len:
            return None;
        return symbols_solutions;


    def luby_decoding_by_solutions(self,Ecc_solutions,code):
        info_len=len(Ecc_solutions);
        info_list=[0]*info_len;
        if self.bits<=64:
            code_list=code.tolist();
        else:
            code_list=code;
        solved_symbols=[False]*info_len;
        for current_solution in Ecc_solutions:
            current_solution_len=len(current_solution);
            symbol=code_list[current_solution[0]];
            symbol_index=current_solution[1];
            solved_symbols[symbol_index]=True;
            if(current_solution_len>2):
                for j in range(2,current_solution_len):
                    #if solved_symbols[current_solution[j]]==False:
                    #    print("Error");
                    symbol-=info_list[current_solution[j]];
                symbol=symbol%self.fn;
            info_list[symbol_index]=symbol;
        if self.bits<=64:
            info=np.array(info_list,dtype=np.ulonglong);
        else:
            info=info_list;
        return info;



    def cds_decode(self,c,gamma,alpha,beta,beta_support,v):
        beta_c=self.sparse_dot_product(beta,beta_support,c);
        t=(alpha-beta_c);#%self.fn;  #t=alpha-h[I_NOT]c=K+h[I]c
        beta_T=self.sparse_vector_mult_T(beta,beta_support);
        phi=self.sub_vectors(gamma,beta_T);
        s=self.dot_product(phi,v);
        cds_secret=(t-s)%self.fn;
        return cds_secret;


    def add_vectors(self,a,b):
        if self.bits<64:
            result=(a+b)%self.fn;
            return result;
        if self.bits==64:
            a_list=a.tolist();
            b_list=b.tolist();
        else:
            a_list=a;
            b_list=b;
        result_len=len(a_list);
        result_list=[(a_list[i]+b_list[i])%self.fn for i in range(0,result_len)];
        if self.bits==64:
            result=np.array(result_list,dtype=np.ulonglong);
        else:
            result=result_list;
        return result;



    def scalar_mult_and_add_vector(self,x,a,b):
        if self.bits<32:
            result=(x*a+b)%self.fn;
            return result;
        if self.bits<=64:
            a_list=a.tolist();
            b_list=b.tolist();
        else:
            a_list=a;
            b_list=b;
        result_len=len(a_list);
        result_list=[(x*a_list[i]+b_list[i])%self.fn for i in range(0,result_len)];
        if self.bits<=64:
            result=np.array(result_list,dtype=np.ulonglong);
        else:
            result=result_list;
        return result;





    def sub_vectors(self,a,b):
        if self.bits<64:
            c=a.astype('int64');
            d=b.astype('int64');
            result=(c-d)%self.fn;
            return result;
        if self.bits==64:
            a_list=a.tolist();
            b_list=b.tolist();
        else:
            a_list=a;
            b_list=b;
        result_len=len(a_list);
        result_list=[(a_list[i]-b_list[i])%self.fn for i in range(0,result_len)];
        if self.bits==64:
            result=np.array(result_list,dtype=np.ulonglong);
        else:
            result=result_list;
        return result;





    def dot_product(self,a,b): 
        if self.bits<32:
            result=int((a.dot(b))%self.fn);
            return result;
        if self.bits<=64:
            a_list=a.tolist();
            b_list=b.tolist();
        else:
            a_list=a;
            b_list=b;
        result=0;
        for x,y in zip(a_list,b_list):
            result+=x*y;
        return result%self.fn;




        
    def sparse_dot_product(self,a,a_support,b):    
        if self.bits<32:
            result=int((a.dot(b))%self.fn);
            return result;
        if self.bits<=64:
            a_list=a.tolist();
            b_list=b.tolist();
        else:
            a_list=a;
            b_list=b;
        result=0;
        for i in a_support:
            x=a_list[i];
            y=b_list[i];
            result+=x*y;
        return result%self.fn;



    def HL_dot_product(self,a,b):
        (a_H,a_L)=self.decompose_vector_high_low(a);
        (b_H,b_L)=self.decompose_vector_high_low(b);

        aHbH=(a_H.dot(b_H));
        aHbL=(a_H.dot(b_L));
        aLbH=(a_L.dot(b_H));
        aLbL=(a_L.dot(b_L));
        aHbL_aLbH=aHbL+aLbH;

        result=(self.factor_HL_squared*int(aHbH)+factor_HL*int(aHbL_aLbH)+int(aLbL))%self.fn;
        return result;




    def vector_mult_T(self,h):
        if self.bits>64:
            return self.vector_mult_T_neighbors(h);
        if self.bits==64:
            left=self.vector_mult_matrix_neighbors(h,self.M_neighbors);
            h_right=h[self.u:self.m];
            Ecc_csr_transpose=np.transpose(self.Ecc_csr);
            right=self.Ecc_matrix_mult_vector_HL(Ecc_csr_transpose,h_right);
            result=np.concatenate((left,right), axis=0);
            return result;                            
        if self.bits==32:
            left=self.matrix_HL_mult_vector(self.transposed_M_HL_tuple,h);
        else:
            left=(np.transpose(self.M_csr).dot(h))%self.fn;
        h_right=h[self.u:self.m];
        right=(np.transpose(self.Ecc_csr).dot(h_right))%self.fn;
        result=np.concatenate((left,right), axis=0)
        return result;



    def vector_mult_T_neighbors(self,h): 
        left=self.vector_mult_matrix_neighbors(h,self.M_neighbors);
        h_right=h[self.u:self.m];
        right=self.vector_mult_Ecc_neighbors(h_right,self.Ecc_neighbors);

        if self.bits<=64:
            result=np.concatenate((left,right), axis=0)
        else:
            result=left+right;
        return result;

    def vector_mult_matrix_neighbors(self,h,M_neighbors):
        if self.bits<=64:
            h_list=h.tolist();
        else:
            h_list=h;
        h_len=len(h);
        result_len=self.k;
        result_list=[0]*result_len;
        rows_neighbors=M_neighbors[0];
        data_neighbors=M_neighbors[1];
        for i in range(0,h_len):
            h_i=h_list[i];
            if h_i!=0:
                row_i=rows_neighbors[i];
                data_i=data_neighbors[i];
                for col,data in zip(row_i,data_i):
                    result_list[col]+=h_i*data;
        result_list=[x%self.fn for x in result_list];
        if self.bits<=64:
            result=np.array(result_list,dtype=np.ulonglong);
        else:
            result=result_list;
        return result;




    def vector_mult_Ecc_neighbors(self,h,Ecc_neighbors):  
        if self.bits<=64:
            h_list=h.tolist();
        else:
            h_list=h;
        h_len=len(h);
        result_len=self.w;
        result_list=[0]*result_len;
        for i in range(0,h_len):
            h_i=h_list[i];
            if h_i!=0:
                row_i=Ecc_neighbors[i];
                for col in row_i:
                    result_list[col]+=h_i;
        result_list=[x%self.fn for x in result_list];
        if self.bits<=64:
            result=np.array(result_list,dtype=np.ulonglong);
        else:
            result=result_list;
        return result;





    def sparse_vector_mult_T(self,h,h_support):
        if self.bits>=64:
            result=self.sparse_vector_mult_T_neighbors(h,h_support);
            return result;
        if self.bits==32:
            left=self.matrix_HL_mult_vector(self.transposed_M_HL_tuple,h);
        else:
            left=(np.transpose(self.M_csr).dot(h))%self.fn;
        h_right=h[self.u:self.m];
        right=(np.transpose(self.Ecc_csr).dot(h_right))%self.fn;  
        result=np.concatenate((left,right), axis=0)
        return result;




    def sparse_vector_mult_T_neighbors(self,h,h_support):
        left=self.sparse_vector_mult_matrix_neighbors(h,h_support,self.M_neighbors);
        h_right=h[self.u:self.m];
        h_right_support=[x-self.u for x in h_support if x>=self.u];
        right=self.sparse_vector_mult_Ecc_neighbors(h_right,h_right_support,self.Ecc_neighbors);
        if self.bits<=64:
            result=np.concatenate((left,right), axis=0)
        else:
            result=left+right;
        return result;



    def sparse_vector_mult_matrix_neighbors(self,h,h_support,M_neighbors):   
        if self.bits<=64:
            h_list=h.tolist();
        else:
            h_list=h;
        result_len=self.k;
        result_list=[0]*result_len;
        rows_neighbors=M_neighbors[0];
        data_neighbors=M_neighbors[1];
        for i in h_support:
            h_i=h_list[i];
            row_i=rows_neighbors[i];
            data_i=data_neighbors[i];
            for col,data in zip(row_i,data_i):
                result_list[col]+=h_i*data;
        result_list=[x%self.fn for x in result_list];
        if self.bits<=64:
            result=np.array(result_list,dtype=np.ulonglong);
        else:
            result=result_list;
        return result;



    def sparse_vector_mult_Ecc_neighbors(self,h,h_support,Ecc_neighbors):  
        if self.bits<=64:
            h_list=h.tolist();
        else:
            h_list=h;
        result_len=self.w;
        result_list=[0]*result_len;
        for i in h_support:
            h_i=h_list[i];
            row_i=Ecc_neighbors[i];
            for col in row_i:
                result_list[col]+=h_i;
        result_list=[x%self.fn for x in result_list];

        if self.bits<=64:
            result=np.array(result_list,dtype=np.ulonglong);
        else:
            result=result_list;
        return result;






    def vector_mult_coo_matrix(self,h,M_coo):    
        vec_len=len(h);
        result_len=M_coo.shape[1];
        result_list=[0]*result_len;
        h_list=h.tolist();
        prev_i=-1;
        for i,j,data in zip(M_coo.row,M_coo.col,M_coo.data):
            h_i=h_list[i];
            if h_i!=0:
                result_list[j]=result_list[j]+int(data)*h_i;
                prev_i=i;
        result_list=[x%self.fn for x in result_list];
        result=np.array(result_list,dtype=np.ulonglong);
        return result;



    def print_message_with_time(self,message,start,end):
        print(message+": {0:.3f}".format(1000*(end - start))+"msec");



    def send_scalar(self,sock,x):
        sent=sock.sendall(x.to_bytes(self.bytes,'little'));
        if sent == 0:
            return False;
        #sent=client_socket.sendall(b'Hello');
        return True;


    def recv_scalar(self,sock):
        x_bytes=sock.recv(self.bytes);
        if x_bytes == b'':
            return None;
        while len(x_bytes)<self.bytes:
            delta=self.bytes-len(x_bytes);
            x_bytes=x_bytes+sock.recv(delta);
        x=int.from_bytes(x_bytes,'little');
        return x;



    def send_vector(self,sock,v):
        if self.bits<=64:
            v_list=v.tolist();
        else:
            v_list=v;
        data=pickle.dumps(v_list);
        v_size=len(data);
        sent=sock.sendall(v_size.to_bytes(packet_size,'little'));
        if sent == 0:
            return False;
        sent=sock.sendall(data);
        if sent == 0:
            return False;
        return True;

    def recv_vector(self,sock):
        size_message=sock.recv(packet_size);
        if size_message == b'':
            return None;
        v_size=int.from_bytes(size_message,'little');
        message=sock.recv(v_size);
        while len(message)<v_size:
            delta=v_size-len(message);
            message=message+sock.recv(delta);
        v_list=pickle.loads(message);
        if self.bits<=64:
            v=np.array(v_list,dtype=np.ulonglong);
        else:
            v=v_list;
        return v;


    def print_matrix(self,mat):
        for row in mat:
            print(*row);
            print();



    def M_I_top(self,I):
        if self.bits<=64:
            M_top_I=np.zeros((self.u,self.k),dtype=np.ulonglong);
            for i,j,data in zip(self.M_coo.row,self.M_coo.col,self.M_coo.data):
                if i>=self.u:
                    break;
                if I[i]==1:
                    M_top_I[i,j]=data;
            return M_top_I;
        
        rows_neighbors=self.M_neighbors[0];
        data_neighbors=self.M_neighbors[1];
        result_list=[];
        for i in range(0,self.u):
            result_row=[0]*self.k;
            if I[i]==1:
                current_row=rows_neighbors[i];
                current_data=data_neighbors[i];
                for j,data in zip(current_row,current_data):
                    result_row[j]=data;
            result_list.append(result_row);
        return result_list;

    def M_I_top_neighbors(self,I):
        (rows_neighbors,data_neighbors)=self.M_neighbors;

        result_rows_neighbors=[];
        result_data_neighbors=[];
        for i in range(0,self.u):
            current_result_row=[];
            current_result_data=[];
            if I[i]==1:
                current_row=rows_neighbors[i];
                current_data=data_neighbors[i];
                current_result_row=current_row[:];
                current_result_data=current_data[:];
            result_rows_neighbors.append(current_result_row);
            result_data_neighbors.append(current_result_data);
        result_neighbors=(result_rows_neighbors,result_data_neighbors);
        return result_neighbors;



    def matrix_top_I_compressed(self,M_neighbors,I_top,matrix_cols):
        top_rows=len(I_top);
        M_top_I=np.zeros((top_rows,matrix_cols),dtype=np.ulonglong);
        hamming_weight=0;
        current_row_index=0;
        rows_neighbors=M_neighbors[0];
        data_neighbors=M_neighbors[1];
        for i in range(0,top_rows):
            if I_top[i]==1:
                matrix_row=rows_neighbors[i];
                matrix_row_data=data_neighbors[i];
                for j,data in zip(matrix_row,matrix_row_data):
                    M_top_I[current_row_index,j]=data;
                current_row_index+=1;
        return M_top_I[0:current_row_index];





            






    def receiver_oblivious_transfer(self,num_of_OTs,choices,bits,ip_addr,port,times):
        print("Receiver starts OT");
        start=time.time();
        #ot_outputs_list=OT.Receiver(num_of_OTs,choices,ip_addr,port);
        ot_outputs_list=OT.ActiveReceiver(num_of_OTs,choices,bits,ip_addr,port);
        end = time.time();
        times['Receiver Oblivious Transfer']=(end-start)*1000;
        return ot_outputs_list;



    def sender_oblivious_transfer(self,num_of_OTs,ot_0_inputs,ot_1_inputs,bits,ip_addr,port,times):
        print("Sender starts OT");
        if self.bits<=64:
            inputs_0_list=ot_0_inputs.tolist();
            inputs_1_list=ot_1_inputs.tolist();
        else:
            inputs_0_list=ot_0_inputs;
            inputs_1_list=ot_1_inputs;
        start = time.time();
        #OT.Sender(num_of_OTs,inputs_0_list,inputs_1_list,ip_addr,port);
        OT.ActiveSender(num_of_OTs,inputs_0_list,inputs_1_list,bits,ip_addr,port);
        end = time.time();
        times['Sender Oblivious Transfer']=(end-start)*1000;







#Threading tries
    
    def M_mult_vector_threading(self,r):
        M_r_list=[0]*self.m;
        #t1=threading.Thread(target=self.M_mult_vector_thread,args=(r,0,self.m//2,M_r_list));
        #t1.start();
        #t2=threading.Thread(target=self.M_mult_vector_thread,args=(r,self.m//2,self.m,M_r_list));
        #t2.start();
        t1=threading.Thread(target=self.M_mult_vector_range,args=(r,0,self.m,M_r_list));
        start=time.time();
        t1.start();
        t1.join();
        end=time.time();
        self.print_message_with_time("Here",start,end);
        #t2.join();
        if self.bits<=64:
            M_r=np.array(M_r_list,dtype=np.ulonglong);
        else:
            M_r=M_r_list;
        return M_r;


    def M_mult_vector_range(self,r,start_index,end_index,result):
        if self.bits<=64:
            r_list=r.tolist();
        else:
            r_list=r;
        rows_neighbors=self.M_neighbors[0];
        data_neighbors=self.M_neighbors[1];
        for i in range(start_index,end_index):
            current_row=rows_neighbors[i];
            current_data=data_neighbors[i]
            current_result=0;
            for col,data in zip(current_row,current_data):
                current_result+=data*r_list[col];
            result[i]=current_result%self.fn;












def main():
    k=182;
    w=10000;
    mu=0.25;
    d_max=10;
    bits=128;
    u_factor=1.4;


    vole=VOLE(k,w,mu,d_max,bits,u_factor);
    vole.robust_soliton_distribution(10000);
    vole.robust_soliton_distribution(20000);

if __name__ == "__main__":
    main();












