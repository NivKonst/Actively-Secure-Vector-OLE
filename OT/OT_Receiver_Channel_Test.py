import socket
import OT
import time


def oblivious_transfer(num_of_OTs,choices,ip_addr,port,time_sum):
    start = time.time();
    OT.ActiveReceiver(num_of_OTs,choices,8,ip_addr,port);
    end = time.time();
    delta=(end-start)*1000;
    time_sum[0]+=delta;



ip_addr=socket.gethostbyname(socket.gethostname());
ot_port=2000;
repetitions=300;
time_sum=[0];
for i in range(1,repetitions+1):
    print(i);
    ot_outputs_vector=oblivious_transfer(1,[1],ip_addr,ot_port,time_sum);
    time.sleep(7);
time_sum[0]=time_sum[0]/repetitions;
print(time_sum[0]);
