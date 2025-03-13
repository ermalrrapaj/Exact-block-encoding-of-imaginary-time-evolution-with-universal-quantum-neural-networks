import numpy as np
import re
from qiskit.circuit import QuantumCircuit, Parameter, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import AerSimulator
from datetime import datetime
import argparse


def measure(circ,n,m):
    circ.measure(n,m)

def reset(circ,n,m):
    circ.measure(n,m)
    circ.reset(n)

def apply_X_step(circ,n,m,phi1,phi2):
    circ.rxx(2*phi1,n,m)
    circ.rx(2*phi2,m)
        
def apply_ZZ_step(circ,n1,n2,m,phi1,phi2):
    circ.rzx(2*phi1,n1,m)
    circ.rzx(2*phi2,n2,m)

def apply_H(circ,n):
    circ.h(n)
        
def TwoBD_rbm(K,t):
    C = np.arccos(np.exp(-2*abs(K*t)))/2
    A = np.exp(abs(K*t))/2
    s = np.sign(K)
    return A,C,s

def mc_counts(counts,n,nTrot):
    cred = {};
    str1 = (2*nTrot + 1)*n*"0"
    for strtot in counts.keys():
        assert(len(strtot)==(2*nTrot + 2)*n)
        str2 = re.sub("^"+str1, "", strtot)
        #if (len(str2) == len(strtot)):
        #    continue
        if len(str2)==n:
            if str2 not in cred:
                cred[str2] = counts[strtot]
            else:
                cred[str2] += counts[strtot]
    ctot = sum(cred.values())
    for str2 in cred.keys():
        cred[str2] /= ctot
    
    return cred

def get_params(n,Jx,Jzz,dT,T):
    nTrot = int(T/dT) ;
    Ax1,Cx1,sx = TwoBD_rbm(Jx,dT/2);
    Ax2,Cx2,sx2 = TwoBD_rbm(Jx,dT);
    Azz,Czz,szz = TwoBD_rbm(Jzz,dT);
    
    return Ax1,Cx1,sx,Ax2,Cx2,sx2,Azz,CZZ,szz

def build_circ(n,nTrot,JX1,SX,JX2,SX2,JZZ,SZZ,Cx1,Cx2,Czz,applyH):
    qubits = QuantumRegister(n+1);
    clbits = ClassicalRegister((2*nTrot + 2)*n);
    # print((2*nTrot + 2)*n)
    circ = QuantumCircuit(qubits, clbits) # quantum, classical bits
    
    # create initial state |+++...>
    for i in range(n):
        apply_H(circ,n)
    
    #First step
    # apply X
    counter=0
    for i in range(n):
        apply_X_step(circ,i,n,JX1,SX*JX1)
        reset(circ,n,n+counter)
        counter+=1
    # apply ZZ
    for i in range(n):
        apply_ZZ_step(circ,i,(i+1)%n,n,JZZ,SZZ*JZZ)
        reset(circ,n,n+counter)
        counter+=1

    # Trotter steps
    for i in range(1, nTrot):
        # apply X
        for i in range(n):
            apply_X_step(circ,i,n,JX2,SX2*JX2)
            reset(circ,n,n+counter)
            counter+=1
        # apply ZZ
        for i in range(n):
            apply_ZZ_step(circ,i,(i+1)%n,n,JZZ,SZZ*JZZ)
            reset(circ,n,n+counter)
            counter+=1
    
    # final step
    for i in range(n):
        apply_X_step(circ,i,n,JX1,SX*JX1)
        reset(circ,n,n+counter)
        counter+=1
            
    # rotate to X basis
    if applyH:
        for i in range(n):
            apply_H(circ,n)
        
    # measure the physical qubits
    for i in range(n):
        counter+=1
        circ.measure(i,i)

    assert((2*nTrot + 2)*n==counter)

    return circ

def get_states(n,Jx,Jzz,dT,Tlist,nShots,seed,applyH=False):

    Ax1,Cx1,sx = TwoBD_rbm(Jx,dT/2);
    Ax2,Cx2,sx2 = TwoBD_rbm(Jx,dT);
    Azz,Czz,szz = TwoBD_rbm(Jzz,dT);

    # couplings for Transverse Ising model
    JX1 = Parameter('JX1')
    JX2 = Parameter('JX2')
    SX = Parameter('SX')
    SX2 = Parameter('SX2')
    JZZ = Parameter('JZZ1')
    SZZ = Parameter('SZ')
    

    # simulator = Aer.get_backend('aer_simulator',seed=10)
    simulator=AerSimulator(seed_simulator=seed)
    simulator.set_options(method='statevector', device='CPU', #cuStateVec_enable = True,
                          batched_shots_gpu=True, shot_branching_enable=False, max_shot_size = 100000)
    

    states=[];
    # ps=[]

    counter=0
    for T in Tlist:
        nTrot = int(T/dT);
        # Create circuit
        circ = build_circ(n,nTrot,JX1,SX,JX2,SX2,JZZ,SZZ,Cx1,Cx2,Czz,applyH);
        if (nTrot == 1):
            circ_p = circ.assign_parameters({JX1: Cx1, SX: sx, JZZ: Czz, SZZ: szz})
        else:
            circ_p = circ.assign_parameters({JX1: Cx1, JX2: Cx2, SX: sx, SX2: sx2, JZZ: Czz, SZZ: szz})
        circ_p = transpile(circ_p, simulator);
        result = simulator.run(circ_p, shots = nShots).result()
        counts = result.get_counts()

        counts = mc_counts(counts,n,nTrot)
        states.append(counts)
    return states
    
def get_ZZ_stats(n,counts):
    result=0
    result_sq=0
    spin={'0':1, '1': -1}
    for key in counts:
        tmp=0.0
        tmp_sq=0.0
        for i in range(n):
            tmp+=spin[key[i]]*spin[key[(i+1)%n]]
            tmp_sq+=(spin[key[i]]*spin[key[(i+1)%n]])**2
        result+=tmp*counts[key]
        result_sq+=tmp_sq*counts[key]
    return result, np.sqrt(result_sq-result**2)

def get_X_stats(n,counts):
    result=0
    result_sq=0
    spin={'0':1, '1': -1}
    for key in counts:
        tmp=0.0
        tmp_sq=0.0
        for i in range(n):
            tmp+=spin[key[i]]
            tmp_sq+=(spin[key[i]])**2
        result+=tmp*counts[key]
        result_sq+=tmp_sq*counts[key]
    return result, np.sqrt(result_sq-result**2)
        

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--n", default=3, type=int, help="system size")
    parser.add_argument("--jx", default=1.0, type=float, help="jx coupling")
    parser.add_argument("--jzz", default=1.0, type=float, help="jzz coupling")
    parser.add_argument("--tmax", default=1.0, type=float, help="max imaginary time")
    parser.add_argument("--nsteps", default=6, type=int, help="number of equidistant timesteps to sample") 
    parser.add_argument("--dt", default=0.01, type=float, help="time step size for Trotter steps")
    parser.add_argument("--nshots", default=10000, type=int, help="number of samples")
    parser.add_argument("--which", default=1, type=int, help="which time step to sample")
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args(); 
    # Tranverse Ising model 
    n = args.n; # system size
    
    np.random.seed(int(datetime.now().timestamp()))
    seed = np.random.randint(1,10**6,1)[0]
    Jx = args.jx;
    Jzz = args.jzz;
    dT = args.dt;
    Tlist=np.linspace(0,args.tmax,args.nsteps)[1:];
    nShots = args.nshots;#1000000;
    states_Z=get_states(n,Jx,Jzz,dT,[Tlist[args.which]],seed,nShots);
    states_X=get_states(n,Jx,Jzz,dT,[Tlist[args.which]],seed,nShots,True);
    np.savez('Ising_data_'+str(args.which)+'_seed_'+str(seed),\
                 z=states_Z,x=states_X,n=n,jx=Jx,jzz=Jzz,nshots=nShots,dt=dT,t=Tlist[args.which])

if __name__ == "__main__":
    main()