from qiskit.quantum_info import SparsePauliOp,Statevector
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector,Parameter
from qiskit.circuit.library import PauliEvolutionGate
from WarmStartUtils import *
import numpy as np
from functools import reduce


"""
Indexed Pauli Ops
"""
def indexedZ(i,n):
    return SparsePauliOp("I" * (n-i-1) + "Z" + "I" * i)
def indexedX(i,n):
    return SparsePauliOp("I" * (n-i-1) + "X" + "I" * i)
def indexedY(i,n):
    return SparsePauliOp("I" * (n-i-1) + "Y" + "I" * i)

"""
Hamiltonian from adjacency matrix A
"""
def getHamiltonian(A):
    n = len(A)
    H = 0 * SparsePauliOp("I" * n)
    for i in range(n):
        for j in range(n):
            H -= 1/4 * A[i][j] * indexedZ(i,n) @ indexedZ(j,n)
    return H.simplify()

"""
Default mixer
"""
def default_mixer(n):
    H_mix = reduce(lambda a,b: a+b, [indexedX(i,n) for i in range(n)])
    default_mixer = QuantumCircuit(n)
    default_mixer.append( PauliEvolutionGate(H_mix, Parameter('t')), range(n))
    return default_mixer

"""
PSC????
"""
# def PSC_init(z,theta):
#     n = len(z)
#     qc=QuantumCircuit(n)
#     thetas = {-1:theta,1:np.pi-theta}
#     for i,v in enumerate(z):
#         qc.ry(thetas[v],i)
#     return qc

# def PSC_mixer(z,theta):
#     n = len(z)
#     H_list = []
#     thetas = {-1:theta,1:np.pi-theta}
#     for i,v in enumerate(z):
#         H_list.append(np.sin(thetas[v])*indexedX(i,n) + np.cos(thetas[v])*indexedZ(i,n))
#     return  sum(H_list).simplify()

# def PSC_mixer_qc(z,theta):
#     n = len(z)
#     H_list = []
#     thetas = {-1:theta,1:np.pi-theta}
#     p = Parameter('t')
#     qc = QuantumCircuit(n)
#     for i,v in enumerate(z):
#         op = np.sin(thetas[v])*SparsePauliOp("X") + np.cos(thetas[v])*SparsePauliOp("Z")
#         qc.append( PauliEvolutionGate(op, p),[i])
#     return qc


def Q2_init(theta_list,rotation=None):
    angles = vertex_on_top(theta_list,rotation)
    n = len(angles)
    qc=QuantumCircuit(n)
    for i,v in enumerate(angles):
        qc.ry(v,i)
        qc.p(-np.pi/2,i)
    return qc

def Q2_mixer(theta_list,rotation=None):
    angles = vertex_on_top(theta_list,rotation)
    n = len(angles)
    H_list = []
    for i,v in enumerate(angles):
        H_list.append(-np.sin(v)*indexedY(i,n)+np.cos(v)*indexedZ(i,n))
    return sum(H_list).simplify()

def Q2_mixer_qc(theta_list,rotation=None):
    angles = vertex_on_top(theta_list,rotation)
    n = len(angles)
    p = Parameter('t')
    qc = QuantumCircuit(n)
    for i,v in enumerate(angles):
        op = -np.sin(v)*SparsePauliOp("Y")+np.cos(v)*SparsePauliOp("Z")
        qc.append( PauliEvolutionGate(op, p),[i])
    return qc


def Q3_init(theta_list,rotation=None):
    angles = vertex_on_top(theta_list,rotation)
    n = len(angles)
    qc=QuantumCircuit(n)
    for i,v in enumerate(angles):
        qc.ry(v[0],i)
        qc.p(v[1],i)
    return qc

def Q3_mixer(theta_list,rotation=None):
    angles = vertex_on_top(theta_list,rotation)
    n = len(angles)
    H_list = []
    for i,v in enumerate(angles):
        theta = v[0]
        phi = v[1]
        H_list.append(np.sin(theta)*np.cos(phi)*indexedX(i,n)+np.sin(theta)*np.sin(phi)*indexedY(i,n)+np.cos(theta)*indexedZ(i,n))
    return sum(H_list).simplify()

def Q3_mixer_qc(theta_list,rotation=None):
    angles = vertex_on_top(theta_list,rotation)
    n = len(angles)
    p = Parameter('t')
    qc = QuantumCircuit(n)
    for i,v in enumerate(angles):
        theta = v[0]
        phi = v[1]
        op = np.sin(theta)*np.cos(phi)*SparsePauliOp("X")+np.sin(theta)*np.sin(phi)*SparsePauliOp("Y")+np.cos(theta)*SparsePauliOp("Z")
        qc.append( PauliEvolutionGate(op, p),[i])
    return qc

def QAOA_Ansatz(cost,mixer=None,p=1,initial=None):
    n=cost.num_qubits
    qc=QuantumCircuit(n)
    if(mixer is None):
       qaoa_mixer =  default_mixer(n)
    else:
        qaoa_mixer = default_mixer(n)
    if(initial is None):
        qc.h(range(n))
    else:
        qc = initial.copy()
    Gamma = ParameterVector('γ',p)
    Beta = ParameterVector('β',p)
    for i in range(p):
        if(cost is not None):
            qc.append(PauliEvolutionGate(cost,Gamma[i]),range(n))
        qc.append(qaoa_mixer.assign_parameters([Beta[i]]),range(n))
    return qc


def single_circuit_optimization(ansatz,H,opt):
    history = {"cost": [], "params": []}
    def compute_expectation(x):
        psi = Statevector(ansatz.assign_parameters(x))
        l = psi.expectation_value(H).real
        history["cost"].append(l)
        history["params"].append(x)
        return -l
    res = opt.minimize(fun= compute_expectation, x0 = np.random.random(ansatz.num_parameters))
    return -res.fun,res.x,history

def circuit_optimization(ansatz,H,opt,reps=10,name=None):
    if(name is None):
        print(f"------------Beginning Optimization------------")
    else:
        print(f"------------Beginning Optimization: {name}------------")

    history_list = []
    param_list = []
    cost_list = []
    for i in range(reps):
        cost,params,history = single_circuit_optimization(ansatz,H,opt)
        history_list.append(history)
        param_list.append(params)
        cost_list.append(cost)
        print(f"Iteration {i} complete")
    return np.array(cost_list),np.array(param_list),history_list

def optimal_sampling_prob(ansatz,params,H,z):
    n = len(z)
    index = np.sum(np.array([2**(i-1) * (v+1) for i,v in enumerate(z)],dtype=int))
    s1 = np.zeros(2**n)
    s2 = np.zeros(2**n)
    
    s1[index] = 1
    s2[2**n-1-index]=1
    
    s1 = Statevector(s1)
    s2 = Statevector(s2)

    psi = Statevector(ansatz.assign_parameters(params))
    return abs((psi.inner(s1)))**2 + abs((psi.inner(s2)))**2