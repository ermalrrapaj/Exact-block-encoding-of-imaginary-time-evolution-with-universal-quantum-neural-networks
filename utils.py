import numpy as np
import qiskit
from qiskit.circuit import Parameter
# from qiskit.opflow import I, X, Y, Z
# from qiskit.quantum_info import Pauli
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
 
X = SparsePauliOp("X")
Y = SparsePauliOp('Y')
Z = SparsePauliOp("Z")
I = SparsePauliOp("I")

# from qiskit.opflow import PauliTrotterEvolution, Suzuki
#from qiskit.opflow.primitive_ops.pauli_op import PauliOp

# X=Pauli('X')
# Y=Pauli('Y')
# Z=Pauli('Z')
# I=Pauli('I')

def two_paulis_rotation(theta: Parameter, pauliOp1: SparsePauliOp, pauliOp2: SparsePauliOp):
    if pauliOp1 not in [I, X, Y, Z]:
        raise Exception('pauliOp1 is not one of I, X, Y, or Z')
    if pauliOp2 not in [I, X, Y, Z]:
        raise Exception('pauliOp2 is not one of I, X, Y, or Z')

    H = 0.5 * (pauliOp2 ^ pauliOp1) # Qiskit uses "little endian" bit ordering
    # evolution_op = (theta * H).exp_i() # exp(-iθH)
    evolution_op = PauliEvolutionGate(H, time=theta) # exp(-iθH)
    return evolution_op

    # trotterized_op = PauliTrotterEvolution(trotter_mode = Suzuki(order = 1)).convert(evolution_op)
    # circuit = trotterized_op.to_circuit()
    # return circuit.to_gate()

def reset(circ,qubits):
    """ reset qubits"""
    for i in range(len(qubits)):
        circ.reset(qubits[i])

def measure(circ,qubits,cbits):
    """ measure qubits and store outputs to cbits"""
    assert len(qubits)<=len(cbits)
    for i in range(len(qubits)):
        circ.measure(qubits[i],cbits[i])
        
def apply_h(circ,qubits):
    """ apply Hadamard gate on qubits """
    for i in range(len(qubits)):
        circ.h(qubits[i])
        
def TwoBD(K,t):
    k=K*t
    """two body RBM identity for coupling K and time t"""
    C = np.arccos(np.exp(-2*abs(k)))/2
    A = np.exp(abs(k))/2
    s=np.sign(k)
    return A,C,s

def apply_X(circ,physical_qubits,ancilla_qubits,angles):
    k = len(physical_qubits)
    assert k == len(ancilla_qubits)
    for i in range(k):
        circ.append(two_paulis_rotation(2*angles[i][0], X, X), [physical_qubits[i],ancilla_qubits[i]])
        circ.rx(2*angles[i][1],ancilla_qubits[i])
        
def apply_Y(circ,physical_qubits,ancilla_qubits,angles):
    k = len(physical_qubits)
    assert k == len(ancilla_qubits)
    for i in range(k):
        circ.append(two_paulis_rotation(2*angles[i][0], Y, X), [physical_qubits[i],ancilla_qubits[i]])
        circ.rx(2*angles[i][1],ancilla_qubits[i])
        
def apply_Z(circ,physical_qubits,ancilla_qubits,angles):
    k = len(physical_qubits)
    assert k == len(ancilla_qubits)
    for i in range(k):
        circ.append(two_paulis_rotation(2*angles[i][0], Z, X), [physical_qubits[i],ancilla_qubits[i]])
        circ.rx(2*angles[i][1],ancilla_qubits[i])
        
def apply_ZZ(circ,physical_qubits,ancilla_qubits,angles):
    k = len(physical_qubits)
    assert k == len(ancilla_qubits)
    for i in range(k):
        circ.append(two_paulis_rotation(2*angles[i][0], Z, X), [physical_qubits[i][0],ancilla_qubits[i]])
        circ.append(two_paulis_rotation(2*angles[i][1], Z, X), [physical_qubits[i][1],ancilla_qubits[i]])