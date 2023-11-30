''' 
THIS IS THE CONFIGURATION FILE FOR THE EXPERIMENT!

For consistency and ease of configuration, these constants will be used accross all notebook.

'''

from qiskit.quantum_info import SparsePauliOp
import numpy as np

'''Configure Dataset'''
DATA_SIZE = 500
FEATURE_DIM = 4
MAX_ITER = 10
MAX_INST = 10

'''Configure Quantum Circuit properties'''
MAX_QUBITS = 12
MAX_REPS = 12
MIN_REPS = 1
MAX_IDENTITIES_BLOCKS = 2  # <---- Number of identity blocks, depth values of identity blocks is close to that of normal ansatz (5 qubits)
ENTANGLEMENT = 'linear'

'''Configure Measurement for Quantum Circuit, support different types of Operators, and their position'''
GLOBAL_OPERATOR = SparsePauliOp.from_list([('Z'*MAX_QUBITS, 1)])

# How many Qubits to be measured
LOCAL_MEASUREMENT = 2

# [...IIZZ]
LOCAL_OPERATOR_BOTTOM = SparsePauliOp.from_list([('I' * (MAX_QUBITS - LOCAL_MEASUREMENT)+'Z'*LOCAL_MEASUREMENT, 1)])

# [ZZII...]
LOCAL_OPERATOR_TOP = SparsePauliOp.from_list([('Z'*LOCAL_MEASUREMENT+'I' * (MAX_QUBITS - LOCAL_MEASUREMENT), 1)])

# [...IZZI...]
LOCAL_OPERATOR_MIDDLE = SparsePauliOp.from_list([('I'*(np.floor(MAX_QUBITS/2)-np.floor(LOCAL_MEASUREMENT/2)).astype(int)+'Z'*LOCAL_MEASUREMENT + 'I'*(MAX_QUBITS-(np.floor(MAX_QUBITS/2)-np.floor(LOCAL_MEASUREMENT/2)).astype(int)-LOCAL_MEASUREMENT),1)])

LOG_PATH = './Logs-MNIST-v4'