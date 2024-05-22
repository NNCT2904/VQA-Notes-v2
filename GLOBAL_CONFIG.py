''' 
THIS IS THE CONFIGURATION FILE FOR THE EXPERIMENT!

For consistency and ease of configuration, these constants will be used accross all notebook.

'''

from qiskit.quantum_info import SparsePauliOp
import numpy as np

'''Configure Dataset'''
DATA_SIZE = 500
FEATURE_DIM = 2
MAX_ITER = 100
MAX_INST = 10

'''Configure Quantum Circuit properties'''
MAX_QUBITS = 4
MAX_QUBITS_CLASSIFICATION = 2
MAX_REPS = 0
MIN_REPS = 2
MAX_IDENTITIES_BLOCKS = 3  # <---- Number of identity blocks, depth values of identity blocks is close to that of normal ansatz 
IDENTITY_BLOCKS_OVERLAY = 3
ENTANGLEMENT = 'linear'

'''Configure Measurement for Quantum Circuit, support different types of Operators, and their position'''
GLOBAL_OPERATOR = SparsePauliOp.from_list([('Z'*MAX_QUBITS_CLASSIFICATION, 1)])

# How many Qubits to be measured
LOCAL_MEASUREMENT = 2

# [...IIZZ]
LOCAL_OPERATOR_BOTTOM = SparsePauliOp.from_list([('I' * (MAX_QUBITS_CLASSIFICATION - LOCAL_MEASUREMENT)+'Z'*LOCAL_MEASUREMENT, 1)])

# [ZZII...]
LOCAL_OPERATOR_TOP = SparsePauliOp.from_list([('Z'*LOCAL_MEASUREMENT+'I' * (MAX_QUBITS_CLASSIFICATION - LOCAL_MEASUREMENT), 1)])

# [...IZZI...]
LOCAL_OPERATOR_MIDDLE = SparsePauliOp.from_list([('I'*(np.floor(MAX_QUBITS_CLASSIFICATION/2)-np.floor(LOCAL_MEASUREMENT/2)).astype(int)+'Z'*LOCAL_MEASUREMENT + 'I'*(MAX_QUBITS_CLASSIFICATION-(np.floor(MAX_QUBITS_CLASSIFICATION/2)-np.floor(LOCAL_MEASUREMENT/2)).astype(int)-LOCAL_MEASUREMENT),1)])

LOG_PATH = './Logs-Cancer-v4'