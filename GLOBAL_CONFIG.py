from qiskit.opflow import PauliSumOp

DATA_SIZE = 500
FEATURE_DIM = 4
MAX_ITER = 120
MAX_INST = 10

MAX_QUBITS = 9
MAX_REPS = 9
MIN_REPS = 1
MAX_IDENTITIES_BLOCKS = 2  # <---- Number of identity blocks, depth values of identity blocks is close to that of normal ansatz (5 qubits)
ENTANGLEMENT = 'linear'

GLOBAL_OPERATOR = PauliSumOp.from_list([('Z'*MAX_QUBITS, 1)])
LOCAL_OPERATOR = PauliSumOp.from_list([('I' * (MAX_QUBITS - 2)+'Z'*2, 1)])

LOG_PATH = './Logs-MNIST-v4'