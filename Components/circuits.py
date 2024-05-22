import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter
# from qiskit.opflow import PauliExpectation, CircuitSampler, StateFn, Gradient, Z, X, I

# from qiskit.quantum_info import Z, X, I

from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp, Statevector

from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit_algorithms import optimizers
from qiskit_algorithms.optimizers import COBYLA

from GLOBAL_CONFIG import *

def circuitBuilder(feature_map:QuantumCircuit, ansatz:QuantumCircuit, feature_map_position:str = 'middle'):
    ''' Generate a circuit with custom ansatz width and feature map width

    Args: 
        feature_map - The Feature map
        ansatz - The Ansatz, assuming that the number of qubits is larger than the feature map
        feature_map_position - position of the feature map, select one from 'top', 'middle', 'bottom', or specify which qubits as an array, eg. [1, 2, 3]
    
    Return:
        A parameterised quantum circuit, to be used as a neural network
    '''

    num_qubits = ansatz.num_qubits

    qc = QuantumCircuit(num_qubits)

    if feature_map_position == 'top':
        qc.append(feature_map, qargs = range(0, feature_map.num_qubits))
    elif feature_map_position == 'bottom':
        qc.append(feature_map, qargs = range(num_qubits - feature_map.num_qubits, num_qubits))
    elif feature_map_position == 'middle':
        mid_index_begin = (np.floor(num_qubits/2) - np.floor(feature_map.num_qubits/2)).astype(int)
        mid_index_end = mid_index_begin + feature_map.num_qubits
        qc.append(feature_map, qargs = range(mid_index_begin, mid_index_end))
    else:
        qc.append(feature_map, qargs = feature_map_position)

    qc.append(ansatz, qargs=range(0, num_qubits))

    return qc

def featureMapGenerator(num_qubits:int):
    feature_map = ZZFeatureMap(feature_dimension=num_qubits,
                               reps=1
                               ).decompose()
    return feature_map


def AnsatzGenerator(num_qubits:int, reps:int=1, entanglement:str='linear'):
    ansatz = RealAmplitudes(
        num_qubits=num_qubits,
        entanglement=entanglement,
        reps=reps,
        insert_barriers=False
    )

    return ansatz

def preTrainedBlockGenerator(num_qubits:int, num_blocks:int, overlay:int=0, skip_last_barrier:bool=False, insert_barriers:bool=False, entanglement:str='linear'):
    ''' Generates an identity block ansatz 
    
    Argumens:
        num_qubits - Number of qubits for ansatz construction
        num_blocks - Number of identity blocks
        entanglement - entanglement type (linear, full, etc.) (not implemented, linear entanglement is enforced)
        skip_final_rotation_layer - indicates if the final rotation should be omitted (not implemented)
        insert_barriers - when True inserts the barrier between two parts of the block
        overlay - indicates how many qubits are overlaid in two rotational slices
                  it allows to add extra rotations to the ansatz
    Returns:
        - a dictionary of
            * The ansatz circuit
            * The ansatz parameter values
    '''
    qc = QuantumCircuit(num_qubits)
    params = [Parameter(f"θ[{i}]") for i in range(num_blocks*(num_qubits*2+overlay)*2)]
    index = 0
    values = []
    for i in range(num_blocks):
        
        ### First half
        
        # Rotation layer
        for j in range(num_qubits):
            qc.ry(params[index], j)
            index += 1

        # Entanglement block
        for j in range(num_qubits-1):
            qc.cx(j, (j+1))
        
        # Rotation layer
        for j in range(num_qubits):
            qc.ry(params[index], j)
            index += 1

        # Optional barrier
        if (insert_barriers):
            qc.barrier([j for j in range(num_qubits)])

        # Overlay block
        for j in range(overlay):
            qc.cx(num_qubits-j-1, num_qubits-j-2)          

        for j in range(overlay):
            qc.ry(params[index], num_qubits-j-1)
            index += 1           

        # The first half of the block has its parameters randomized
        first_half_params = np.random.uniform(0, np.pi*2, int(num_qubits*2+overlay))
        
        # Optional barrier
        if (insert_barriers):
            qc.barrier([j for j in range(num_qubits)])
            qc.barrier([j for j in range(num_qubits)])

        ### Second half in reverse
        
        # Reverse overlay block
        for j in range(overlay-1, -1, -1):
            qc.ry(params[index], num_qubits-j-1)
            index += 1           

        for j in range(overlay-1, -1, -1):
            qc.cx(num_qubits-j-1, num_qubits-j-2)          

        # Optional barrier
        if (insert_barriers):
            qc.barrier([j for j in range(num_qubits)])

        # Rotation layer
        for j in range(num_qubits-1, -1, -1):
            qc.ry(params[index], j)
            index += 1

        # Entanglement block reversed
        for j in range(num_qubits-2, -1, -1):
            qc.cx(j, (j+1))

        # Rotation layer
        for j in range(num_qubits-1, -1, -1):
            qc.ry(params[index], j)
            index += 1
            
        # Barrier
        if (not skip_last_barrier):
            qc.barrier([j for j in range(num_qubits)])

        # Configure the parameters of the second half to inverse the first half
        second_half_params = - np.flip(first_half_params, axis=0)
        block_parameters = np.append(first_half_params, second_half_params)
        values = np.append(values, block_parameters)

    params_values = dict(zip(params, values))
    return {
        'circuit': qc,
        'params_values': params_values
    }

def preTrainedBlockGenerator_old(num_qubits, num_blocks, skip_final_rotation_layer=False, entanglement='full', insert_barriers=False):
    qc = QuantumCircuit(num_qubits)
    params = [Parameter(f"θ[{i}]") for i in range(num_blocks*num_qubits*4)]
    index = 0
    values = []
    for i in range(num_blocks):
        # Begin the first half with a rotation layer
        for j in range(num_qubits):
            qc.ry(params[index], j)
            index += 1

        # Entanglement block
        if (entanglement == 'full'):
            for j in range(num_qubits-1):
                for i in range(j+1, num_qubits):
                    qc.cx(j, i)
        elif (entanglement == 'linear'):
            for j in range(num_qubits-1):
                qc.cx(j, (j+1))

        # End of first half with a rotation layer
        for j in range(num_qubits):
            qc.ry(params[index], j)
            index += 1

        # The first half of the block have its parameters randomized
        first_half_params = np.random.uniform(0, np.pi*2, int(num_qubits*2))
        
        # Optional barrier
        if (insert_barriers):
            qc.barrier([j for j in range(num_qubits)])

        # Begin the second half with a rotation layer
        for j in range(num_qubits):
            qc.ry(params[index], j)
            index += 1

        # Entanglement block reversed
        if (entanglement == 'full'):
            for j in reversed(range(num_qubits-1)):
                for i in reversed(range(j+1, num_qubits)):
                    qc.cx(j, i)
        elif (entanglement == 'linear'):
            for j in range(num_qubits-1):
                qc.cx(j, (j+1))

        # End of second half with a rotation layer
        for j in range(num_qubits):
            qc.ry(params[index], j)
            index += 1
        qc.barrier([j for j in range(num_qubits)])

        # Configure the parameters of the second half to inverse the first half
        first_half_params_2d = np.reshape(first_half_params, (-1, num_qubits))
        second_half_2d = - np.flip(first_half_params_2d, axis=0)
        second_half_params = np.reshape(second_half_2d, (1, -1))[0]

        block_parameters = np.append(first_half_params, second_half_params)
        values = np.append(values, block_parameters)

    params_values = dict(zip(params, values))
    return {
        'circuit': qc,
        'params_values': params_values
    }

def LLMinimizePrimitives(circuit: QuantumCircuit, optimiser:optimizers, estimator: Estimator):
    operator = SparsePauliOp.from_list([('I' * (circuit.num_qubits - 2)+'Z'*2, 1)])

    initial_point = np.zeros(circuit.num_parameters)

    bouned_circuit = circuit.assign_parameters(initial_point)

    expectation = Statevector(bouned_circuit)
    
    state = expectation.evolve(operator)

    PShiftGradient = ParamShiftEstimatorGradient(state)

    def loss(x):
        # values = dict(zip(circuit.parameters, x))
        return np.real(estimator.run([circuit], observables=[operator], parameter_values=[x]).result().values)
    
    def gradient(x):
        # values = dict(zip(circuit.parameters, x))
        return np.real(PShiftGradient.run(parameter_values=[x]))

    res = optimiser.minimize(fun=loss, jac=gradient, x0=initial_point)

    return res


def LayerwiseTrainingPrimitives(ansatz: QuantumCircuit, max_num_layers: int, optimizer = COBYLA(maxiter=5), estimator = Estimator()):
    optimal_parameters = []

    for reps in range(1, max_num_layers+1):
        ansatz.reps = reps

        # bind already optimized parameters
        values_dict = dict(zip(ansatz.parameters, optimal_parameters))
        partially_bound = ansatz.assign_parameters(values_dict)

        res = LLMinimizePrimitives(partially_bound, optimizer, estimator)
        print('Circuit rep:', reps, 'best value:', res.fun)
        optimal_parameters += list(res.x)

    return optimal_parameters

# def LLMinimize(circuit, optimizer, q_instance):
#     initial_point = np.zeros(circuit.num_parameters)

#     operator = (I ^ (circuit.num_qubits-2)) ^ (Z ^ 2)

#     expectation = StateFn(operator, is_measurement=True) @ StateFn(circuit)
#     grad = Gradient().convert(expectation)

#     # Pauli basis
#     expectation = PauliExpectation().convert(expectation)
#     grad = PauliExpectation().convert(grad)

#     sampler = CircuitSampler(q_instance, caching='all')

#     def loss(x):
#         values = dict(zip(circuit.parameters, x))
#         return np.real(sampler.convert(expectation, values).eval())

#     def gradient(x):
#         values = dict(zip(circuit.parameters, x))
#         return np.real(sampler.convert(grad, values).eval())

#     return optimizer.minimize(loss, jac=gradient, x0=initial_point)


# def LLMinimize_old(circuit, optimizer, q_instance):
#     initial_point = np.zeros(circuit.num_parameters)

#     operator = (I ^ (circuit.num_qubits-2)) ^ (Z ^ 2)

#     expectation = StateFn(operator, is_measurement=True) @ StateFn(circuit)
#     grad = Gradient().convert(expectation)

#     # Pauli basis
#     expectation = PauliExpectation().convert(expectation)
#     grad = PauliExpectation().convert(grad)

#     sampler = CircuitSampler(q_instance, caching='all')

#     def loss(x):
#         values = dict(zip(circuit.parameters, x))
#         return np.real(sampler.convert(expectation, values).eval())

#     def gradient(x):
#         values = dict(zip(circuit.parameters, x))
#         return np.real(sampler.convert(grad, values).eval())

#     return optimizer.minimize(loss, jac=gradient, x0=initial_point)


def layerwise_training(ansatz:QuantumCircuit, max_num_layers:int, optimizer:optimizers, q_instance):
    optimal_parameters = []

    for reps in range(1, max_num_layers+1):
        ansatz.reps = reps

        # bind already optimized parameters
        values_dict = dict(zip(ansatz.parameters, optimal_parameters))
        partially_bound = ansatz.bind_parameters(values_dict)

        res = LLMinimize(partially_bound, optimizer, q_instance)
        # print('Circuit rep:', reps, 'best value:', res.fun)
        optimal_parameters += list(res.x)

    return optimal_parameters
