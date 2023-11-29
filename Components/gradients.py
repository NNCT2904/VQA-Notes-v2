import numpy as np
from qiskit.opflow import Gradient, CircuitSampler, StateFn, PauliExpectation
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator
from matplotlib import pyplot as plt

from GLOBAL_CONFIG import *


def sampleGradientRuntime(ansatzes, operators, ansatzes_parameters=[]):
    ansatzes_to_run = []
    operator_to_run = []
    parameters_to_run = []

    # 100 parameters for ansatzes
    num_values = 100

    # If no parameters are given, create 100 random parameters, or else use the given parameters to scan current gradient
    if len(ansatzes_parameters) == 0:
        for i in range(len(ansatzes)):
            for j in range(num_values):
                ansatzes_to_run.append(ansatzes[i])
                operator_to_run.append(operators[i])
                parameters_to_run.append(np.random.uniform(0, np.pi, ansatzes[i].num_parameters))
    else:
        for i in range(len(ansatzes)):
            for j in range(num_values):
                points = []
                for k in range(ansatzes[i].num_parameters):
                    low = ansatzes_parameters[i][k] - np.pi/2
                    high = ansatzes_parameters[i][k] + np.pi/2
                    points.append(np.random.uniform(low, high))
                parameters_to_run.append(points)
                ansatzes_to_run.append(ansatzes[i])
                operator_to_run.append(operators[i])
    
    # Estimate all ansatzes with provided parameters
    with Estimator(circuits=ansatzes_to_run, observables=operator_to_run, service=service, options=options) as estimator:
        
        estimator_results = estimator(
            circuits=ansatzes_to_run,
            observables=operator_to_run,
            parameter_values=parameters_to_run
        )

    # Get the expectation values for each ansatz, each 100 entry is one ansatz
    parsed_estimator_result = []
    for i in range(0, len(estimator_results.values), num_values):
        parsed_estimator_result.append(estimator_results.values[i:i+num_values])
        
    print(f"Number of ansatzes, parameters, operator to run: {len(ansatzes_to_run)}")
    return parsed_estimator_result

def sampleAnsatz(estimator, ansatzes, operators, ansatzes_parameters=[],):
    ''' 
    This function sample the ansatz with a range of parameters. The result can be used to calculate the gradient of the ansatz.

    Args:
        estimator - the Qiskit estimator, to either emulate quantum device, or connect to an actual device from IBMQ

        ansatzes - the ansatzes to execute in batch

        ansatzes_parameters - the parameters of the ansatz, by default this function will generate 100 random sets of parameters to scan the gradient.

    '''

    ansatzes_to_run = []
    operator_to_run = []
    parameters_to_run = []

    # 100 parameters for ansatzes
    num_values = 100

    # If no parameters are given, create 100 random parameters, or else use the given parameters to scan current gradient
    if len(ansatzes_parameters) == 0:
        for i in range(len(ansatzes)):
            for j in range(num_values):
                ansatzes_to_run.append(ansatzes[i])
                operator_to_run.append(operators[i])
                parameters_to_run.append(np.random.uniform(0, np.pi, ansatzes[i].num_parameters))
    else:
        for i in range(len(ansatzes)):
            for j in range(num_values):
                points = []
                for k in range(ansatzes[i].num_parameters):
                    low = ansatzes_parameters[i][k] - np.pi/2
                    high = ansatzes_parameters[i][k] + np.pi/2
                    points.append(np.random.uniform(low, high))
                parameters_to_run.append(points)
                ansatzes_to_run.append(ansatzes[i])
                operator_to_run.append(operators[i])
    
    # Estimate all ansatzes with provided parameters
    
    job = estimator.run(
        ansatzes_to_run, 
        parameter_values=parameters_to_run, 
        observables=operator_to_run
    )

    job_result = job.result().values

    # Get the expectation values for each ansatz, each 100 entry is one ansatz
    parsed_estimator_result = []
    for i in range(0, len(job_result), num_values):
        parsed_estimator_result.append(job_result[i:i+num_values])
        
    print(f"Number of ansatzes, parameters, operator to run: {len(ansatzes_to_run)}")
    return parsed_estimator_result


def getVariance(data, num_qubits):
    '''
    Calculate the gradient of the given data, then calculate and the variance of the gradient.
    This function will also draw a graph of variance line.

    Args: 
        data - the given data, can be obtained from function sampleAnsatz()

        num_qubits - range of number of qubits, eg. range(2, MAX_QUBITS)

    '''

    g=[]
    for d in data:
        g.append(np.gradient(d))

    fit = np.polyfit(num_qubits, np.log(np.var(g, axis=1)), deg=1)
    x = np.linspace(num_qubits[0], num_qubits[-1], 200)

    plt.figure(figsize=(12, 6))
    plt.semilogy(num_qubits, np.var(g, axis=1), 'o-', label='measured variance')
    plt.semilogy(x, np.exp(fit[0] * x + fit[1]), 'r--', label=f'exponential fit w/ {fit[0]:.2f}')
    plt.xlabel('Numer of qubits')
    plt.ylabel(r'$\mathrm{Var}[\partial_{\theta 1} \langle E(\theta) \rangle]$')
    plt.legend(loc='best')

    return np.var(g, axis=1)