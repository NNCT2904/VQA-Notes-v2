import numpy as np
from qiskit.circuit import QuantumCircuit
import qiskit.primitives
from qiskit.primitives import Estimator
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from matplotlib import pyplot as plt
from qiskit_machine_learning.connectors import TorchConnector
import torch
from torch import tensor, optim
import copy

from Components.utils import *

from GLOBAL_CONFIG import *


# def sampleGradientRuntime(ansatzes, operators, ansatzes_parameters=[]):
#     ansatzes_to_run = []
#     operator_to_run = []
#     parameters_to_run = []

#     # 100 parameters for ansatzes
#     num_values = 100

#     # If no parameters are given, create 100 random parameters, or else use the given parameters to scan current gradient
#     if len(ansatzes_parameters) == 0:
#         for i in range(len(ansatzes)):
#             for j in range(num_values):
#                 ansatzes_to_run.append(ansatzes[i])
#                 operator_to_run.append(operators[i])
#                 parameters_to_run.append(np.random.uniform(0, np.pi, ansatzes[i].num_parameters))
#     else:
#         for i in range(len(ansatzes)):
#             for j in range(num_values):
#                 points = []
#                 for k in range(ansatzes[i].num_parameters):
#                     low = ansatzes_parameters[i][k] - np.pi/2
#                     high = ansatzes_parameters[i][k] + np.pi/2
#                     points.append(np.random.uniform(low, high))
#                 parameters_to_run.append(points)
#                 ansatzes_to_run.append(ansatzes[i])
#                 operator_to_run.append(operators[i])
    
#     # Estimate all ansatzes with provided parameters
#     with Estimator(circuits=ansatzes_to_run, observables=operator_to_run, service=service, options=options) as estimator:
        
#         estimator_results = estimator(
#             circuits=ansatzes_to_run,
#             observables=operator_to_run,
#             parameter_values=parameters_to_run
#         )

#     # Get the expectation values for each ansatz, each 100 entry is one ansatz
#     parsed_estimator_result = []
#     for i in range(0, len(estimator_results.values), num_values):
#         parsed_estimator_result.append(estimator_results.values[i:i+num_values])
        
#     print(f"Number of ansatzes, parameters, operator to run: {len(ansatzes_to_run)}")
#     return parsed_estimator_result

def sampleAnsatz(estimator: qiskit.primitives.Estimator, ansatzes:list[QuantumCircuit], operators:list[SparsePauliOp], ansatzes_parameters:list[float|np.float64]=[]) -> list:
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


def getVariance(data:list[float|np.float64], num_qubits:list[int]):
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

    variance = np.var(g, axis=1)

    fit = np.polyfit(num_qubits, np.log(variance), deg=1)
    x = np.linspace(num_qubits[0], num_qubits[-1], 200)

    plt.figure(figsize=(12, 6))
    plt.semilogy(num_qubits, variance, 'o-', label='measured variance')
    plt.semilogy(x, np.exp(fit[0] * x + fit[1]), 'r--', label=f'exponential fit w/ {fit[0]:.2f}')
    plt.xlabel('Numer of qubits')
    plt.ylabel(r'$\mathrm{Var}[\partial_{\theta 1} \langle E(\theta) \rangle]$')
    plt.xticks(num_qubits)
    plt.legend(loc='best')

    return variance

def plotMeanVariance(data:list[float|np.float64], num_qubits:list[int], smooth_weight=0):
    '''
    Plot the variance of the gradient obtained in batch data.

    Args: 
        data - the given data, can be obtained from function getVariance()

        num_qubits - range of number of qubits, eg. range(2, MAX_QUBITS)

        smooth_weight - within range [0, 1], apply smooth to graph, can cause information loss

    '''

    color = {
        'm1': 'tab:blue', 
        'm2': 'tab:orange' ,
        'm3': 'tab:green', 
        'm4':'tab:red'
        }
    for c in data:
        select = pd.DataFrame(np.reshape(data[c], (len(data[c]), data[c][0].shape[1])))
        
        max = smooth(select.max(), smooth_weight)
        min = smooth(select.min(), smooth_weight)
        mean = smooth(select.mean(), smooth_weight)

        fit = np.polyfit(num_qubits, np.log(mean), deg=1)

        plt.semilogy(num_qubits, mean, color = color[c], label=f'Method {c} Variance, exp fit w/ {fit[0]:.2f}')

        plt.fill_between(num_qubits, max, min, color = color[c], alpha = 0.2)
    
    # plt.ylim(0.2, 1.1)
    plt.title('Gradient Variances vs Qubits')
    plt.xlabel('Qubits')
    plt.xticks(num_qubits)
    plt.ylabel('Var')
    plt.legend(loc='best')

def sampleWeightLoss(model: TorchConnector, X_train :tensor, y_train:tensor, optimizer: optim.Optimizer, loss_function:torch.nn.modules.loss._Loss):
        
    # zero the parameter gradients
    optimizer.zero_grad()
    outp = model(X_train)
    loss = loss_function(outp.flatten(), y_train)

    # log result
    loss = loss.detach().flatten()[0]
    weight = copy.deepcopy(model.weight.data)

    return loss, weight