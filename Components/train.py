import numpy as np

from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import Gradient
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.quantum_info import SparsePauliOp

import copy, math
from sklearn.metrics import accuracy_score
import torch

from Components.circuits import preTrainedBlockGenerator, layerwise_training, featureMapGenerator, AnsatzGenerator
from Components.utils import classification_callback

import time
from IPython.display import clear_output

def create_qnn(feature_dim, reps, entanglement, operator):
    feature_map = featureMapGenerator(feature_dim)
    ansatz = AnsatzGenerator(feature_dim, reps=reps, entanglement=entanglement)
    circuit = feature_map.compose(ansatz)

    qnn = EstimatorQNN(
        circuit=circuit,
        observables=SparsePauliOp.from_operator(operator),
        input_params=list(feature_map.parameters),
        weight_params=list(ansatz.parameters),
        gradient=Gradient(),
    )

    print(f'num_weight (d): {qnn.num_weights}')

    return qnn

def create_identity_blocks_qnn(feature_dim, num_blocks, entanglement, operator):
    feature_map = featureMapGenerator(feature_dim)

    identity_block = preTrainedBlockGenerator(feature_dim, num_blocks, entanglement=entanglement)

    ansatz = identity_block['circuit']
    initial_point = list(identity_block['params_values'].values())

    circuit = feature_map.compose(ansatz)

    qnn = EstimatorQNN(
        circuit=circuit,
        observables=SparsePauliOp.from_operator(operator),
        input_params=list(feature_map.parameters),
        weight_params=list(ansatz.parameters),
        gradient=Gradient(),
    )

    print(f'num_weight (d): {qnn.num_weights}')

    return qnn, initial_point

def sampling_experiment(qnn, max_iter, loss, train_features, test_features, train_labels, test_labels, initial_point = None):
    callback_results = []
    scores = []

    for i in range(10):
        print(f'Sample number {i+1}')

        classifier = NeuralNetworkClassifier(
            qnn, 
            optimizer=COBYLA(maxiter=max_iter),
            loss=loss,
            initial_point=initial_point
        )


        callback = classification_callback()

        classifier.callback = callback.collect

        classifier.fit(train_features, train_labels)
        score = classifier.score(test_features, test_labels)

        callback_results.append(callback)
        scores.append(score)

        clear_output(wait=True)

    return callback_results, scores

def sampling_ll_experiment(qnn, max_iter, loss, train_features, test_features, train_labels, test_labels, initial_point = None):
    callback_results = []
    scores = []

    for i in range(10):
        print(f'Sample number {i+1}')

        initial_point = layerwise_training(ansatz_method_2, MAX_REPS, COBYLA(maxiter=50), q_instance)

        classifier = NeuralNetworkClassifier(
            qnn, 
            optimizer=COBYLA(maxiter=max_iter),
            loss=loss,
            initial_point=initial_point
        )


        callback = classification_callback()

        classifier.callback = callback.collect

        classifier.fit(train_features, train_labels)
        score = classifier.score(test_features, test_labels)

        callback_results.append(callback)
        scores.append(score)

        clear_output(wait=True)

    return callback_results, scores

@torch.no_grad()
def predict_batch(dataloader, model):
    model.eval()
    predictions = np.array([])
    for x_batch, _ in dataloader:
        outp = model(x_batch)
        probs = torch.sigmoid(outp)
        preds = ((probs > 0.5).type(torch.long))
        predictions = np.hstack((predictions, preds.numpy().flatten()))
    predictions = predictions
    return predictions.flatten()

@torch.no_grad()
def predict(data, model):
    model.eval()
    output = model(data)
    probs = torch.sigmoid(output)
    preds = ((probs > 0.5).type(torch.long)*2-1)
    return preds



def train_batch(model, epochs, train_dataloader, val_dataloader, optimizer, loss_function):
    accuracy_train = []
    accuracy_test = []
    losses = []
    weights = []
    max_epochs = epochs
    print ("{:<10} {:<10} {:<20} {:<16} {:<16}".format('Epoch', 'Batch','Loss','Train Accuracy', 'Test Accuracy'))
    for epoch in range(max_epochs):  # loop over the dataset multiple times
        for it, (X_batch, y_batch) in enumerate(train_dataloader):
            # zero the parameter gradients
            optimizer.zero_grad()
            outp = model(X_batch)
            loss = loss_function(outp.flatten(), y_batch)
            loss.backward()
            losses.append(loss.detach().flatten()[0])

            # log result
            a_train = accuracy_score(train_dataloader.dataset.tensors[1], predict(train_dataloader.dataset.tensors[0], model))
            a_test = accuracy_score(val_dataloader.dataset.tensors[1], predict(val_dataloader.dataset.tensors[0], model))
            accuracy_train.append(a_train)
            accuracy_test.append(a_test)
            weights.append(copy.deepcopy(model.weight.data))
            
            # optimiser next step
            optimizer.step()
            print ("{:<10} {:<10} {:<20} {:<16} {:<16}".format(f'[ {epoch} ]', it, loss.detach().flatten()[0].numpy().round(5), a_train.round(5), a_test.round(5)))
    return model, losses, accuracy_train, accuracy_test, weights

def train(model, epochs, X_train, y_train, X_val, y_val, optimizer, loss_function):
    accuracy_train = []
    accuracy_test = []
    losses = []
    weights = []
    max_epochs = epochs
    print ("{:<10} {:<20} {:<16} {:<16}".format('Epoch','Loss','Train Accuracy', 'Test Accuracy'))
    for epoch in range(max_epochs):
        # zero the parameter gradients
        optimizer.zero_grad()
        outp = model(X_train)
        loss = loss_function(outp.flatten(), y_train)
        loss.backward()
        losses.append(loss.detach().flatten()[0])

        # log result
        a_train = accuracy_score(y_train, predict(X_train, model))
        a_test = accuracy_score(y_val, predict(X_val, model))
        accuracy_train.append(a_train)
        accuracy_test.append(a_test)
        weights.append(copy.deepcopy(model.weight.data))

        # optimiser next step
        optimizer.step()
        
        if (epoch % 10 == 0) or (epoch+1 == max_epochs):
            print ("{:<10} {:<20} {:<16} {:<16}".format(f'[ {epoch} ]', loss.detach().flatten()[0].numpy().round(5), a_train.round(5), a_test.round(5)))
    return model, losses, accuracy_train, accuracy_test, weights