import numpy as np

from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import Gradient
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector

import copy, math
from sklearn.metrics import accuracy_score
import torch
from torch import tensor, optim

from Components.utils import classification_callback

import time
from IPython.display import clear_output

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



def train_batch(model: TorchConnector, epochs:int, train_dataloader, val_dataloader, optimizer, loss_function):
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

def train(model: TorchConnector, epochs:int, X_train :tensor, y_train:tensor, X_val:tensor, y_val:tensor, optimizer: optim.Optimizer, loss_function:torch.nn.modules.loss._Loss):
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
            print ("{:<10} {:<20} {:<16} {:<16}".format(f'[ {epoch} ]', loss.detach().flatten()[0].numpy().round(5), round(a_train, 5), round(a_test, 5)))
    return model, losses, accuracy_train, accuracy_test, weights