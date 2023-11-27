import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from sklearn.model_selection import KFold
from qiskit.algorithms.optimizers import COBYLA, NELDER_MEAD, SLSQP, SPSA

from typing import List


def plot_loss(loss, label='loss', ylim=[0, 1.4]):
    plt.figure(figsize=(12, 6))
    plt.plot(loss, 'tab:blue', label=label)
    plt.ylim(ylim[0], ylim[1])
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.legend(loc='best')


def score(predicted, test_labels):
    # arr = []
    predicted = predicted.reshape(-1)
    return np.mean(predicted == test_labels)


# parity maps bitstrings to 0 or 1
def parity(x):
    return "{:b}".format(x).count("1")

# Classification callback


class classification_callback:
    name = "class_callback"

    # Initialise callback for objfun silent collection
    def __init__(self, log_interval=50):
        self.objfun_vals = []
        self.weight_vals = []
        self.log_interval = log_interval
        print('Callback initialted')

    # Find the first minimum objective fun value
    def min_obj(self):
        if self.objfun_vals == []:
            return (-1, 0)
        else:
            minval = min(self.objfun_vals)
            minvals = [(i, v)
                       for i, v in enumerate(self.objfun_vals) if v == minval]
            return minvals[0]

    # Plots the objfun chart
    def plot(self):
        clear_output(wait=True)
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.title("Objective function")
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.plot(range(len(self.objfun_vals)), self.objfun_vals)
        plt.show()

    # Store objective function values and weights and plot
    def graph(self, weights, obj_func_eval):
        self.objfun_vals.append(obj_func_eval)
        self.plot()
        print(f'Step: {len(self.objfun_vals)}')
        print(f'Current value: {obj_func_eval} ')

    # Collects objfun values and prints their values at log intervals
    # When finished the "plot" function can produce the chart
    def collect(self, weights, obj_func_eval):
        self.objfun_vals.append(obj_func_eval)
        self.weight_vals.append(weights)
        current_batch_idx = len(self.objfun_vals)
        if current_batch_idx % self.log_interval == 0:
            prev_batch_idx = current_batch_idx-self.log_interval
            last_batch_min = np.min(
                self.objfun_vals[prev_batch_idx:current_batch_idx])
            print('Prev=', prev_batch_idx, ', Curr=', current_batch_idx)
            print('Classification callback(',
                  current_batch_idx, ') = ', obj_func_eval)

# Exponential Moving Target used to smooth the lines 
def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed


def plot_objfn_range(objective_fn, smooth_weight=0, d = None, xlabel='Iterations', ylabel='Loss', title='Loss Function vs Iteration'):
    color = ['tab:blue', 'tab:orange' ,'tab:green', 'tab:red']

    for c in range(len(objective_fn)):
        select = objective_fn[c]
        max = smooth(select.max(), smooth_weight)
        min = smooth(select.min(), smooth_weight)
        mean = smooth(select.mean(), smooth_weight)
        
        if d:
            plt.plot(range(0, objective_fn[c].shape[1]), mean, color = color[c], label=f'Method {c}, d={d[c]}')
        else:
            plt.plot(range(0, objective_fn[c].shape[1]), mean, color = color[c], label=f'Method {c}')

        plt.fill_between(range(0, objective_fn[c].shape[1]), max, min, color = color[c], alpha = 0.2)
    
    # plt.ylim(0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')

def plot_score_range(scores, smooth_weight=0, xlabel='Iterations', ylabel='Score', title='Loss Function vs Iteration'):
    color = ['tab:blue', 'tab:orange' ,'tab:green', 'tab:red']

    for c in range(len(scores)):
        select = scores[c]
        max = smooth(select.max(), smooth_weight)
        min = smooth(select.min(), smooth_weight)
        mean = smooth(select.mean(), smooth_weight)
        

        plt.plot(range(0, scores[c].shape[1]), mean, color = color[c], label=f'Method {c} Average')

        plt.fill_between(range(0, scores[c].shape[1]), max, min, color = color[c], alpha = 0.2)
    
    plt.ylim(0.2, 1.1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')

def result_to_objfun_dataframes(callback_results):
    dataframes = []
    for c in range(len(callback_results)):
        objfn_val_df = pd.DataFrame([callback_results[c][i].objfun_vals for i in range(len(callback_results[c]))])
        # objfn_val_df.to_csv(f'./Saves/LossFunction/m{c}.csv')
        objfn_val_df = objfn_val_df.fillna(objfn_val_df.min())
        
        dataframes.append(objfn_val_df)
        
    return dataframes

def save_results(callback_results = None, objfn_val = None, accuracy_train = None, accuracy_test = None, weights = None):
    if callback_results:
        for c in range(len(callback_results)):
            objfn_val_df = pd.DataFrame([callback_results[c][i].objfun_vals for i in range(len(callback_results[c]))])
            objfn_val_df.to_csv(f'./Saves/LossFunction/m{c}.csv')

            weight_val = [callback_results[c][i].weight_vals for i in range(len(callback_results[c]))]

            for wr in range(len(weight_val)):
                weight_record = pd.DataFrame(weight_val[wr])
                weight_record.to_csv(f'./Saves/Weights/m{c}/sample_{wr}.csv')
    
    if objfn_val:
        for o in range(len(objfn_val)):
            objfn_val_df = pd.DataFrame(objfn_val[o][i] for i in range(len(objfn_val[o]))).astype('float')
            objfn_val_df.to_csv(f'./Saves/LossFunction/m{o}.csv')
    
    if accuracy_train:
        for a in range(len(accuracy_train)):
            accuracy_train_df = pd.DataFrame(accuracy_train[a][i] for i in range(len(accuracy_train[a])))
            accuracy_train_df.to_csv(f'./Saves/Scores/Train/m{a}.csv')

    if accuracy_test:
        for a in range(len(accuracy_test)):
            accuracy_test_df = pd.DataFrame(accuracy_test[a][i] for i in range(len(accuracy_test[a])))
            accuracy_test_df.to_csv(f'./Saves/Scores/Test/m{a}.csv')
    
    if weights:
        for w in range(len(weights)):
            for wr in range(len(weights[w])):
                weight_record = pd.DataFrame(weights[w][wr]).astype('float')
                weight_record.to_csv(f'./Saves/Weights/m{w}/sample_{wr}.csv')


def cross_validate(qnn, features, labels, K=5, loss='cross_entropy', maxiter=250):

    kf = KFold(n_splits=K, random_state=None)
    kfsplit = kf.split(features, labels)

    kf_score = []
    callback_results = []

    for k, (train, test) in enumerate(kfsplit):
        cf_callback_loop = classification_callback()

        classification_loop = NeuralNetworkClassifier(
            qnn,
            optimizer=COBYLA(maxiter),
            loss=loss,
            # one_hot=True,
            callback=cf_callback_loop.collect,
            warm_start=False
        )

        classification_loop.fit(features[train], labels[train])
        score = classification_loop.score(features[test], labels[test])
        kf_score.append(score)
        callback_results.append(classification_loop)
        print(f'Fold: {k+1}, Accuracy: {score}')

    return kf_score, callback_results
