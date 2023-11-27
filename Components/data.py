#from qiskit_machine_learning.datasets import breast_cancer
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import GLOBAL_CONFIG


def cancer_data(PCA_n=GLOBAL_CONFIG.FEATURE_DIM, scale=True, split=True):
    X, y = datasets.load_breast_cancer(return_X_y=True)

    if PCA_n:
        pca = PCA(n_components=PCA_n)
        X = pca.fit_transform(X)
        

    if scale:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = scaler.fit_transform(X)

    train_features, test_features, train_labels, test_labels = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=42, stratify=y
    )

    # The training labels are in {0, 1}, we'll encode them {-1, 1}!
    train_labels = train_labels * 2 - 1
    test_labels = test_labels * 2 - 1

    print(f'Training set: {len(train_features)} samples')
    print(f'Testing set: {len(test_features)} samples')
    print(f'Number of features: {train_features.shape[-1]}')
    print(f'Classes:{np.unique(y)}; Encoded as: {np.unique(train_labels)}')

    return train_features, test_features, train_labels, test_labels