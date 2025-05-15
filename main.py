import os
import getData
import numpy as np
import pandas as pd
import torch
import chart
from NeuralNetwork import NeuralNetwork
from config import config_data
from train import train
from test import test

os.environ["XDG_SESSION_TYPE"] = "xcb"

pd.set_option('future.no_silent_downcasting', True)

neural_network = NeuralNetwork()

dataSet = getData.get_data_set()

classLabel = config_data["classLabel"]
labelEncoder = config_data["labelEncoder"]

# Convert categorical labels to numerical values
dataSet[classLabel] = dataSet[classLabel].replace(labelEncoder)

# Train Test and Split
X = getData.get_all_independent_variables(dataSet)
X_reduced = getData.dimension_reduction(X)
y = getData.get_class_label_row(dataSet)

X_train, X_test, y_train, y_test = getData.get_splitted_train_test_data(X_reduced, y)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = np.array(y_train, dtype=np.int64)
y_train = torch.LongTensor(y_train)  

y_test = np.array(y_test, dtype=np.int64)
y_test = torch.LongTensor(y_test)   

neural_network = NeuralNetwork()

train(neural_network, X_train, y_train)

chart.draw_neural_network()

test(neural_network, X_test, y_test)