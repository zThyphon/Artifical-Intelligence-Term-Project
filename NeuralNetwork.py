
import torch
import torch.nn as nn
from config import config_data

# Configuration
numberOfIndependentVariables = config_data["numberOfIndependentVariables"]
numberOfClassLabels = config_data["numberOfClassLabels"]
numberOfInputNeurons = config_data["neuralNetworkLayers"][0]
numberOfFirstHiddenLayerNeurons = config_data["neuralNetworkLayers"][1]
numberOfSecondHiddenLayerNeurons = config_data["neuralNetworkLayers"][2]
dropOutRate = config_data["dropOutRate"]
randomSeed = config_data["randomSeed"]
batchSize = config_data["batchSize"]
numberOfEpochs = config_data["numberOfEpochs"]
learningRate = config_data["learningRate"]

class NeuralNetwork(nn.Module):
    def __init__(self, input_features=numberOfInputNeurons, h1=numberOfFirstHiddenLayerNeurons,
                 h2=numberOfSecondHiddenLayerNeurons, output_features=numberOfClassLabels):
        super().__init__()
        self.function1 = nn.Sequential(
            nn.Linear(input_features, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropOutRate)
        )
        self.function2 = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropOutRate)
        )
        self.output = nn.Linear(h2, output_features)

    def forward(self, x):
        x = self.function1(x)
        x = self.function2(x)
        return self.output(x)

randomSeed = config_data["randomSeed"]
torch.manual_seed(randomSeed)
