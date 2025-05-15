config_data = {
    "dataSet": "./firewall_data.csv",
    "numberOfIndependentVariables": 11,
    "reducedNumberOfDimensions": 4,
    "classLabel": "Action",
    "numberOfClassLabels": 3,
    "featureLabel": "Action",
    "trainTestSplitRatio": 0.3,
    "validationSplit": 0.2,
    "labelEncoder": {"allow": 0, "drop": 1, "deny": 2},
    "learningRate": 0.001,
    "weightDecay": 0.01,
    "numberOfEpochs": 20,
    "batchSize": 32,
    "dropOutRate": 0.2,
    "neuralNetworkLayers": [4, 9, 10, 3], # [input neurons, hidden layers, output neurons]
    "randomSeed": 36
}