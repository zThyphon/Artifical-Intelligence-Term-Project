from config import config_data
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

def get_data_set():
    dataSetPath = config_data["dataSet"] 
    dataSet = pd.read_csv(dataSetPath)
    return dataSet

def get_all_independent_variables(dataSet):
    classLabel = config_data["classLabel"]
    independentVariables = dataSet.drop(classLabel, axis=1).values
    return independentVariables

def get_class_label_row(dataSet):
    classLabel = config_data["classLabel"]
    classLabelRow = dataSet[classLabel].values
    return classLabelRow

def get_splitted_train_test_data(x,y):
    trainTestSplitRatio = config_data["trainTestSplitRatio"]
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=trainTestSplitRatio, random_state=0)
    return X_train, X_test, y_train, y_test


def get_class_label(class_result):
    class_label = ""
    
    if class_result == 0:
        class_label = "allow"

    elif class_result == 1:
        class_label = "drop"

    elif class_result == 2:
        class_label = "deny"

    return class_label

def dimension_reduction(x):
    reduced_number_of_dimensions = config_data["reducedNumberOfDimensions"]
    pca = PCA(n_components=reduced_number_of_dimensions)
    X_reduced = pca.fit_transform(x)

    return X_reduced