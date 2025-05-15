from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
from config import config_data
import torch
import torch.nn as nn
import getData 

def test(neural_network, x_test, y_test):
    lossFunction = nn.CrossEntropyLoss()    
    batchSize = config_data["batchSize"]

    neural_network.eval()

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batchSize)

    total_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            outputs = neural_network(inputs)
            loss = lossFunction(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

            for i in range(len(labels)):
                actual_class = labels[i].item()
                predicted_class = preds[i].item()
                
                actual_label = getData.get_class_label(actual_class)
                predicted_label = getData.get_class_label(predicted_class)

                print(f"""
                    {batch_idx*batchSize + i + 1}) 
                      Actual Class: {actual_label} \t
                      Predicted Class: {predicted_label}""")

                if actual_class == predicted_class:
                    correct += 1

    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    print(f"\nAverage Test Loss: {avg_loss:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(all_labels, all_preds))

    total = len(y_test)
    print(f"\nCorrect Predictions: {correct}/{total}")
    print(f"Accuracy: {(correct/total)*100:.2f}%")