import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import chart
from config import config_data

def train(neural_network, x_train, y_train):
    lossFunction = nn.CrossEntropyLoss()

    learningRate = config_data["learningRate"]
    batchSize = config_data["batchSize"]
    numberOfEpochs = config_data["numberOfEpochs"]
    validationSplit = config_data["validationSplit"]
    weightDecay = config_data["weightDecay"]

    optimizationFunction = torch.optim.Adam(neural_network.parameters(), lr=learningRate, weight_decay=weightDecay)

    full_dataset = TensorDataset(x_train, y_train)

    total_size = len(full_dataset)
    val_size = int(total_size * validationSplit)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)

    losses = []
    val_losses = []

    # training
    for epoch in range(numberOfEpochs):
        neural_network.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_dataloader:
            y_pred = neural_network(batch_x)
            loss = lossFunction(y_pred, batch_y)

            optimizationFunction.zero_grad()
            loss.backward()
            optimizationFunction.step()

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(train_dataloader)
        losses.append(average_loss)

        # validation
        neural_network.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_dataloader:
                y_pred = neural_network(batch_x)
                loss = lossFunction(y_pred, batch_y)
                val_loss += loss.item()

        average_val_loss = val_loss / len(val_dataloader)
        val_losses.append(average_val_loss)

        print(f"Epoch: {epoch+1}/{numberOfEpochs}, "
              f"Training Loss: {average_loss:.4f}, Validation Loss: {average_val_loss:.4f}")

    chart.draw(numberOfEpochs, losses, "Training Loss", "Epochs", "Loss")
    chart.draw(numberOfEpochs, val_losses, "Validation Loss", "Epochs", "Loss")
