import matplotlib.pyplot as plt
import numpy as np
from config import config_data

def draw(x, y, xLabel, yLabel, title):
    plt.title(title)
    plt.plot(range(x), y)
    plt.ylabel(xLabel)
    plt.xlabel(yLabel)
    plt.show()


def draw_neural_network():
    neuralNetworkLayers = config_data["neuralNetworkLayers"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-1, len(neuralNetworkLayers))  # Set x-axis limits
    ax.set_ylim(-1, max(neuralNetworkLayers))  # Set y-axis limits
    ax.axis("off")

    layer_positions = []
    for i, layer_size in enumerate(neuralNetworkLayers):
        x = i  
        y_positions = np.linspace(0, max(neuralNetworkLayers) - 1, layer_size) 
        layer_positions.append(list(zip([x] * layer_size, y_positions)))

    # Draw connections (weights)
    for i in range(len(neuralNetworkLayers) - 1):
        for neuron1 in layer_positions[i]: 
            for neuron2 in layer_positions[i + 1]: 
                ax.plot([neuron1[0], neuron2[0]], [neuron1[1], neuron2[1]], "gray", lw=0.5)

    # Draw neurons
    for i, layer in enumerate(layer_positions):
        for (x, y) in layer:
            ax.add_patch(plt.Circle((x, y), 0.3, color="blue", ec="black", lw=1.5))

    # Add layer labels
    layer_labels = ["Input"] + [f"H{i}" for i in range(1, len(neuralNetworkLayers) - 1)] + ["Output"]
    
    for i, (label, layer_size) in enumerate(zip(layer_labels, neuralNetworkLayers)):
        ax.text(i, max(neuralNetworkLayers), f"{label}\n({layer_size} neurons)", ha="center", fontsize=12, fontweight="bold")

    plt.show()