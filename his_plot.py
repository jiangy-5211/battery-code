import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history):
    """
    Plot training history curves including loss and accuracy metrics in separate plots.

    Parameters:
    history: Training history object containing loss and metric data
    """
    # Set style and color scheme
    plt.style.use('default')
    plt.rcParams['font.family'] = 'DejaVu Sans'  # Use a common cross-platform font
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFC53A']

    # Extract history data
    loss = history['loss']
    val_loss = history['val_loss']
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    # --- Plot Loss ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'o-', color=colors[0], linewidth=2, markersize=4, label='Training Loss')
    plt.plot(epochs, val_loss, 's-', color=colors[1], linewidth=2, markersize=4, label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Plot Accuracy ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracy, 'o-', color=colors[2], linewidth=2, markersize=4, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 's-', color=colors[3], linewidth=2, markersize=4, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# --- Example Usage ---
# Simulate some training history data for demonstration
if __name__ == '__main__':
    np.random.seed(42)
    num_epochs = 20
    example_history = {
        'loss': np.exp(-np.linspace(0, 3, num_epochs)) + np.random.normal(0, 0.05, num_epochs),
        'val_loss': np.exp(-np.linspace(0, 3, num_epochs)) + np.random.normal(0, 0.05, num_epochs) + 0.1,
        'accuracy': 1 - np.exp(-np.linspace(0, 2, num_epochs)) - np.random.normal(0, 0.02, num_epochs),
        'val_accuracy': 1 - np.exp(-np.linspace(0, 2, num_epochs)) - np.random.normal(0, 0.02, num_epochs) - 0.05
    }

    # Call the function with the example data
    plot_training_history(example_history)



