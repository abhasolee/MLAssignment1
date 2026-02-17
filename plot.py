import matplotlib.pyplot as plt

# --- 1. Data Entry ---
# I have transcribed the numbers from your logs into this dictionary
data = {
    "Adult Income (Tabular)": {
        "MLP": {
            "loss": [141.5597, 131.3464, 128.8628, 127.6916, 126.2020, 125.5439, 124.9700, 123.8110, 123.1308, 120.8052],
            "val_acc": [0.8621, 0.8600, 0.8532, 0.8578, 0.8566, 0.8612, 0.8575, 0.8606, 0.8590, 0.8600]
        },
        "CNN": {
            "loss": [151.2514, 132.1430, 130.0974, 128.4980, 127.8666, 127.7865, 127.2273, 126.9094, 126.4685, 126.3887],
            "val_acc": [0.8421, 0.8566, 0.8523, 0.8566, 0.8541, 0.8492, 0.8590, 0.8566, 0.8566, 0.8557]
        },
        "Attention": {
            "loss": [137.3751, 132.8502, 131.9284, 131.7929, 131.5393, 131.6066, 131.2607, 131.3807, 131.0225, 131.2388],
            "val_acc": [0.8517, 0.8550, 0.8477, 0.8501, 0.8504, 0.8544, 0.8535, 0.8477, 0.8526, 0.8529]
        }
    },
    "CIFAR-10 (Natural Images)": {
        "MLP": {
            "loss": [1353.1689, 1223.3876, 1164.6698, 1130.0642, 1096.7899, 1077.5387, 1056.9168, 1035.6318, 1019.6260, 1006.1204],
            "val_acc": [0.4676, 0.4880, 0.5071, 0.5228, 0.5289, 0.5305, 0.5422, 0.5433, 0.5475, 0.5467]
        },
        "CNN": {
            "loss": [1048.4017, 772.6650, 676.8879, 614.8790, 563.7642, 524.9404, 489.6249, 455.9533, 429.3322, 404.9282],
            "val_acc": [0.6141, 0.6529, 0.6733, 0.7020, 0.7045, 0.7105, 0.7189, 0.7247, 0.7208, 0.7245]
        },
        "Attention": {
            "loss": [1438.7733, 1384.3512, 1367.6751, 1356.7534, 1348.9223, 1341.1164, 1334.1288, 1331.8603, 1328.0413, 1322.9882],
            "val_acc": [0.3679, 0.3935, 0.3886, 0.3894, 0.3908, 0.3909, 0.3873, 0.3872, 0.3986, 0.3951]
        }
    },
    "PCam (Medical Images)": {
        "MLP": {
            "loss": [2143.9188, 1991.8986, 1840.1256, 1674.5914, 1512.3262, 1383.5489, 1267.8890, 1173.1324, 1096.6532, 1023.5524],
            "val_acc": [0.7766, 0.7553, 0.7293, 0.7064, 0.6669, 0.6785, 0.6852, 0.6263, 0.6721, 0.7115]
        },
        "CNN": {
            "loss": [1848.8507, 1522.4417, 1335.4796, 1224.9380, 1139.6533, 1073.4755, 1029.2540, 985.1154, 948.9202, 919.8739],
            "val_acc": [0.7720, 0.7946, 0.7839, 0.8343, 0.8156, 0.8278, 0.8161, 0.7890, 0.8009, 0.8226]
        },
        "Attention": {
            "loss": [2661.0346, 2522.3062, 2468.9157, 2420.1578, 2386.1905, 2388.7906, 2350.5886, 2370.3725, 2349.5431, 2398.7565],
            "val_acc": [0.6802, 0.6714, 0.6854, 0.7101, 0.7083, 0.7157, 0.7168, 0.7122, 0.7143, 0.7180]
        }
    }
}

epochs = list(range(1, 11))
colors = {'MLP': 'red', 'CNN': 'blue', 'Attention': 'green'}
markers = {'MLP': 'o', 'CNN': 's', 'Attention': '^'}

# --- 2. Plotting Loop ---
for dataset_name, models_data in data.items():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Title for the whole figure
    fig.suptitle(f"Performance on {dataset_name}", fontsize=16, weight='bold')
    
    # Subplot 1: Training Loss
    for model_name, metrics in models_data.items():
        axes[0].plot(epochs, metrics['loss'], 
                     label=model_name, color=colors[model_name], 
                     marker=markers[model_name], linestyle='-', linewidth=2)
    
    axes[0].set_title("Training Loss (Lower is Better)")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # Subplot 2: Validation Accuracy
    for model_name, metrics in models_data.items():
        axes[1].plot(epochs, metrics['val_acc'], 
                     label=model_name, color=colors[model_name], 
                     marker=markers[model_name], linestyle='-', linewidth=2)
    
    axes[1].set_title("Validation Accuracy (Higher is Better)")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    # Adjust layout and show
    plt.tight_layout()
    plt.show()