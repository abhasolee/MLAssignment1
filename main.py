import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_dataloaders
from src.models import MLP, CNN, SimpleAttention
from src.utils import train_model, evaluate_model

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

datasets_list = ['adult', 'cifar', 'pcam'] 
models_list = ['MLP', 'CNN', 'Attention']

results = []

for d_name in datasets_list:
    print(f"\n--- Loading Dataset: {d_name} ---")
    train_loader, val_loader, test_loader = get_dataloaders(d_name, config['experiment']['batch_size'])
    
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.view(sample_x.shape[0], -1).shape[1] # Flattened size
    
    if d_name == 'adult':
        num_classes = 2
        img_size = input_dim
        channels = 1
        is_tabular = True
    elif d_name == 'cifar':
        num_classes = 10
        img_size = 32
        channels = 3
        is_tabular = False
    elif d_name == 'pcam':
        num_classes = 2
        img_size = 96
        channels = 3
        is_tabular = False

    for m_name in models_list:
        print(f"Training {m_name} on {d_name}...")
        
        if m_name == 'MLP':
            model = MLP(input_dim, num_classes).to(device)
        elif m_name == 'CNN':
            model = CNN(channels, img_size, num_classes, is_1d=is_tabular).to(device)
        elif m_name == 'Attention':
            model = SimpleAttention(input_dim, num_classes).to(device)
    
        optimizer = optim.Adam(model.parameters(), lr=config['experiment']['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        _, _, time_taken = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, 
            config['experiment']['num_epochs']
        )
        
        test_acc, test_f1 = evaluate_model(model, test_loader, device)
        
        results.append({
            "Dataset": d_name,
            "Architecture": m_name,
            "Accuracy": f"{test_acc:.4f}",
            "F1": f"{test_f1:.4f}",
            "Time": f"{time_taken:.1f}s"
        })

print("\nFinal Results Table:")
print(f"{'Dataset':<10} | {'Architecture':<12} | {'Accuracy':<10} | {'F1':<10} | {'Time':<10}")
print("-" * 60)
for r in results:
    print(f"{r['Dataset']:<10} | {r['Architecture']:<12} | {r['Accuracy']:<10} | {r['F1']:<10} | {r['Time']:<10}")