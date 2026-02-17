import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def get_dataloaders(dataset_name, batch_size, data_dir="./data"):
    
    # --- Dataset A: Adult Income (Tabular) ---
    if dataset_name == 'adult':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        columns = [
            "age", "workclass", "fnlwgt", "education", "education-num", 
            "marital-status", "occupation", "relationship", "race", "sex", 
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
        ]

        df = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)

        df.dropna(inplace=True)
        
        df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)
        y = df['income'].values
        X_df = df.drop('income', axis=1)
        
        num_cols = X_df.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = X_df.select_dtypes(include=['object']).columns
        
        X_encoded = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)
        
        scaler = StandardScaler()
        X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])
        
        X_tensor = torch.tensor(X_encoded.values.astype(np.float32), dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # First split: 80% Train, 20% Temp
        X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
        # Second split
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Dataset B: CIFAR-10 (Image) ---
    elif dataset_name == 'cifar':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Dataset C: PCam (Medical) ---
    elif dataset_name == 'pcam':
        print("Loading PCam Dataset...") 
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((96, 96))
        ])
        train_dataset = datasets.PCAM(root=data_dir, split='train', download=True, transform=transform)
        val_dataset = datasets.PCAM(root=data_dir, split='val', download=True, transform=transform)
        
        test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader