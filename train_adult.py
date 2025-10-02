import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class AdultDataset(Dataset):
    def __init__(self, dataframe):
        self.X = dataframe.drop('income', axis=1).values.astype(float)
        self.y = dataframe['income'].values.astype(int)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

class AdultNN(nn.Module):
    def __init__(self, input_dim):
        super(AdultNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Binary classification output
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def preprocess_adult_data(filepath):
    # Load CSV
    df = pd.read_csv(filepath)

    # Encode labels: income >50K
