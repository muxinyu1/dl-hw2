import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import GINConv, global_mean_pool
import torch.optim as optim
from sklearn.metrics import r2_score
import numpy as np

# Load QM9 dataset
path = './data/QM9'
dataset = QM9(path)
DIPOLE_INDEX = 0  # Dipole moment index in target `y`

# Split dataset
train_dataset = dataset[:10000]
val_dataset = dataset[10000:11000]
test_dataset = dataset[11000:12000]

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define GIN model
class GIN(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super(GIN, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(dataset.num_features + 3, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)

        nn2 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim))
        self.conv2 = GINConv(nn2)

        self.lin = torch.nn.Linear(hidden_dim, 1)  # Output layer for dipole moment

    def forward(self, data):
        x = torch.cat([data.x, data.pos], dim=1)  # Concatenate node features and coordinates
        edge_index = data.edge_index
        batch = data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # Aggregate node features to graph features
        x = self.lin(x)  # Final prediction
        return x

# Set up device, model, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN(hidden_dim=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and evaluation functions
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y[:, DIPOLE_INDEX].unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def evaluate(loader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_true = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = F.mse_loss(out, data.y[:, DIPOLE_INDEX].unsqueeze(1))
            total_loss += loss.item() * data.num_graphs
            all_preds.append(out.cpu().numpy())
            all_true.append(data.y[:, DIPOLE_INDEX].unsqueeze(1).cpu().numpy())

    # Flatten predictions and true values for R² calculation
    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    r2 = r2_score(all_true, all_preds)
    return total_loss / len(loader.dataset), r2

# Train and evaluate the model
train_losses, val_losses = [], []
for epoch in range(1, 51):
    train_loss = train()
    val_loss, val_r2 = evaluate(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}')

# Test performance
test_loss, test_r2 = evaluate(test_loader)
print(f'Test Loss: {test_loss:.4f}, Test R²: {test_r2:.4f}')
