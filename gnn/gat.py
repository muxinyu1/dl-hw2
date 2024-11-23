import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import GATConv, global_mean_pool
import torch.optim as optim
import numpy as np
import wandb
from sklearn.metrics import r2_score

# 初始化 wandb 项目
wandb.init(project="QM9-GAT", name="GAT-dipole-prediction")

path = './data/QM9'
dataset = QM9(path)
DIPOLE_INDEX = 0  # 偶极矩在 y 中的位置

train_dataset = dataset[:10000]
val_dataset = dataset[10000:11000]
test_dataset = dataset[11000:12000]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class GAT(torch.nn.Module):
    def __init__(self, hidden_dim=64, heads=4):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dataset.num_features + 3, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.lin = torch.nn.Linear(hidden_dim, 1)  # 输出偶极矩

    def forward(self, data):
        # 将节点特征和坐标拼接
        x = torch.cat([data.x, data.pos], dim=1)
        edge_index = data.edge_index
        batch = data.batch
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(hidden_dim=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
    avg_loss = total_loss / len(train_loader.dataset)
    wandb.log({"Train Loss": avg_loss})  # 记录到 wandb
    return avg_loss

def evaluate(loader, mode="Validation"):
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
    avg_loss = total_loss / len(loader.dataset)
    wandb.log({f"{mode} Loss": avg_loss})  # 记录到 wandb

    # 将预测值和真实值拼接为完整数组
    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    if mode == "Test":
        # 计算 R² 分数
        r2 = r2_score(all_true, all_preds)
        wandb.log({"Test R2": r2})
        print(f"R² Score on Test Set: {r2:.4f}")
    
    return avg_loss

# 训练模型
train_losses, val_losses = [], []
for epoch in range(1, 51):  
    train_loss = train()
    val_loss = evaluate(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# 测试集评估
test_loss = evaluate(test_loader, mode="Test")
print(f'Test Loss: {test_loss:.4f}')

# 可选：记录最终损失到 wandb
wandb.log({
    "Final Train Loss": train_losses,
    "Final Validation Loss": val_losses,
})

# 结束 wandb 运行
wandb.finish()
