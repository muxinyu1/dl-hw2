import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import GINConv, global_mean_pool
import torch.optim as optim
import wandb
import numpy as np
from sklearn.metrics import r2_score

# 初始化 wandb
wandb.init(project="QM9-GIN", name="GIN-dipole-prediction")

# 加载 QM9 数据集
path = './data/QM9'
dataset = QM9(path)
DIPOLE_INDEX = 0  # 偶极矩在目标 y 中的位置

# 数据集划分
train_dataset = dataset[:10000]
val_dataset = dataset[10000:11000]
test_dataset = dataset[11000:12000]

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义 GIN 模型
class GIN(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super(GIN, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(dataset.num_features + 3, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)

        nn2 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim))
        self.conv2 = GINConv(nn2)

        self.lin = torch.nn.Linear(hidden_dim, 1)  # 输出偶极矩预测值

    def forward(self, data):
        x = torch.cat([data.x, data.pos], dim=1)  # 拼接节点特征和坐标
        edge_index = data.edge_index
        batch = data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # 图级别聚合
        x = self.lin(x)  # 最终预测
        return x

# 初始化设备、模型和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN(hidden_dim=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
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
    wandb.log({"Train Loss": avg_loss})  # 记录训练损失到 wandb
    return avg_loss

# 评估函数
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
    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    r2 = r2_score(all_true, all_preds)
    wandb.log({f"{mode} Loss": avg_loss, f"{mode} R²": r2})  # 记录到 wandb
    return avg_loss, r2

# 训练和验证
train_losses, val_losses = [], []
for epoch in range(1, 201):
    train_loss = train()
    val_loss, val_r2 = evaluate(val_loader, mode="Validation")
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}')

# 测试集评估
test_loss, test_r2 = evaluate(test_loader, mode="Test")
print(f'Test Loss: {test_loss:.4f}, Test R²: {test_r2:.4f}')

# 结束 wandb 运行
wandb.finish()
