import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
import wandb

# ----------------------------
# 数据加载
# ----------------------------
with open("../../car_racing_data_32x32_120.pkl", "rb") as f:
    data = pickle.load(f)

class WorldModel(nn.Module):
    def __init__(self, action_size, hidden_size, output_size):
        super(WorldModel, self).__init__()
        self.fc_state = nn.Linear(3 * 32 * 32, hidden_size)
        self.fc_action = nn.Linear(action_size, hidden_size)
        self.fc_input = nn.Linear(hidden_size * 2, hidden_size * 4)
        self.fc_output = nn.Linear(hidden_size, output_size)

    def forward(self, state, action, hidden=None):
        batch_size = state.size(0)
        state_flat = state.view(batch_size, -1)
        state_proj = self.fc_state(state_flat)
        action_proj = self.fc_action(action)
        combined = torch.cat([state_proj, action_proj], dim=-1)

        if hidden is None:
            h_t = torch.zeros(1, batch_size, hidden_size, device=state.device)
            c_t = torch.zeros(1, batch_size, hidden_size, device=state.device)
        else:
            h_t, c_t = hidden

        gates = self.fc_input(combined)
        i_t, f_t, o_t, g_t = torch.chunk(gates, 4, dim=-1)
        i_t, f_t, o_t = torch.sigmoid(i_t), torch.sigmoid(f_t), torch.sigmoid(o_t)
        g_t = torch.tanh(g_t)
        c_t = f_t * c_t.squeeze(0) + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        hidden = (h_t.unsqueeze(0), c_t.unsqueeze(0))

        next_state_pred = self.fc_output(h_t)
        return next_state_pred, hidden


class WorldModelDataLoader:
    def __init__(self, data, batch_size, sequence_length, device):
        self.data = data
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device
        split_train = int(0.8 * len(self.data))
        split_valid = int(0.1 * len(self.data))
        self.train_data = self.data[:split_train]
        self.valid_data = self.data[split_train:split_train + split_valid]
        self.test_data = self.data[split_train + split_valid:]
        self.set_train()

    def set_train(self):
        self.current_data = self.train_data
        self.index, self.sub_index = 0, 0

    def set_valid(self):
        self.current_data = self.valid_data
        self.index, self.sub_index = 0, 0

    def set_test(self):
        self.current_data = self.test_data
        self.index, self.sub_index = 0, 0

    def get_batch(self):
        states, actions = [], []
        batch_data = self.current_data[self.index: self.index + self.batch_size]
        for sequence in batch_data:
            state_seq = [torch.tensor(step[0]) for step in sequence[self.sub_index:self.sub_index + self.sequence_length]]
            action_seq = [torch.tensor(step[1]) for step in sequence[self.sub_index:self.sub_index + self.sequence_length]]
            if len(state_seq) < self.sequence_length:
                pad_len = self.sequence_length - len(state_seq)
                state_seq += [torch.zeros_like(state_seq[0])] * pad_len
                action_seq += [torch.zeros_like(action_seq[0])] * pad_len
            states.append(torch.stack(state_seq))
            actions.append(torch.stack(action_seq))
        self.sub_index += self.sequence_length
        if self.sub_index >= len(self.current_data[self.index]):
            self.index += self.batch_size
            self.sub_index = 0
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        end_flag = self.index >= len(self.current_data)
        return states, actions, end_flag


# ----------------------------
# 参数初始化
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_size, hidden_size, output_size = 3, 128, 32 * 32 * 3
batch_size, sequence_length = 16, 10
learning_rate, num_epochs = 3e-4, 50

data_loader = WorldModelDataLoader(data, batch_size, sequence_length, device)
model = WorldModel(action_size=action_size, hidden_size=hidden_size, output_size=output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ----------------------------
# wandb 初始化
# ----------------------------
wandb.init(project="world_model_rollout", config={
    "action_size": action_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "batch_size": batch_size,
    "sequence_length": sequence_length,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate
})

# ----------------------------
# 训练过程
# ----------------------------
def train():
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        data_loader.set_train()
        total_train_loss = 0
        total_train_samples = 0
        while True:
            states, actions, end_flag = data_loader.get_batch()
            batch_size_actual = states.size(0)
            hidden = (torch.zeros(1, batch_size_actual, hidden_size).to(device),
                      torch.zeros(1, batch_size_actual, hidden_size).to(device))
            for t in range(sequence_length - 1):
                current_state = states[:, t]
                action = actions[:, t]
                next_state = states[:, t + 1].view(batch_size_actual, -1)
                next_state_pred, hidden = model(current_state, action, hidden)
                loss = criterion(next_state_pred, next_state)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                hidden = tuple([h.detach() for h in hidden])
                total_train_loss += loss.item()
                total_train_samples += 1
            if end_flag:
                break
        avg_train_loss = total_train_loss / total_train_samples
        val_loss = evaluate()
        wandb.log({"Train Loss": avg_train_loss, "Validation Loss": val_loss, "Epoch": epoch + 1})
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "world_model_best.pth")
            wandb.save("world_model_best.pth")
            print("Best model saved.")

def evaluate():
    model.eval()
    data_loader.set_valid()
    total_val_loss = 0
    total_val_samples = 0
    with torch.no_grad():
        while True:
            states, actions, end_flag = data_loader.get_batch()
            batch_size_actual = states.size(0)
            hidden = (torch.zeros(1, batch_size_actual, hidden_size).to(device),
                      torch.zeros(1, batch_size_actual, hidden_size).to(device))
            for t in range(sequence_length - 1):
                current_state = states[:, t]
                action = actions[:, t]
                next_state = states[:, t + 1].view(batch_size_actual, -1)
                next_state_pred, hidden = model(current_state, action, hidden)
                loss = criterion(next_state_pred, next_state)
                total_val_loss += loss.item()
                total_val_samples += 1
            if end_flag:
                break
    return total_val_loss / total_val_samples

# ----------------------------
# Rollout Strategies
# ----------------------------
def teacher_forcing_rollout(model, initial_state, actions, ground_truth_states, sequence_length):
    model.eval()
    predictions = []
    state = initial_state.unsqueeze(0)
    hidden = (torch.zeros(1, 1, hidden_size).to(device), torch.zeros(1, 1, hidden_size).to(device))
    with torch.no_grad():
        for t in range(sequence_length - 1):
            action = actions[t].unsqueeze(0).to(device)
            next_state_gt = ground_truth_states[t + 1].view(1, -1).to(device)
            next_state_pred, hidden = model(state, action, hidden)
            predictions.append(next_state_pred.squeeze(0).cpu())
            state = next_state_gt.unsqueeze(0)
    return torch.stack(predictions)


def autoregressive_rollout(model, initial_state, actions, sequence_length):
    model.eval()
    predictions = []
    state = initial_state.unsqueeze(0)
    hidden = (torch.zeros(1, 1, hidden_size).to(device), torch.zeros(1, 1, hidden_size).to(device))
    with torch.no_grad():
        for t in range(sequence_length - 1):
            action = actions[t].unsqueeze(0).to(device)
            next_state_pred, hidden = model(state, action, hidden)
            predictions.append(next_state_pred.squeeze(0).cpu())
            state = next_state_pred.unsqueeze(0)  # 使用模型的预测作为下一步输入
    return torch.stack(predictions)

# ----------------------------
# 可视化和误差分析
# ----------------------------
def plot_rollout_comparison(ground_truth, pred_teacher_forcing, pred_autoregressive, steps=10):
    plt.figure(figsize=(15, 5))
    for t in range(steps):
        plt.subplot(3, steps, t + 1)
        plt.imshow(ground_truth[t].permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f"Ground Truth {t}")
        plt.subplot(3, steps, steps + t + 1)
        plt.imshow(pred_teacher_forcing[t].view(3, 32, 32).permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f"Teacher Forcing {t}")
        plt.subplot(3, steps, 2 * steps + t + 1)
        plt.imshow(pred_autoregressive[t].view(3, 32, 32).permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f"Autoregressive {t}")
    plt.tight_layout()
    plt.show()

# ----------------------------
# 测试 Rollout
# ----------------------------
def test_rollout():
    model.eval()
    data_loader.set_test()
    states, actions, _ = data_loader.get_batch()
    initial_state = states[0][0]  # 使用测试集中的一个序列
    sequence_actions = actions[0]
    ground_truth_states = states[0]

    # Teacher Forcing Rollout
    teacher_forcing_predictions = teacher_forcing_rollout(
        model, initial_state, sequence_actions, ground_truth_states, sequence_length
    )

    # Autoregressive Rollout
    autoregressive_predictions = autoregressive_rollout(
        model, initial_state, sequence_actions, sequence_length
    )

    # 可视化对比
    plot_rollout_comparison(
        ground_truth_states[:sequence_length], 
        teacher_forcing_predictions, 
        autoregressive_predictions, 
        steps=sequence_length
    )

    # 误差分析
    mse_teacher_forcing = torch.mean(
        (ground_truth_states[1:sequence_length] - teacher_forcing_predictions) ** 2, dim=(1, 2, 3)
    )
    mse_autoregressive = torch.mean(
        (ground_truth_states[1:sequence_length] - autoregressive_predictions) ** 2, dim=(1, 2, 3)
    )

    print("MSE 每步误差对比:")
    print("Teacher Forcing:", mse_teacher_forcing)
    print("Autoregressive:", mse_autoregressive)

    print("\n累计误差:")
    print("Teacher Forcing:", torch.sum(mse_teacher_forcing).item())
    print("Autoregressive:", torch.sum(mse_autoregressive).item())

    # 将结果记录到 wandb
    wandb.log({
        "Test Rollout Teacher Forcing MSE": torch.sum(mse_teacher_forcing).item(),
        "Test Rollout Autoregressive MSE": torch.sum(mse_autoregressive).item()
    })

# ----------------------------
# 运行代码
# ----------------------------
if __name__ == "__main__":
    train()  # 训练模型
    test_rollout()  # 测试并比较两种 Rollout 策略
    wandb.finish()

