import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth', optimizer=None, n_games=0):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'n_games': n_games,
        }
        torch.save(checkpoint, os.path.join(model_folder_path, file_name))
        print(f"Checkpoint saved to {file_name}.")

    def load(self, file_name='model.pth', optimizer=None):
        file_path = os.path.join('./model', file_name)
        checkpoint = torch.load(file_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        n_games = checkpoint.get('n_games', 0)
        return n_games

class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done, weights):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        weights = torch.tensor(weights, dtype=torch.float)
    
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            # Removed: done = (done,)
    
        done = torch.tensor(done, dtype=torch.float)  # Convert done to tensor
    
        # Ensure action tensor has the correct shape
        action = action.unsqueeze(-1)
    
        # Ensure state and action tensors have the same batch dimension
        if state.shape[0] != action.shape[0]:
            action = action.expand(state.shape[0], -1)
    
        pred = self.model(state).gather(1, action).squeeze(-1)
        next_action = self.model(next_state).argmax(1).unsqueeze(-1)
        next_pred = self.target_model(next_state).gather(1, next_action).squeeze(-1)
        target = reward + (1 - done) * self.gamma * next_pred
    
        loss = (weights * (pred - target.detach()).pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        return loss.item()