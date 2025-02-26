# modelo de red neuronal profunda (DQN) y un entrenador asociado para implementar el algoritmo de aprendizaje por refuerzo

import os
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from colorama import Fore, Style

# Configure logging if it has not been set up
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def save(self, file_name, folder_path='./model_Model', n_games=0, optimizer=None, loss=None, last_record_game=None):
        # Creates the directory if it does not exist
        os.makedirs(folder_path, exist_ok=True)
        checkpoint = {
            'n_games': n_games,
            'model_state_dict': self.state_dict(),
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer
        if loss is not None:
            checkpoint['loss'] = loss
        if last_record_game is not None:
            checkpoint['last_record_game'] = last_record_game
        checkpoint_path = os.path.join(folder_path, file_name)
        try:
            torch.save(checkpoint, checkpoint_path)
            print(Fore.GREEN + f"Modelo guardado en {checkpoint_path}" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error al guardar el modelo: {e}" + Style.RESET_ALL)



    def load(self, file_name, folder_path='./model_Model'):
        folder_fullpath = folder_path
        if not os.path.exists(folder_fullpath):
            logging.error(f"Directory {folder_fullpath} does not exist.")
            return 0, None, None, 0
        file_path = os.path.join(folder_fullpath, file_name)
        try:
            checkpoint = torch.load(file_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            n_games = checkpoint.get('n_games', 0)
            loss = checkpoint.get('loss', None)
            optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
            last_record_game = checkpoint.get('last_record_game', 0)
            logging.info(f"Unified checkpoint '{file_name}' loaded from {folder_fullpath}. n_games: {n_games}")
            return n_games, loss, optimizer_state_dict, last_record_game
        except FileNotFoundError:
            logging.error(f"File {file_path} not found.")
            return 0, None, None, 0
        except Exception as e:
            logging.error(f"Error loading unified checkpoint '{file_name}': {e}")
            return 0, None, None, 0

class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def _prepare_tensor(self, data, dtype, unsqueeze=True):
        tensor = torch.tensor(data, dtype=dtype).to(device)
        if unsqueeze and len(tensor.shape) == 1:
            tensor = torch.unsqueeze(tensor, 0)
        return tensor

    def train_step(self, state, action, reward, next_state, done, weights):
        # Convierte las entradas utilizando la función auxiliar
        state = self._prepare_tensor(state, torch.float32)
        next_state = self._prepare_tensor(next_state, torch.float32)
        action = self._prepare_tensor(action, torch.int64)
        reward = self._prepare_tensor(reward, torch.float32)
        done = self._prepare_tensor(done, torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)  # No se aplica unsqueeze a weights
    
        # Asegura que action tenga forma [batch, 1]
        if action.dim() == 1:
            action = action.unsqueeze(1)
    
        # Calcula predicciones y objetivos
        pred = self.model(state).gather(1, action).squeeze(-1)
        next_action = self.model(next_state).argmax(1).unsqueeze(-1)
        next_pred = self.target_model(next_state).gather(1, next_action).squeeze(-1)
        target = reward + (1 - done) * self.gamma * next_pred
    
        # Calcula pérdida y actualiza pesos. Se utiliza target.detach() para evitar la retropropagación a la red target.
        loss = (weights * (pred - target.detach()).pow(2)).mean()
    
        # Obtén estadísticas previas (p.ej. pesos antes de actualizar)
        old_weights = {name: param.clone() for name, param in self.model.named_parameters()}
    
        self.optimizer.zero_grad()
        loss.backward()
    
        # Logear la magnitud de los gradientes
        grad_norms = {name: param.grad.norm().item() for name, param in self.model.named_parameters() if param.grad is not None}
        logging.debug(f"Gradient norms: {grad_norms}")
    
        self.optimizer.step()
    
        # Calcula y logea los cambios en los pesos
        weight_changes = {name: (param - old_weights[name]).norm().item() for name, param in self.model.named_parameters()}
        logging.debug(f"Weight changes: {weight_changes}")
    
        logging.debug(f"Train step completed. Loss: {loss.item()}")
        return loss.item()