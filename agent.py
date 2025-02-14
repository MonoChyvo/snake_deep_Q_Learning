import os
import torch
import pickle
import random
import numpy as np
import pandas as pd
from helper import plot
from collections import deque
from tabulate import tabulate
from datetime import datetime
from model import DQN, QTrainer
from colorama import Fore, Style
from game import SnakeGameAI, Direction, Point
from torch.utils.tensorboard import SummaryWriter

def save_memory(memory, file_name='memory.pkl'):
    with open(file_name, 'wb') as f:
        pickle.dump(memory, f)
    print(f"Replay memory saved to {file_name}.")

def load_memory(file_name='./model/memory.pkl'):
    with open(file_name, 'rb') as f:
        memory = pickle.load(f)
    print(Fore.MAGENTA + f"Replay memory loaded from {file_name}.\n" + Style.RESET_ALL)
    return memory

MAX_MEMORY = 100_000
LR = 0.001           # Reducir la tasa de aprendizaje
GAMMA = 0.95         # Cambiar el factor de descuento para recompensas futuras
BATCH_SIZE = 500     # Reducir o aumentar según experimentación
TAU = 0.005          # Parámetro de actualización suave

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        return batch, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

class Agent:
    def __init__(self):
        self.n_games = 0
        self.gamma = GAMMA

        self.memory = PrioritizedReplayMemory(MAX_MEMORY)
        self.model = DQN(11, 256, 3)
        self.target_model = DQN(11, 256, 3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)

        # Boltzmann exploration parameters
        self.temperature = 1.0

        # Initialize game results list
        self.game_results = []

    def save_checkpoint(self, file_name='checkpoint.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_path = os.path.join(model_folder_path, file_name)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'n_games': self.n_games,
        }
        torch.save(checkpoint, file_path)
        save_memory(self.memory, os.path.join(model_folder_path, 'memory.pkl'))
        print(f"Agent checkpoint saved to {file_path}.\n")

    def load_checkpoint(self, checkpoint):
        print(Fore.GREEN + f"Loading checkpoint from {checkpoint}...\n" + Style.RESET_ALL)

        checkpoint = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.n_games = checkpoint.get('n_games', 0)
        print(Fore.RED + f"Loaded checkpoint from game {self.n_games}\n" + Style.RESET_ALL)
        
        try:
            self.memory = load_memory()
        except FileNotFoundError:
            print(Fore.MAGENTA + "Replay memory file not found. Starting with an empty memory.\n" + Style.RESET_ALL)
            self.memory = deque(maxlen=MAX_MEMORY)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            # Move direction
            dir_l, dir_r, dir_u, dir_d,
            # Food location
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y,
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory.memory) < BATCH_SIZE:
            return
        mini_sample, indices, weights = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = mini_sample

        actions = np.array([np.argmax(a) for a in actions])
        states = np.array(states)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        weights = np.array(weights)

        loss = self.trainer.train_step(states, actions, rewards, next_states, dones, weights)
        priorities = np.full(len(indices), loss + 1e-5, dtype=np.float32)
        self.memory.update_priorities(indices, priorities)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Convert one-hot action to index
        action_idx = np.argmax(action)  # new
        weights = np.ones((1,), dtype=np.float32)
        self.trainer.train_step(state, action_idx, reward, next_state, done, weights)

    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float)
        q_values = self.model(state_tensor).detach().numpy()
        probabilities = np.exp(q_values / self.temperature) / np.sum(np.exp(q_values / self.temperature))
        action = np.random.choice(len(q_values), p=probabilities)
        final_move = [0, 0, 0]
        final_move[action] = 1
        return final_move

    def update_target_network(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
    
    def update_temperature(self, decay_rate, min_temperature):
        # Actualiza la temperatura con decaimiento exponencial
        self.temperature = max(self.temperature * decay_rate, min_temperature)

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    checkpoint_path = './model/checkpoint.pth'
    print(Fore.CYAN + f"\nCheckpoint path: {checkpoint_path}\n" + Style.RESET_ALL)

    if os.path.exists(checkpoint_path):
        agent.load_checkpoint(checkpoint_path)
        print(Fore.YELLOW + f"Resuming training from game {agent.n_games}.\n" + Style.RESET_ALL)
        print(Fore.GREEN + "Model state loaded successfully.\n" + Style.RESET_ALL)
    else:
        print("No checkpoint found. Starting training from scratch.")

    writer = SummaryWriter('runs/snake_training')

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move, agent.n_games, record)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            agent.update_target_network()

            # Actualización de la temperatura
            agent.update_temperature(0.995, 0.1)

            
                

            if score > record:
                record = score
                agent.model.save()
                agent.last_record_game = agent.n_games

            timestamp = datetime.now()
            agent.game_results.append({
                'game': agent.n_games,
                'score': score,
                'record': record,
                'timestamp': timestamp
            })

            table_data = [
                [Fore.RED + "Game" + Style.RESET_ALL, agent.n_games],
                [Fore.GREEN + "Score" + Style.RESET_ALL, score],
                [Fore.YELLOW + "Record" + Style.RESET_ALL, record],
                [Fore.BLUE + "Temperature" + Style.RESET_ALL, f"{agent.temperature:.4f}"]
            ]
            print(tabulate(table_data, tablefmt="fancy_grid"))

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            save_plot = (agent.n_games % 100 == 0)
            plot(plot_scores, plot_mean_scores, save_plot=save_plot, save_path='plots', filename=f'training_progress_game_{agent.n_games}.png')

            writer.add_scalar('Score', score, agent.n_games)
            writer.add_scalar('Mean Score', mean_score, agent.n_games)
            writer.add_scalar('Temperature', agent.temperature, agent.n_games)
            writer.add_scalar('Steps per Game', game.steps, agent.n_games)
            writer.add_scalar('Cumulative Reward', total_score, agent.n_games)

            if agent.n_games % 10 == 0:
                agent.save_checkpoint()
                

            if agent.n_games % 100 == 0:
                df_game_results = pd.DataFrame(agent.game_results)
                df_game_results.to_csv('results/game_results.csv', index=False)

if __name__ == '__main__':
    train()
