import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os
import pickle
from colorama import Fore, Style
from tabulate import tabulate
import pandas as pd
from datetime import datetime

def save_memory(memory, file_name='memory.pkl'):
    with open(file_name, 'wb') as f:
        pickle.dump(memory, f)
    print(f"Replay memory saved to {file_name}.")

def load_memory(file_name='memory.pkl'):
    with open(file_name, 'rb') as f:
        memory = pickle.load(f)
    print(f"Replay memory loaded from {file_name}.")
    return memory

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.002

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0  # Initial exploration rate
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.01
        self.epsilon_decay_steps = 1000
        self.epsilon_step = (self.initial_epsilon - self.final_epsilon) / self.epsilon_decay_steps
        self.gamma = 0.99

        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        # Contadores de acciones
        self.exploration_count = 0  # Explorations in the current game
        self.exploitation_count = 0  # Explotaciones en el juego actual
        self.global_exp = 0          # Exploraciones globales
        self.global_expl = 0         # Explotaciones globales

        # Parámetros de decaimiento adaptativo de epsilon
        self.last_record_game = 0
        self.max_epsilon = 0.1
        self.adaptive_increase_factor = 1.5
        self.stuck_threshold = 50

        # Almacenamiento de datos
        self.game_results = []
        self.checkpoint_results = []

    def save_checkpoint(self, file_name='checkpoint.pth'):
        self.model.save(file_name, optimizer=self.trainer.optimizer, n_games=self.n_games, epsilon=self.epsilon)
        save_memory(self.memory, 'model/memory.pkl')
        print(f"Agent checkpoint saved to {file_name}.\n")
        print(f"Global Explorations: {self.global_exp}\nGlobal Exploitations: {self.global_expl}")

    def load_checkpoint(self, file_name='checkpoint.pth'):
        n_games, epsilon = self.model.load(file_name, optimizer=self.trainer.optimizer)
        self.n_games = n_games
        self.epsilon = epsilon
        print(f"Loaded checkpoint from game {self.n_games} with epsilon={self.epsilon:.4f}.")
        try:
            self.memory = load_memory('model/memory.pkl')
        except FileNotFoundError:
            print("Replay memory file not found. Starting with an empty memory.")
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
            # Peligro en línea recta
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            # Peligro a la derecha
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            # Peligro a la izquierda
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            # Dirección de movimiento
            dir_l, dir_r, dir_u, dir_d,
            # Ubicación de la comida
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y,
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        mini_sample = random.sample(self.memory, BATCH_SIZE) if len(self.memory) > BATCH_SIZE else self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Si el agente lleva muchos juegos sin mejorar, aumenta epsilon
        if self.n_games - self.last_record_game > self.stuck_threshold:
            self.epsilon = min(self.max_epsilon, self.epsilon * self.adaptive_increase_factor)
            print(f"Agent is stuck! Increasing epsilon to {self.epsilon:.4f}.")

        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
            self.exploration_count += 1
            self.global_exp += 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            self.exploitation_count += 1
            self.global_expl += 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    checkpoint_file = 'checkpoint.pth'
    checkpoint_path = os.path.join('./model', checkpoint_file)
    if os.path.exists(checkpoint_path):
        agent.load_checkpoint(checkpoint_file)
        print(f"Resuming training from game {agent.n_games}.")
        print(f"Epsilon value: {agent.epsilon:.4f}")
        print("Model state loaded successfully.")
    else:
        print("No checkpoint found. Starting training from scratch.")

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

            if score > record:
                record = score
                agent.model.save()
                agent.last_record_game = agent.n_games

            # Almacena resultados del juego actual
            timestamp = datetime.now()
            agent.game_results.append({
                'game': agent.n_games,
                'epsilon': agent.epsilon,
                'exploration': agent.exploration_count,
                'exploitation': agent.exploitation_count,
                'score': score,
                'record': record,
                'timestamp': timestamp
            })

            table_data = [
                [Fore.RED + "Game" + Style.RESET_ALL, agent.n_games],
                [Fore.GREEN + "Score" + Style.RESET_ALL, score],
                [Fore.YELLOW + "Record" + Style.RESET_ALL, record],
                [Fore.BLUE + "Epsilon" + Style.RESET_ALL, f"{agent.epsilon:.4f}"],
                [Fore.MAGENTA + "Explorations" + Style.RESET_ALL, agent.exploration_count],
                [Fore.MAGENTA + "Exploitations" + Style.RESET_ALL, agent.exploitation_count]
            ]
            print(tabulate(table_data, tablefmt="fancy_grid"))

            # Reinicia contadores para el siguiente juego
            agent.exploration_count = 0
            agent.exploitation_count = 0

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            save_plot = (agent.n_games % 100 == 0)
            plot(plot_scores, plot_mean_scores, save_plot=save_plot, save_path='plots',
                 filename=f'training_progress_game_{agent.n_games}.png')

            # Decaimiento fijo de epsilon al finalizar cada juego
            agent.epsilon = max(agent.final_epsilon, agent.epsilon - agent.epsilon_step)

            print(f'Game #{agent.n_games}')
            if agent.n_games % 10 == 0:
                agent.save_checkpoint()
                agent.checkpoint_results.append({
                    'games_range': f"{agent.n_games - 9}-{agent.n_games}",
                    'epsilon': agent.epsilon,
                    'gamma': agent.gamma,
                    'LR': LR,
                    'total_exploration': sum(res['exploration'] for res in agent.game_results[-10:]),
                    'total_exploitation': sum(res['exploitation'] for res in agent.game_results[-10:]),
                })

            if agent.n_games % 100 == 0:
                df_game_results = pd.DataFrame(agent.game_results)
                df_checkpoint_results = pd.DataFrame(agent.checkpoint_results)
                df_game_results.to_csv('results/game_results.csv', index=False)
                df_checkpoint_results.to_csv('results/checkpoint_results.csv', index=False)

if __name__ == '__main__':
    train()
