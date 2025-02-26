import torch
import numpy as np
from helper import *
from model import DQN, QTrainer
from colorama import Fore, Style
from game import SnakeGameAI, Direction, Point

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_MEMORY = 150_000  # Larger memory for better experience diversity
LR = 0.0003  # Smaller learning rate for more stable training
GAMMA = 0.99  # Higher discount factor to better value future rewards
BATCH_SIZE = 512  # Larger batch size for better gradient estimates
TAU = 0.001  # Slower target network updates for stability


class PrioritizedReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.position = 0

    def push(self, experience):
        max_priority = max(self.priorities, default=1.0)
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
            self.priorities.append(max_priority)
        else:
            self.memory[self.position] = experience
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == 0:
            raise ValueError("Memory is empty")
        if len(self.priorities) != len(self.memory):
            self.priorities = np.ones(len(self.memory), dtype=np.float32).tolist()

        probabilities = np.array(self.priorities, dtype=np.float32)
        probabilities = probabilities / probabilities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        mini_sample = [self.memory[i] for i in indices]
        weights = probabilities[indices]
        return mini_sample, indices, weights

    def update_priorities(self, batch_indices, batch_priorities, max_priority=100.0):
        for idx, priority in zip(batch_indices, batch_priorities):
            clamped_priority = np.clip(priority, a_min=0, a_max=max_priority)
            self.priorities[idx] = clamped_priority


class Agent:
    def __init__(self):
        self.n_games = 0
        self.last_record_game = 0
        self.memory = PrioritizedReplayMemory(MAX_MEMORY)
        self.model = DQN(11, 256, 3).to(device)
        self.target_model = DQN(11, 256, 3).to(device)

        n_games_loaded, _, _, last_recorded_game = self.model.load("model_MARK_VII.pth")  # TODO: explain what this function does (from version 1.1 onward)
        
        self.n_games = n_games_loaded
        self.last_record_game = last_recorded_game
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=GAMMA)
        self.temperature = 1.0
        self.game_results = []
        
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
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # Danger right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            # Danger left
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y,
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory.push(experience)

    def train_long_memory(self):
        mini_sample, indices, weights = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        print(Fore.GREEN + f"Actions:\n{actions}\n" + Style.RESET_ALL)
        print(Fore.YELLOW + f"Rewards:\n{rewards}\n" + Style.RESET_ALL)

        actions = np.array([np.argmax(a) for a in actions])
        states = np.array(states)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        weights = np.array(weights)

        loss = self.trainer.train_step(states, actions, rewards, next_states, dones, weights)

        priorities = np.full(len(indices), loss + 1e-5, dtype=np.float32)
        self.memory.update_priorities(indices, priorities)

        return loss  # Return the loss value

    def train_short_memory(self, state, action, reward, next_state, done):
        action_idx = np.array([np.argmax(action)])
        weights = np.ones((1,), dtype=np.float32)
        self.trainer.train_step(state, action_idx, reward, next_state, done, weights)

    def get_action(self, state: np.ndarray) -> list:
        state_tensor = torch.tensor(state, dtype=torch.float)
        q_values = self.model(state_tensor).detach().numpy()
        
        # Improved numerical stability: subtract max value
        exp_q = np.exp((q_values - np.max(q_values)) / self.temperature)
        
        probabilities = exp_q / np.sum(exp_q)
        action = np.random.choice(len(q_values), p=probabilities)
        final_move = [0, 0, 0]
        final_move[action] = 1
        return final_move

    def update_target_network(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

    def update_temperature(self, decay_rate, min_temperature):
        self.temperature = max(self.temperature * decay_rate, min_temperature)

def train(max_games: int) -> None:
    agent = Agent()
    game = SnakeGameAI()
    
    record = agent.last_record_game
    total_score = 0
    plot_mean_scores = []
    plot_scores = []

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

            # Long term training and capturing loss
            loss = agent.train_long_memory()
            save_checkpoint(agent, loss)

            # Calculate reward statistics for the current game
            episode_reward = sum(game.reward_history)
            avg_reward = episode_reward / len(game.reward_history) if game.reward_history else 0

            if score > record:
                record = score
                agent.last_record_game = agent.n_games 
                print(Fore.GREEN + f"New record at game {agent.last_record_game}!" + Style.RESET_ALL)

            total_score = update_plots(agent, score, total_score, plot_scores, plot_mean_scores)
            
            # Auxiliary functions to print information
            print(Fore.YELLOW + f"Game #{agent.n_games} ended with loss: {loss}" + Style.RESET_ALL)
            print(Fore.MAGENTA + f"Game {agent.n_games} - Total Reward: {episode_reward}, Avg Reward: {avg_reward:.2f}" + Style.RESET_ALL)
            print_weight_norms(agent)
            log_game_results(agent, score, record)
            print_game_info(reward, score, agent.last_record_game)
            
            # Terminate training if max_games reached
            if agent.n_games >= max_games:
                print(Fore.GREEN + "            Training complete." + Style.RESET_ALL)
                break


if __name__ == "__main__":
    train(max_games=1000)