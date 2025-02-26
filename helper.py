import os
import sys
import pandas as pd
from IPython import display
from datetime import datetime
import matplotlib.pyplot as plt
from colorama import Fore, Style

def plot_training_progress(scores, mean_scores, save_plot=False, save_path="plots", filename="training_progress.png"):
    # Solo actualiza dinámicamente si se ejecuta en un entorno interactivo
    if "ipykernel" in sys.modules:
        display.clear_output(wait=True)
        display.display(plt.gcf())
        
    plt.clf()
    plt.title("Training Progress")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores, label="Scores")
    plt.plot(mean_scores, label="Mean Scores")
    plt.ylim(ymin=0)
    plt.legend()
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    
    if save_plot:
        try:
            os.makedirs(save_path, exist_ok=True)
            full_path = os.path.join(save_path, filename)
            plt.savefig(full_path)
            print(f"Plot saved to {full_path}.")
        except Exception as e:
            print(f"Error saving plot: {e}")
    
    plt.show(block=False)
    plt.pause(0.1)

def log_game_results(agent, score, record):
    # Registra y guarda resultados del juego en CSV cada 25 juegos.
    timestamp = datetime.now()
    agent.game_results.append({"game": agent.n_games, "score": score, "record": record, "timestamp": timestamp})
    
    if agent.n_games % 25 == 0:
        df_game_results = pd.DataFrame(agent.game_results)
        os.makedirs("results", exist_ok=True)
        df_game_results.to_csv("results/game_results.csv", index=False)

def update_plots(agent, score, total_score, plot_scores, plot_mean_scores):
    # Actualiza las listas de puntuaciones y genera el gráfico de progreso
    plot_scores.append(score)
    total_score += score
    mean_score = total_score / agent.n_games
    plot_mean_scores.append(mean_score)
    
    save_plot = agent.n_games % 100 == 0
    plot_training_progress(
        plot_scores,
        plot_mean_scores,
        save_plot=save_plot,
        save_path="plots",
        filename=f"training_progress_game_{agent.n_games}.png",
    )
    return total_score

def save_checkpoint(agent, loss):
    # Guarda el modelo (checkpoint) incluyendo algunos estados del entrenamiento
    agent.model.save(
        "model_MARK_VII.pth",
        n_games=agent.n_games,
        optimizer=agent.trainer.optimizer.state_dict(),
        loss=loss,
        last_record_game=agent.last_record_game
    )

def print_game_info(reward, score, last_record_game):
    # Imprime información del juego
    print("-" * 30)
    print(f"Último récord logrado en la partida: {last_record_game}")
    print(Fore.CYAN + "Reward: " + Style.RESET_ALL, reward)
    print(Fore.MAGENTA + "Score: " + Style.RESET_ALL, score)
    print("-" * 30)

def print_weight_norms(agent):
    # Muestra las normas de los pesos para dar seguimiento al entrenamiento
    w1_norm = agent.model.linear1.weight.data.norm().item()
    w2_norm = agent.model.linear2.weight.data.norm().item()
    w3_norm = agent.model.linear3.weight.data.norm().item()
    print(Fore.CYAN + f"Weight norms - linear1: {w1_norm:.4f}, linear2: {w2_norm:.4f}, linear3: {w3_norm:.4f}" + Style.RESET_ALL)