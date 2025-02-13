import matplotlib.pyplot as plt
from IPython import display
import os

def plot(scores, mean_scores, save_plot=False, save_path='plots', filename='training_progress.png'):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Scores')
    plt.plot(mean_scores, label='Mean Scores')
    plt.ylim(ymin=0)
    plt.legend()
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    
    if save_plot:
        # Ensure the directory exists
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path)  # Save the plot as an image file
        print(f"Plot saved to {full_path}.")
    
    plt.show(block=False)
    plt.pause(.1)