import numpy as np
import matplotlib.pyplot as plt

def plot_scores(array, title):
    
    plt.title(title)
    plt.ylabel('Scores')
    plt.xlabel('Games')
    plt.plot(array)
    plt.yticks(np.arange(min(array), max(array)+1, 3.0))
    plt.savefig('./plot_avg_scores.png')


def plot_avg_scores(file):
    scores_array = []
    with open(file) as f:
        for line in f.readlines():
            currentline = line.split(",")
            for score in currentline:
                scores_array.append(float(score))
    
    scr_array = np.expand_dims(scores_array, axis=1)
    plot_scores(scr_array, 'Plot Scores')

