# A2C (Advantage Actor-Critic)
The folder contains the files needed to run the synchronous version of Actor-Critic [Mnih et al](https://proceedings.mlr.press/v48/mniha16.html?ref=.) with the PongNoFrameSkip-v4 gym environment. Similar to the already developed [DQN](https://github.com/gianluca-maselli/DQN_Atari) present in our repository we used the same environment to replicate the steps performed in the original Atari2600 game. For this reason, the preprocessing functions are almost the same. The difference with many other versions of the same algorithm is that we decide to implement the necessary functions to run the A2C over multiple environments from scratch and without the use of the Gym Library. 

# Usage
Running A2C with PongNoFrameSkip-v4 env is straightforward by launching the command
```
python main.py
```
This will train the agent on 16 serialized PongNoFrameSkip-v4 environments. 

Examples of Plots of both losses and scores average are reported below:

Losses             |  Scores AVG
:-------------------------:|:-------------------------:
![](https://github.com/gianluca-maselli/A2C/blob/main/Plots/plot_losses.png)  |  ![](https://github.com/gianluca-maselli/A2C/blob/main/Plots/plot_avg_scores.png)
