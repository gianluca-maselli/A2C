import torch 
from collections import deque
import gym

from A2C_model import A2C
from train_test import train, test
from show_plot import *
from tqdm import tqdm

if __name__ == '__main__':

    useGPU = 0
    if torch.cuda.is_available():
        dev = "cuda:0"
        useGPU = 1
    else: 
        dev = "cpu" 
        useGPU = 0
    
    device = torch.device(dev) 

    print('Device: ', device)
    
    use_pre_trained = True
    save_plot_scores = True
    
    
    #generate the environment
    env_name = "PongNoFrameskip-v4"
    #env_name = "PongDeterministic-v4"
    env_t = gym.make(env_name)
    #get the dimension of the env 
    space = env_t.observation_space.shape
    print('Space dim: ', space)
    #get the available actions
    actions = env_t.action_space.n
    print('n. of actions: \n', actions)
    actions_name = env_t.unwrapped.get_action_meanings()
    print('Available actions: \n', actions_name)
    
    if use_pre_trained == False:
        # --- SERIALIZED ENVS --- #
        #create envs where doing experience
        n_envs = 16
        envs = [gym.make(env_name) for _ in range(n_envs)]
        
        # ---- MODEL ---- #
        n_frames = 4
        #conv net dim
        hidden_dim1 = 32
        kernel_size1 = 8
        stride1 = 4
        hidden_dim2 = 64
        kernel_size2 = 4
        stride2 = 2
        hidden_dim3 = 64
        kernel_size3 = 3
        stride3 = 1
        #fully_connected dims
        fc1 = 512
        fc2 = actions
        out_actor_dim = actions
        out_critic_dim = 1
        
        a2c = A2C(input_shape=n_frames, layer1=hidden_dim1, kernel_size1=kernel_size1, stride1=stride1, layer2=hidden_dim2, 
            kernel_size2=kernel_size2, stride2=stride2, layer3=hidden_dim3, kernel_size3=kernel_size3, stride3=stride3, 
            fc1_dim=fc1, out_actor_dim=out_actor_dim, out_critic_dim=out_critic_dim).to(device)
        #loss function 
        learning_rate = 0.0001
        alpha = 0.99
        optimizer = torch.optim.Adam(a2c.parameters(), lr=learning_rate)
        
        # ---- TRAIN ---- #
        n_updates = int(4e10)
        queue = deque(maxlen=4)
        gamma = 0.99
        entropy_coef = 0.01
        value_coeff= 0.5 
        rollout_size = 5 
        #file to save scores
        file_scores = 'episodes_score.txt'
        
        trained_a2c, history_of_rewards, a2c_losses, actor_losses, critic_losses, entropy_list = train(n_updates=n_updates, rollout_size = rollout_size, n_envs=n_envs, envs=envs, queue=queue, model=a2c, 
                                                                                                    gamma=gamma, value_coeff=value_coeff, opt=optimizer,entropy_coef=entropy_coef, n_frames=n_frames, device=device, actions_name=actions_name, file=file_scores)

        if save_plot_scores:
            file_name = './episodes_score.txt'
            plot_avg_scores(file_name)
            
            

    print('load the model...')
    trained_model = torch.load('./model.pt', map_location=torch.device(device))
    # ----- TEST ----- # 
    print('\n')
    print(' ----- TEST PHASE ----- ')
    print('\n')

    max_games = 5
    t_queue = deque([np.zeros((84, 84), dtype=np.float32)] * 4, maxlen=4)
    scores = []
    test_render_interval = 5

    for play_i in tqdm(range(1,max_games+1)):
        score = test(env=env_t, queue=t_queue, n_frames=4, model=trained_model, device=device, play_i=play_i, render_interval=test_render_interval, actions_name=actions_name)
        scores.append(score)
        print(f' Game {play_i}, Score: {score}')

    print(f'Best score: {max(scores)}')
        
    

