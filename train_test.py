
from utilis import *
from A2C_utils import *
import torch.nn.functional as F

def train(n_updates, rollout_size, n_envs, envs, queue, model, gamma, value_coeff, opt, entropy_coef, n_frames, device, actions_name, file):
    model.train()
    #keep the rewards
    history_of_rewards = []
    a2c_losses = []
    actor_losses = []
    critic_losses = []
    entropy_list = []
    images = []
    steps = []
    mean_reward = 18.0
    games = 0

    #create an array where storing all the initial states 
    #reset envs and store all the initial states in the init_states list
    current_states = []
    #dictionary to keep track of the queues for each env
    list_queues = {}
    dict_rewards = {}
    dict_episode_rewards = {}
    env_count = 1
    #termination of the run
    finished = 0

    #initialize and reset envs
    for env in envs:
        in_state_i = env.reset() # Reset environments
        #initialize a queue for each env, preprocess each frame and obtain a vecotr of 84,84,4
        frame_queue = initialize_queue(queue, n_frames, in_state_i, env, actions_name)
        list_queues['env'+str(env_count)] = frame_queue
        dict_rewards['env'+str(env_count)] = 0
        dict_episode_rewards['env'+str(env_count)] = []
        #stack the frames together
        input_frames = stack_frames(frame_queue)
        current_state = input_frames
        current_states.append(current_state)
        env_count+=1

    update_i = 0
    while update_i < n_updates:
        # ----- ROLLOUT STEP ----- #    
        steps, games, images, finished, current_states  = rollout(rollout_size=rollout_size, envs=envs, model=model, steps_array=steps, current_states=current_states , 
                    games=games, images=images, dict_rewards=dict_rewards, dict_episode_rewards=dict_episode_rewards, list_queues=list_queues, 
                    mean_reward=mean_reward, history_of_rewards=history_of_rewards, n_envs=n_envs, queue=queue, n_frames=n_frames, device=device, actions_name=actions_name, finished=finished, file=file)

        if finished == 0:
            #compute expected returns
            actions, policies, values, expected_returns, advantages = compute_returns(steps,gamma, n_envs=n_envs, lambd=1.0, device=device)
            #update parameters
            a2c_loss, policy_loss, value_loss, entropy_loss = update_parameters(policy=policies, values=values, actions=actions, returns=expected_returns, advantages=advantages, value_coef=value_coeff, entropy_coef=entropy_coef, opt=opt, model=model, device=device)

            #empty the lists after gradient computation for the next rollout
            steps = []

            #print the current results after certain steps
            if update_i % 20 == 0:
                print(f'Update: {update_i} \n Policy_Loss: {policy_loss.item()} \n Value_Loss: {value_loss.item()} \n A2C loss: {a2c_loss.item()} \n Entropy: {entropy_loss}')

        
            #save losses 
            a2c_losses.append(a2c_loss.detach().item())
            actor_losses.append(policy_loss.detach().item())
            critic_losses.append(value_loss.detach().item())
            entropy_list.append(entropy_loss.detach().item())
            update_i +=1

        if finished==1:
            #plot losses
            plot([np.array(actor_losses),np.array(critic_losses),np.array(entropy_list), np.array(a2c_losses)], ['Plot Policy Loss','Plot Critic Loss','Plot Entropy Loss', 'Plot A2C Loss'])
            break

    return model, history_of_rewards, a2c_losses, actor_losses, critic_losses,entropy_list


# ----- TEST ------ #

def test(env, queue, n_frames, model, device, play_i, render_interval, actions_name):
    
    model.eval()
    init_state = env.reset()
    frame_queue =  initialize_queue(queue, n_frames, init_state, env, actions_name)
    #stack the frames together
    input_frames = stack_frames(frame_queue)
    current_state = np.expand_dims(input_frames,0)
    tot_score = 0
    done = False
    
    #continue till the game is in progress
    while(done==False):
        current_state = torch.from_numpy(current_state).permute(0,3,1,2).to(device)
        #compute the Q-values for this state given by the CNN
        with torch.no_grad():
            logits_v, value_v = model(current_state) 
        #chose the action with the highest q-val
        probs = F.softmax(logits_v, dim=-1)
        action = probs.max(1, keepdim=True)[1].numpy()
        next_state, reward, done, _ , _ = skip_frames(action[0, 0],env=env,skip_frame=4)
        #increment the score according to the reward obtained
        tot_score+=reward
        new_frame = frame_preprocessing(next_state)
        frame_queue.append(new_frame)
        next_frames = stack_frames(frame_queue)
        next_state = np.expand_dims(next_frames,0)
        current_state = next_state
        
        if play_i % render_interval == 0:
            env.render()
        env.close()

    return tot_score
