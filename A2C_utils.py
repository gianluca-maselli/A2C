
import torch
from utilis import *

#used to gather experience to compute the gradients 
def rollout(rollout_size, envs, model, steps_array, current_states, games, images, dict_rewards, dict_episode_rewards, list_queues, mean_reward, history_of_rewards, n_envs, queue, n_frames, device, actions_name, finished, file):
    
    for i in range(rollout_size):
        if finished == 1: break
        #compute policy and values for each state, resulting shape (10,6) and (10,1)
        current_states = torch.tensor(np.array(current_states)).permute(0,3,1,2).to(device)
        logits_v, value_v = model(current_states) 
        #compute actions probs
        probs_v = torch.nn.functional.softmax(logits_v, dim=-1)
        action_pd = torch.distributions.Categorical(probs=probs_v) 
        actions_ = action_pd.sample().cpu().data.numpy()

        next_states = [] #list used to create the list of stacked frames for the next step
        rews = [] #rewards collected for each env for the current step
        dones = [] #dones collected for each env for the current step
        #perform corresponding action in each env

        # interact
        for j, env_j in enumerate(envs):
            #loop in each env and perform the associated action 
            next_state_j, reward_j, done_j, _, info_j = skip_frames(actions_[j],env=env_j, skip_frame=4)
            dict_rewards['env'+str(j+1)] += reward_j
            rews.append(reward_j) #for each env we add its current rewards
            dones.append(done_j) #for each env we add its current status
      
            #if a certain episode is terminated we need to reset i
            if done_j:
                #store the tot reward for the env
                dict_episode_rewards['env'+str(j+1)].append(dict_rewards['env'+str(j+1)])
                #reset env
                init_state_j = env_j.reset()
                #empty the queue
                frame_queue_j = initialize_queue(queue, n_frames, init_state_j, env_j, actions_name)
                #substitute the empy queue to the old one
                list_queues['env'+str(j+1)] = frame_queue_j
                #empty the rewards
                dict_rewards['env'+str(j+1)] = 0

                #compute the avg sum of rewards for episode
                lists_all_rewards = list(dict_episode_rewards.values())
                #check if all the list have length > 1, then compute the avg score for the oldest episode
                score = 0
                if all(len(l) >= 1 for l in lists_all_rewards):
                    games +=1
                    print('-----------------------------------')
                    print('Current envs rewards: ', lists_all_rewards)
                    #sum all the elements at the first position
                    #used to delete element in the correspective list
                    for k in range(1,n_envs+1):
                        score+= dict_episode_rewards['env'+str(k)].pop(0)

                    #compute avg
                    avg_score = score/n_envs
                    print(f'Game {games}: {avg_score}')
                    #add avg for episode
                    history_of_rewards.append(avg_score)
                    print('--------------------------------')
                    
                    if len(history_of_rewards) > 100:
                        avg = np.mean(np.array(history_of_rewards[-100:]))
                        print('----------------------------')
                        print('AVG last 100 scores: ',  avg)
                        print('----------------------------')
                        with open(file, 'a') as f:
                            str_write = str(avg)+','
                            f.write(str_write)
                        if avg >= mean_reward:
                            print('Save Model...')
                            torch.save(model,'./model.pt')
                            print("Solved with %d games!" % games)
                            finished = 1
                            break
                             

            #if not terminal state continue as usual
            else:
                next_frame = frame_preprocessing(next_state_j)
                list_queues['env'+str(j+1)].append(next_frame)
                frame_queue_j = list_queues['env'+str(j+1)]

            stacked_frames_j =  stack_frames(frame_queue_j)
            #for each env we create the unique frame used as next frame
            next_states.append(stacked_frames_j)

        #move to the next state
        current_states=next_states
        #append to the current step all the lists collected
        steps_array.append((torch.tensor(np.array(rews)).to(device), torch.tensor(dones).to(device), actions_, logits_v.to(device), value_v.to(device)))
  
    #compute the next value with the model
    with torch.no_grad():
        final_state = torch.tensor(np.array(next_states)).permute(0,3,1,2).to(device)
        _, next_value = model(final_state) #.to(device))
    #append the next value to the steps 
    steps_array.append((None, None, None, None, next_value))
    return steps_array, games, images, finished, current_states


#compute the expected returns using GAE
def compute_returns(steps, gamma, n_envs, lambd, device):
    #get the last value to perform bootstrapping
    _, _, _, _, last_values = steps[-1]
    returns = last_values
    advantages = torch.zeros(n_envs, 1).to(device)
    #print('return', returns.shape)
    out = [None] * (len(steps) - 1) 
    #loop in reverse mode excluding the last element (i.e. next value)
    for t in reversed(range(len(steps) - 1)):
        #extract the stored quantities in reverse order
        rewards, dones, actions, policies, values = steps[t]
        #next value becomes the value one step ahead
        _, _, _, _, next_values = steps[t + 1]
        #computation of normal returns
        returns = rewards.unsqueeze(1) + returns * gamma * torch.logical_not(dones).unsqueeze(1)
        #computation of advantages using GAE
        deltas = rewards.unsqueeze(1) + next_values * gamma * torch.logical_not(dones).unsqueeze(1) - values
        advantages = advantages * gamma * lambd *  torch.logical_not(dones).unsqueeze(1) + deltas
        #add elements from the last one to the first one, in this way we do not need to reverse log probs and values array
        out[t] = torch.tensor(actions), policies, values, returns, advantages

    return map(lambda x: torch.cat(x, 0), zip(*out))


#update parameters function
def update_parameters(policy, values, actions, returns, advantages, value_coef, entropy_coef, opt, model, device):
    #compute action probs
    probs = torch.nn.functional.softmax(policy, dim=-1)
    #compute log probs
    log_probs = torch.nn.functional.log_softmax(policy, dim=-1)
    #gather logprobs with respect the chosen actions
    action_log_probs = log_probs.gather(1, actions.unsqueeze(1).to(device)) #.squeeze()
    #policy loss
    policy_loss = (-action_log_probs * advantages.detach()).mean() 
    #crtic loss
    value_loss = torch.nn.functional.mse_loss(values, returns.float())
    #entropy loss
    entropy_loss = (probs * log_probs).sum(dim=1).mean()
    #overall loss
    a2c_loss = policy_loss + value_coef* value_loss + entropy_coef * entropy_loss 
    opt.zero_grad()
    a2c_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    opt.step()
    
    return a2c_loss, policy_loss, value_loss, entropy_loss
                
