from unityagents import UnityEnvironment
import numpy as np


# please write the path of Banana.exe here)
env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana_Windows_x86_64/Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]



from model import QNetwork
from bufer import Replay_Buffer
from Agent import Agent
agent = Agent(state_size = 37, action_size = 4, seed = 0)

def dqn_unity(num_episodes = 2000,  eps_start = 1, eps_decay=0.995, eps_end = 0.01):
    
    scores = [] # list of scores from each episode
    score_window = deque(maxlen = 100) # a deque of 100 episode scores to average
    eps = eps_start
    
    for i_episode in range(1,num_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        
        score = 0
        while True:
       
                   
            action = agent.select_act(state,eps)           # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state,action,reward,next_state,done)
            score += reward
            state = next_state
            
            if done:
                break
                
        scores.append(score)
        score_window.append(score)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(score_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(score_window)))
        if np.mean(score_window)>=13:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(score_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'Banana_saved_model.pth')
            break
            
    return scores


scores = dqn_unity()
