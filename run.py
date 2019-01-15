# run this file to see how my trained models work.


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


agent.qnetwork_local.load_state_dict(torch.load('Banana_saved_model.pth'))

eps = 0.
scores = []
for i in range(5):

    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    while True:
        action = agent.select_act(state,eps)           # select an action
               
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
    scores.append(score)
    #print("Score: {}".format(score))
print('Avg score:',np.mean(scores))