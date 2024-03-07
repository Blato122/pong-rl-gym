""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gymnasium as gym
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

"""
refactor that later!! functions etc
"""

# spr. różnicę z:
#-bias
#-more hidden layers
#-different lr?
#-CUDA!!!

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
torch.manual_seed(42)

"""
lr to 1e-3
crop photo better!!!
"""

# hyperparameters
N_IN = 80 * 80
N_OUT = 3
N_HIDDEN = 200
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False

# model = nn.Sequential(
#   nn.Linear(N_IN, N_HIDDEN, bias=False),
#   F.relu(),
#   nn.Linear(N_HIDDEN, N_OUT, bias=False),
#   F.softmax()
# )


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(N_IN, N_HIDDEN)
        self.lin2 = nn.Linear(N_HIDDEN, N_OUT)
        # dropout? or sth
        
        self.log_probs = []
        self.rewards_pre_discount = []
    
    def forward(self, x):
        x = self.lin1(x)
        h = F.relu(x)
        logits = self.lin2(h)
        probs = F.softmax(logits) # check if dim ok!
        return probs
    
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
model = torch.load('3save-torch.pt') if resume else Policy()
model.to(device)
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

# remove numpy later! only pytorch maybe

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[34:194] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return torch.tensor(I.astype(np.float32).ravel()) # CLEANER!!!!!!!!!!!!!!!!!!!

"""
add eps to discount rewards division by std!!!
"""

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  print("rewards shape:", r.shape)
  discounted_r = torch.zeros_like(r)
  running_add = 0
  for t in reversed(range(r.shape[0])):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  discounted_r -= torch.mean(discounted_r)
  discounted_r /= torch.std(discounted_r)
  return discounted_r

env = gym.make("Pong-v0")#, render_mode="human")
observation = env.reset()
prev_x = None # used in computing the difference frame
running_mean = None
running_wins = None
reward_sum = 0
episode_number = 0

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle(f'learning rate={learning_rate}, n_hidden={N_HIDDEN}')
ax1.set_xlabel("Episode number")
ax1.set_ylabel("Running reward average of 100 episodes")
ax1.grid(True)
ax2.set_xlabel("Episode number")
ax2.set_ylabel("Running win average of 100 episodes")
ax2.grid(True)

plot_running_rewards = []
plot_running_wins = []

while True:
  if render: env.render()

  # preprocess the observation, set input to network to be difference image (to capture motion)
  if len(observation) == 2: observation = observation[0]
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else torch.zeros(N_IN)
  prev_x = cur_x
  x.to(device) # HUH???????????

  # 3 actions: up/down or stay in place
  NOOP, RIGHT, LEFT = 0, 2, 3

  # forward the policy network and sample an action from the returned probability
  aprob = model(x)
  # action = np.random.choice(range(len(aprob)), p=aprob)
#   action = random.choices(range(len(aprob)), weights=aprob, k=1)
  # <=> action = m.sample()
  action = torch.multinomial(aprob, num_samples=1)
  model.log_probs.append(aprob[action].log())
  action.to(device) # HUH???????
  # imo cleaner and more intuitive than the Categorical way

  # cross entropy loss derivative
#   y = np.zeros_like(aprob)
#   y[action] = 1

#   dlogps.append(aprob - y) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
  # dlogps.append(aprob - y) # + softmax jednak? i backprop potem (ale to już chyba jest backprop cross entropy)
  # dlogps shape - list of 1234 or so elements
  # aprob shape - 3

  # step the environment and get new measurements
  if action[0] == 0: a = NOOP
  elif action[0] == 1: a = RIGHT
  elif action[0] == 2: a = LEFT

  observation, reward, done, _, _ = env.step(a)
  reward_sum += reward

  model.rewards_pre_discount.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
  
  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
    
  if done: # an episode finished
    episode_number += 1

    # compute the discounted reward backwards through time
    # vstack?!?! ++ its different from the np version!
    #calc the shapes again to see why its necessary
    discounted_epr = discount_rewards(np.vstack(model.rewards_pre_discount))

    # print("PG magic:", epdlogp.shape, discounted_epr.shape, "\n")
    #epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    
    optimizer.zero_grad()
    policy_loss = -np.vstack(model.log_probs) * discounted_epr
    policy_loss = torch.tensor(policy_loss).sum() 
    policy_loss.to(device)
    policy_loss.backward()
    optimizer.step()

    # https://stackoverflow.com/questions/30561194/what-is-the-difference-between-del-a-and-a-when-i-want-to-empty-a-list-c
    model.rewards_pre_discount = []
    model.log_probs = []

    # boring book-keeping
    running_mean = reward_sum if running_mean is None else running_mean * 0.99 + reward_sum * 0.01
    running_wins = (reward_sum > 0)if running_wins is None else running_wins * 0.99 + (reward_sum > 0) * 0.01
    print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_mean) )
    plot_running_rewards.append(running_mean)
    if episode_number % 100 == 0: 
        fig.clear(keep_observers=True) # ??
        ax1.plot(range(episode_number), plot_running_rewards)
        ax2.plot(range(episode_number), plot_running_wins)
        fig.savefig('3torchplot.png')
        torch.save(model, open('3torchsave.p', 'wb'))
    
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None