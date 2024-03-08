""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
# import numpy as np
import gymnasium as gym

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

# in Policy? as tensors(prob naah, some cat instead of append)?
log_probs = []
rewards_pre_discount = []

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(N_IN, N_HIDDEN)
        self.lin2 = nn.Linear(N_HIDDEN, N_OUT)
        # dropout? or sth
    
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

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[34:194] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return torch.tensor(I, dtype=torch.float32).ravel().to(device) # CLEANER!!!!!!!!!!!!!!!!!!!

"""
add eps to discount rewards division by std!!!
"""

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  print("rewards shape:", r.shape)
  discounted_r = torch.zeros_like(r).to(device)
  running_add = 0
  for t in reversed(range(r.shape[0])):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  discounted_r -= torch.mean(discounted_r).to(device)
  discounted_r /= torch.std(discounted_r).to(device)
  return discounted_r

env = gym.make("Pong-v0") if not render else gym.make("Pong-v0", render_mode="human")
observation = env.reset()
prev_x = None # used in computing the difference frame
running_mean = None
running_wins = None
reward_sum = 0
episode_number = 0

plot_running_rewards = []
plot_running_wins = []

while True:
  if render: env.render()

  # preprocess the observation, set input to network to be difference image (to capture motion)
  if len(observation) == 2: observation = observation[0]
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else torch.zeros(N_IN).to(device)
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
  log_probs.append(aprob[action].log())
  action.to(device) # HUH???????
  # imo cleaner and more intuitive than the Categorical way

  # step the environment and get new measurements
  if action[0] == 0: a = NOOP
  elif action[0] == 1: a = RIGHT
  elif action[0] == 2: a = LEFT

  observation, reward, done, _, _ = env.step(a)
  reward_sum += reward

  rewards_pre_discount.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
  
  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
    
  if done: # an episode finished
    episode_number += 1

    # compute the discounted reward backwards through time
    # vstack?!?! ++ its different from the np version!
    #calc the shapes again to see why its necessary
    # discounted_epr = discount_rewards(torch.tensor(np.vstack(rewards_pre_discount)).to(device))
    discounted_epr = discount_rewards(torch.tensor(rewards_pre_discount).unsqueeze(1).to(device))

    # print("PG magic:", epdlogp.shape, discounted_epr.shape, "\n")
    #epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    
    optimizer.zero_grad()
    # policy_loss = -np.vstack(model.log_probs) * discounted_epr
    policy_loss = -torch.tensor(log_probs).unsqueeze(1).to(device) * discounted_epr
    policy_loss = policy_loss.sum().to(device)
    policy_loss.to(device)
    policy_loss.backward()
    optimizer.step()

    # https://stackoverflow.com/questions/30561194/what-is-the-difference-between-del-a-and-a-when-i-want-to-empty-a-list-c
    rewards_pre_discount = []
    log_probs = []

    # boring book-keeping
    running_mean = reward_sum if running_mean is None else running_mean * 0.99 + reward_sum * 0.01
    running_wins = (reward_sum > 0)if running_wins is None else running_wins * 0.99 + (reward_sum > 0) * 0.01
    print('resetting env. episode reward total was %f. running mean: %f. running wins: %f' % (reward_sum, running_mean, running_wins) )
    plot_running_rewards.append(running_mean)
    plot_running_wins.append(running_wins)
    if episode_number % 100 == 0: 
        # fig like in 3actions
        torch.save(model, open('3torchsave.p', 'wb'))
    
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None