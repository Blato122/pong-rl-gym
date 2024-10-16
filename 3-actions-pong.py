import numpy as np
import pickle
import gymnasium as gym
import random
from matplotlib import pyplot as plt

# hyperparameters
H = 120 # number of hidden layer neurons in the 1st hidden layer
H2 = 60 # number of hidden layer neurons in the 2nd hidden layer
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-3 # changed from 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = True

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
if resume:
  model = pickle.load(open('11800ep-6.1reward-86.5win-save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(D,H) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H,H2) / np.sqrt(H)
  model['W3'] = np.random.randn(H2,3) / np.sqrt(H)
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def softmax(x):
  # x - list of len 3
  """Compute softmax values for each sets of scores in x."""
  # e_x = np.exp(x - np.max(x))
  # return e_x / e_x.sum()
  return np.exp(x) / np.sum(np.exp(x), axis=0)

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[34:194] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float32).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  print("rewards shape:", r.shape)
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(r.shape[0])):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  discounted_r -= np.mean(discounted_r)
  discounted_r /= np.std(discounted_r)
  return discounted_r

def policy_forward(x):
  """ forward pass, returning probabilities of taking each of the 3 actions and the hidden states (needed later) """
  # matmul docs:
  # If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions.
  # After matrix multiplication the prepended 1 is removed.
  # If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions.
  # After matrix multiplication the appended 1 is removed.

  h = x @ model['W1'] # (6400,) @ (6400, 200) ---> (1, 6400) @ (6400, 200) ---> (1, 200) ---> (200,) 
  h[h<0] = 0 # ReLU nonlinearity
  h2 = h @ model['W2'] # (200,) @ (200, 100) ---> (1, 200) @ (200, 100) ---> (1, 100) -> (100,)
  h2[h2<0] = 0 # ReLU nonlinearity
  logits = h2 @ model['W3'] # (100,) @ (100, 3) ---> (1, 100) @ (100, 3) ---> (1, 3) ---> (3,)
  p = softmax(logits)
  return p, h, h2

def policy_backward(ep_hs, ep_h2s, ep_dL_dlogits):
  """ backward pass (ep_hs and ep_h2s are arrays of intermediate hidden states) """
  dL_dW3 = ep_h2s.T @ ep_dL_dlogits # (100, 1234) @ (1234, 3) ---> (100, 3)
  dL_dh2 = ep_dL_dlogits @ model["W3"].T # (1234, 3) @ (3, 100) ---> (1234, 100)
  dL_dh2[ep_h2s <= 0] = 0 # dL_dh2: (1234, 100), ep_h2s: (1234, 100), OK
  dL_dW2 = ep_hs.T @ dL_dh2 # (200, 1234) @ (1234, 100) ---> (200, 100)       umm more or less, shapes are ok but idk
  # dL_dW3 = np.dot(ep_hs.T, ep_dL_dlogits).ravel()
  # print("dL_dW3, ep_hs.T, ep_dL_dlogits", dL_dW3.shape, ep_hs.T.shape, ep_dL_dlogits.shape)
  # print("pre dL_dh: ep_dL_dlogits, modelW2", ep_dL_dlogits.shape, model['W3'].shape)
  dL_dh = dL_dh2 @ model["W2"].T # (1234, 100) @ (100, 200) ---> (1234, 200)     shapes ok but idk
  # dL_dh = np.outer(ep_dL_dlogits, model['W3'])
  # print("dL_dh, ep_dL_dlogits, modelW2.T", dL_dh.shape, ep_dL_dlogits.shape, model['W3'].T.shape)
  dL_dh[ep_hs <= 0] = 0 # dL_dh: (1234, 200), ep_hs: (1234, 200), OK
  dL_dW1 = ep_xs.T @ dL_dh # (6400, 1234) @ (1234, 200) ---> (6400, 200)
  #dL_dW1 = np.dot(dL_dh.T, ep_xs)
  # print("dL_dW1, dL_dh.T, ep_xs", dL_dW1.shape, dL_dh.T.shape, ep_xs.shape)
  return {'W1':dL_dW1, 'W2':dL_dW2, 'W3':dL_dW3}

env = gym.make("Pong-v0") if not render else gym.make("Pong-v0", render_mode="rgb_array")
# env.metadata["render_fps"] = 60
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,h2s,dL_dlogits,rewards = [],[],[],[],[]
running_mean = None
running_wins = None
reward_sum = 0
episode_number = 0

plot_running_rewards = []
plot_running_wins = []
img = None

while True:
  if render: 
    if img is None:
      img = plt.imshow(env.render())
    elif plt.fignum_exists(1): # sys exit if doesn't exist?
      img.set_data(env.render())
    plt.pause(0.001)

  # preprocess the observation, set input to network to be difference image
  if len(observation) == 2: observation = observation[0]
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # 3 actions now! up/down or stay in place
  NOOP, RIGHT, LEFT = 0, 2, 3

  # forward the policy network and sample an action from the returned probability
  action_probs, h, h2 = policy_forward(x)
  # action = np.random.choice(range(len(action_probs)), p=action_probs)
  action = random.choices(range(len(action_probs)), weights=action_probs, k=1)

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  h2s.append(h2) # hidden state

  # cross entropy loss derivative
  y = np.zeros_like(action_probs)
  y[action] = 1

  # https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/
  dL_dlogits.append(y - action_probs) 
  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
  # dL_dlogits shape - list of 1234 or so elements, action_probs shape - 3

  """
  the gradient that we calculate is a value not a function!
  we calculate the gradient of a loss function for a given
  loss value.
  it tells us the direction of the fastest ascent while standing
  in a given point.
  """

  # step the environment and get new measurements
  if action[0] == 0: a = NOOP
  elif action[0] == 1: a = RIGHT
  elif action[0] == 2: a = LEFT

  observation, reward, done, _, _ = env.step(a)
  reward_sum += reward

  rewards.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
  
  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
    
  if done: # an episode (game of 21) finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    ep_xs = np.vstack(xs)
    ep_hs = np.vstack(hs)
    ep_h2s = np.stack(h2s)
    ep_dL_dlogits = np.vstack(dL_dlogits)
    ep_rewards = np.vstack(rewards)
    xs,hs,h2s,dL_dlogits,rewards = [],[],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_ep_rewards = discount_rewards(ep_rewards)

    print("PG magic:", ep_dL_dlogits.shape, discounted_ep_rewards.shape, "\n")
    ep_dL_dlogits *= discounted_ep_rewards # modulate the gradient with advantage (the discounted rewards)
    print("after PG magic:", ep_dL_dlogits.shape, "\n")
    grad = policy_backward(ep_hs, ep_h2s, ep_dL_dlogits)
    # for k in model: 
    #   print(grad_buffer[k].shape)
    #   print(grad[k].shape)
    for k in model: 
      grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
      
    # if episode_number % batch_size == 0:
    #       print("update")
    #       for k, v in model.items():
    #           model[k] += learning_rate * grad_buffer[k] # update the parameters
    #           grad_buffer[k] = np.zeros_like(v) # reset the gradients

    # boring book-keeping
    running_mean = reward_sum if running_mean is None else running_mean * 0.99 + reward_sum * 0.01
    running_wins = (reward_sum > 0) if running_wins is None else running_wins * 0.99 + (reward_sum > 0) * 0.01
    print('resetting env. episode reward total was %f. running mean: %f. running wins: %f' % (reward_sum, running_mean, running_wins * 100) )
    plot_running_rewards.append(running_mean)
    plot_running_wins.append(running_wins * 100)

    if episode_number % 100 == 0: # save the model and update the plots
      # hmm maybe add eval mode?
      if not render: # since render is basically eval mode, I don't want to create a training progress plot
        fig, ax1 = plt.subplots(figsize=(8, 6))

        # Plot the line with normal scale on primary y-axis
        p1, = ax1.plot(range(episode_number), plot_running_rewards, label='Running reward average')

        # Create a secondary y-axis for percentage data
        ax2 = ax1.twinx()

        # Plot the line with percentage on secondary y-axis (set limits from 0 to 100)
        p2, = ax2.plot(range(episode_number), plot_running_wins, label='Percentage of wins', color='red')
        ax2.set_ylim(0, 100)  # Set limits for percentage axis

        # Set labels and title
        ax1.set_xlabel('Episode number')
        ax1.set_ylabel('Running reward average over past 100 episodes')
        ax2.set_ylabel('Percentage of wins over past 100 episodes')
        plt.title(f'learning_rate={learning_rate}, n_hidden={H, H2}')

        # Show legend
        plt.legend(handles=[p1, p2])
        # Adjust layout to avoid overlapping labels
        plt.tight_layout()
        plt.grid()
        fig.savefig('plot.png')

      pickle.dump(model, open('save.p', 'wb'))
    
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None