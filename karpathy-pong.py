""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gymnasium as gym

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
if resume:
  model = pickle.load(open('change1_save.p', 'rb'))
else:
  model = {}
  # - - - - - -
  """ CHANGE 1 - H,D TO D,H"""
  # - - - - - -
  model['W1'] = np.random.randn(D,H) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[34:194] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 255 # everything else (paddles, ball) just set to 1
  return I.astype(np.float32).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  print(r.shape)
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
  # - - - - - -
  """ CHANGE 2 - @ NOTATION """
  # - - - - - -

  # matmul docs, applies below: 
  # If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions.
  # After matrix multiplication the prepended 1 is removed.
  h = x @ model['W1'] # (6400,) @ (6400, 200) ---> (1, 6400) @ (6400, 200) ---> (1, 200) ---> (200,) 
  # tested - ok

  h[h<0] = 0 # ReLU nonlinearity

  # matmul docs, both apply below:
  # If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions.
  # After matrix multiplication the prepended 1 is removed.
  # If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions.
  # After matrix multiplication the appended 1 is removed.
  logits = h @ model['W2'] # (200,) @ (200,) ---> (1, 200) @ (200, 1) ---> (1, 1) ---> (), scalar
  # tested - ok

  p = sigmoid(logits)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

env = gym.make("Pong-v0")#, render_mode="human")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

while True:
  if render: env.render()

  # preprocess the observation, set input to network to be difference image
  # print(len(observation))
  # print(observation, observation[0].shape, len(observation[0]))
  if len(observation) == 2: observation = observation[0]
  # print(observation.shape)
  cur_x = prepro(observation)
  # print(cur_x.shape)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if action == 2 else 0 # a "fake label"
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  observation, reward, done, _, _ = env.step(action)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
  
  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
    
  if done: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(eph, epdlogp)
    for k in model: 
      print(grad_buffer[k].shape)
      print(grad[k].shape)
      grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward) )
    if episode_number % 100 == 0: pickle.dump(model, open('change1_save.p', 'wb'))
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None