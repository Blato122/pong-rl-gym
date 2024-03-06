""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gymnasium as gym
import random

"""
change W2 - now (H,3)
change 1st backprop step
change backward and forward passes???

DO TETRISA TEZ INFO - dodac na koncu forward pass jakiegos softmaxa czy cos XD
jak ja chcialem niby bez tego to zrobic w ogole

refactor that later!! functions etc
"""

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
if resume:
  model = pickle.load(open('3save.p', 'rb'))
else:
  model = {}
  # - - - - - -
  """ CHANGE 1 - H,D TO D,H"""
  # - - - - - -
  model['W1'] = np.random.randn(D,H) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H,3) / np.sqrt(H)
  
  # - - - - - -
  """ CHANGE 4(?) - sgd instead of this rmsprop thing - NAAH"""
  # - - - - - -
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

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
  I[I != 0] = 255 # everything else (paddles, ball) just set to 1
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
  # - - - - - -
  """ CHANGE 2 - @ NOTATION """
  # - - - - - -
  # matmul docs:
  # If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions.
  # After matrix multiplication the prepended 1 is removed.
  # If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions.
  # After matrix multiplication the appended 1 is removed.

  h = x @ model['W1'] # (6400,) @ (6400, 300) ---> (1, 6400) @ (6400, 300) ---> (1, 300) ---> (300,) 
  # tested - ok

  h[h<0] = 0 # ReLU nonlinearity
  logits = h @ model['W2'] # (300,) @ (300, 3) ---> (1, 300) @ (300, 3) ---> (1, 3) ---> (3,)
  # tested - ok

  # p = sigmoid(logits)
  p = softmax(logits)
  return p, h # return probability of taking each of the 3 actions and the hidden states

def policy_backward(eph, epdlogp):
    # - - - - - -
  """ CHANGE 2 - SHAPE COMPATIBLE WITH the model + shapes like in the notebook """
  # - - - - - -
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = eph.T @ epdlogp 
  # dW2 = np.dot(eph.T, epdlogp).ravel()
  print("dW2, eph.T, epdlogp", dW2.shape, eph.T.shape, epdlogp.shape)
  # print("pre dh: epdlogp, modelW2", epdlogp.shape, model['W2'].shape)
  dh = epdlogp @ model["W2"].T 
  # dh = np.outer(epdlogp, model['W2'])
  print("dh, epdlogp, modelW2.T", dh.shape, epdlogp.shape, model['W2'].T.shape)
  dh[eph <= 0] = 0 # backprop relu
  dW1 = epx.T @ dh #dW1 = np.dot(dh.T, epx)
  print("dW1, dh.T, epx", dW1.shape, dh.T.shape, epx.shape)
  return {'W1':dW1, 'W2':dW2}

env = gym.make("Pong-v0")#, render_mode="human")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_mean = None
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

  # 3 actions now! up/down or stay in place
  # actions = {
  #   0: 0, # noop
  #   1: 2, # right
  #   2: 3, # left
  # }
  NOOP, RIGHT, LEFT = 0, 2, 3

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  # print(aprob, aprob.sum())
  # action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
  # action = np.random.choice(range(len(aprob)), p=aprob)
  # wei = softmax(aprob)
  action = random.choices(range(len(aprob)), weights=aprob, k=1)
  # print(action)
  # print(aprob, action[0])

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  # y = 1 if action == 2 else 0 # a "fake label"
  # cross entropy loss derivative
  y = np.zeros_like(aprob)
  y[action] = 1

  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
  # dlogps.append(aprob - y) # + softmax jednak? i backprop potem (ale to już chyba jest backprop cross entropy)
  # dlogps shape - list of 1234 or so elements
  # aprob shape - 3

  # step the environment and get new measurements
  if action[0] == 0: a = NOOP
  elif action[0] == 1: a = RIGHT
  elif action[0] == 2: a = LEFT

  observation, reward, done, _, _ = env.step(a)
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

    print("PG magic:", epdlogp.shape, discounted_epr.shape, "\n")
    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    print("after PG magic:", epdlogp.shape, "\n")
    grad = policy_backward(eph, epdlogp)
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
    print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_mean) )
    if episode_number % 100 == 0: pickle.dump(model, open('3save.p', 'wb'))
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None