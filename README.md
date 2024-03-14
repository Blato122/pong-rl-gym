# Policy gradient pong

## 1. Introduction:
This program trains a reinforcement learning agent to play a game of Pong using the policy gradient algorithm. It is based on an excellent article and a lecture by Andrej Karpathy. /insert links here - gist, blog, yt/
My goal was to add some more functionalities to his code in order to understand the whole concept even better. The changes I introduced:
* added third possible action (do nothing) besides moving up/down
  * modified the output layer so that it consists of 3 neurons, instead of 1
  * changed the activation function in the output layer from sigmoid to softmax
  * modified the action sampling method (since now there are 3 probabilities instead of 1)
  * modified the loss function so that it works with 3 actions (one-hot encoding)
* added a second hidden layer
  * modified the forward and backward pass
* messed with the numpy array shapes so that everything works again after these changes
* now using the @ operator instead od np.dot()
* the program now creates and updates plots of running average reward and running average wins over the last 100 episodes
* in render mode, the program now displays the game using matplotlib's imshow() function, allowing it to be rendered on WSL2 using XServer on Windows

## 2. How does it work?
The program uses [gymnasium](https://gymnasium.farama.org/index.html) which provides many reinforcement learning environments, such as Atari Pongv0 in my case. There are 3 main environment functions:
* [step()](https://gymnasium.farama.org/api/env/#gymnasium.Env.step) - performs a single action provided by an agent. In this case, it moves the paddle up/down or does nothing.
* [reset()](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset) - resets the environment to an initial state - ball and paddles in the middle. Called in the very beginning and then after every episode (an episode is one game of 21, e.g. 21-18).
* [render()](https://gymnasium.farama.org/api/env/#gymnasium.Env.render) - returns a single frame representing the current state of the environment. I'm using the "rgb_array" mode so that I can later display it using matplotlib's imshow() function. You might as well try the "human" mode, it didn't work on my WSL2.

The [action space](https://gymnasium.farama.org/environments/atari/pong/#actions) consists of 6 possible moves but I only use 3 of them: 0 (NOOP - do nothing), 1 (LEFT - down), 2 (UP - right).

The [observation space](https://gymnasium.farama.org/environments/atari/pong/#observations) is a numpy array of shape (210, 160, 3) - 210x160 px rgb (0-255) Pong frame.

First, the program resets the environment to obtain the initial observation

prepro
difference of frames

## 3. Forward pass maths:

## 4. Backpropagation maths:

## 5. Performance and results:

## 6. Sources and useful links:
1. Andrej Karpathy Pong policy gradient blog post - https://karpathy.github.io/2016/05/31/rl/
2. Andrej Karpathy Pong policy gradient lecture - https://www.youtube.com/watch?v=tqrcjHuNdmQ&t=1870s
3. Andrej Karpathy Pong policy gradient gist - https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
