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
* learning rate changed from 1e-4 to 1e-3 (seems to have sped up the learning)
* the program now creates and updates plots of running average reward and running average wins over the last 100 episodes
* in render mode, the program now displays the game using matplotlib's imshow() function, allowing it to be rendered on WSL2 using XServer on Windows

## 2. How does it work?
The program uses [gymnasium](https://gymnasium.farama.org/index.html) which provides many reinforcement learning environments, such as Atari Pong-v0 in this case. There are 3 main environment functions:
* [step()](https://gymnasium.farama.org/api/env/#gymnasium.Env.step) - performs a single action provided by an agent. In this case, it moves the paddle up/down or does nothing.
* [reset()](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset) - resets the environment to an initial state - ball and paddles in the middle. Called in the very beginning and then after every episode (an episode is one game of 21, e.g. 21-18).
* [render()](https://gymnasium.farama.org/api/env/#gymnasium.Env.render) - returns a single frame representing the current state of the environment. I'm using the "rgb_array" mode so that I can later display it using matplotlib's imshow() function. You might as well try the "human" mode, it didn't work on my WSL2.

The [action space](https://gymnasium.farama.org/environments/atari/pong/#actions) consists of 6 possible moves but I only use 3 of them: 0 (NOOP - do nothing), 1 (LEFT - down), 2 (UP - right).

The [observation space](https://gymnasium.farama.org/environments/atari/pong/#observations) is a numpy array of shape (210, 160, 3) - 210x160 px rgb (0-255) Pong frame.

change he to they/it?

The game consists of episodes - games of 21. Each episode is made up of many points and each point is made up of tens or even hundreds of individual actions (up/down/stay). The ultimate goal is winning a game (an episode) with the hardcoded computer opponent. We can achieve that by teaching our agent how to score points. 

After each point, the agent receives a reward that tells him how good his actions were. He gets a +1/-1 reward for the final move that led to winning/losing and a 0 reward for the rest of his actions. These rewards are then stored. After `batch_size` episodes, they get discounted. What that means, is that for each reward in a sequence, we assign a new reward, based on how far it is from the +1/-1 reward that led to scoring/losing a point. For example, if we set the `discount_factor` to 0.9 and the reward sequence is \[0, 0, 0, -1], we get \[-0.729 , -0.81, -0.9, -1] after discounting them (remember that each reward has an action associated with it - DOPISAĆ COŚ O TYM 1ST BACKPROP STEP BO MOŻE NIEJASNE TO JEST). 

rewards, discounting

encouraging an action

loss function

The flow:
* set everything up and obtain the initial observation
* enter the main loop
    * render the current frame
    * preprocess the observation (crop the unnecessary frame contents, downsample, set background to 0, set paddles and the ball to 1)
    * set the input to the neural network to be the difference of two last frames in order to capture motion
    * perform the forward pass and get the probabilities of performing each action
    * sample an action from the returned probabilities
    * compute the cross entropy loss derivative (??????????????????????????????????????????)
    * ?
    * perform the sampled action, get a new observation and the reward
    * if an episode has ended:
        * discount the rewards (??? explain how it works)
        * ?
        * modulate the gradient with the advantage (the discounted rewards??)
        * ?
        * perform the rest of the backward pass
        * update the parameters

sign of parameter grad update

## 3. Forward pass maths:

## 4. Backpropagation maths:

## 5. Performance and results:
<p align="center">
   <img src="https://github.com/Blato122/pong-rl-gym/blob/main/3plot.png" alt="Plot showing the performance of a neural network - running average reward and wins over the last 100 episodes" width=80% height=80%>
   <img src="https://github.com/Blato122/pong-rl-gym/blob/main/pong-gif-15sec.gif" alt="Gif showcasing two episodes of a Pong game, after 6700 episodes of training" />
</p>

## 6. Sources and useful links:
1. Andrej Karpathy Pong policy gradient blog post - https://karpathy.github.io/2016/05/31/rl/
2. Andrej Karpathy Pong policy gradient lecture - https://www.youtube.com/watch?v=tqrcjHuNdmQ&t=1870s
3. Andrej Karpathy Pong policy gradient gist - https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
