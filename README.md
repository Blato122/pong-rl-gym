# Policy gradient pong

## 1. Introduction:
This program trains a reinforcement learning agent to play a game of Pong using the policy gradient algorithm. It is based on an excellent article and a lecture by Andrej Karpathy.
My goal was to add some more functionalities to his code in order to understand the whole concept even better. The changes I introduced:
* added third possible action (do nothing) besides moving up/down
    * modified the output layer so that it consists of 3 neurons, instead of 1
    * changed the activation function in the output layer from sigmoid to softmax
    * modified the action sampling method (since now there are 3 probabilities instead of 1)
    * modified the loss function so that it works with 3 actions (one-hot encoding)
* added a second hidden layer
    * modified the forward and backward passes
* messed with the numpy array shapes so that everything works again after these changes
* now using the @ operator instead od np.dot()
* learning rate changed from 1e-4 to 1e-3 (seems to have sped up the learning)
* the program now creates and updates plots of running average reward and running average wins over the last 100 episodes
* in render mode, the program now displays the frames using matplotlib's imshow() function, allowing it to be rendered on WSL2 using XServer on Windows

## 2. How does it work?
The program uses [gymnasium](https://gymnasium.farama.org/index.html) which provides many reinforcement learning environments, such as Atari Pong-v0 in this case. There are 3 main environment functions:
* [step()](https://gymnasium.farama.org/api/env/#gymnasium.Env.step) - performs a single action provided by an agent. In this case, it moves the paddle up/down or does nothing.
* [reset()](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset) - resets the environment to an initial state - ball and paddles in the middle. Called in the very beginning and then after every episode (an episode is one game of 21, e.g. 21-18).
* [render()](https://gymnasium.farama.org/api/env/#gymnasium.Env.render) - returns a single frame representing the current state of the environment. I'm using the "rgb_array" mode so that I can later display it using matplotlib's imshow() function. You might as well try the "human" mode, it didn't work on my WSL2.

The [action space](https://gymnasium.farama.org/environments/atari/pong/#actions) consists of 6 possible moves but I only use 3 of them: 0 (NOOP - do nothing), 1 (LEFT - down), 2 (UP - right).

The [observation space](https://gymnasium.farama.org/environments/atari/pong/#observations) is a numpy array of shape (210, 160, 3) - 210x160 px rgb (0-255) Pong frame.

---

Each episode - a game of 21 - is made up of many points and each point is made up of tens or even hundreds of individual actions (up/down/stay). The ultimate goal is to win a game (an episode) with the hardcoded computer opponent. We can achieve that by teaching our agent how to score points. 

After each point, the agent receives a reward that tells him how good his actions were. It gets a +1/-1 reward for the final move that led to winning/losing and a 0 reward for the rest of its actions. These rewards are then stored. After `batch_size` episodes, they get discounted. What that means, is that for each reward in a sequence, we assign a new reward, based on how far it is from the +1/-1 reward that led to scoring/losing a point. For example, if we set the `discount_factor` to 0.9 and the reward sequence is \[0, +1, 0, 0, 0, -1], we get \[+0.9, +1, -0.729 , -0.81, -0.9, -1] after discounting them (remember that each reward has an action associated with it). We do that because it usually doesn't feel right to blame all the actions equally - it is often the latest actions that cause the particular outcome. Plus, we need to assign some reward to the actions that had a reward of 0 (remember - we assigned a reward of 0 to all of the actions that didn't directly lead to scoring/losing a point) before because they almost certainly weren't neutral for the outcome.

---

Okay, but what about the loss function, its gradient and the parameter updates? How do these rewards encourage the agent to make correct decisions? In each iteration of the algorithm, before performing an action and receiving a reward, we perform a forward pass. This forward pass gives us the probabilities of performing each one of the 3 actions. We then sample an action from these probabilities and calculate the cross entropy loss. 

umm... The outcome probabilities (all 3 of them) are simply a function of the MODEL PARAMETERS! so we can fill in the gradient for only one of them and decide whether we want to encourage it or discourage it by choosing a sign!!! ten fill in the gradient to po prostu chyba chodzi o to, że w cross entropy jego indeks dajemy jako 1, a pozostałe jako 0. wtedy loss zależy tylko od tej podjętej akcji, nie składają się na niego inne akcje i licząc gradient lossu, możemy zaktualizować parametry tak, aby tylko ta jedna akcja została dis/encouraged

Minimizing the cross entropy loss, ∑ilogp(yi∣xi), is equivalent to minimizing the negative log likelihood loss or maximizing the log likelihood loss. Log likelihood is, in turn, basically an equivalent of the product of all pre-log probabilities. Since we want the probabilties (of choosing the right action given some observation) to be as high as possible, the loss should be as close to 1 (perfect) as possible - we have to maximize it (because the probabilities are in range \[0,1], the likelihood is often a very small number). But because it is more convenient to use the log probabilities, we'll stick to maximizing the log likelihood loss.

There is one problem, though. Since we're not doing supervised learning and we have no correct labels, this wouldn't work. We cannot encourage all the actions that we took because some of them might be incorrect. This is why treat the sampled action as a fake label that might not necessarily be correct. Now, the **most important part** - we multiply the loss of an action by a reward received for performing it. The actions that received a negative reward will now actually be minimizing the negative log likelihood loss making them less likely in the future!

Does the gradient of the loss function change when we modulate it with these numbers (called advantage)? To my understanding, no, because it is simply a scalar - the loss is not a function of advantage. That's why we can first calculate the derivative of the pre-advantage loss and then, after discounting the rewards, modulate it with them. 

The flow:
* set everything up and obtain the initial observation
* enter the main loop
    * render the current frame
    * preprocess the observation (crop the unnecessary frame contents, downsample, set background to 0, set paddles and the ball to 1)
    * set the input to the neural network to be the difference of two last frames in order to capture motion
    * perform the forward pass and get the probabilities of performing each action
    * sample an action from the returned probabilities
    * compute the cross entropy loss derivative (pre-advantage)
    * perform the sampled action, get a new observation and the reward
    * if an episode has ended:
        * discount the rewards
        * modulate the gradient with the advantage (the discounted rewards)
        * perform the rest of the backward pass
        * update the parameters

**check the sign of parameter grad update**

## 3. Performance and results:
<p align="center">
   <img src="https://github.com/Blato122/pong-rl-gym/blob/main/3plot.png" alt="Plot showing the performance of a neural network - running average reward and wins over the last 100 episodes" width=80% height=80%>
   <img src="https://github.com/Blato122/pong-rl-gym/blob/main/pong-gif-15sec.gif" alt="Gif showcasing a part of a Pong game, after 6700 episodes of training" />
</p>

## 4. Sources and useful links:
1. Andrej Karpathy Pong policy gradient blog post - https://karpathy.github.io/2016/05/31/rl/
2. Andrej Karpathy Pong policy gradient lecture - https://www.youtube.com/watch?v=tqrcjHuNdmQ&t=1870s
3. Andrej Karpathy Pong policy gradient gist - https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
4. Cross entropy loss and softmax derivatives - https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/
5. David Silver's lectures on reinforcement learning - https://www.davidsilver.uk/teaching/ and https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ
