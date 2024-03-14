## Policy gradient pong

# 1. Introduction:
This program trains a reinforcement learning agent to play a game of Pong using the policy gradient algorithm. It is based on an excellent article and a lecture by Andrej Karpathy. /insert links here - gist, blog, yt/
My goal was to add some more functionalities to his code in order to understand the whole concept even better. The changes I introduced:
* added third possible action (do nothing) besides moving up/down
  * modified the output layer so that it consists of 3 neurons, instead of 1
  * modified the action sampling method (since now there are 3 probabilities instead of 1)
  * modified the loss function so that it works with 3 actions (one-hot encoding)
* added a second hidden layer
  * modified the forward and backward pass
* the program now creates and updates a plot of running average reward over the last 100 episodes

# x. Sources:
1. Andrej Karpathy Pong policy gradient blog post -
2. Andrej Karpathy Pong policy gradient lecture -
3. Andrej Karpathy Pong policy gradient
