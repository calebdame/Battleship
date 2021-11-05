# Battleship
### Various Methods to automate playing Battleship

1. Statistical Sampler : Random battleship formations conditioned on known hits, misses, and sinks are randomly generated and the most likely location of a hit is chosen as the next target
2. Naive Reinforcement Learning in Tensorflow: make Neural Network play each round, but use reward function to vary the network's learning rate
3. Deep Reinforcement Learning : Estimate the Q-values (**see Q-learning**) using the A2C Method (Advantage Actor Critic)
