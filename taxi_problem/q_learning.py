# Initialize the Q-table by all zeros.
# Start exploring actions: For each state, select any one among all possible actions for the current state (S).
# Travel to the next state (S') as a result of that action (a).
# For all possible actions from the state (S') select the one with the highest Q-value.
# Update Q-table values using the equation.
# Set the next state as the current state.
# If goal state is reached, then end and repeat the process.


import gym
import numpy as np
import random
from time import sleep

# ============== Setup env ============
env = gym.make('Taxi-v3').env
# env.reset()

# ======== Traing the Agent ===============
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        print(f"Episode: {i}")

print("Training finished.\n")
print(f"Results during training:")
print(f"Average penalties per episode: {penalties}")

# =======Evaluate agent's performance after Q-learning =======

total_epochs, total_penalties = 0, 0
episodes = 100
frames = [] # for animation

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
        }
        )

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")

def print_frames(frames):
        for i, frame in enumerate(frames):
            print(frame['frame'])
            print(f"Timestep: {i + 1}")
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            print(f"Reward: {frame['reward']}")
            sleep(.1)

print_frames(frames)

