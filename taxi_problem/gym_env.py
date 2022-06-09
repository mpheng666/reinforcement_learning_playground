import gym

env = gym.make('Taxi-v3').env

env.reset()


# south, north, east, west, pickup, dropoff
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

# (taxi row, taxi col, passenger index, destination index)
state = env.encode(3, 2, 1, 0)
print("state: ", state)

env.s = state

print(env.P[state])
# {action: [(probability, nextstate, reward, done)]}
# {0: [(1.0, 428, -1, False)],
#  1: [(1.0, 228, -1, False)],
#  2: [(1.0, 348, -1, False)],
#  3: [(1.0, 328, -1, False)],
#  4: [(1.0, 328, -10, False)],
#  5: [(1.0, 328, -10, False)]}

# probability is always 1.0 in this environment
# done is used to tell us when a passenger is dropped off to the right location


for i in range(1000):
    env.render()
    