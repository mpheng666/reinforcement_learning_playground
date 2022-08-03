# pip3 install gym[box2d]
import gym

from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make("LunarLander-v2")
env.reset()
# print("sample action:" , env.action_space.sample())
# print("observation space shape", env.observation_space.shape)
# print("sample observation", env.observation_space.sample())

# # Instantiate the agent
# model = A2C("MlpPolicy", env, verbose = 1)

# # Train the agent
# model.learn(total_timesteps=int(2e5))

# # Save the agent
# model.save("dqn_lunar_a2c")

# # delete the trained model to demonstrate loading
# del model 

# Load the trained agent
model = A2C.load("dqn_lunar_a2c", env=env)

# Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

obs = env.reset()
for step in range(6000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render() 

# env.close()