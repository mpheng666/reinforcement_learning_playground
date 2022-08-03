from math import gamma
import gym
import numpy as np
import pandas as pd
from collections import deque
import random

from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.models import load_model

import pickle
from matplotlib import pyplot as plt

class DQN:
    def __init__(self, env, learning_rate, gamme, epsilon, epsilon_decay):

        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.counter = 0

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.reward_list = []

        self.replay_memory_buffer = deque(maxlen=500000)
        self.batch_file = 64
        self.epsilon_min  = 0.01
        self.num_of_observation_space = env.observation_space.shape[0]
        self.num_of_action_space = env.action_space.n
        self.model = self.initialize_model()


        
        