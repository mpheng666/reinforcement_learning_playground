from configparser import Interpolation
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# setup environment
env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device: ", device)
print("cuda: ", torch.cuda.is_available())

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # print("w", w)
        # print("h", h)

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # print("convw", convw)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # print("convh", convh)
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
        # print("head:", self.head)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # print("forward return: ", self.head(x.view(x.size(0), -1)))
        return self.head(x.view(x.size(0), -1))


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, T.InterpolationMode.BICUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    # print("env.render.shape: ", env.render(mode='rgb_array').shape)
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    # print("screen.shape: ", screen.shape)
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    # print("screen2.shape: ", screen.shape)
    view_width = int(screen_width * 0.6)
    # print("view_width: ", view_width)
    cart_location = get_cart_location(screen_width)
    # print("cart location: ", cart_location)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # print("slice_range: ", slice_range)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    # print("screen continuous shape: ", screen.shape)
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)


env.reset()
# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
#            interpolation='none')
# plt.title('Example extracted screen')
# plt.show()

BATCH_SIZE = 128
GAMMA = 0.9999
EPS_START = 0.9
EPS_END = 0.015
EPS_DECAY = 1000
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
# print("policy net: ", policy_net)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
# print("target net: ", target_net)

# optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.Adam(policy_net.parameters())
# print("optimizer: ", optimizer)
memory = ReplayMemory(10000)


steps_done = 0
actions = []

def select_action(state):
    global steps_done
    sample = random.random()
    # print("sample: ", sample)
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    # print("eps_threshold: ", eps_threshold)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            temp = 1
            return policy_net(state).max(1)[1].view(1, 1), temp
    else:
        temp = 0
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), temp

episode_durations = []

def plot_durations():
    plt.figure(2)
    # plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # print("durations_t: ", durations_t.numpy())

    # Take 100 episode averages and plot them too
    if len(durations_t) >= 50:
        means = durations_t.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(49), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def plot_loss(ave_losses):
    plt.figure(3)
    # plt.clf()
    plt.title('Loss over t')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(ave_losses)
    # plt.pause(0.001)

def plot_action_selected(actions):
    plt.figure(4)
    plt.title("Action selected over t")
    plt.xlabel('Episode')
    plt.ylabel('Action')
    plt.plot(actions)
    # plt.pause(0.001)

def optimize_model():
    loss = 0
    if len(memory) < BATCH_SIZE: #128
        return loss
    transitions = memory.sample(BATCH_SIZE)
    # print("transitions: ", transitions)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    # print("non final mask: ", non_final_mask)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    # print("non final next states: ", non_final_next_states.shape)
    state_batch = torch.cat(batch.state)
    # print(type(state_batch))
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # print("reward type: ", type(batch.reward[0]))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # print("state_action_values size", state_action_values.size())

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # print(type(next_state_values))
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # print("expected_state_action_values: ", expected_state_action_values.size())

    # Compute Huber loss
    # criterion = nn.SmoothL1Loss()
    criterion = nn.HuberLoss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))
    # print("loss: ", loss.item())

    # Optimize the model
    optimizer.zero_grad() # zero previos grad from backpropagation
    loss.backward() # back propagation
    for param in policy_net.parameters():
        # print("param.grad.data before clamp: ", param.grad.data)
        param.grad.data.clamp_(-1, 1)
        # print("param.grad.data after clamp: ", param.grad.data)
    optimizer.step() # gradient descent
    return loss.item()


num_episodes = 1000000
# num_episodes = 10
ave_losses = []
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    losses = []
    temp_actions = []
    for t in count():
        # print("t: ", t)
        # Select and perform an action
        action, temp = select_action(state)
        temp_actions.append(temp)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        losses.append(optimize_model())
        if done:
            episode_durations.append(t + 1)
            # average_loss = sum(losses)/(t+1)
            ave_losses.append(sum(losses)/(t+1))
            actions.append(sum(temp_actions)/(t+1))
            plot_durations()
            plot_loss(ave_losses)
            plot_action_selected(actions)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
