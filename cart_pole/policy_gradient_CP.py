import tensorflow as tf
import numpy as np
import gym
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()
plt.ion()

def plot_rewards(reward):
    plt.figure(1)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward)
    plt.pause(0.001)
    # # Take 100 episode averages and plot them too
    # if len(durations_t) >= 50:
    #     means = durations_t.unfold(0, 50, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(49), means))
    #     plt.plot(means.numpy())


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network.
    for size in sizes[:-1]:
        # TODO: migrate to tf2
        x = tf.compat.v1.layers.dense(x, units=size, activation=activation)
    return tf.compat.v1.layers.dense(x, units=sizes[-1], activation=output_activation)

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    obs_ph = tf.compat.v1.placeholder(shape=(None, obs_dim), dtype=tf.float32)
    logits = mlp(obs_ph, sizes=hidden_sizes+[n_acts])

    # make action selection op (outputs int actions, sampled from policy)
    actions = tf.squeeze(tf.compat.v1.multinomial(logits=logits, num_samples=1), axis=1)

    # make loss function whose gradient, for the right data, is policy gradient
    weights_ph = tf.compat.v1.placeholder(shape=(None,), dtype=tf.float32)
    print("weight ph: ", weights_ph.shape)
    act_ph = tf.compat.v1.placeholder(shape=(None,), dtype=tf.int32)
    print("act ph: ", act_ph.shape)
    action_masks = tf.one_hot(act_ph, n_acts)
    print("action mask: ", action_masks.shape)
    log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
    print("log_probs: ", log_probs.shape)
    loss = -tf.reduce_mean(weights_ph * log_probs)
    print("loss: ", loss.shape)

    # make train op
    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    # train_op = tf.optimizers.Adam(learning_rate=lr)

    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())

    rewards_for_viz = []

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = sess.run(actions, {obs_ph: obs.reshape(1, -1)})[0]
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True
                # print("weight ph: ", weights_ph)
                # print("act ph: ", act_ph)
                # print("action mask: ", action_masks)
                # print("log_probs: ", log_probs)
                # print("loss: ", loss)

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        batch_loss, _ = sess.run([loss, train_op],
                                 feed_dict={
            obs_ph: np.array(batch_obs),
            act_ph: np.array(batch_acts),
            weights_ph: np.array(batch_weights)
        })
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        rewards_for_viz.append(np.mean(batch_rets))
        plot_rewards(rewards_for_viz)
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
