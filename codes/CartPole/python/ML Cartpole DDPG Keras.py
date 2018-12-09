import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
import sys
import random
import time
import tensorflow as tf
from collections import deque
from keras.layers import Dense, Input, Add, GaussianNoise,Concatenate
from keras.optimizers import Adam, SGD, Nadam
from keras.models import Model
from keras import backend as K
from keras import regularizers

from mlagents.envs import UnityEnvironment

print("Python version:")
print(sys.version)

# check Python version
if (sys.version_info[0] < 3):
    raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")

env_name = "../env/CartPole"  # Name of the Unity environment binary to launch

env = UnityEnvironment(file_name=env_name)

# Set the default brain to work with
default_brain = env.brain_names[0]
brain = env.brains[default_brain]


class OU_noise():
    def __init__(self, action_size, mu=0, theta=0.1, sigma=0.1):
        self.action_size = action_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_size) * self.mu
        # self.shape = np.shape(self.action_size)
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_size) * self.mu

    def noise(self):
        x = self.state

        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(self.action_size[0], self.action_size[1])
        self.state = x + dx
        # print(self.state)
        return self.state


class DDPGAgent:
    def __init__(self, state_size, agent_size, action_size):
        self.state_size = state_size
        self.agent_size = agent_size
        self.action_size = action_size
        self.load_model = True
        self.Gausian_size = 0.01
        self.gard_clip_radious = 100.0

        # build networks
        self.actor = self.build_actor()
        self.actor_target = self.build_actor()
        self.critic = self.build_critic()
        self.critic_target = self.build_critic()
        self.actor_updater = self.actor_optimizer()

        self.memory = deque(maxlen=50000)
        self.batch_size = 256
        self.discount_factor = 0.99
        self.epsilon = 1
        self.epsilon_decay = 0.9999

        self.noiser = OU_noise([agent_size, self.action_size])  # 수정한 부분

        if self.load_model:
            self.actor.load_weights("3DBall_actor.h5")
            self.actor_target.load_weights("3DBall_actor.h5")
            self.critic.load_weights("3DBall_critic.h5")
            self.critic_target.load_weights("3DBall_critic.h5")

    def build_actor(self):
        print("building actor network")
        input = Input(shape=[self.state_size])
        h1 = Dense(512, activation='elu')(input)
        h1 = Dense(512, activation='elu')(h1)
        h1 = Dense(512, activation='elu')(h1)
        h1 = Dense(512, activation='elu')(h1)
        h1 = Dense(512, activation='elu')(h1)
        h1 = Dense(512, activation='elu')(h1)
        h1 = Dense(512, activation='elu')(h1)
        action = Dense(self.action_size, activation='tanh')(h1)
        actor = Model(inputs=input, outputs=action)
        actor.summary()
        return actor

    def actor_optimizer(self):
        actions = self.actor.output
        dqda = tf.gradients(self.critic.output, self.critic.input)
        loss = actions * tf.clip_by_value(-dqda[1], -self.gard_clip_radious, self.gard_clip_radious)

        optimizer = Adam(lr=0.00001)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, self.critic.input[0],
                            self.critic.input[1]], [], updates=updates)
        return train

    def build_critic(self):
        print("building critic network")
        state = Input(shape=[self.state_size], name='state_input')
        action = Input(shape=[self.action_size], name='action_input')
        w1 = Dense(64, activation='elu')(state)
        w1 = Dense(64, activation='elu')(w1)
        a1 = Dense(64, activation='elu')(action)
        a1 = Dense(64, activation='elu')(a1)
        c = Concatenate()([w1, a1])
        # c = Add()([w1,a1])
        h2 = Dense(512, activation='elu')(c)
        h2 = Dense(512, activation='elu')(h2)
        h2 = Dense(512, activation='elu')(h2)
        h2 = Dense(512, activation='elu')(h2)
        h2 = Dense(512, activation='elu')(h2)
        h2 = Dense(512, activation='elu')(h2)
        h2 = Dense(512, activation='elu')(h2)
        Velue = Dense(1, activation='linear', kernel_regularizer=regularizers.l1_l2(0.0001, 0.0001))(h2)
        critic = Model(inputs=[state, action], outputs=Velue)
        critic.compile(loss='mse', optimizer=Adam(lr=0.00001))
        critic.summary()
        return critic

    def get_action(self, state):
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.05)
        # print(state)
        action = self.actor.predict(state)

        real = action + self.epsilon * self.noiser.noise()
        return np.clip(real, -1.1, 1.1)

    def gat_action_nonoise(self, state):
        action = self.actor.predict(state)

        real = action
        return np.clip(real, -1.1, 1.1)

    def append_sample(self, state, action, reward, next_state, done):

        for i in range(self.agent_size):
            self.memory.append((state[i], action[i], reward[i], next_state[i], done[i]))
        # self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        # make mini-batch from replay memory
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.asarray([e[0] for e in mini_batch])
        actions = np.asarray([e[1] for e in mini_batch])
        rewards = np.asarray([e[2] for e in mini_batch])
        next_states = np.asarray([e[3] for e in mini_batch])
        dones = np.asarray([e[4] for e in mini_batch])

        # update critic network
        critic_action_input = self.actor_target.predict(next_states)
        target_q_values = self.critic_target.predict([next_states, critic_action_input])

        targets = np.zeros([self.batch_size, 1])
        for i in range(self.batch_size):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = rewards[i] + self.discount_factor * target_q_values[i]

        self.critic.train_on_batch([states, actions], targets)

        # update actor network
        a_for_grad = self.actor.predict(states)
        self.actor_updater([states, states, a_for_grad])
        # self.actor_updater([states, states, actions])

    def train_critic(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.asarray([e[0] for e in mini_batch])
        actions = np.asarray([e[1] for e in mini_batch])
        rewards = np.asarray([e[2] for e in mini_batch])
        next_states = np.asarray([e[3] for e in mini_batch])
        dones = np.asarray([e[4] for e in mini_batch])

        self.critic.train_on_batch([states, actions], rewards)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

# Reset the environment
env_info = env.reset(train_mode=True)[default_brain]

# Examine the state space for the default brain
print("Agent state looks like: \n{}".format(env_info.vector_observations[0]))

print("Agent shape looks like: \n{}".format(np.shape(env_info.vector_observations)))

agent = DDPGAgent(5,10, 1)

train_mode = False  # Whether to run the environment in training or inference mode

reward_memory = deque(maxlen=20)
agents_reward = np.zeros(agent.agent_size)
env_info = env.reset(train_mode=train_mode)[default_brain]
for episode in range(10000000):
    # env_info = env.reset(train_mode=train_mode)[default_brain]
    state = env_info.vector_observations
    done = False
    episode_rewards = 0
    for i in range(100):

        if not train_mode:
            action = agent.gat_action_nonoise(state)
        else:
            action = agent.get_action(state)
        # action = np.column_stack([np.random.randint(0, action_size[i], size=(len(env_info.agents))) for i in range(len(action_size))])
        env_info = env.step(action)[default_brain]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        agents_reward += reward
        # episode_rewards += reward#env_info.rewards[0]
        done = env_info.local_done
        for idx, don in enumerate(done):
            if don:
                reward_memory.append(agents_reward[idx])
                agents_reward[idx] = 0
        if train_mode:
            agent.append_sample(state, action, reward, next_state, done)
        state = next_state

        if len(agent.memory) > agent.batch_size * 10 and train_mode:
            agent.train_model()

    agent.noiser.reset()
    agent.update_target_model()
    if agent.epsilon <= 0.05:
        agent.epsilon = 0.99

    if episode % 50 == 0 and not episode == 0 and train_mode:
        #agent.actor.save("3DBall_actor.h5")
        #agent.critic.save("3DBall_critic.h5")
        print("model saved")
    print("episode_{} reward: {} episilon: {}".format(episode, np.mean(reward_memory), agent.epsilon))