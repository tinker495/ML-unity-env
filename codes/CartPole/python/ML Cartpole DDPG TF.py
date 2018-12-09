import numpy as np
import numpy.random as nr
import random
import tensorflow as tf
import tensorflow.layers as layer
from collections import deque
from mlagents.envs import UnityEnvironment

######################################
state_size = 5
agent_size = 10
action_size = 1
save_path = ".ddpg"
load_model = True
Viewtrain = False
batch_size = 200
hidden_layer_size = 512
mem_maxlen = 1000000
discount_factor = 0.97
train_late = 0.0001
train_mode = False
run_episode = 1000000
update_interval = 100
update_target_rate = 0.001
print_interval = 100
save_interval = 500
numGoals = 3
epsilon_refresh = True
epsilon_refresh_trig = 0.01
epsilon_decay = 0.99995
env_name = "../env/CartPole"
logdir = "../Summary/ddpg"
######################################

def UpdateTargetGraph(tfVars,target_update_rate):
    var_lens = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:var_lens // 2]):
        print(tfVars[idx + var_lens//2].name , var.name)
        op_holder.append(tfVars[idx + var_lens // 2].assign((var.value() * target_update_rate) + ((1 - target_update_rate) * tfVars[idx + var_lens // 2].value())))
    return op_holder

def update(Session,update_op_holder):
    for op in update_op_holder:
        Session.run(op)

class OU_noise():
    def __init__(self,action_size,mu=0,theta=0.1,sigma=0.2):
        self.action_size = action_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_size)*self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_size)*self.mu

    def noise(self):
        x = self.state
        dx = self.theta*(self.mu - x) + self.sigma*nr.randn(self.action_size[0],self.action_size[1])
        self.state = x + dx
        return  self.state

class ActorNetwork():
    def __init__(self,state_size,action_size,hidden_layer_size,learning_rate,name):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_layer_size
        self.name = name
        self.learning_rate = learning_rate
        with tf.variable_scope(name):
            self.observation = tf.placeholder(tf.float32,shape=[None,self.state_size],name="actor_observation")
            self.L1 = layer.dense(self.observation,self.hidden_size,activation=tf.nn.leaky_relu)
            self.L2 = layer.dense(self.L1,self.hidden_size,activation=tf.nn.leaky_relu)
            self.L3 = layer.dense(self.L2,self.hidden_size,activation=tf.nn.leaky_relu)
            self.action = layer.dense(self.L3,self.action_size,activation=tf.nn.tanh)



        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)


    def trainer(self,critic,batch_size):
        action_Grad = tf.gradients(critic.value,critic.action)
        self.policy_Grads = tf.gradients(ys=self.action,xs=self.trainable_var,grad_ys=action_Grad)
        for idx,grads in enumerate(self.policy_Grads):
            self.policy_Grads[idx] = -grads/batch_size

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.Adam = tf.train.AdamOptimizer(self.learning_rate)
            self.Update = self.Adam.apply_gradients(zip(self.policy_Grads,self.trainable_var))

class CriticNetwork():
    def __init__(self,state_size,action_size,hidden_layer_size,learning_rate,name):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layer_size = hidden_layer_size
        self.name = name
        with tf.variable_scope(name):
            self.observation = tf.placeholder(tf.float32,shape=[None,self.state_size],name="critic_observation")
            self.O1 = layer.dense(self.observation,self.hidden_layer_size//2,activation=tf.nn.leaky_relu)
            self.action = tf.placeholder(tf.float32,shape=[None,self.action_size],name="critic_action")
            self.A1 = layer.dense(self.action,self.hidden_layer_size//2,activation=tf.nn.leaky_relu)
            self.L1 = tf.concat([self.O1,self.A1],1)
            self.L1 = layer.dense(self.L1,self.hidden_layer_size,activation=tf.nn.leaky_relu)
            self.L2 = layer.dense(self.L1,self.hidden_layer_size,activation=tf.nn.leaky_relu)
            self.L3 = layer.dense(self.L2,self.hidden_layer_size,activation=tf.nn.leaky_relu)
            self.value = layer.dense(self.L3,1,activation=None)

        self.true_value = tf.placeholder(tf.float32,name="true_value")
        self.loss = tf.losses.huber_loss(self.true_value,self.value)
        self.Adam = tf.train.AdamOptimizer(learning_rate)
        self.Update = self.Adam.minimize(self.loss)
        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

class DDPGAgent():
    def __init__(self,state_size,action_size,hidden_layer_size,mem_maxlen,save_path,learning_rate,load_model,batch_size,epsilon,epsilon_decay,update_target_rate,epsilon_min=0.01,discount_factor=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_laeyr_size = hidden_layer_size

        self.actor_model = ActorNetwork(self.state_size,self.action_size,self.hidden_laeyr_size,learning_rate,"actor_model")
        self.critic_model = CriticNetwork(self.state_size,self.action_size,self.hidden_laeyr_size,learning_rate,"critic_model")

        self.actor_target = ActorNetwork(self.state_size,self.action_size,self.hidden_laeyr_size,learning_rate,"actor_target")
        self.critic_target = CriticNetwork(self.state_size,self.action_size,self.hidden_laeyr_size,learning_rate,"critic_target")
        self.tvar = tf.trainable_variables()
        self.actor_model.trainer(self.critic_model,batch_size)
        self.update_op_holder = UpdateTargetGraph(self.tvar,update_target_rate)
        self.memory = deque(maxlen=mem_maxlen)
        self.Session = tf.Session()
        self.load_model = load_model

        self.init = tf.global_variables_initializer()
        self.batch_size = batch_size

        self.Session.run(self.init)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.discount_factor = discount_factor
        self.Saver = tf.train.Saver(max_to_keep=5)
        self.save_path = save_path
        self.Summary,self.Merge = self.make_Summary()
        self.noiser = OU_noise([agent_size,self.action_size])

        if self.load_model == True:
            ckpt = tf.train.get_checkpoint_state(self.save_path)
            self.Saver.restore(self.Session,ckpt.model_checkpoint_path)

    def get_action(self,state,train_mode=True):
        #print(state)
        out = self.Session.run(self.actor_model.action,feed_dict={self.actor_model.observation:state})
        if train_mode:
            return out + self.noiser.noise()*self.epsilon
        else:
            return out
    def append_sample(self,state,action,reward,next_state,done):
        for i in range(agent_size):
            self.memory.append((state[i],action[i],reward[i],next_state[i],done[i]))

    def save_model(self):
        self.Saver.save(self.Session,self.save_path + "\model.ckpt")

    def train_model(self,print_debug=False):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory,self.batch_size)
        states = np.asarray([e[0] for e in mini_batch])
        actions = np.asarray([e[1] for e in mini_batch])
        rewards = np.asarray([e[2] for e in mini_batch])
        next_states = np.asarray([e[3] for e in mini_batch])
        dones = np.asarray([e[4] for e in mini_batch])

        critic_action_input = self.Session.run(self.actor_target.action,feed_dict={self.actor_target.observation:next_states})
        target_q_value  = self.Session.run(self.critic_target.value,feed_dict={self.critic_target.observation:next_states,self.critic_target.action:critic_action_input})

        targets = np.zeros([self.batch_size,1])
        for i in range(self.batch_size):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = rewards[i] + self.discount_factor*target_q_value[i]

        _,loss = self.Session.run([self.critic_model.Update,self.critic_model.loss],feed_dict={self.critic_model.observation:states,self.critic_model.action:actions,self.critic_model.true_value:targets})

        action_for_train = self.Session.run(self.actor_model.action,feed_dict={self.actor_model.observation:states})
        _,grad = self.Session.run([self.actor_model.Update,self.actor_model.policy_Grads],feed_dict={self.actor_model.observation:states,self.critic_model.observation:states,self.critic_model.action:action_for_train})
        #print(grad)
        return loss

    def update_target(self):
        update(self.Session,self.update_op_holder)

    def make_Summary(self):
        self.summary_loss = tf.placeholder(dtype=tf.float32)
        self.summary_reward = tf.placeholder(dtype=tf.float32)
        tf.summary.scalar("loss", self.summary_loss)
        tf.summary.scalar("reward",self.summary_reward)
        return tf.summary.FileWriter(logdir=logdir,graph=self.Session.graph),tf.summary.merge_all()

    def Write_Summray(self,reward,loss,episode):
        self.Summary.add_summary(self.Session.run(self.Merge,feed_dict={self.summary_loss:loss,self.summary_reward:reward}),episode)


if __name__ == '__main__':
    env = UnityEnvironment(file_name=env_name)

    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    agent = DDPGAgent(state_size,
                      action_size,
                      hidden_layer_size,
                      mem_maxlen,
                      save_path,
                      train_late,
                      load_model,
                      batch_size,
                      1.0,
                      epsilon_decay=epsilon_decay,
                      update_target_rate=update_target_rate,
                      discount_factor=discount_factor)

    env_info = env.reset(train_mode=train_mode and not Viewtrain)[default_brain]

    print("Agent shape looks like: \n{}".format(np.shape(env_info.visual_observations)))

    reward_memory = deque(maxlen=20)
    agent_reward = np.zeros(agent_size)
    losses = deque(maxlen=20)
    frame_count = 0;
    for episode in range(run_episode):
        state = env_info.vector_observations

        for i in range(update_interval):
            frame_count += 1
            action = agent.get_action(state,train_mode)
            env_info = env.step(action)[default_brain]
            next_state = env_info.vector_observations
            reward = env_info.rewards - 0.5*np.reshape(np.abs(action),[10,])
            agent_reward += reward
            done = env_info.local_done
            for idx,don in enumerate(done):
                if don:
                    reward_memory.append(agent_reward[idx])
                    agent_reward[idx] = 0
            if train_mode:
                agent.append_sample(state,action,reward,next_state,done)
            state = next_state

            if train_mode and len(agent.memory) > agent.batch_size * 2:
                loss = agent.train_model()
                losses.append(loss)
                agent.update_target()

        if epsilon_refresh and agent.epsilon < epsilon_refresh_trig:
            agent.epsilon = 0.9

        if episode%print_interval == 0:
            print("episode({}) - reward: {:.2f}     loss: {:.4f}      epsilon: {:.3f}       memory_len:{}".format(
                episode,np.mean(reward_memory),np.mean(losses),agent.epsilon,len(agent.memory)))
            agent.Write_Summray(np.mean(reward_memory), np.mean(losses), episode)

        if episode%save_interval == 0 and episode != 0:
            print("model saved")
            agent.save_model()
