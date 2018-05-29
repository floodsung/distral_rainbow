import tensorflow as tf

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models,distill_network
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer,BasicPlayer
from anyrl.spaces import gym_space_vectorizer

from sonic_util import AllowBacktracking, make_env
import numpy as np
import csv
import ray
import time
import os

import logging
import sys
import pdb

np.random.seed(0)

# class StreamToLogger(object):
#    """
#    Fake file-like stream object that redirects writes to a logger instance.
#    """
#    def __init__(self, logger, log_level=logging.INFO):
#       self.logger = logger
#       self.log_level = log_level
#       self.linebuf = ''

#    def write(self, buf):
#       for line in buf.rstrip().splitlines():
#          self.logger.log(self.log_level, line.rstrip())

# logging.basicConfig(
#    level=logging.DEBUG,
#    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
#    filename="out.log",
#    filemode='a'
# )

# stdout_logger = logging.getLogger('STDOUT')
# sl = StreamToLogger(stdout_logger, logging.INFO)
# sys.stdout = sl

# stderr_logger = logging.getLogger('STDERR')
# sl = StreamToLogger(stderr_logger, logging.ERROR)
# sys.stderr = sl

THREAD_NUM = 8
NUM_ITER  = 5000000
AGENT_NUM_PER_THREAD = 6

@ray.remote(num_cpus=AGENT_NUM_PER_THREAD,num_gpus=1)
class MultiAgent():
    """docstring for MultiAgent"""
    def __init__(self,thread_index,num_agent=AGENT_NUM_PER_THREAD):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ray.get_gpu_ids()))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        env = AllowBacktracking(make_env())
        action_space = env.action_space.n
        observation_space = env.observation_space

        distill_net = distill_network(sess,
                                  action_space,
                                  gym_space_vectorizer(observation_space),
                                  min_val=-200,
                                  max_val=200)

        self.agents = [DistralAgent(sess,i,thread_index,distill_net,action_space,observation_space) for i in range(num_agent)]
        sess.run(tf.global_variables_initializer())
        [sess.run(agent.dqn.update_target) for agent in self.agents]

    def train(self,distill_policy_weights):
        self.agents[0].dqn.set_distill_policy_weights(distill_policy_weights)
        distill_grads_list = [agent.train() for agent in self.agents]

        return distill_grads_list

@ray.remote
class SonicEnv():

    def __init__(self,env_index):
        train_file = csv.reader(open('./sonic-train.csv','r'),delimiter=',')
        self.games = []
        for i,row in enumerate(train_file):
            if i == 0:
                continue
            self.games.append(row)

        self.env = AllowBacktracking(make_env(game=self.games[env_index][0],state=self.games[env_index][1]))

    def step(self,action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def action_space(self):
        return self.env.action_space.n

    def observation_space(self):
        return self.env.observation_space


class DistralAgent():

    def __init__(self,sess,agent_index,thread_index,distill_network,action_space,observation_space):
        # step 1: init env
        self.env_index = agent_index + thread_index*AGENT_NUM_PER_THREAD
        print("create env:",self.env_index)
        self.env = SonicEnv.remote(self.env_index)

        self.sess = sess
        self.dqn = DQN(*rainbow_models(self.sess,
                                          agent_index,
                                          action_space,
                                          gym_space_vectorizer(observation_space),
                                          min_val=-200,
                                          max_val=200),distill_network)
        self.player = NStepPlayer(BasicPlayer(self.env, self.dqn.online_net), 3)

        self.replay_buffer = PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1)
        self.steps_taken = 0
        self.train_interval=1
        self.target_interval=8192
        self.batch_size=32
        self.min_buffer_size=20000
        self.handle_ep=lambda steps, rew: None
        self.next_target_update = self.target_interval
        self.next_train_step = self.train_interval

    def train(self):

        transitions = self.player.play()
        distill_grads = 0
        for trans in transitions:
                if trans['is_last']:
                    self.handle_ep(trans['episode_step'] + 1, trans['total_reward'])
                self.replay_buffer.add_sample(trans)
                self.steps_taken += 1
                if self.replay_buffer.size >= self.min_buffer_size and self.steps_taken >= self.next_train_step:
                    self.next_train_step = self.steps_taken + self.train_interval
                    batch = self.replay_buffer.sample(self.batch_size)

                    grad_names = []
                    for grad in self.dqn.distill_grads:
                        if grad[0] != None:
                            grad_names.append(grad[0])

                    _,losses,distill_grads = self.sess.run((self.dqn.optim,self.dqn.losses,grad_names,self.dqn.target_preds,self.dqn.target_dists,self.dqn.distill_kl),
                                         feed_dict=self.dqn.feed_dict(batch))

                    self.replay_buffer.update_weights(batch, losses)

                if self.steps_taken >= self.next_target_update:
                    self.next_target_update = self.steps_taken + self.target_interval
                    self.sess.run(self.dqn.update_target)

        return distill_grads


def main():

    ray.init()


    # create an env for env info
    env = AllowBacktracking(make_env())
    action_space = env.action_space.n
    observation_space = env.observation_space

    # init multi agents
    agents = [MultiAgent.remote(i) for i in range(THREAD_NUM)]

    # init session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # init local dqn with distill network
    local_dqn = DQN(*rainbow_models(sess,0,
                                  action_space,
                                  gym_space_vectorizer(observation_space),
                                  min_val=-200,
                                  max_val=200),
                    distill_network(sess,
                                  action_space,
                                  gym_space_vectorizer(observation_space),
                                  min_val=-200,
                                  max_val=200))
    sess.run(tf.global_variables_initializer())

    distill_network_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="distill")
    saver = tf.train.Saver(distill_network_variables)
    checkpoint = tf.train.get_checkpoint_state("./models")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print ("Could not find old network weights")

    grad_names = []
    for grad in local_dqn.distill_grads:
        if grad[0] != None:
            grad_names.append(grad[0])

    weights = local_dqn.get_distill_policy_weights()


    for iteration in range(NUM_ITER):
        #start = time.time()
        if iteration % 1000 == 0:
            print("iter:",iteration)

        weights_id = ray.put(weights)
        gradients_ids = [agent.train.remote(weights_id) for agent in agents]
        gradients_raw = ray.get(gradients_ids)
        gradients_list = []
        for gradients in gradients_raw:
            for gradient in gradients:
                gradients_list.append(gradient)

        if not 0 in gradients_list:
            mean_grads = [sum([gradients[i] for gradients in gradients_list]) / len(gradients_list) for i in range(len(gradients_list[0]))]
            feed_dict = {grad: mean_grad for (grad, mean_grad) in zip(grad_names, mean_grads)}
            sess.run(local_dqn.train_distill_policy, feed_dict=feed_dict)

        weights = local_dqn.get_distill_policy_weights()

        if (iteration+1)% 20000 == 0:
            if not os.path.exists("./models"):
                os.makedirs("./models")
            saver.save(sess, './models/' + 'network', global_step = 10000)


        #print("iter:",iteration, "time:",time.time() - start)


if __name__ == '__main__':
    main()






