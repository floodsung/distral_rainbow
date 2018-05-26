import tensorflow as tf

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer

from sonic_util import AllowBacktracking, make_env
import numpy as np
import csv
import ray
import os
import time

THREAD_NUM = 20
NUM_ITER  = 5000000

class DistralAgent():

    def __init__(self):
        # step 1: init env
        self.env = self.init_env()

        # step 2: init rainbow
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.dqn = DQN(*rainbow_models(self.sess,
                                  self.env.action_space.n,
                                  gym_space_vectorizer(self.env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        self.player = NStepPlayer(BatchedPlayer(self.env, self.dqn.online_net), 3)
        self.sess.run(tf.global_variables_initializer())

        distill_network_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="distill")
        saver = tf.train.Saver(distill_network_variables)
        checkpoint = tf.train.get_checkpoint_state("./models")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print ("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print ("Could not find old network weights")

        self.replay_buffer = PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1)
        self.sess.run(self.dqn.update_target)
        self.steps_taken = 0
        self.train_interval=1
        self.target_interval=8192
        self.batch_size=32
        self.min_buffer_size=20000
        self.handle_ep=lambda steps, rew: None
        self.next_target_update = self.target_interval
        self.next_train_step = self.train_interval


    def init_env(self):

        env = AllowBacktracking(make_env())

        env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
        return env

    def train(self):
        transitions = self.player.play()
        for trans in transitions:
                if trans['is_last']:
                    self.handle_ep(trans['episode_step'] + 1, trans['total_reward'])
                self.replay_buffer.add_sample(trans)
                self.steps_taken += 1
                if self.replay_buffer.size >= self.min_buffer_size and self.steps_taken >= self.next_train_step:
                    self.next_train_step = self.steps_taken + self.train_interval
                    batch = self.replay_buffer.sample(self.batch_size)

                    _,losses = self.sess.run((self.dqn.optim,self.dqn.losses),
                                         feed_dict=self.dqn.feed_dict(batch))
                    self.replay_buffer.update_weights(batch, losses)
                    # if self.steps_taken % 100 == 0:
                    #     print("steps:",self.steps_taken,"distill_loss:",distill_loss)
                if self.steps_taken >= self.next_target_update:
                    self.next_target_update = self.steps_taken + self.target_interval
                    self.sess.run(self.dqn.update_target)



def main():

    agent = DistralAgent()

    for iteration in range(NUM_ITER):
        agent.train()







if __name__ == '__main__':
    main()






