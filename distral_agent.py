import tensorflow as tf

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
from sonic_util import AllowBacktracking, make_env
import gym_remote.exceptions as gre
import numpy as np
import csv
import os
import time


def main():
    env = AllowBacktracking(make_env())
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)

    # step 2: init rainbow
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    dqn = DQN(*rainbow_models(sess,
                              env.action_space.n,
                              gym_space_vectorizer(env.observation_space),
                              min_val=-200,
                              max_val=200))
    player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
    sess.run(tf.global_variables_initializer())

    distill_network_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="distill")
    saver = tf.train.Saver(distill_network_variables)
    checkpoint = tf.train.get_checkpoint_state("./models")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print ("Could not find old network weights")

    dqn.train(sess=sess,
              num_steps=2000000, # Make sure an exception arrives before we stop.
              player=player,
              replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
              train_interval=1,
              target_interval=8192,
              batch_size=32,
              min_buffer_size=20000)





if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)






