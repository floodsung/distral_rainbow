"""
The core implementation of deep Q-learning.
"""

import time

import tensorflow as tf
import os
import numpy as np
import ray
import pdb

# pylint: disable=R0902,R0903



class DQN:
    """
    Train TFQNetwork models using Q-learning.
    """
    def __init__(self, online_net, target_net, distill_net, discount=0.99):
        """
        Create a Q-learning session.

        Args:
          online_net: the online TFQNetwork.
          target_net: the target TFQNetwork.
          discount: the per-step discount factor.
        """
        self.online_net = online_net
        self.target_net = target_net
        self.distill_net = distill_net
        self.discount = discount

        obs_shape = (None,) + online_net.obs_vectorizer.out_shape
        self.obses_ph = tf.placeholder(online_net.input_dtype, shape=obs_shape)
        self.actions_ph = tf.placeholder(tf.int32, shape=(None,))
        self.rews_ph = tf.placeholder(tf.float32, shape=(None,))
        self.new_obses_ph = tf.placeholder(online_net.input_dtype, shape=obs_shape)
        self.terminals_ph = tf.placeholder(tf.bool, shape=(None,))
        self.discounts_ph = tf.placeholder(tf.float32, shape=(None,))
        self.weights_ph = tf.placeholder(tf.float32, shape=(None,))

        self.log_distill_policy = self.distill_net.log_policy(self.new_obses_ph)
        losses,self.distill_loss,self.target_preds_mean,self.target_dists,self.target_preds,self.tile_policies = online_net.transition_loss(target_net,self.log_distill_policy, self.obses_ph, self.actions_ph,
                                            self.rews_ph, self.new_obses_ph, self.terminals_ph,
                                            self.discounts_ph)
        self.losses = self.weights_ph * losses
        self.loss = tf.reduce_mean(self.losses)

        self.distill_variables = ray.experimental.TensorFlowVariables(self.log_distill_policy, self.online_net.session)

        assigns = []
        for dst, src in zip(target_net.variables, online_net.variables):
            assigns.append(tf.assign(dst, src))
        self.update_target = tf.group(*assigns)

        self.optim = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1.5e-4).minimize(self.loss)

        with tf.variable_scope("adam", reuse=tf.AUTO_REUSE):
            distill_optim = tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1.5e-4)
            self.distill_grads = distill_optim.compute_gradients(self.distill_loss)
            self.train_distill_policy = distill_optim.apply_gradients(self.distill_grads)


    def feed_dict(self, transitions):
        """
        Generate a feed_dict that feeds the batch of
        transitions to the DQN loss terms.

        Args:
          transition: a sequence of transition dicts, as
            defined in anyrl.rollouts.ReplayBuffer.

        Returns:
          A dict which can be fed to tf.Session.run().
        """
        obs_vect = self.online_net.obs_vectorizer
        res = {
            self.obses_ph: obs_vect.to_vecs([t['obs'] for t in transitions]),
            self.actions_ph: [t['model_outs']['actions'][0] for t in transitions],
            self.rews_ph: [self._discounted_rewards(t['rewards']) for t in transitions],
            self.terminals_ph: [t['new_obs'] is None for t in transitions],
            self.discounts_ph: [(self.discount ** len(t['rewards'])) for t in transitions],
            self.weights_ph: [t['weight'] for t in transitions]
        }
        new_obses = []
        for trans in transitions:
            if trans['new_obs'] is None:
                new_obses.append(trans['obs'])
            else:
                new_obses.append(trans['new_obs'])
        res[self.new_obses_ph] = obs_vect.to_vecs(new_obses)
        return res

    def get_distill_policy_weights(self):
        return self.distill_variables.get_weights()

    def set_distill_policy_weights(self,weights):
        self.distill_variables.set_weights(weights)

    def _discounted_rewards(self, rews):
        res = 0
        for i, rew in enumerate(rews):
            res += rew * (self.discount ** i)
        return res
