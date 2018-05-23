import tensorflow as tf

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer

from sonic_util import AllowBacktracking, make_env
import csv
import ray

THREAD_NUM = 25
NUM_ITER  = 5000000

class DistralAgent():

    def __init__(self,env_index):
        # step 1: init env
        self.env_index = env_index
        self.env = self.init_env(env_index)

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
        # online_network_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="online")
        # saver = tf.train.Saver(online_network_variables)
        # checkpoint = tf.train.get_checkpoint_state("./models")
        # if checkpoint and checkpoint.model_checkpoint_path:
        #     saver.restore(sess, checkpoint.model_checkpoint_path)
        #     print ("Successfully loaded:", checkpoint.model_checkpoint_path)
        # else:
        #     print ("Could not find old network weights")

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


    def init_env(self,env_index):
        train_file = csv.reader(open('./sonic-train.csv','r'),delimiter=',')
        self.games = []
        for i,row in enumerate(train_file):
            if i == 0:
                continue
            self.games.append(row)

        env = AllowBacktracking(make_env(game=self.games[env_index][0],state=self.games[env_index][1]))

        env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
        return env

    def train(self,distilled_policy_weights):

        self.dqn.set_distilled_policy_weights(distilled_policy_weights)

        transitions = self.player.play()
        actual_grads = 0
        batch = 0
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
                if self.steps_taken >= self.next_target_update:
                    self.next_target_update = self.steps_taken + self.target_interval
                    self.sess.run(self.dqn.update_target)

        info = {"id": self.env_index}

        return batch,info


def main():

    ray.init()

    remote_agent = ray.remote(DistralAgent)
    agents = [remote_agent.remote(i) for i in range(THREAD_NUM)]

    #joint agent
    env = AllowBacktracking(make_env())
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    sess = tf.Session(config=config)
    joint_dqn = DQN(*rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200))
    sess.run(tf.global_variables_initializer())

    distilled_network_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="distilled")
    saver = tf.train.Saver(distilled_network_variables)
    checkpoint = tf.train.get_checkpoint_state("./models")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print ("Could not find old network weights")


    parameters = joint_dqn.get_distilled_policy_weights()

    batch_list = [agent.train.remote(parameters) for agent in agents]

    for iteration in range(NUM_ITER):
        if iteration % 1000 == 0:
            print("iter:",iteration)
        done_ids,batch_list = ray.wait(batch_list,num_returns=5)
        batchs,infos = ray.get(done_ids)

        if not 0 in batchs:
            batchs = np.stack(batchs)
            _,loss = sess.run((joint_dqn.policy_optim,joint_dqn.distilled_policy_loss),
                            feed_dict=self.dqn.feed_dict(batchs))

        parameters = joint_dqn.get_distilled_policy_weights()
        for info in infos:
            batch_list.extend([agents[info["id"]].train.remote(parameters)])

        if (iteration+1)% 20000 == 0:
            if not os.path.exists("./models"):
                os.makedirs("./models")
            saver.save(sess, './models/' + 'network', global_step = 10000)



if __name__ == '__main__':
    main()






