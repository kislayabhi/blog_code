from nn_tf import NN
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys
from flat_game import carmunk
import tensorflow as tf
from collections import deque


class Agent():

    def __init__(self):
        self.game = carmunk.GameState()
        self.episodes_length = 10000
        self.nn = NN(7, 3)
        self.gamma = 0.9

        # Generate the necessary tensorflow ops
        self.inputs1, self.nextQ = self.nn.placeholder_inputs(None)
        self.Qout = self.nn.inference(self.inputs1, 128, 32)
        self.loss = self.nn.loss_val(self.Qout, self.nextQ)
        self.train_op = self.nn.training(self.loss, learning_rate=0.01)
        self.init = tf.initialize_all_variables()
        self.saver = tf.train.Saver()

        # Generate the requisite buffer
        self.experience_memory = 10000
        self.experience_buffer = deque()

        self.minibatch_size = 128

        self.sess = tf.Session()
        self.sess.run(self.init)

    # Maps state to Q values
    def StoQ_FApprox(self, state):
        return self.sess.run(self.Qout, feed_dict={self.inputs1: state})

    def StoQ_FApprox_train(self, current_input, target_output):
        return self.sess.run(self.train_op, feed_dict={self.inputs1: current_input, self.nextQ: target_output})

    def experience_replay(self, new_experience):
        buff_size = len(self.experience_buffer)
        if( buff_size >= self.experience_memory):
            self.experience_buffer.popleft()
        self.experience_buffer.append(new_experience)
        buff_size += 1

        # Sample minibatches from the deque
        if(buff_size > self.minibatch_size):
            sampled_experience = random.sample(self.experience_buffer, self.minibatch_size)
        else:
            sampled_experience = random.sample(self.experience_buffer, buff_size)

        return sampled_experience

    def learn(self):

        epsilon = 0.9
        for e in xrange(self.episodes_length):
            r, s = self.game.frame_step(2)
            while (r != 500):

                # Choose a from s using policy derived from Q (epsilon-greedy)
                Qs = self.StoQ_FApprox(s)
                if(np.random.rand(1) > max(epsilon, 0.1)):
                    a = np.where(Qs == Qs.max())
                else:
                    a = np.random.randint(3)

                # Take action a, observe r and snext.
                r, snext = self.game.frame_step(a)

                # Tweak the Function approximator to behave according to the
                # current rewards.
                # target_reward = Qs.copy()
                # print "target_reward:", target_reward, target_reward[0,0]

                # if(r == 500):
                #     target_reward[0, a] = r
                # else:
                #     Qnext = self.StoQ_FApprox(snext)
                #     target_reward[0, a] = r + self.gamma * np.amax(Qnext)

                # Instead of training from the current state and target_reward,
                # we will train using a mini-batch.
                history = self.experience_replay((s, a, r, snext))

                # print history

                hist_states=[]
                hist_target_reward=[]

                for experience in history:
                    h_s = experience[0]
                    h_a = experience[1]
                    h_r = experience[2]
                    h_snext = experience[3]

                    Q_h_s = self.StoQ_FApprox(h_s)
                    target = Q_h_s.copy()

                    if(h_r == 500):
                        # Terminal stage
                        target[0, a] = h_r
                    else:
                        # Non terminal stage
                        Q_h_snext = self.StoQ_FApprox(h_snext)
                        target[0, a] = h_r + self.gamma * np.amax(Q_h_snext)

                    hist_states.append(h_s)
                    hist_target_reward.append(target)

                # print history
                self.StoQ_FApprox_train(np.array(hist_states).reshape(len(history), -1), np.array(hist_target_reward).reshape(len(history), -1))
                s = snext
            save_path = self.saver.save(self.sess, "/tmp/model.ckpt")
            epsilon = epsilon - 0.05 / 4
            print "episode: ", e, " and new epsilon ", epsilon


if __name__ == '__main__':
    movetraj = []
    agent = Agent()
    agent.learn()
