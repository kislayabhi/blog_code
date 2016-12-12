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
        # self.experience_buffer = deque()

        self.minibatch_size = 128



        # self.saver.restore(self.sess, "newmodel1.ckpt")

        self.logs_path = '/tmp/tensorflow_logs/example17'
        # Create a summary to monitor loss tensor
        tf.scalar_summary("lossloss", self.loss)
        # Merge all summaries into a single op
        self.merged_summary_op = tf.merge_all_summaries()

    # Maps state to Q values
    def StoQ_FApprox(self, state):
        return self.sess.run(self.Qout, feed_dict={self.inputs1: state})

    def StoQ_FApprox_train(self, current_input, target_output):
        return self.sess.run([self.train_op, self.merged_summary_op], feed_dict={self.inputs1: current_input, self.nextQ: target_output})

    def epsilon_greedy(self, qval):
        ran = random.randint(0, 100) / 100.0
        if (ran < self.epsilon):
            return random.randint(0, qval.shape[1] - 1)
        else:
            return np.argmax(qval)

    def learn(self):
        self.sess = tf.Session()
        self.sess.run(self.init)

        # op to write logs to Tensorboard
        summary_writer = tf.train.SummaryWriter(self.logs_path, graph=tf.get_default_graph())

        replay = []

        self.epsilon = 0.9
        total_time = 0
        for e in range(1, self.episodes_length+1):
            r, s = self.game.frame_step(2)

            # while (r != 500):
            while True :

                # print "current episode: ", e
                orgstate = s.copy()

                # Choose a from s using policy derived from Q (epsilon-greedy)
                Qs = self.StoQ_FApprox(s)

                a = self.epsilon_greedy(Qs)


                # Take action a, observe r and snext.
                r, s = self.game.frame_step(a)

                # Tweak the Function approximator to behave according to the
                # current rewards.

                # Instead of training from the current state and target_reward,
                # we will train using a mini-batch.
                replay.append((orgstate.copy(), a, r, s.copy()))
                # history = self.experience_replay((s.copy(), a, r, snext.copy()))

                if(r==500):
                    save_path = self.saver.save(self.sess, "/tmp/model.ckpt")
                    print "episode: ", e, " and new epsilon ", self.epsilon, "len_experience buffer", len(replay)
                #     break

                if(len(replay)>=10000 and self.epsilon > 0.1):
                    self.epsilon = self.epsilon - 1. / total_time


                if(len(replay)>self.minibatch_size):
                    if(len(replay)>self.experience_memory):
                        replay.pop(0)
                    minibatch = random.sample(replay, self.minibatch_size)

                    hist_states=[]
                    hist_target_reward=[]

                    for experience in minibatch:
                        h_s = experience[0]
                        h_a = experience[1]
                        h_r = experience[2]
                        h_snext = experience[3]

                        oldq = self.StoQ_FApprox(h_s)
                        newq = self.StoQ_FApprox(h_snext)
                        maxq = np.max(newq)
                        target = oldq.copy()

                        # print "target", target

                        if(h_r == 500):
                            # Terminal stage
                            target[0][h_a] = h_r
                        else:
                            # Non terminal stage
                            target[0][h_a] = h_r + self.gamma * maxq

                        hist_states.append(h_s.copy())
                        hist_target_reward.append(target.copy())

                    hist_states = np.array(hist_states).reshape(self.minibatch_size, -1)
                    hist_target_reward = np.array(hist_target_reward).reshape(self.minibatch_size, -1)

                    _, summary = self.StoQ_FApprox_train(hist_states, hist_target_reward)
                    summary_writer.add_summary(summary, total_time)
                total_time+=1




    def exp_replay(self,replay, minibatch):
        x, y = [], []
        # print "len exp", len(minibatch)
        for exp in minibatch:
            h_s = exp[0]
            h_a = exp[1]
            h_r = exp[2]
            h_snext = exp[3]

            # get old state qval
            # (changed) oldq = self.nn.predict(old_state)
            # oldq = self.sess.run(self.Qout, feed_dict={self.inputs1: old_state})
            oldq = self.StoQ_FApprox(h_s)
            # get new state qval
            # (changed) newq = self.nn.predict(new_state)
            # newq = self.sess.run(self.Qout, feed_dict={self.inputs1: new_state})
            newq = self.StoQ_FApprox(h_snext)
            # best action qval
            maxq = np.max(newq)
            target = oldq.copy()

            # print "newq", newq

            # print target
            if (h_r == 500):
                # terminal state
                target[0][h_a] = h_r
            else:
                # non-terminal state
                target[0][h_a] = h_r + self.gamma * maxq
            x.append(h_s.copy())
            y.append(target.copy())
            # print target

        x = np.array(x).reshape(self.minibatch_size, -1)
        y = np.array(y).reshape(self.minibatch_size, -1)

        # print "x", x
        # print "y", y
        # raw_input("wait...")
        # self.nn.train(x, y, self.batchsize)
        # _, summary = self.sess.run([self.train_op, self.merged_summary_op], feed_dict={self.inputs1: x, self.nextQ: y})
        _, summary = self.StoQ_FApprox_train(x, y)

        return summary


if __name__ == '__main__':
    movetraj = []
    agent = Agent()
    agent.learn()
