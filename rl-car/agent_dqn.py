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

        self.time_per_epoch = tf.placeholder(tf.float32, shape=())
        self.init = tf.initialize_all_variables()
        self.saver = tf.train.Saver()

        # Generate the requisite buffer
        self.experience_memory = 10000
        self.replay = []

        self.minibatch_size = 128
        self.epsilon = 0.9

        # self.saver.restore(self.sess, "newmodel1.ckpt")
        self.logs_path = '/tmp/tensorflow_logs/example21'

        # Create a summary to monitor loss tensor
        tf.scalar_summary("timeperepoch", self.time_per_epoch)

        # Merge all summaries into a single op
        self.merged_summary_op = tf.merge_all_summaries()

    # Maps state to Q values
    def StoQ_FApprox(self, state):
        return self.sess.run(self.Qout, feed_dict={self.inputs1: state})

    def StoQ_FApprox_train(self, current_input, target_output):
        self.sess.run([self.train_op], feed_dict={self.inputs1: current_input, self.nextQ: target_output})

    def epsilon_greedy(self, qval):
        ran = random.random()
        if (ran < self.epsilon):
            return random.randint(0, qval.shape[1] - 1)
        else:
            return np.argmax(qval)

    def experience_replay(self, current_state, current_action, current_reward, next_state):

        self.replay.append((current_state, current_action, current_reward, next_state))
        states_history=[]
        targetQ_history=[]

        if(len(self.replay) > self.minibatch_size):
            if(len(self.replay) > self.experience_memory):
                self.replay.pop(0)
            minibatch = random.sample(self.replay, self.minibatch_size)

            for experience in minibatch:
                hState = experience[0]
                hAction = experience[1]
                hReward = experience[2]
                hNextState = experience[3]

                oldq = self.StoQ_FApprox(hState)
                newq = self.StoQ_FApprox(hNextState)
                maxq = np.max(newq)
                target = oldq.copy()

                if(hReward == 500):
                    # Terminal stage
                    target[0][hAction] = hReward
                else:
                    # Non terminal stage
                    target[0][hAction] = hReward + self.gamma * maxq

                states_history.append(hState.copy())
                targetQ_history.append(target.copy())

            states_history = np.array(states_history).reshape(self.minibatch_size, -1)
            targetQ_history = np.array(targetQ_history).reshape(self.minibatch_size, -1)

        return states_history, targetQ_history

    def learn(self):
        self.sess = tf.Session()
        self.sess.run(self.init)

        # op to write logs to Tensorboard
        summary_writer = tf.train.SummaryWriter(self.logs_path, graph=tf.get_default_graph())

        total_time = 0
        for episode_number in range(1, self.episodes_length+1):

            reward, state = self.game.frame_step(2)
            epoch_time = 0
            while True :
                orgstate = state.copy()

                # Choose a from s using policy derived from Q (epsilon-greedy)
                Qs = self.StoQ_FApprox(state)
                action = self.epsilon_greedy(Qs)
                # Take action action, observe reward and snext.
                reward, state = self.game.frame_step(action)

                states_history, targetQ_history = self.experience_replay(orgstate.copy(), action, reward, state.copy())

                if(len(self.replay) > self.minibatch_size):
                    self.StoQ_FApprox_train(states_history, targetQ_history)

                if(len(self.replay) >= 10000 and self.epsilon > 0.1):
                    self.epsilon = self.epsilon - 1. / total_time

                epoch_time += 1
                total_time += 1

                if(reward == 500):
                    save_path = self.saver.save(self.sess, "/tmp/model.ckpt")
                    print "episode: ", episode_number, " and new epsilon ", self.epsilon, "len_experience buffer", len(self.replay), "total time", total_time, "epoch time", epoch_time
                    # This game ends now.
                    break;

            summary = self.sess.run(self.merged_summary_op, feed_dict={self.time_per_epoch: epoch_time})
            summary_writer.add_summary(summary, episode_number)

if __name__ == '__main__':
    movetraj = []
    agent = Agent()
    agent.learn()
