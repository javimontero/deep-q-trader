'''
Author: Javier Montero 
License: MIT
'''

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.layers.core import Flatten
from keras.optimizers import Adam
from keras import backend as K
from time import time

import numpy as np
import random

class Agent:
    def __init__(self, state_size, action_size, memory_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.98    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9
        self.learning_rate = 0.0005
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    """Huber loss for Q Learning

    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # DRQN Architecture as "Financial trading as a game: a Depp RL approach"
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(LSTM(256, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), None
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]), np.max(act_values[0]) # return action, qvalue
        #act_values[0][np.argmax(act_values[0])] # returns action and qvalue

    def act_legal (self, state):
        p = np.argmax (state[0:2]) # get position feature from state
        
        if np.random.rand() <= self.epsilon: # Enter exploration
            if p == 0 : return np.random.choice ((1,2)), None
            if p == 1 : return np.random.choice ((0,1,2)), None
            if p == 2 : return np.random.choice ((0,1)), None
        # Enter exploitation
        act_values = self.model.predict(state)
        a = np.argmax(act_values[0])
        if (a == p and a!=1): # ilegal action return second most valued index and q-value
            return np.argsort(-act_values[0])[1], act_values[0][np.argsort(-act_values[0])[1]]
        else:
            return np.argmax(act_values[0]), np.max(act_values[0]) # return action, qvalue

    def replay(self, batch_size):
        if len(self.memory) < batch_size: 
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)

            #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

