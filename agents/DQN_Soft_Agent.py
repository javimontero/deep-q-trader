'''
towardsdatascience implementation of DQN
https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c

'''
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from keras import initializers

from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size, memory_size):
        self.state_size = state_size
        self.action_size = action_size
        #self.env     = env
        self.memory  = deque(maxlen=memory_size)
        
        self.gamma = 0.98
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.tau = .125

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
            error = y_true - y_pred
            cond  = K.abs(error) <= clip_delta

            squared_loss = 0.5 * K.square(error)
            quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

            return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def create_model(self):
        model   = Sequential()
        #state_shape  = self.env.observation_space.shape
        model.add(Dense(36, input_dim=self.state_size, activation="relu",
            kernel_initializer=initializers.RandomNormal(stddev=0.001),
            bias_initializer='zeros'))
        model.add(Dense(64, activation="relu",
            kernel_initializer=initializers.RandomNormal(stddev=0.001),
            bias_initializer='zeros'))
        model.add(Dense(24, activation="relu",
            kernel_initializer=initializers.RandomNormal(stddev=0.001),
            bias_initializer='zeros'))
        model.add(Dense(self.action_size))
        #model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate))

        return model
        # model.add(Dense(self._units, input_dim=self._state_size, activation='relu',
        #                kernel_initializer=initializers.RandomNormal(stddev=0.001, seed=3456),
        #                bias_initializer='zeros'))
        # model.add(Dense(self._units, activation='relu',
        #                kernel_initializer=initializers.RandomNormal(stddev=0.001, seed=3456),
        #                bias_initializer='zeros'))
        # model.add(Dense(self._action_size, activation='linear',
        #                kernel_initializer=initializers.RandomNormal(stddev=0.001, seed=3456),
        #                bias_initializer='zeros'))
        # model.compile(loss=self.__huber_loss, optimizer="adam")
    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size), None
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]), np.max(act_values[0]) # return action, qvalue

    def act_legal (self, state):
        # Position is one hot encoded within state
        if state[0][0] == 1: p = 0
        if state[0][1] == 1: p = 1
        if state[0][2] == 1: p = 2
        
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


    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self, batch_size):
        #batch_size = 32
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

    def update_target_model(self): # wrapper for target_train
        self.target_train()