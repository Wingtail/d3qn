import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import numpy as np
from replay_buffer import ReplayBuffer, PriorityExperienceReplay

class D3QN(keras.Model):
    def __init__(self, model, n_actions, lr=1e-4, is_noisy=False):
        super(D3QN, self).__init__()
        self.model = model
        self.is_noisy = is_noisy
        if is_noisy:
            self.V = keras.layers.DenseFlipout(1, activation=None)
            self.A = keras.layers.DenseFlipout(n_actions, activation=None)
        else:
            self.V = keras.layers.Dense(1, activation=None)
            self.A = keras.layers.Dense(n_actions, activation=None)

        self.opt = tf.keras.optimizers.Adam(lr=lr)

    def call(self, x):
        x = self.model(x)

        V = self.V(x)
        A = self.A(x)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))
        return Q

    def advantage(self, x, decision_mask=None):
        x = self.model.predict(x)
        A = self.A(x).numpy() #breaks tf auto gradient tape

        if decision_mask is not None:
            A = A+decision_mask
        return A

    def debug(self, x, A):
        V = self.V(x).numpy()
        Q = (V+(A-np.mean(A, axis=1, keepdims=True)))
        print("Q-value: ", Q)

    def loss_func(self, y_true, y_pred, weights=None):
        difference = y_true - y_pred
        abs_diff = tf.math.abs(difference)
        abs_diff = tf.math.reduce_sum(abs_diff, axis=-1)

        if weights is not None:
            abs_diff = weights*abs_diff

        return 0.5*tf.math.reduce_sum(abs_diff, axis=-1)

    def train_func(self, x, y, weights=None):
        with tf.GradientTape() as tape:
            q = self.call(x)
            loss = self.loss_func(y, q, weights=weights)
            if self.is_noisy:
                loss = loss+self.losses

        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

class Agent():
    def __init__(self, eval_model, next_model, lr, n_actions, obs_shape, epsilon=1.0, gamma=0.99, batch_size=64, epsilon_dec=1e-3, eps_min=0.01, mem_size=1000000, replace=100, test_mode=False, is_per=True, per_alpha=0.6, per_beta=0.4, save_weight_name="d3qn"):
        self.action_space = list(range(n_actions))
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps_dec = epsilon_dec
        self.eps_min = eps_min
        self.replace = replace
        self.batch_size = batch_size

        if test_mode:
            self.epsilon = 0
        else:
            self.epsilon = epsilon

        self.test_mode = test_mode
        self.save_weight_name = save_weight_name

        self.learn_step_counter = 0

        self.is_per = is_per

        if is_per:
            self.memory = PriorityExperienceReplay(mem_size, obs_shape, alpha=per_alpha, beta=per_beta)
        else:
            self.memory = ReplayBuffer(mem_size, obs_shape)

        self.q_eval = D3QN(eval_model, n_actions, lr=lr)
        self.q_next = D3QN(next_model, n_actions, lr=lr)

        #For tf-keras formality
        self.q_eval.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse') #actual loss is not mse
        self.q_next.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

    def store_transition(self, obs, action, reward, new_obs, done):
        return self.memory.store_transition(obs, action, reward, new_obs, done)

    def choose_action(self, obs, decision_mask=None):
        if np.random.random() < self.epsilon:
            valid_actions = np.argwhere(decision_mask==0)[:,0]
            action = np.random.choice(valid_actions)
        else:
            if len(obs.shape) < 3:
                obs = np.array([obs]) #Adding batch dimension if there is no batch dimension
            actions = self.q_eval.advantage(obs, decision_mask=decision_mask)
            action = np.squeeze(np.argmax(actions, axis=1))

        if not self.test_mode:
            self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)

        return action

    def learn(self):
        obs, actions, rewards, new_obs, dones, _, _ = self.memory.sample_buffer(self.batch_size)

        if self.learn_step_counter % self.replace == 0:
            if self.learn_step_counter == 0:
                self.q_eval.predict(obs)
                self.q_next.predict(new_obs)
            self.q_next.set_weights(self.q_eval.get_weights())

        q_target = self.q_eval.predict(obs)
        q_next = self.q_next.predict(new_obs)

        if self.is_per:
            td_error = np.copy(q_target)

        max_actions = np.argmax(self.q_eval.predict(new_obs), axis=1)

        for idx, terminal in enumerate(dones):
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma*q_next[idx, max_actions[idx]]*(1-int(dones[idx]))

        self.q_eval.train_func(obs, q_target, weights=weights)

        if self.is_per:
            td_error = np.sum(np.abs(q_target-td_error), axis=-1)
            self.memory.update_priority(index, td_error)

        self.learn_step_counter += 1

    def save_model(self, model_name=None):
        if model_name is None:
            model_name = self.save_weight_name
        self.q_eval.save_weights(model_name)

    def load_model(self, model_name=None):
        if model_name is None:
            model_name = self.save_weight_name
        self.q_eval.load_weights(model_name)
