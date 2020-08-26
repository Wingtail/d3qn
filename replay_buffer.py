import numpy as np
from queue import Queue
from segment_tree import DynamicSegmentTree

class ReplayBuffer():
    def __init__(self, mem_size, obs_shape):
        self.mem_size = mem_size
        self.obs_shape = obs_shape
        self.mem_cntr = 0
        self.init_memory()

    def init_memory(self):
        self.obs_memory = np.zeros((self.mem_size, *self.obs_shape), dtype=np.float32) #dtype should be equal to environment obs dtype
        self.new_obs_memory = np.zeros((self.mem_size, *self.obs_shape), dtype=np.float32) #dtype should be equal to environment obs dtype
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.done_memory = np.zeros(self.mem_size, dtype=np.bool)

    def get_index(self):
        return self.mem_cntr%self.mem_size

    def store_transition(self, obs, action, reward, new_obs, done):
        index = self.get_index()

        self.obs_memory[index] = obs
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_obs_memory[index] = new_obs
        self.done_memory[index] = done

        self.mem_cntr += 1
        return index

    def uniform_sampling(self, batch_size):
        filled_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(filled_mem, batch_size, replace=True)

        obs = self.obs_memory[batch]
        action = self.action_memory[batch]
        reward = self.reward_memory[batch]
        new_obs = self.new_obs_memory[batch]
        done = self.done_memory[batch]

        return obs, action, reward, new_obs, done,  None, None #obs, action, reward, new_obs, done, index, weights
        #index, weights are for Priority Experience Replay

    def sample_buffer(self, batch_size):
        return self.uniform_sampling(batch_size)

class PriorityExperienceReplay(ReplayBuffer):
    def __init__(self, mem_size, obs_shape, alpha=0.4, beta=0.4):
        super(PriorityExperienceReplay, self).__init__(mem_size, obs_shape)
        self.segment_tree = DynamicSegmentTree(alpha=alpha, beta=beta)

    def store_transition(self, obs, action, reward, new_obs, done):
        index = super().store_transition(obs, action, reward, new_obs, done)
        self.segment_tree.update_node(index)

    def sample_buffer(self, batch_size):
        index, weight = self.segment_tree.sample(batch_size)

        obs = self.obs_memory[index]
        action = self.action_memory[index]
        reward = self.reward_memory[index]
        new_obs = self.new_obs_memory[index]
        done = self.done_memory[index]

        return obs, action, reward, new_obs, done, index, weight

    def update_priority(self, index, priority):
        assert len(index) == len(priority)
        for i in range(len(index)):
            self.segment_tree.update_node(index[i], priority=priority[i])
