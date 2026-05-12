import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, state_shape, action_dim):
        self.capacity = capacity
        self.state_shape = state_shape
        self.action_shape = action_dim

        self.ptr = 0 # This points to the elements in the buffer where we want to append data
        self.size = 0 # This just keeps track of the amount of data

        # Pre allocating memory for the replay buffer
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32) # *state_shape unpacks the tuple so we end up with (capacity, 4, 96, 96)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32) 
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity # This will go to the end of the capacity and then start replacing data at the start of the array
        self.size = min(self.size + 1, self.capacity) # Updates size to the new amount of data

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size) # Returns an array of random indicies of the size batch size (with max index self.size)
        
        return (self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.dones[idxs]) # Returns the sample