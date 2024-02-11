# the Echo State Network is a specific architecture in Reservoir Computing
# I implemented this by following https://martinuzzifrancesco.github.io/posts/a-brief-introduction-to-reservoir-computing/

import numpy as np


class ESN:
    def __init__(self, 
                 n_input: int, 
                 n_reservoir: int, 
                 input_mixing: str="none",
                 reservoir_mixing: str="none",
                 sigma: float=1.,
                 p: float=0.5,
                 alpha: float=0.5,
                 beta: float=0.1,
                 warmup: int=0):
        self.n_reservoir = n_reservoir
        self.alpha = alpha
        self.beta = beta
        self.warmup = warmup
        
        # input i is connected to n_reservoir / n_input reservoir nodes
        # this means each reservoir node only receive new information from one input signal
        # nonzero weights are uniformly sampled from [-sigma, sigma]
        blocks = []
        assert n_reservoir % n_input == 0
        block_height = int(n_reservoir / n_input)
        for i in range(n_input):
            block = np.zeros((block_height, n_input))
            entries = np.random.rand(block_height)*sigma - (sigma/2)
            block[:, i] = entries
            blocks.append(block)
        self.W_in = np.vstack(blocks)

        # reservoir connections are connected in an Erdős–Rényi config
        # (this just means they're independently randomly sampled)
        self.W = (np.random.rand(n_reservoir, n_reservoir) > p) * 1.
        # nonzero weights are uniformly sampled from [-1, 1]
        nz_W = np.random.rand(n_reservoir, n_reservoir)*2 - 1
        self.W *= nz_W

        # output weights are fit using the normal equations, we initialise now
        self.W_out = np.zeros((n_input, n_reservoir))

    def update(self, X: np.ndarray) -> np.ndarray:
        states = [np.zeros(self.n_reservoir)]
        for u in X:
            prev_state = states[-1]
            cur_state = (1-self.alpha)*prev_state + self.alpha*np.tanh(self.W @ prev_state + self.W_in @ u)
            states.append(cur_state)
        return np.array(states[1:])

    def fit(self, X: np.ndarray, Y: np.ndarray):
        states = self.update(X)
        states = states[self.warmup:]
        Y = Y[self.warmup:]
        # solve the system Y = W_out @ states using the normal equations (regularised)
        self.W_out = Y.T @ states @ np.linalg.inv(states.T @ states + self.beta*np.eye(self.n_reservoir))

    def predict(self, X: np.ndarray) -> np.ndarray:
        states = self.update(X)
        out = states @ self.W_out.T
        return out