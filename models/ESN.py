# the Echo State Network is a specific architecture in Reservoir Computing
# I implemented this by following https://martinuzzifrancesco.github.io/posts/a-brief-introduction-to-reservoir-computing/
# (it's not a great post to be honest but gives you the gist of it and includes useful references)

from warnings import warn
import numpy as np


class ESN:

    def spectral_normalising(self, W: np.ndarray) -> np.ndarray:
        """
        To maintain the Echo State Property, we need the spectral radius to be < 1.
        Intuition:
        - Think of the state update equation.
        - At time t the state has a term: (W @ W @ ... @ W) @ states[0] i.e., t applications of W.
        - If the spectral radius of W is < 1, this matrix power will shrink to zero as t increases.
        - This means the effect of previous states slowly dies away.
        """
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        W *= self.spectral_radius / radius
        return W

    def init_W(self,
               layer: str,
               mixing: str,
               sigma: float=1.,
               p: float=0.5) -> np.ndarray:
        """
        Initialise frozen weight matrices.
    
        layer: "input" or "reservoir" weights
        mixing:
        - "none": each output neuron receives information from only one input neuron
        - "random": each output neuron receives information from a random subset of input neurons
        - "full": each output neuron receives infromation from all input neurons
        - "moc": each output neuron receives information from one input neuron *and* the moc strength signal
        """
        if layer not in ["input", "reservoir"]: raise Exception("'layer' must be 'input' or 'reservoir'.")
        if mixing not in ["none", "random", "full", "moc"]: raise Exception("'mixing' must be 'none', 'random', 'full', or 'moc'.")
        if layer == "input" and mixing == "none": assert self.n_reservoir % self.n_input == 0
        if mixing == "full": p = 0.

        # W_in is (n_reservoir, n_input) and W_reservoir is (n_reservoir, n_reservoir)
        height = self.n_reservoir
        width = self.n_input if layer == "input" else self.n_reservoir

        if mixing == "none":
            # input i is connected to n_reservoir / n_input reservoir nodes
            # this means each reservoir node only receive new information from one input signal
            # nonzero weights are uniformly sampled from [-sigma, sigma]
            blocks = []
            block_height = int(height / width)
            for i in range(width):
                block = np.zeros((block_height, width))
                entries = np.random.rand(block_height)*sigma*2 - sigma
                block[:, i] = entries
                blocks.append(block)
            W = np.vstack(blocks)
        elif mixing in ["random", "full"]:
            W = (np.random.rand(height, width) > p) * 1.
            # nonzero weights are uniformly sampled from [-sigma, sigma]
            nz_W = np.random.rand(height, width)*sigma*2 - sigma
            W *= nz_W
        elif mixing == "moc":
            warn("'moc' mixing assumes moc strength is provided in the *last* column of your data.")
            blocks = []
            block_height = int(height / width)
            for i in range(width):
                block = np.zeros((block_height, width))
                # input signal
                entries = np.random.rand(block_height)*sigma*2 - sigma
                block[:, i] = entries
                # moc strength
                entries = np.random.rand(block_height)*sigma*2 - sigma
                block[:, -1] = entries
                blocks.append(block)
            W = np.vstack(blocks)

        if layer == "reservoir": W = self.spectral_normalising(W)
        return W

    def __init__(self, 
                 n_input: int, 
                 n_reservoir: int, 
                 input_mixing: str="none",
                 reservoir_mixing: str="random",
                 sigma: float=1.,
                 p: float=0.5,
                 alpha: float=0.5,
                 beta: float=0.1,
                 spectral_radius: float=0.95,
                 warmup: int=0):
        """
        Standard Echo State Network (reservoir computing) with leaky updating.
        See https://martinuzzifrancesco.github.io/posts/a-brief-introduction-to-reservoir-computing/ and references.

        n_input: number of input signals (will also equal the number of output signals)
        n_reservoir: size of the reservoir (hidden state)
        input_mixing: how to mix information between input signals - see init_W
        reservoir_mixing: how to mix information between hidden state signals - see init_W
        sigma: nonzero weights will be initialised to values on the interval [-sigma, sigma]. defaults to 1 for reservoir weights
        p: during 'random' mixing, edges between neurons are randomly sampled from a uniform distribution w/ P(inclusion) = p 
        alpha: leaking rate - controls the weighted updating of the reservoir between previous state and new input
        beta: regularisation parameter for Tikhonov Regularisation - used for a closed-form solution for the output weights
        spectral_radius: desired spectral radius of the reservoir weights - see spectral_normalising
        warmup: reservoir is initialised at zero. to combat this, a number of sampled can be used as a warmup before fitting
        """
        warn("Data must be sorted by time for ESN.")
        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.alpha = alpha
        self.beta = beta
        if spectral_radius >= 1: raise ValueError("'spectral_radius' must be < 1 to maintain the Echo State Property.")
        self.spectral_radius = spectral_radius
        self.warmup = warmup

        self.W_in = self.init_W(layer="input", mixing=input_mixing, sigma=sigma, p=p)
        self.W = self.init_W(layer="reservoir", mixing=reservoir_mixing, sigma=1., p=p)

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