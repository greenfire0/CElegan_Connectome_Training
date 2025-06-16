import numpy as np
from numba import njit

# ---------------------------------------------------------------
# Helper: build dense (N×N) weight matrix from 1-D genome vector
# ---------------------------------------------------------------
def vector_to_dense(weight_vec, weight_graph, name2idx, N):
    W = np.zeros((N, N), dtype=np.float64)
    idx = 0
    for pre, posts in weight_graph.items():     # insertion order preserved
        i = name2idx[pre]
        for post in posts:
            j = name2idx[post]
            W[i, j] = weight_vec[idx]
            idx += 1
    if idx != len(weight_vec):
        raise ValueError("Genome length doesn’t match number of synapses")
    return W


# ---------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------
@njit
def _dendrite_accumulate(post, W, src_idx, dst_state):
    """Add weighted outputs of one presynaptic neuron to all posts."""
    post[:, dst_state] += W[src_idx]

@njit
def _motor_control(post, m_left_idx, m_right_idx, state):
    left = 0.0
    right = 0.0
    for i in m_left_idx:
        left  += post[i, state]
        post[i, state] = 0.0
    for i in m_right_idx:
        right += post[i, state]
        post[i, state] = 0.0
    return left, right


# ---------------------------------------------------------------
#  Drop-in class  (constructor unchanged)
# ---------------------------------------------------------------
class WormConnectome:
    """
    Dense-array implementation – same API as your original class.
    """

    # ----- constructor signature is unchanged ------------------
    def __init__(self, weight_matrix, all_neuron_names, threshold=30):
        from Worm_Env.weight_dict import dict as weight_graph
        from Worm_Env.weight_dict import mLeft, mRight, muscleList, muscles

        self.names = all_neuron_names
        self.N     = len(self.names)

        # name ↔ index maps
        self.weight_matrix = np.asarray(weight_matrix, dtype=np.float64)
        self.name2idx = {n: i for i, n in enumerate(self.names)}

        # dense weights built from the original connection order
        self.W = vector_to_dense(
            np.asarray(weight_matrix, dtype=np.float64),
            weight_graph,
            self.name2idx,
            self.N
        )

        # index arrays for fast look-ups inside njit code
        self.m_left_idx  = np.array([self.name2idx[n] for n in mLeft],  dtype=np.int32)
        self.m_right_idx = np.array([self.name2idx[n] for n in mRight], dtype=np.int32)

        # postsynaptic buffer: shape (N, 2); we just flip columns
        self.post = np.zeros((self.N, 2), dtype=np.float64)
        self.this_state = 0
        self.next_state = 1
        self.threshold  = threshold

    # ----- behaviour identical to original .move ----------------
    def move(self, dist, sees_food, mLeft, mRight, muscleList, muscles):
        """
        Keeps the same call signature you already use from GA code.
        """
        # Stimulate sensory neurons (same sets you used before)
        if 0 < dist < 100:
            for name in ("FLPR", "FLPL", "ASHL", "ASHR",
                         "IL1VL", "IL1VR", "OLQDL", "OLQDR", "OLQVR", "OLQVL"):
                idx = self.name2idx[name]
                _dendrite_accumulate(self.post, self.W, idx, self.next_state)

        elif sees_food:
            for name in ("ADFL", "ADFR", "ASGR", "ASGL",
                         "ASIL", "ASIR", "ASJR", "ASJL"):
                idx = self.name2idx[name]
                _dendrite_accumulate(self.post, self.W, idx, self.next_state)

        # Fire all neurons above threshold
        active = np.abs(self.post[:, self.this_state]) > self.threshold
        if np.any(active):
            # posts[:, next] += Wᵀ @ active_mask
            self.post[:, self.next_state] += self.W[active].T @ np.ones(np.sum(active))

        # Motor control – sum & zero muscle rows
        left, right = _motor_control(
            self.post, self.m_left_idx, self.m_right_idx, self.next_state
        )

        # Advance time: copy column pointer & zero next
        self.post[:, self.this_state] = self.post[:, self.next_state]
        self.post[:, self.next_state].fill(0.0)
        self.this_state, self.next_state = self.next_state, self.this_state

        return (left, right)
