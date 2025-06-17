import numpy as np
from numba import njit

# ------------------------------------------------------------
# 1. build edge order EXACTLY like the old typed.Dict walk
# ------------------------------------------------------------
def build_edge_order(weight_graph, name2idx):
    src = []
    dst = []
    for pre in weight_graph:                 # insertion order
        for post in weight_graph[pre]:
            src.append(name2idx[pre])
            dst.append(name2idx[post])
    return np.asarray(src, dtype=np.int32), np.asarray(dst, dtype=np.int32)

def vector_to_dense(genome, src_idx, dst_idx, N):
    W = np.zeros((N, N), dtype=np.float64)
    W[src_idx, dst_idx] = genome
    return W

# ------------------------------------------------------------
# 2. Numba kernels (sensory line removed)
# ------------------------------------------------------------
@njit
def _inject_sensory(post, W, sensory_idx, nxt):
    for idx in sensory_idx:
        post[:, nxt] += W[idx]          # add entire row
        # DO NOT zero post[idx,nxt] – matches original behaviour

@njit
def _one_hop_propagate(post, W, threshold, muscle_mask, cur, nxt):
    N = post.shape[0]
    for i in range(N):
        if muscle_mask[i]:
            continue
        if np.abs(post[i, cur]) > threshold:
            post[:, nxt] += W[i]
            post[i, nxt] = 0.0          # spike only once this tick

@njit
def _motor_sum_and_clear(post, left_idx, right_idx, col):
    left = 0.0
    right = 0.0
    for i in left_idx:
        left  += post[i, col]
        post[i, col] = 0.0
    for i in right_idx:
        right += post[i, col]
        post[i, col] = 0.0
    return left, right

@njit
def _step_once(post, W, threshold,
               muscle_mask, left_idx, right_idx,
               sensory_idx, cur, nxt):
    _inject_sensory(post, W, sensory_idx, nxt)
    _one_hop_propagate(post, W, threshold, muscle_mask, cur, nxt)
    left, right = _motor_sum_and_clear(post, left_idx, right_idx, nxt)

    # copy next→current (no zeroing)
    post[:, cur] = post[:, nxt]
    return left, right

# ------------------------------------------------------------
# 3. Drop-in class (API unchanged)
# ------------------------------------------------------------
class WormConnectome:
    def __init__(self, weight_matrix, all_neuron_names, threshold=30):
        from Worm_Env.weight_dict import dict      as weight_graph
        from Worm_Env.weight_dict import mLeft, mRight, muscleList

        self.names  = all_neuron_names
        self.N      = len(self.names)
        self.threshold = threshold

        # genome kept for GA
        self.weight_matrix = np.asarray(weight_matrix, dtype=np.float64)

        # name↔index
        self.name2idx = {n: i for i, n in enumerate(self.names)}

        # --- build W with the *same* edge order as old code ----
        src_idx, dst_idx = build_edge_order(weight_graph, self.name2idx)
        self.W = vector_to_dense(self.weight_matrix, src_idx, dst_idx, self.N)

        # pre-computed masks / indices
        self.left_idx  = np.array([self.name2idx[n] for n in mLeft],  dtype=np.int32)
        self.right_idx = np.array([self.name2idx[n] for n in mRight], dtype=np.int32)
        muscle_prefixes = {n[:3] for n in muscleList}
        self.muscle_mask = np.array([name[:3] in muscle_prefixes
                                     for name in self.names], dtype=np.bool_)

        # double buffer
        self.post   = np.zeros((self.N, 2), dtype=np.float64)
        self.curcol = 0
        self.nextcol = 1

        # sensory neuron index arrays
        self.touch_idx = np.array([self.name2idx[n] for n in
            ("FLPR","FLPL","ASHL","ASHR","IL1VL","IL1VR","OLQDL","OLQDR","OLQVR","OLQVL")],
            dtype=np.int32)
        self.food_idx = np.array([self.name2idx[n] for n in
            ("ADFL","ADFR","ASGR","ASGL","ASIL","ASIR","ASJR","ASJL")],
            dtype=np.int32)

    def move(self, dist, sees_food, *_unused):
        sensory_idx = self.touch_idx if (0 < dist < 100) else (
                      self.food_idx  if sees_food else np.empty(0, dtype=np.int32))

        left, right = _step_once(
            self.post, self.W, self.threshold,
            self.muscle_mask, self.left_idx, self.right_idx,
            sensory_idx,
            self.curcol, self.nextcol
        )
        self.curcol, self.nextcol = self.nextcol, self.curcol
        return (left, right)