import numpy as np
from numba import njit

# ------------------------------------------------------------
# 1. Edge utilities (unchanged)
# ------------------------------------------------------------
def build_edge_order(weight_graph, name2idx):
    src, dst = [], []
    for pre in weight_graph:
        for post in weight_graph[pre]:
            src.append(name2idx[pre])
            dst.append(name2idx[post])
    return np.asarray(src, dtype=np.int32), np.asarray(dst, dtype=np.int32)

def vector_to_dense(genome, src_idx, dst_idx, N):
    W = np.zeros((N, N), dtype=np.float64)
    W[src_idx, dst_idx] = genome
    return W

# ------------------------------------------------------------
# 2. Numba kernels – LIF dynamics
# ------------------------------------------------------------
@njit
def _add_row(V, W_row):
    """In-place V += W_row (Numba helper)."""
    for j in range(V.size):
        V[j] += W_row[j]

@njit
def _update_potential(V, W, sensory_idx, spiked_prev, leak):
    # leak
    for i in range(V.size):
        V[i] *= leak

    # external sensory input
    for idx in sensory_idx:
        _add_row(V, W[idx])

    # synaptic input from last-step spikes
    for pre in range(V.size):
        if spiked_prev[pre]:
            _add_row(V, W[pre])

@njit
def _compute_spikes(V, threshold, muscle_mask):
    spiked = np.zeros(V.size, dtype=np.bool_)
    for i in range(V.size):
        if muscle_mask[i]:
            continue
        if np.abs(V[i]) > threshold:
            spiked[i] = True
            V[i] = 0.0         # reset after spike
    return spiked

@njit
def _motor_sum_and_clear(V, left_idx, right_idx):
    left = 0.0
    right = 0.0
    for i in left_idx:
        left += V[i]
        V[i] = 0.0
    for i in right_idx:
        right += V[i]
        V[i] = 0.0
    return left, right

@njit
def _lif_step(V, W, sensory_idx, threshold, leak,
              muscle_mask, left_idx, right_idx, spiked_prev):
    _update_potential(V, W, sensory_idx, spiked_prev, leak)
    spiked = _compute_spikes(V, threshold, muscle_mask)
    left, right = _motor_sum_and_clear(V, left_idx, right_idx)
    return left, right, spiked

# ------------------------------------------------------------
# 3. Drop-in class
# ------------------------------------------------------------
class WormConnectome:
    def __init__(self, weight_matrix, all_neuron_names,
                 threshold=30.0, tau_ms=20.0):
        """
        threshold : spike threshold (absolute value)
        tau_ms    : membrane time constant (ms).  dt is 1 ms per step.
        """
        from Worm_Env.weight_dict import dict as weight_graph
        from Worm_Env.weight_dict import mLeft, mRight, muscleList

        self.names  = all_neuron_names
        self.N      = len(self.names)
        self.threshold = float(threshold)

        # genome kept for GA
        self.weight_matrix = np.asarray(weight_matrix, dtype=np.float64)

        # name↔index
        self.name2idx = {n: i for i, n in enumerate(self.names)}

        # dense weight matrix with fixed edge order
        src_idx, dst_idx = build_edge_order(weight_graph, self.name2idx)
        self.W = vector_to_dense(self.weight_matrix, src_idx, dst_idx, self.N)

        # pre-computed masks / indices
        self.left_idx  = np.array([self.name2idx[n] for n in mLeft],  dtype=np.int32)
        self.right_idx = np.array([self.name2idx[n] for n in mRight], dtype=np.int32)
        muscle_prefixes = {n[:3] for n in muscleList}
        self.muscle_mask = np.array([name[:3] in muscle_prefixes
                                     for name in self.names], dtype=np.bool_)

        # LIF state -------------------------------------------------
        self.V      = np.zeros(self.N, dtype=np.float64)   # membrane potentials
        self.spiked = np.zeros(self.N, dtype=np.bool_)     # spikes from previous step
        self.leak   = np.exp(-1.0 / tau_ms)                # dt = 1 ms
        # -----------------------------------------------------------

        # sensory neuron indices
        self.touch_idx = np.array([self.name2idx[n] for n in
            ("FLPR","FLPL","ASHL","ASHR","IL1VL","IL1VR","OLQDL",
             "OLQDR","OLQVR","OLQVL")], dtype=np.int32)
        self.food_idx = np.array([self.name2idx[n] for n in
            ("ADFL","ADFR","ASGR","ASGL","ASIL","ASIR","ASJR","ASJL")],
            dtype=np.int32)
        self._edge_w   = self.weight_matrix              # 1-D genome vector
        self._edge_ptr = np.column_stack(                # (dummy, src, dst)
            (np.zeros_like(src_idx), src_idx, dst_idx)
        ).astype(np.int32)

    # inside WormConnectome.move()
    def move(self, dist, sees_food, *_unused):
        # ── 1) choose the set that actually injects current ──────────
        if 0 < dist < 100:
            sensory_idx = self.touch_idx
        elif sees_food:
            sensory_idx = self.food_idx
        else:
            sensory_idx = np.empty(0, dtype=np.int32)

        # ── 2) build a DISPLAY-ONLY mask that can include *both* sets ─
        self._sens_mask = np.zeros(self.N, dtype=np.bool_)
        if 0 < dist < 100:                 # head touches something
            self._sens_mask[self.touch_idx] = True
        if sees_food:                      # chemosensors detect food
            self._sens_mask[self.food_idx] = True

        # ── 3) run the original LIF step (UNCHANGED) ─────────────────
        left, right, spk = _lif_step(
            self.V, self.W, sensory_idx, self.threshold, self.leak,
            self.muscle_mask, self.left_idx, self.right_idx,
            self.spiked
        )
        self.spiked = spk                      # for the solver next tick

        # ── 4) what the viewer should consider a “spike” ─────────────
        self.spiked_vis = spk | self._sens_mask

        return left, right