
from Worm_Env.weight_dict import dict
import copy
from numba import njit, typed, types
from numba.typed import List
import numpy as np


@njit
def dendrite_accumulate(post_synaptic, combined_weights, neuron_name, next_state):
    for neuron in combined_weights[neuron_name].keys():
        weight = combined_weights[neuron_name][neuron]
        post_synaptic[neuron][next_state] += weight

@njit
def motor_control(post_synaptic, mLeft, mRight, muscleList, next_state):
    accumleft = 0
    accumright = 0
    for muscle in muscleList:
        if muscle in mLeft:
            accumleft += post_synaptic[muscle][next_state]
            post_synaptic[muscle][next_state] = 0
        elif muscle in mRight:
            accumright += post_synaptic[muscle][next_state]
            post_synaptic[muscle][next_state] = 0
    return accumleft, accumright

@njit
def run_connectome(post_synaptic, combined_weights, threshold, muscles, muscleList, mLeft, mRight, thisState, nextState):
    for ps in post_synaptic.keys():
        if ps[:3] not in muscles and abs(post_synaptic[ps][thisState]) > threshold:
            dendrite_accumulate(post_synaptic, combined_weights, ps, nextState)
            post_synaptic[ps][nextState] = 0
    
    movement = motor_control(post_synaptic, mLeft, mRight, muscleList, nextState)
    
    for ps in post_synaptic.keys():
        post_synaptic[ps][thisState] = post_synaptic[ps][nextState]
    
    thisState, nextState = nextState, thisState
    return movement, thisState, nextState

class WormConnectome:
    def __init__(self, weight_matrix, all_neuron_names, threshold=30):
        self.combined_weights = typed.Dict.empty(
            key_type=types.unicode_type,
            value_type=types.DictType(types.unicode_type, types.float64)
        )
        self.weight_matrix = weight_matrix.astype(np.float64)
        for neuron in dict:
            self.combined_weights[neuron] = typed.Dict.empty(
                key_type=types.unicode_type,
                value_type=types.float64
            )
            for post_neuron in dict[neuron]:
                self.combined_weights[neuron][post_neuron] = 0.0

        index = 0
        for pre_neuron, connections in self.combined_weights.items():
            for post_neuron in connections:
                self.combined_weights[pre_neuron][post_neuron] = weight_matrix[index]
                index += 1
        
        self.all_neuron_names = all_neuron_names
        self.postSynaptic = typed.Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float64[:]
        )
        self.threshold = threshold
        self.thisState = 0  # Initialize thisState
        self.nextState = 1  # Initialize nextState
        
        self.create_post_synaptic()
        
    def create_post_synaptic(self):
        for neuron in self.all_neuron_names:
            self.postSynaptic[neuron] = np.zeros(2)
    
    def move(self, dist, sees_food, mLeft, mRight, muscleList, muscles):
        # Convert lists to numba.typed.List
        mLeft_typed = List(mLeft)
        mRight_typed = List(mRight)
        muscleList_typed = List(muscleList)
        muscles_typed = List(muscles)
        
        if 0 < dist < 100:
            for dneuron in ["FLPR", "FLPL", "ASHL", "ASHR", "IL1VL", "IL1VR", "OLQDL", "OLQDR", "OLQVR", "OLQVL"]:
                dendrite_accumulate(self.postSynaptic, self.combined_weights, dneuron, self.nextState)
        elif sees_food:
            for dneuron in ["ADFL", "ADFR", "ASGR", "ASGL", "ASIL", "ASIR", "ASJR", "ASJL"]:
                dendrite_accumulate(self.postSynaptic, self.combined_weights, dneuron, self.nextState)
        
        movement, self.thisState, self.nextState = run_connectome(
            self.postSynaptic,
            self.combined_weights,
            self.threshold, muscles_typed,
            muscleList_typed,
            mLeft_typed,
            mRight_typed,
            self.thisState,
            self.nextState
        )
        
        return movement