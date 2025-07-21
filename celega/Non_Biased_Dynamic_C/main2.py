from Worm_Env.connectome import WormConnectome
from graphs.graph_worm import Genetic_Dyn_Video
import numpy as np
import ray


gv = Genetic_Dyn_Video()        # defaults: patterns 0 & 1
gv.run_video_simulation()