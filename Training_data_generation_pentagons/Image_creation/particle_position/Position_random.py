import numpy as np

def position():
    margin = 30  # px, increase if shapes get cut off at edges
    pos_sampler = lambda: (
    np.random.uniform(margin, 2048 - margin),
    np.random.uniform(margin, 2048 - margin),
)
    return pos_sampler

def rotation():
    # random rotations (radians)
    rot_sampler = lambda: np.random.uniform(0, 2*np.pi)
    return rot_sampler

