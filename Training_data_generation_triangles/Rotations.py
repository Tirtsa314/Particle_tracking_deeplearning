import numpy as np


def rotation_matrix_xyz(rx, ry, rz):
    """
    Standard rotation convention.

    rx: rotation around x-axis
    ry: rotation around y-axis
    rz: rotation around z-axis

    Object -> image/world rotation:
        R = Rz @ Ry @ Rx

    Because y points downward in image coordinates, positive rz appears
    clockwise when plotted with imshow.
    """
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([
        [1,  0,   0],
        [0, cx, -sx],
        [0, sx,  cx],
    ], dtype=float)

    Ry = np.array([
        [ cy, 0, sy],
        [  0, 1,  0],
        [-sy, 0, cy],
    ], dtype=float)

    Rz = np.array([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [ 0,   0, 1],
    ], dtype=float)

    return Rz @ Ry @ Rx

