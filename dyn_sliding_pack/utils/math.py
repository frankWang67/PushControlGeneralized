import casadi as cs
import numpy as np
import scipy as sp


def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    rot_mat = np.array([[c, -s],
                        [s,  c]])
    return rot_mat
