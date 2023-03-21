import copy
import numpy as np
import os
import sys
from scipy.interpolate import interp1d
from autolab_core import RigidTransform


class HiddenPrints:
    def __init__(self, activated=True):
        self.activated = activated
        self.original_stdout = None

    def open(self):
        sys.stdout.close()
        sys.stdout = self.original_stdout

    def close(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __enter__(self):
        if self.activated:
            self.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.activated:
            self.open()

def rotation_matrix(theta):
    """
    Get the rotation matrix (CCW)
    :param theta: the rotation angle
    """
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def angle_diff(angle1, angle2):
    """
    Calculate the difference (angle2-angle1) ∈ [-pi, pi]
    """
    # restrict angle in [-pi, pi]
    angle1 = restrict_angle_in_unit_circle(angle1)
    angle2 = restrict_angle_in_unit_circle(angle2)
    diff1 = angle2 - angle1
    diff2 = 2 * np.pi - np.abs(diff1)
    if diff1 > 0:
        diff2 = -diff2
    if np.abs(diff1) < np.abs(diff2):
        return diff1
    else:
        return diff2

def angle_array_diff(angle_array1, angle_array2):
    angle_array1 = restrict_angle_in_unit_circle(angle_array1)
    angle_array2 = restrict_angle_in_unit_circle(angle_array2)
    diff1 = angle_array2 - angle_array1
    diff2 = 2 * np.pi - np.abs(diff1)
    diff = np.empty_like(diff1)
    diff2[diff1>0] = -diff2[diff1>0]
    diff[np.abs(diff1)<np.abs(diff2)] = diff1[np.abs(diff1)<np.abs(diff2)]
    diff[np.abs(diff1)>=np.abs(diff2)] = diff2[np.abs(diff1)>=np.abs(diff2)]
    return diff

def restrict_angle_in_unit_circle(angle):
    """
        restrict angle in [-pi, pi]
    """
    return (angle - (-np.pi)) % (2 * np.pi) + (-np.pi)

def make_angle_continuous(angle):
    """
    Angle restricted in [-pi, pi] might be discrete,
        this function makes consecutive angles in a
        sequence continuous
    Example:
        >>> [0.8*pi, 0.9*pi, 1.0*pi, -0.9*pi, -0.8*pi]
        >>> [0.8*pi, 0.9*pi, 1.0*pi, 1,1*pi, 1.2*pi]
    """
    seq_len = len(angle)
    ret = copy.deepcopy(angle)
    for i in range(seq_len):
        if i == 0:
            ret[i] = angle[i]
        else:
            ret[i] = last_angle + angle_diff(last_angle, angle[i])
        last_angle = ret[i]
    return ret

def interpolate_path(path, N):
    """
    Interpolate path
    :param path: (N, M), N is time steps, M is items
    """
    tau = np.linspace(0, 1., N)
    f_interp = interp1d(np.linspace(0., 1., path.shape[0]), path, axis=0)
    ret = f_interp(tau)
    return ret

def make_rigid_transform(translation, rotation, from_frame):
    return RigidTransform(translation=translation, rotation=rotation, from_frame=from_frame)
