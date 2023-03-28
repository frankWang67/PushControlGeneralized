import casadi as cs
import numpy as np
from scipy.interpolate import interp1d
import dyn_sliding_pack


def numerical_diff(a, b):
    return np.array(a) - np.array(b)

def make_initialization_solutions(pbm:dyn_sliding_pack.params.DynPusherProblem,
                                  md:dyn_sliding_pack.model.DynamicPusherModel,
                                  N):
    # linear interpolate the position
    r0 = pbm.traj.r0
    rf = pbm.traj.rf
    r_interp = interp1d([0, N-1], np.c_[r0, rf], 'linear')(np.arange(0, N, 1))

    # terminal time
    tf = (pbm.traj.tf_min + pbm.traj.tf_max)/2

    # constant velocity
    v_traj = (numerical_diff(rf, r0) / tf).reshape(-1, 1).repeat(N, -1)

    # the state guess
    x_traj = np.concatenate((r_interp, v_traj), axis=0)
    assert x_traj.shape[0] == md.Nx

    # the input guess
    kMass = pbm.system.m
    kMuGround = pbm.env.mu_g
    kGravity = pbm.env.g
    u_traj = np.zeros((md.Nu, N))
    u_traj[0, :] = np.ones(N) * (kMuGround * kMass * kGravity)  # norm contact force

    # the auxiliary guess
    z_traj = np.zeros((md.Nz, N))
    f_ground = pbm.env.mu_g * pbm.system.m * pbm.env.g  # ground friction
    z_traj[0, :] = -f_ground * v_traj[0, :] / np.linalg.norm(v_traj[0:2, :], axis=0)
    z_traj[1, :] = -f_ground * v_traj[1, :] / np.linalg.norm(v_traj[0:2, :], axis=0)

    return (x_traj, u_traj, z_traj)
