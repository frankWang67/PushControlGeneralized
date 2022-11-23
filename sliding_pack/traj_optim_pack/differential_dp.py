# Author: Yongpeng Jiang
# Contact: jyp19@mails.tsinghua.edu.cn
# Date: 11/22/2022
# -------------------------------------------------------------------
# Description:
# 
# Class for the trajectory optimization (TO) for the pusher-slider 
# problem using a Differential Dynamic Program (DDP) approach
# -------------------------------------------------------------------

# import libraries
import sys
import os
import time
import numpy as np
import casadi as cs
import cvxopt as cvx
import sliding_pack

class buildDDPOptObj():
    def __init__(self, dyn_class, timeHorizon, configDict) -> None:
        """
            configDict: the key ['TO'] of .yaml file
        """
        # init parameters
        self.dyn = dyn_class
        self.TH = timeHorizon
        self.beta = None
        
        self.W_x = cs.diag(cs.SX(configDict['W_x']))
        self.W_u = cs.diag(cs.SX(configDict['W_u']))
        
        # input constraints
        self.A_u = np.array([[ 1, 0, 0],
                             [-1, 0, 0],
                             [ 0, 1, 0],
                             [ 0,-1, 0],
                             [ 0, 0, 1],
                             [ 0, 0,-1]])
        self.b_u = np.array([[0],
                             [-self.dyn.f_lim],
                             [-self.dyn.f_lim],
                             [-self.dyn.f_lim],
                             [-self.dyn.psi_dot_lim],
                             [-self.dyn.psi_dot_lim]])
        
        # dynamic functions
        self.f_xu = cs.Function(
            'f_xu',
            [self.dyn.x, self.dyn.u, self.dyn.beta],
            [self.dyn.f_(self.dyn.x, self.dyn.u, self.dyn.beta)],
            ['x', 'u', 'b'],
            'f_xu'
        )
        
    def build_value_approx(self):
        """
            build the symbol of quadratic approximation matrix Q
            with Casadi
        """
        # auxiliar symbolic variables
        # -------------------------------------------------------------------
        # dx - state vector
        __dx_slider = cs.SX.sym('dx_slider')
        __dy_slider = cs.SX.sym('dy_slider')
        __dtheta = cs.SX.sym('dtheta')
        __dpsi = cs.SX.sym('dpsi')
        __dx = cs.vertcat(__dx_slider, __dy_slider, __dtheta, __dpsi)
        # du - control vector
        __df_norm = cs.SX.sym('df_norm')
        __df_tan = cs.SX.sym('df_tan')
        __dpsi_dot = cs.SX.sym('dpsi_dot')
        __du = cs.veccat(__df_norm, __df_tan, __dpsi_dot)
        # [1, dx, du] - concat input vector
        __dxu = cs.SX(8, 1)
        __dxu[0, 0] = 1
        __dxu[1:5, :] = __dx
        __dxu[5:, :] = __du
        
        #  -------------------------------------------------------------------
        __cost_l = cs.Function(
            'cost_l',
            [self.dyn.x, self.dyn.u],
            [cs.dot(self.dyn.x, cs.mtimes(self.W_x, self.dyn.x))
             + cs.dot(self.dyn.u, cs.mtimes(self.W_u, self.dyn.u))]
        )
        __cost_V = cs.Function(
            'cost_V',
            [self.dyn.x, self.dyn.u],
            [cs.dot(self.f_xu(self.dyn.x, self.dyn.u), cs.mtimes(self.W_x, self.f_xu(self.dyn.x, self.dyn.u)))]
        )
        __lx = cs.jacobian(__cost_l(self.dyn.x, self.dyn.u), self.dyn.x)
        __lu = cs.jacobian(__cost_l(self.dyn.x, self.dyn.u), self.dyn.u)
        __fx = cs.jacobian(self.f_xu(self.dyn.x, self.dyn.u, self.beta), self.dyn.x)
        __fu = cs.jacobian(self.f_xu(self.dyn.x, self.dyn.u, self.beta), self.dyn.u)
        __qx = 
        
        
        
        
        
        
    def project_feasible(self):
        pass
        
    def forward_integration(self, X, U, k, K):
        """
            X: (4, N) is the array of current states
            U: (3, N) is the array of current inputs
            k: (3, 1) is the feedforward vector
            K: (3, 4) is the feedback matrix
        """
        X_hat = np.zeros((4, self.TH + 1))
        X_hat[:, 0] = X[:, 0]  # (4, 1)
        for k in range(0, self.TH):
            uk_hat = U[:, k] + k + K @ (X_hat[:, k] - X[:, k])
            uk_hat = self.project_feasible(uk_hat)  # (3, 1)
            X_hat[:, k+1] = self.f_xu(X_hat[:, k], uk_hat)
        
        return X_hat
        
    def solve(self, x_init, U_init, beta):
        """
            x_init: (4, 1)
            U_init: (3, N)
            beta: list[3]
        """
        self.beta = beta