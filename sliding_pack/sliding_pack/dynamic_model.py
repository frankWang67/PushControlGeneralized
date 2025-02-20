# Author: Joao Moura
# Contact: jpousad@ed.ac.uk
# Date: 19/10/2020
# -------------------------------------------------------------------
# Description:
# 
# Functions modelling the dynamics of an object sliding on a table.
# Based on: Hogan F.R, Rodriguez A. (2020) IJRR paper
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------
import sys
import numpy as np
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.path import Path
import casadi as cs
import sliding_pack

class Sys_sq_slider_quasi_static_ellip_lim_surf():

    def __init__(self, configDict, curve, contactMode='sticking', pusherAngleLim=0., limit_surf_gain=1.):
        self.curve = curve

        # init parameters
        self.mode = contactMode
        # self.sl = configDict['sideLenght']  # side dimension of the square slider [m]
        self.miu = configDict['pusherFricCoef']  # fric between pusher and slider
        self.f_lim = configDict['pusherForceLim']
        self.psi_dot_lim = configDict['pusherAngleVelLim']
        self.Kz_max = configDict['Kz_max']
        self.Kz_min = configDict['Kz_min']
        #  -------------------------------------------------------------------
        # vector of physical parameters
        
        self.Nbeta = 3
        self.beta = cs.MX.sym('beta', self.Nbeta)
        # beta[0] - xl
        # beta[1] - yl
        # beta[2] - r_pusher
        #  -------------------------------------------------------------------
        # self.psi_lim = 0.9*cs.arctan2(self.beta[0], self.beta[1])
        self.psi_lim = pusherAngleLim
        self.limit_surf_gain = limit_surf_gain

        # system constant variables
        self.Nx = 4  # number of state variables

        # vectors of state and control
        #  -------------------------------------------------------------------
        # x - state vector
        # x[0] - x slider CoM position in the global frame
        # x[1] - y slider CoM position in the global frame
        # x[2] - slider orientation in the global frame
        # x[3] - angle of pusher relative to slider
        self.x = cs.MX.sym('x', self.Nx)
        # dx - derivative of the state vector
        self.dx = cs.MX.sym('dx', self.Nx)
        #  -------------------------------------------------------------------

        # auxiliar symbolic variables
        # used to compute the symbolic representation for variables
        # -------------------------------------------------------------------
        # x - state vector
        __x_slider = cs.MX.sym('__x_slider')  # in global frame [m]
        __y_slider = cs.MX.sym('__y_slider')  # in global frame [m]
        __theta = cs.MX.sym('__theta')  # in global frame [rad]
        __psi = cs.MX.sym('__psi')  # in relative frame [rad]
        __x = cs.veccat(__x_slider, __y_slider, __theta, __psi)
        # u - control vector
        __f_norm = cs.MX.sym('__f_norm')  # in local frame [N]
        __f_tan = cs.MX.sym('__f_tan')  # in local frame [N]
        # rel vel between pusher and slider [rad/s]
        __psi_dot = cs.MX.sym('__psi_dot')
        __u = cs.veccat(__f_norm, __f_tan, __psi_dot)
        # beta - dynamic parameters
        __xl = cs.MX.sym('__xl')  # slider x lenght
        __yl = cs.MX.sym('__yl')  # slider y lenght
        __r_pusher = cs.MX.sym('__r_pusher')  # radious of the cilindrical pusher
        __beta = cs.veccat(__xl, __yl, __r_pusher)

        # system model
        # -------------------------------------------------------------------
        # Rotation matrix
        __A = cs.MX(self.curve.lim_surf_A)
        __A = self.limit_surf_gain * __A
        self.A = cs.Function('A', [__beta], [__A], ['b'], ['A'])
        __ctheta = cs.cos(__theta)
        __stheta = cs.sin(__theta)
        __R = cs.MX.zeros(3, 3)  # anti-clockwise rotation matrix (from {Slider} to {World})
        __R[0,0] = __ctheta; __R[0,1] = -__stheta; __R[1,0] = __stheta; __R[1,1] = __ctheta; __R[2,2] = 1.0
        #  -------------------------------------------------------------------
        self.R = cs.Function('R', [__x], [__R], ['x'], ['R'])  # (rotation matrix from {Slider} to {World})
        #  -------------------------------------------------------------------
        __p = cs.MX.sym('p', 2) # pusher position
        __rc_prov = cs.mtimes(__R[0:2,0:2].T, __p - __x[0:2])  # (Real {Pusher Center} in {Slider})
        #  -------------------------------------------------------------------
        # slider position

        t = self.curve.psic_to_t(__psi)
        contact_point = self.curve.curve_func(t)
        __xc = contact_point[0]  # ({Contact Point} in {Slider})
        __yc = contact_point[1]  # ({Contact Point} in {Slider})
        tangent_vec = self.curve.tangent_func(t)
        normal_vec = self.curve.normal_func(t)
        # `normal_vec` is the normal vector of the shape contour pointing inwards
        __rc = cs.MX(2,1); __rc[0] = __xc-__r_pusher*normal_vec[0]; __rc[1] = __yc-__r_pusher*normal_vec[1]  # ({Pusher Center} in {Slider})
        #  -------------------------------------------------------------------
        __psi_prov = cs.atan2(__rc_prov[0], __rc_prov[1])  # (Real {φ_c})
            
        # pusher position
        __p_pusher = cs.mtimes(__R[0:2,0:2], __rc)[0:2] + __x[0:2]  # ({Pusher Center} in {World})
        #  -------------------------------------------------------------------
        self.psi_ = cs.Function('psi_', [__x,__p], [__psi_prov])  # compute (φ_c) from state variables, pusher coordinates and slider geometry
        self.psi = cs.Function('psi', [self.x,__p], [self.psi_(self.x, __p)])
        #  -------------------------------------------------------------------
        self.p_ = cs.Function('p_', [__x,__beta], [__p_pusher], ['x', 'b'], ['p'])  # compute (pusher_center_coordinate) from state variables and slider geometry
        self.p = cs.Function('p', [self.x, self.beta], [self.p_(self.x, self.beta)], ['x', 'b'], ['p'])
        #  -------------------------------------------------------------------
        self.s = cs.Function('s', [self.x], [self.x[0:3]], ['x'], ['s'])  # compute (x, y, θ) from state variables
        #  -------------------------------------------------------------------
        
        # dynamics
        __Jc = cs.MX(2,3)
        __Jc[0,0] = 1; __Jc[1,1] = 1; __Jc[0,2] = -__yc; __Jc[1,2] = __xc;  # contact jacobian
        
        self.RAJc = cs.Function('RAJc', [__x,__beta], [cs.mtimes(cs.mtimes(__R, __A), __Jc.T)], ['x', 'b'], ['f'])
        __force = tangent_vec * __f_tan + normal_vec * __f_norm
        __f = cs.MX(cs.vertcat(cs.mtimes(cs.mtimes(__R,__A),cs.mtimes(__Jc.T, __force)),__u[2]))
        # __f = cs.MX(cs.vertcat(cs.mtimes(cs.mtimes(__R,__A), cs.DM([1, 1, 1])),__u[2]))
        #  -------------------------------------------------------------------
        self.f_ = cs.Function('f_', [__x,__u,__beta], [__f], ['x', 'u', 'b'], ['f'])  # compute (f(x, u)) from state variables, input variables and slider geometry
        #  -------------------------------------------------------------------

        # control constraints
        #  -------------------------------------------------------------------
        if self.mode == 'sliding_cc':
            # complementary constraint
            # u - control vector
            # u[0] - normal force in the local frame
            # u[1] - tangential force in the local frame
            # u[2] - rel sliding vel between pusher and slider counterclockwise(φ_c(-))
            # u[3] - rel sliding vel between pusher and slider clockwise(φ_c(+))
            self.Nu = 4  # number of action variables
            self.u = cs.MX.sym('u', self.Nu)
            self.Nz = 0
            self.z0 = []
            self.lbz = []
            self.ubz = []
            # discrete extra variable
            self.z_discrete = False
            empty_var = cs.MX.sym('empty_var')
            self.g_u = cs.Function('g_u', [self.u, empty_var], [cs.vertcat(
                # friction cone edges
                self.miu*self.u[0]+self.u[1],  # lambda(+)>=0
                self.miu*self.u[0]-self.u[1],  # lambda(-)>=0
                # complementarity constraint
                (self.miu * self.u[0] - self.u[1])*self.u[3],  # lambda(-)*φ_c(+)=0
                (self.miu * self.u[0] + self.u[1])*self.u[2]  # lambda(+)*φ_c(-)=0
            )], ['u', 'other'], ['g'])
            self.g_lb = [0., 0., 0., 0.]
            self.g_ub = [cs.inf, cs.inf, 0., 0.]
            self.Ng_u = 4
            # cost gain for extra variable
            __Ks_max = self.Kz_max
            __Ks_min = self.Kz_min
            __i_th = cs.MX.sym('__i_th')
            self.kz_f = cs.Function('ks', [__i_th], [__Ks_max * cs.exp(__i_th * cs.log(__Ks_min / __Ks_max))])  # decrease from Ks_max to Ks_min
            # state and acton limits
            #  -------------------------------------------------------------------
            self.lbx = [-cs.inf, -cs.inf, -cs.inf, -self.psi_lim]
            self.ubx = [cs.inf, cs.inf, cs.inf, self.psi_lim]
            self.lbu = [0.0,  -self.f_lim, 0.0, 0.0]
            self.ubu = [self.f_lim, self.f_lim, self.psi_dot_lim, self.psi_dot_lim]
            #  -------------------------------------------------------------------
            # dynamics equation
            self.f = cs.Function('f', [self.x, self.u, self.beta], [self.f_(self.x, cs.vertcat(self.u[0:2], self.u[2]-self.u[3]), self.beta)],  ['x', 'u', 'b'], ['f'])
        elif self.mode == 'sliding_cc_slack':
            # complementary constraint + slack variables
            # u - control vector
            # u[0] - normal force in the local frame
            # u[1] - tangential force in the local frame
            # u[2] - rel sliding vel between pusher and slider counterclockwise
            # u[3] - rel sliding vel between pusher and slider clockwise
            self.Nu = 4  # number of action variables
            self.u = cs.MX.sym('u', self.Nu)
            self.Nz = 2
            self.z = cs.MX.sym('z', self.Nz)
            self.z0 = [1.]*self.Nz
            self.lbz = [-cs.inf]*self.Nz
            self.ubz = [cs.inf]*self.Nz
            # discrete extra variable
            self.z_discrete = False
            self.g_u = cs.Function('g_u', [self.u, self.z], [cs.vertcat(
                # friction cone edges
                self.miu*self.u[0]+self.u[1],
                self.miu*self.u[0]-self.u[1],
                # complementarity constraint
                (self.miu * self.u[0] - self.u[1])*self.u[3] + self.z[0],
                (self.miu * self.u[0] + self.u[1])*self.u[2] + self.z[1]
            )], ['u', 'other'], ['g'])
            self.g_lb = [0., 0., 0., 0.]
            self.g_ub = [cs.inf, cs.inf, 0., 0.]
            self.Ng_u = 4
            # cost gain for extra variable
            __Ks_max = self.Kz_max
            __Ks_min = self.Kz_min
            __i_th = cs.MX.sym('__i_th')
            self.kz_f = cs.Function('ks', [__i_th], [__Ks_max * cs.exp(__i_th * cs.log(__Ks_min / __Ks_max))])
            # state and acton limits
            #  -------------------------------------------------------------------
            self.lbx = [-cs.inf, -cs.inf, -cs.inf, -self.psi_lim]
            self.ubx = [cs.inf, cs.inf, cs.inf, self.psi_lim]
            self.lbu = [0.0,  -self.f_lim, 0.0, 0.0]
            self.ubu = [self.f_lim, self.f_lim, self.psi_dot_lim, self.psi_dot_lim]
            #  -------------------------------------------------------------------
            # dynamics equation
            self.f = cs.Function('f', [self.x, self.u, self.beta], [self.f_(self.x, cs.vertcat(self.u[0:2], self.u[2]-self.u[3]), self.beta)],  ['x', 'u', 'b'], ['f'])
        elif self.mode == 'sliding_mi':
            # mixed integer
            # u - control vector
            # u[0] - normal force in the local frame
            # u[1] - tangential force in the local frame
            # u[2] - rel sliding vel between pusher and slider
            self.Nu = 3  # number of action variables
            self.u = cs.MX.sym('u', self.Nu)
            self.Nz = 3
            self.z = cs.MX.sym('z', self.Nz)
            self.z0 = [0]*self.Nz
            self.lbz = [0]*self.Nz
            self.ubz = [1]*self.Nz
            # discrete extra variable
            self.z_discrete = True
            self.Ng_u = 7
            bigM = 500  # big M for the Mixed Integer optimization
            self.g_u = cs.Function('g_u', [self.u, self.z], [cs.vertcat(
                self.miu*self.u[0]+self.u[1] + bigM*self.z[1],  # friction cone edge
                self.miu*self.u[0]-self.u[1] + bigM*self.z[2],  # friction cone edge
                self.miu*self.u[0]+self.u[1] - bigM*(1-self.z[2]),  # friction cone edge
                self.miu*self.u[0]-self.u[1] - bigM*(1-self.z[1]),  # friction cone edge
                self.u[2] + bigM*self.z[2],  # relative rot constraint
                self.u[2] - bigM*self.z[1],
                cs.sum1(self.z),  # sum of the integer variables
            )], ['u', 'other'], ['g'])
            self.g_lb = [0., 0., -cs.inf, -cs.inf, 0., -cs.inf, 1.]
            self.g_ub = [cs.inf, cs.inf, 0., 0., cs.inf, 0., 1.]
            __i_th = cs.MX.sym('__i_th')
            self.kz_f = cs.Function('ks', [__i_th], [0.])
            # state and acton limits
            #  -------------------------------------------------------------------
            self.lbx = [-cs.inf, -cs.inf, -cs.inf, -self.psi_lim]
            self.ubx = [cs.inf, cs.inf, cs.inf, self.psi_lim]
            self.lbu = [0.0,  -self.f_lim, 0.0]
            self.ubu = [self.f_lim, self.f_lim, self.psi_dot_lim]
            #  -------------------------------------------------------------------
            # dynamics equation
            self.f = cs.Function('f', [self.x, self.u, self.beta], [self.f_(self.x, self.u, self.beta)],  ['x', 'u', 'b'], ['f'])
        elif self.mode == 'sticking':
            # sticking constraint
            # u - control vector
            # u[0] - normal force in the local frame
            # u[1] - tangential force in the local frame
            self.Nu = 2  # number of action variables
            self.u = cs.MX.sym('u', self.Nu)
            empty_var = cs.MX.sym('empty_var')
            self.g_u = cs.Function('g_u', [self.u, empty_var], [cs.vertcat(
                self.miu*self.u[0]+self.u[1],  # friction cone edge
                self.miu*self.u[0]-self.u[1]  # friction cone edge
            )], ['u', 'other'], ['g'])
            self.g_lb = [0.0, 0.0]
            self.g_ub = [cs.inf, cs.inf]
            self.Nz = 0
            self.z0 = []
            self.lbz = []
            self.ubz = []
            # discrete extra variable
            self.z_discrete = False
            self.Ng_u = 2
            # state and acton limits
            #  -------------------------------------------------------------------
            self.lbx = [-cs.inf, -cs.inf, -cs.inf, self.psi_lim]
            self.ubx = [cs.inf, cs.inf, cs.inf, self.psi_lim]
            self.lbu = [0.0,  -self.f_lim]
            self.ubu = [self.f_lim, self.f_lim]
            #  -------------------------------------------------------------------
            # dynamics equation
            self.f = cs.Function('f', [self.x, self.u, self.beta], [self.f_(self.x, cs.vertcat(self.u, 0.0), self.beta)],  ['x', 'u', 'b'], ['f'])
        else:
            print('Specified mode ``{}`` does not exist!'.format(self.mode))
            sys.exit(-1)
        #  -------------------------------------------------------------------

    def set_patches(self, ax, x_data, beta, curve_func):
        t_vals = np.linspace(0, 1, 100)
        contour_pts = np.array([curve_func(t) for t in t_vals]).reshape(-1, 2)
        R_pusher = beta[2]
        x0 = x_data[0, 0]
        y0 = x_data[1, 0]
        contour_pts[:, 0] += x0
        contour_pts[:, 1] += y0
        contour_path = Path(contour_pts)
        self.slider = patches.PathPatch(contour_path, lw=2)
        self.pusher = patches.Circle(
                np.array(self.p(x0, beta)), radius=R_pusher, color='black')
        self.path_past, = ax.plot(x0, y0, color='orange')
        self.path_future, = ax.plot(x0, y0,
                color='blue', linestyle='dashed')
        ax.add_patch(self.slider)
        ax.add_patch(self.pusher)
        self.path_past.set_linewidth(2)

    def animate(self, i, ax, x_data, beta, X_future=None):
        xi = x_data[:, i]
        # compute transformation with respect to rotation angle xi[2]
        trans_ax = ax.transData
        transf_i = transforms.Affine2D().translate(xi[0], xi[1]).rotate_around(xi[0], xi[1], xi[2])
        # Set changes
        self.slider.set_transform(transf_i + trans_ax)
        self.pusher.set_center(np.array(self.p(xi, beta)))
        # Set path changes
        if self.path_past is not None:
            self.path_past.set_data(x_data[0, 0:i], x_data[1, 0:i])
        if (self.path_future is not None) and (X_future is not None):
            self.path_future.set_data(X_future[0, :, i], X_future[1, :, i])
        return []
    #  -------------------------------------------------------------------
