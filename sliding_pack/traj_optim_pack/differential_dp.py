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
from matplotlib import pyplot as plt
import numpy as np
import sympy
import casadi as cs
#  -------------------------------------------------------------------
import sliding_pack
#  -------------------------------------------------------------------

class buildDDPOptObj():
    def __init__(self, dt, dyn_class, timeHorizon, configDict) -> None:
        """
            configDict: the key ['TO'] of .yaml file
        """
        # init parameters
        self.dt = dt
        self.dyn = dyn_class
        self.TH = timeHorizon
        self.miu = dyn_class.miu
        self.beta = dyn_class.beta
        self.beta_value = None
        
        self.W_x = cs.diag(cs.MX(configDict['W_x']))
        self.W_u = cs.diag(cs.MX(configDict['W_u'][:3]))
        self.W_x_arr = np.diag(configDict['W_x'])
        self.W_u_arr = np.diag(configDict['W_u'][:3])
        self.gamma_u = cs.DM(cs.diag(configDict['gamma_u']))
        self.converge_eps = configDict['converge_eps']
        
        # input constraints
        self.A_u = np.array([[ 1, 0, 0],
                             [ 0, 1, 0],
                             [ 0, 0, 1],
                             [self.miu,-1, 0],
                             [self.miu, 1, 0]])

        self.lb_u = np.array([0, -self.dyn.f_lim, -self.dyn.psi_dot_lim, 0, 0])

        self.ub_u = np.array([self.dyn.f_lim, self.dyn.f_lim, self.dyn.psi_dot_lim, np.inf, np.inf])
        
        # dynamic functions
        self.f_xu = cs.Function(
            'f_xu',
            [self.dyn.x, self.dyn.u, self.dyn.beta],
            [self.dyn.x + self.dyn.f_(self.dyn.x, self.dyn.u, self.dyn.beta) * self.dt],
            ['x', 'u', 'b'],
            ['f_xu']
        )
        
        self.build_value_approx()
        
    def build_value_approx(self):
        """
            build the symbol of quadratic approximation matrix Q
            with Casadi
        """
        # auxiliar symbolic variables
        # -------------------------------------------------------------------
        # dx - state vector
        __dx = cs.MX.sym('x', 4)
        # du - control vector
        __du = cs.MX.sym('u', 3)
        # [1, dx, du] - concat input vector
        __dxu = cs.MX(8, 1)
        __dxu[0, 0] = 1
        __dxu[1:5, 0] = __dx
        __dxu[5:, 0] = __du
        # nominal state
        __nom_x = cs.MX.sym('nom_x', 4)
        
        #  -------------------------------------------------------------------
        __cost_l = cs.Function(
            'cost_l',
            [self.dyn.x, self.dyn.u, __nom_x],
            [cs.dot((self.dyn.x - __nom_x), cs.mtimes(self.W_x, (self.dyn.x - __nom_x)))
             + cs.dot(self.dyn.u, cs.mtimes(self.W_u, self.dyn.u))]
        )
        
        __cost_VN = cs.dot((self.dyn.x - __nom_x), cs.mtimes(100 * self.W_x, (self.dyn.x - __nom_x)))

        # first and second derivatives of V_{k+1}, l and f
        __lx = cs.gradient(__cost_l(self.dyn.x, self.dyn.u, __nom_x), self.dyn.x)
        __lu = cs.gradient(__cost_l(self.dyn.x, self.dyn.u, __nom_x), self.dyn.u)
        __lxx = cs.hessian(__cost_l(self.dyn.x, self.dyn.u, __nom_x), self.dyn.x)[0]
        __luu = cs.hessian(__cost_l(self.dyn.x, self.dyn.u, __nom_x), self.dyn.u)[0]
        __lux = cs.jacobian(cs.gradient(__cost_l(self.dyn.x, self.dyn.u, __nom_x), self.dyn.u), self.dyn.x)
        __fx = cs.jacobian(self.f_xu(self.dyn.x, self.dyn.u, self.beta), self.dyn.x)  # checked
        __fu = cs.jacobian(self.f_xu(self.dyn.x, self.dyn.u, self.beta), self.dyn.u)  # checked
        __fxx = [cs.hessian(self.f_xu(self.dyn.x, self.dyn.u, self.beta)[i], self.dyn.x)[0] for i in range(4)]
        __fuu = [cs.hessian(self.f_xu(self.dyn.x, self.dyn.u, self.beta)[i], self.dyn.u)[0] for i in range(4)]
        __fux = [cs.jacobian(cs.gradient(self.f_xu(self.dyn.x, self.dyn.u, self.dyn.beta)[i], self.dyn.u), self.dyn.x) for i in range(4)]

        self.VN = cs.Function(
            'VN',
            [self.dyn.x, __nom_x],
            [__cost_VN],
            ['x', 'nom_x'],
            ['VN']
        )

        __Vx = cs.MX.sym('_Vx', 4, 1)
        __Vxx = cs.MX.sym('_Vxx', 4, 4)
        __VNx = cs.gradient(self.VN(self.dyn.x, __nom_x), self.dyn.x)
        __VNxx = cs.hessian(self.VN(self.dyn.x, __nom_x), self.dyn.x)[0]

        self.VNx = cs.Function(
            'VNx',
            [self.dyn.x, __nom_x],
            [__VNx],
            ['x', 'nom_x'],
            ['VNx']
        )

        self.VNxx = cs.Function(
            'VNxx',
            [self.dyn.x, __nom_x],
            [__VNxx],
            ['x', 'nom_x'],
            ['VNxx']
        )
        
        __qx = __lx + cs.mtimes(__fx.T, __Vx)
        __qu = __lu + cs.mtimes(__fu.T, __Vx)
        __Qxx = __lxx + cs.mtimes(__fx.T, cs.mtimes(__Vxx, __fx)) + \
                sum((__Vx[i] * __fxx[i]) for i in range(4))
        __Quu = __luu + cs.mtimes(__fu.T, cs.mtimes(__Vxx, __fu)) + \
                sum((__Vx[i] * __fuu[i]) for i in range(4))
        __Qux = __lux + cs.mtimes(__fu.T, cs.mtimes(__Vxx, __fx)) + \
                sum((__Vx[i] * __fux[i]) for i in range(4))
        
        self.Q_xx = cs.Function(
            'Q_xx',
            [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
            [__Qxx]
        )

        self.Q_uu = cs.Function(
            'Q_uu',
            [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
            [__Quu]
        )
        
        self.Q_ux = cs.Function(
            'Q_ux',
            [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
            [__Qux]
        )
        
        self.qx = cs.Function(
            'qx',
            [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
            [__qx]
        )

        self.qu = cs.Function(
            'qu',
            [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
            [__qu]
        )
        
        self.fx = cs.Function(
            'fx',
            [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
            [__fx]
        )
        
        self.fu = cs.Function(
            'fu',
            [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
            [__fu]
        )
        
        self.luu = cs.Function(
            'luu',
            [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
            [__luu]
        )
        
        self.fxx = cs.Function(
            'fxx',
            [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
            [__fxx[0], __fxx[1], __fxx[2], __fxx[3]]
        )
        
        self.fuu = cs.Function(
            'fuu',
            [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
            [__fuu[0], __fuu[1], __fuu[2], __fuu[3]]
        )
        
        self.fux = cs.Function(
            'fux',
            [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
            [__fux[0], __fux[1], __fux[2], __fux[3]]
        )
        
        self.cost_l = __cost_l
        self.cost_VN = __cost_VN
        
        # self.lx = cs.Function(
        #     'lu',
        #     [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
        #     [__lx]
        # )
        
        # self.Vxx = cs.Function(
        #     'Vxx',
        #     [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
        #     [__Vxx]
        # )
        
        # test code
        # xx = [1.1, 2.2, 3.3, 4.4]
        # uu = [5, 6, 7]
        # nom_x = [1, 2, 3, 4]
        # beta = [0.07, 0.12, 0.01]
        # Vx_vec = np.random.rand(4, 1)
        # Vxx_mat = np.random.rand(4, 4)
        # Vxx_mat = Vxx_mat @ Vxx_mat.T
        # var = xx, uu, beta, nom_x, Vx_vec, Vxx_mat
        
    def set_nominal_traj(self, X_nom):
        """
            set the tracking states for DDP
            ------input
            X_nom: (4, N+1)
        """
        self.X_nom = X_nom
        
    def compute_cost_function(self, X_opt, U_opt):
        """
            compute cost of the current X, U trajectory
        """
        cost = 0
        for k in range(self.TH):
            cost += (X_opt[:, k] - self.X_nom[:, k]).T @ self.W_x_arr @ (X_opt[:, k] - self.X_nom[:, k])
            cost += U_opt[:, k].T @ self.W_u_arr @ U_opt[:, k]
        cost += (X_opt[:, -1] - self.X_nom[:, -1]).T @ self.W_x_arr @ (X_opt[:, -1] - self.X_nom[:, -1])
        
        return cost
        
    def make_quadratic_approx(self):
        """
            make useful blocks storage of the matrix Q
            ------output
            Dict[List[]]
        """
        self.Q_block = {
            'Q_xx': [],  # DM(4, 4)
            'Q_uu': [],  # DM(3, 3)
            'Q_ux': [],  # DM(3, 4)
            'qx': [],  # DM(4, 1)
            'qu': [],  # DM(3, 1)
        }
        
    def project_feasible(self, k, uk, duk, dx):
        """
            get the inputs that do not violate constraints
            in the forward propagate process
        """
        Q_ux = self.Q_block['Q_ux'][k]
        Q_uu = self.Q_block['Q_uu'][k]
        qu = self.Q_block['qu'][k]
        
        H = Q_uu + self.gamma_u
        G = Q_ux @ dx + qu - self.gamma_u.T @ duk
        
        A_u = cs.DM(self.A_u)  # DM(3, 3)

        lb_a = self.lb_u - self.A_u @ uk
        ub_a = self.ub_u - self.A_u @ uk
        
        qp = {'h': H.sparsity(),
              'a': A_u.sparsity()}
        S = cs.conic('S', 'qpoases', qp, {'printLevel': 'none'})
        try:
            r = S(h=H, \
                g=G, \
                a=A_u, \
                lba = lb_a, \
                uba = ub_a)
        except:
            import pdb; pdb.set_trace()
        
        duk_hat = r['x']  # suppose (3, 1)
        
        return uk + duk_hat.toarray().squeeze()
        
    def coldboot_integration(self, x0, U0):
        """
            ------ input
            x0: (4, 1) is the array of initial states
            U0: (3, N) is the array of initial inputs
            ------ output
            X_hat: (4, N+1) is the array of updated states
        """
        X_hat = np.zeros((4, self.TH + 1))
        X_hat[:, 0] = x0.squeeze()  # (4, 1)
        for k in range(0, self.TH):
            X_hat[:, k+1] = self.f_xu(X_hat[:, k], U0[:, k], self.beta_value).toarray().squeeze()

        return X_hat
    
    def total_cost(self, iter, X, U):
        cost = 0
        X_forward = X
        for k in range(iter, min(iter + 2, self.TH)):
            cost += self.cost_l(X_forward[:, k], U[:, k], self.X_nom[:, k])
            X_forward[:, k+1] = self.f_xu(X_forward[:, k], U[:, k], self.beta_value).toarray().squeeze()
        # cost += self.VN(X_forward[:, -1], self.X_nom[:, -1])
        
        return cost
    
    def line_search_over_k(self, iter, X, U, uk_forward, uk_back):
        alpha = 1.0
        base_cost = self.total_cost(iter, X, U)
        U_new = U
        while True:
            # import pdb; pdb.set_trace()
            U_new[:, iter] = U[:, iter] + alpha * uk_forward.squeeze() + uk_back.squeeze()
            if self.total_cost(iter, X, U_new) >= base_cost and alpha >= 0.0001:
                alpha = 0.5 * alpha
            else:
                break
        
        return alpha

    def backward_propagation(self, X, U):
        """
            ------ input
            X: (4, N+1) is the array of forward integrated states
            U: (4, N) is the array of inputs
        """
        self.make_quadratic_approx()
        
        VN = self.VN(X[:, -1], self.X_nom[:, -1]).toarray()[0, 0]  # double
        V = VN  # value function
        Vx, Vxx = np.zeros((4, 1)), np.zeros((4, 4))
        self.Vx_tensor, self.Vxx_tensor = np.zeros((self.TH, 4, 1)), np.zeros((self.TH, 4, 4))
        self.k_vec, self.K_mat = np.zeros((self.TH, 3, 1)), np.zeros((self.TH, 3, 4))
        
        for k in range(self.TH - 1, -1, -1):
            if k == self.TH - 1:
                Vx = self.VNx(X[:, -1], self.X_nom[:, -1]).toarray()  # (4, 1)
                Vxx = self.VNxx(X[:, -1], self.X_nom[:, -1]).toarray()  # (4, 4)

            # calculate matrix values
            Q_xx = self.Q_xx(X[:, k], U[:, k], self.beta_value, self.X_nom[:, k], Vx, Vxx)
            Q_ux = self.Q_ux(X[:, k], U[:, k], self.beta_value, self.X_nom[:, k], Vx, Vxx)
            Q_uu = self.Q_uu(X[:, k], U[:, k], self.beta_value, self.X_nom[:, k], Vx, Vxx)
            qx = self.qx(X[:, k], U[:, k], self.beta_value, self.X_nom[:, k], Vx, Vxx)
            qu = self.qu(X[:, k], U[:, k], self.beta_value, self.X_nom[:, k], Vx, Vxx)
                        
            self.Q_block['Q_xx'].insert(0, Q_xx)
            self.Q_block['Q_ux'].insert(0, Q_ux)
            self.Q_block['Q_uu'].insert(0, Q_uu)
            self.Q_block['qx'].insert(0, qx)
            self.Q_block['qu'].insert(0, qu)

            A_u = cs.DM(self.A_u)  # DM(3, 3)

            lb_a = self.lb_u - self.A_u @ U[:, k]
            ub_a = self.ub_u - self.A_u @ U[:, k]

            # solve the qp problem for feedforward k
            qp = {'h': Q_uu.sparsity(),
                  'a': A_u.sparsity()}
            S = cs.conic('S', 'qpoases', qp, {'printLevel': 'none'})
            try:
                r = S(h=Q_uu, \
                      g=qu, \
                      a=A_u, \
                      lba=lb_a.reshape(-1, 1), \
                      uba=ub_a.reshape(-1, 1))
            except:
                import pdb; pdb.set_trace()
            du = r['x'].toarray().squeeze()  # suppose (3, 1)
            
            # solve for feedback matrices K
            eps = 1e-8
            """
                (Imple1) Free dimension decomposition.
            """
            # index of active constraints
            act_cst = np.bitwise_or(np.abs(self.A_u @ du - lb_a).squeeze() < eps, np.abs(self.A_u @ du - ub_a).squeeze() < eps)
            act_cst = self.A_u[act_cst].reshape(-1, 3)  # active constraints
            act_null = sympy.Matrix(act_cst).nullspace()
            act_null = [np.array(act_null[i]).squeeze().tolist() for i in range(len(act_null))]
            act_null = np.array(act_null).T.astype(np.double)
            K_mat = -np.linalg.inv(Q_uu.toarray() + 5 * np.eye(3)) @ Q_ux.toarray()
            if len(act_null) == 0:
                K_mat = np.zeros_like(K_mat)
            else:
                K_mat = act_null @ np.linalg.inv(act_null.T @ act_null) @ act_null.T @ K_mat

            self.k_vec[k, :] = np.expand_dims(du, axis=1)
            self.K_mat[k, :] = K_mat
            
            """
                (Imple2) Free dimension decomposition, for box constraints.
            """
            """
            grad_du = qu + Q_uu @ du
            c_du = np.bitwise_or(np.bitwise_and(np.abs(A_u @ du - lb_a) < eps, np.array(A_u @ grad_du) > 0),
                                 np.bitwise_and(np.abs(A_u @ du - ub_a) < eps, np.array(A_u @ grad_du) < 0)).squeeze()
            f_du = np.bitwise_not(c_du)
            
            Q_uuff, Q_uxf = Q_uu.toarray(), Q_ux.toarray()
            Q_uuff[np.ix_(c_du, c_du)] = 0
            Q_uuff[np.ix_(f_du, f_du)] = np.linalg.inv(Q_uuff[np.ix_(f_du, f_du)])
            
            Q_uxf[c_du] = 0
            
            self.k_vec[k, :] = np.expand_dims(du, axis=1)
            self.K_mat[k, :] = -Q_uuff @ Q_uxf
            """
            
            # propagate value
            V += 0.5 * self.k_vec[k, :].T @ Q_uu.toarray() @ self.k_vec[k, :]
            Vx = qx - self.K_mat[k, ...].T @ Q_uu.toarray() @ self.k_vec[k, :]
            Vxx = Q_xx - self.K_mat[k, ...].T @ Q_uu.toarray() @ self.K_mat[k, ...]
            self.Vx_tensor[k, :], self.Vxx_tensor[k, ...] = Vx, Vxx

    def forward_integration(self, X, U):
        """
            X: (4, N+1) is the array of current states
            U: (3, N) is the array of current inputs
        """
        X_hat, U_hat = np.zeros((4, self.TH + 1)), np.zeros((3, self.TH))
        X_hat[:, 0] = X[:, 0]  # (4, 1)
        for k in range(0, self.TH):
            k_vec, K_mat = self.k_vec[k, :], self.K_mat[k, ...]
            dx = np.expand_dims(X_hat[:, k] - X[:, k], 1)
            # alpha = self.line_search_over_k(k, X_hat, U, k_vec, K_mat @ dx)
            U_hat[:, k] = self.project_feasible(k, U[:, k], 0.2 * k_vec + K_mat @ dx, dx)  # (3, 1)
            X_hat[:, k+1] = self.f_xu(X_hat[:, k], U_hat[:, k], self.beta_value).toarray().squeeze()
        
        return X_hat, U_hat
        
    def solve_constrained_ddp(self, x0, U0):
        """
            x0: (4, 1)
            U0: (3, N)
            beta: list[3]
        """
        check_converge = False  # skip convergence check during the first iteration
        X_hat, U_hat = np.zeros((4, self.TH + 1)), \
                       np.zeros((3, self.TH))
        cost = []
                             
        # integrate forward after coldboot
        X = self.coldboot_integration(x0, U0)
        U = U0
        import pdb; pdb.set_trace()
        
        while True:
            self.backward_propagation(X, U)
            X_hat, U_hat = self.forward_integration(X, U)
            
            # import pdb; pdb.set_trace()
            
            cost.append(self.compute_cost_function(X, U))
            
            # convergence check
            # import pdb; pdb.set_trace()
            if check_converge == True:
                print('error X: ', np.linalg.norm([(X_hat[:, i] - X[:, i]).T @ self.W_x_arr @ (X_hat[:, i] - X[:, i]) for i in range(self.TH + 1)]))
                print('error U: ', np.linalg.norm([(U_hat[:, i] - U[:, i]).T @ self.W_u_arr @ (U_hat[:, i] - U[:, i]) for i in range(self.TH)]))
                print('cost: ', cost[-1])
            
            if check_converge == True \
               and \
               np.linalg.norm([(X_hat[:, i] - X[:, i]).T @ self.W_x_arr @ (X_hat[:, i] - X[:, i]) for i in range(self.TH + 1)]) < 0.00001 \
               and \
               np.linalg.norm([(U_hat[:, i] - U[:, i]).T @ self.W_u_arr @ (U_hat[:, i] - U[:, i]) for i in range(self.TH)]) < 0.0001:
                X, U = X_hat, U_hat
                break
            
            X, U = X_hat, U_hat
            check_converge = True
        
        import pdb; pdb.set_trace()
        return X, U
        
        
if __name__ == '__main__':
    planning_config = sliding_pack.load_config('planning_switch_config.yaml')
    dyn = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
        planning_config['dynamics'],
        planning_config['TO']['contactMode']
    )
    T = 2.5  # time of the simulation is seconds
    freq = 25  # number of increments per second
    dt = 1.0/freq  # sampling time
    # N = int(T*freq)  # total number of iterations
    N = 61
    ddpOptObj = buildDDPOptObj(dt=dt,
                               dyn_class=dyn,
                               timeHorizon=N,
                               configDict=planning_config['TO'])
    
    X_nom = np.load('./X_nom.npy')
    U_nom = np.load('./U_nom.npy')
    X_real = np.load('./X_real.npy')
    x0 = X_real[:, 0].reshape(4, 1)
    x0[-1] -= 0.07
    U0 = U_nom
    ddpOptObj.beta_value = [0.07, 0.12, 0.01]
    ddpOptObj.set_nominal_traj(X_real)
    X, U = ddpOptObj.solve_constrained_ddp(x0, U0)
    
    import pdb; pdb.set_trace()
    
    np.save('../../examples/X_opt.npy', X)
    