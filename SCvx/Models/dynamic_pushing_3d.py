import sympy as sp
import numpy as np
import cvxpy as cvx
from pyquaternion import Quaternion

from global_parameters import K


def skew(v):
    """
        3x3 skew-symmetric matrix
        v: angle
    """
    return sp.Matrix([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def dir_cosine(q):
    """
        3x3 rotation matrix
        q: quaternion in [w, x, y, z] order
    """
    return sp.Matrix([
        [1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
        [2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] - q[0] * q[1])],
        [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]
    ])

def angvel(q0, q1, dt):
    """
        3x angular velocity vector
        q0: start orientation
        q1: terminal orientation
        (https://mariogc.com/post/angular-velocity-quaternions/)
    """
    return (2 / dt) * np.array([
            q0[0]*q1[1] - q0[1]*q1[0] - q0[2]*q1[3] + q0[3]*q1[2],
            q0[0]*q1[2] + q0[1]*q1[3] - q0[2]*q1[0] - q0[3]*q1[1],
            q0[0]*q1[3] - q0[1]*q1[2] + q0[2]*q1[1] - q0[3]*q1[0]
        ])

def omega(w):
    """
        4x4 skew-symmetric matrix
        w: 3x vector of angular velocity
    """
    return sp.Matrix([
        [0, -w[0], -w[1], -w[2]],
        [w[0], 0, w[2], -w[1]],
        [w[1], -w[2], 0, w[0]],
        [w[2], w[1], -w[0], 0],
    ])

class Model:
    # dimensions
    n_x = 17
    n_u = 10

    n_eq = 4
    n_ineq = 0

    # indexes
    r_b_idx = range(0, 3)
    q_b_idx = range(3, 7)
    v_b_idx = range(7, 10)
    w_b_idx = range(10, 13)
    r_m_idx = range(13, 15)
    v_m_idx = range(15, 17)

    f_b_idx = range(0, 3)
    a_b_idx = range(3, 6)
    dw_b_idx = range(6, 9)
    k_idx = range(9, 10)

    # system parameters
    m = 0.1
    lx = 0.2
    ly = 0.2
    h = 0.01
    r = 0.02
    v_max = 2.0
    omega_max = 2.0
    a_max = 2.0
    beta_max = 2.0
    f_max = 1.0

    # environment parameters
    g = 9.81
    mu_p = 0.1

    # TO parameters
    q_b0 = Quaternion(axis=[np.sqrt(2)/2, -np.sqrt(2)/2, 0.0], degrees=-45)
    r_b0 = [0.354, 0.354, 0.5] + q_b0.q.tolist()
    q_bf = Quaternion()
    r_bf = [0.0, 0.0, 0.0] + q_bf.q.tolist()
    r_m0 = [0.0, 0.0]
    r_mf = [0.0, 0.0]

    v_b0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    v_bf = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    v_m0 = [0.0, 0.0]
    v_mf = [0.0, 0.0]
    
    t_f_guess = 3.0
    eps = 1e-5

    x_init = np.concatenate((r_b0, v_b0, r_m0, v_m0))
    x_final = np.concatenate((r_bf, v_bf, r_mf, v_mf))

    # variable boundary
    lb_r_b = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
    ub_r_b = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

    lb_v_b = [-v_max, -v_max, -v_max, -omega_max, -omega_max, -omega_max]
    ub_v_b = [v_max, v_max, v_max, omega_max, omega_max, omega_max]

    lb_r_m = [-lx/2, -ly/2]
    ub_r_m = [lx/2, ly/2]

    lb_v_m = [-v_max, -v_max]
    ub_v_m = [v_max, v_max]

    lb_f_b = [-f_max, -f_max, 0.0]
    ub_f_b = [f_max, f_max, f_max]

    lb_a_b = [-a_max, -a_max, -a_max, -beta_max, -beta_max, -beta_max]
    ub_a_b = [a_max, a_max, a_max, beta_max, beta_max, beta_max]

    lbx = lb_r_b + lb_v_b + lb_r_m + lb_v_m
    ubx = ub_r_b + ub_v_b + ub_r_m + ub_v_m

    # restrict the auxiliary variable k in the constraints
    lbu = lb_f_b + lb_a_b + [-np.inf]
    ubu = ub_f_b + ub_a_b + [np.inf]

    def __init__(self):
        pass

    def nondimensionalize(self):
        pass

    def x_nondim(self, x):
        return x

    def u_nondim(self, u):
        return u

    def redimensionalize(self):
        pass

    def x_redim(self, x):
        return x

    def u_redim(self, u):
        return u

    def get_equations(self):
        f = sp.zeros(self.n_x, 1)

        x = sp.Matrix(sp.symbols('x_board y_board z_board \
                                  q0_board q1_board q2_board q3_board \
                                  dx_board dy_board dz_board \
                                  omega0_board omega1_board omega2_board\
                                  x_mass y_mass \
                                  dx_mass dy_mass', real=True))
        u = sp.Matrix(sp.symbols('f_tan0 f_tan1 f_norm \
                                  d2x_board d2y_board d2z_board \
                                  beta0_board beta1_board beta2_board \
                                  k', real=True))

        gravity2G = sp.Matrix([0.0, 0.0, -self.m * self.g])
        rotMatB2G = dir_cosine(x[self.q_b_idx, 0])

        omegaBoard2B = x[self.w_b_idx, 0]
        betaBoard2B = u[self.dw_b_idx, 0]
        posMass2B = sp.Matrix([x[self.r_m_idx, 0], 0.0])
        velMass2B = sp.Matrix([x[self.v_m_idx, 0], 0.0])
        inerBoard2B = -self.m * skew(omegaBoard2B) * skew(omegaBoard2B) * posMass2B \
                        -2*self.m * skew(omegaBoard2B) * velMass2B \
                        -self.m * skew(betaBoard2B) * posMass2B \
                        -self.m * rotMatB2G.T * u[self.a_b_idx, 0]

        f[self.r_b_idx.start:self.r_b_idx.stop, 0] = x[self.v_b_idx, 0]
        f[self.q_b_idx.start:self.q_b_idx.stop, 0] = (1 / 2) * omega(x[self.w_b_idx, 0]) * x[self.q_b_idx, 0]
        f[self.v_b_idx.start:self.v_b_idx.stop, 0] = u[self.a_b_idx, 0]
        f[self.w_b_idx.start:self.w_b_idx.stop, 0] = u[self.dw_b_idx, 0]
        f[self.r_m_idx.start:self.r_m_idx.stop, 0] = x[self.v_m_idx, 0]
        f[self.v_m_idx.start:self.v_m_idx.stop, 0] = (1/self.m) * (u[self.f_b_idx, 0][0:2, 0] + (rotMatB2G.T * gravity2G + inerBoard2B)[0:2, 0])

        f = sp.simplify(f)
        A = sp.simplify(f.jacobian(x))
        B = sp.simplify(f.jacobian(u))

        f_func = sp.lambdify((x, u), f, 'numpy')
        A_func = sp.lambdify((x, u), A, 'numpy')
        B_func = sp.lambdify((x, u), B, 'numpy')

        # equality functions (linearized)
        if self.n_eq > 0:
            eq = sp.zeros(self.n_eq, 1)
            eq[0, 0] = u[self.f_b_idx, 0][2, 0] + (rotMatB2G.T * gravity2G + inerBoard2B)[2, 0]
            eq[1, 0] = (u[self.f_b_idx, 0][0:2, 0].norm() - self.mu_p * u[self.f_b_idx, 0][2, 0]) * (x[self.v_m_idx, 0].norm() ** 2)
            eq[2, 0] = (u[self.f_b_idx, 0][0, 0] - u[self.k_idx, 0][0, 0] * x[self.v_m_idx, 0][0, 0]) * (x[self.v_m_idx, 0].norm() ** 2)
            eq[3, 0] = (u[self.f_b_idx, 0][1, 0] - u[self.k_idx, 0][0, 0] * x[self.v_m_idx, 0][1, 0]) * (x[self.v_m_idx, 0].norm() ** 2)

            eq = sp.simplify(eq)
            D_eq = sp.simplify(eq.jacobian(x))
            E_eq = sp.simplify(eq.jacobian(u))
            r_eq = sp.simplify(eq - D_eq * x - E_eq * u)

            self.eq_func = sp.lambdify((x, u), eq, 'numpy')
            self.D_eq_func = sp.lambdify((x, u), D_eq, 'numpy')
            self.E_eq_func = sp.lambdify((x, u), E_eq, 'numpy')
            self.r_eq_func = sp.lambdify((x, u), r_eq, 'numpy')

            self.s_prime_eq = cvx.Variable((self.n_eq, K), nonneg=False)

        # inequality functions
        if self.n_ineq > 0:
            ineq = sp.zeros(self.n_ineq, 1)

            """
                ADD INEQUALITY FUNCTIONS HERE
            """

            ineq = sp.simplify(ineq)
            D_ineq = sp.simplify(ineq.jacobian(x))
            E_ineq = sp.simplify(ineq.jacobian(u))
            r_ineq = sp.simplify(ineq - D_ineq * x - E_ineq * u)

            self.ineq_func = sp.lambdify((x, u), ineq, 'numpy')
            self.D_ineq_func = sp.lambdify((x, u), D_ineq, 'numpy')
            self.E_ineq_func = sp.lambdify((x, u), E_ineq, 'numpy')
            self.r_ineq_func = sp.lambdify((x, u), r_ineq, 'numpy')

            self.s_prime_ineq = cvx.Variable((self.n_ineq, K), nonneg=True)

        return f_func, A_func, B_func

    def initialize_trajectory(self, X, U):
        for k in range(K):
            alpha1 = (K - k) / K
            alpha2 = k / K

            r_board = alpha1 * self.x_init[self.r_b_idx] + alpha2 * self.x_final[self.r_b_idx]
            q_board = Quaternion.slerp(Quaternion(self.x_init[self.q_b_idx]), \
                                       Quaternion(self.x_final[self.q_b_idx]), \
                                       alpha2).q
            v_board = (self.x_final[self.r_b_idx] - self.x_init[self.r_b_idx]) / self.t_f_guess
            w_board = angvel(self.x_init[self.q_b_idx], self.x_final[self.q_b_idx], self.t_f_guess)

            r_mass = alpha1 * self.x_init[self.r_m_idx] + alpha2 * self.x_final[self.r_m_idx]
            v_mass = (self.x_final[self.r_m_idx] - self.x_init[self.r_m_idx]) / self.t_f_guess

            gravity = self.m * self.g * np.array([0.0, 0.0, -1.0])
            rotmat_board = np.array(dir_cosine(q_board).tolist(), dtype=np.float)
            norm_board = rotmat_board[:, 2]
            f_norm = -norm_board.dot(gravity)
            f_tan = -np.matmul(rotmat_board.T, (gravity + f_norm*norm_board))[0:2].flatten()

            if np.linalg.norm(v_mass) > 0:
                aux_k = -self.mu_p * f_norm / np.linalg.norm(v_mass)
            else:
                aux_k = -self.eps
            
            X[:, k] = np.concatenate((r_board, q_board, v_board, w_board, r_mass, v_mass))
            U[:, k] = np.concatenate((f_tan, [f_norm], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, aux_k]))

        return X, U

    def get_objective(self, X_v, U_v, X_last_p, U_last_p):
        """
        Get model specific objective to be minimized.

        :param X_v: cvx variable for current states
        :param U_v: cvx variable for current inputs
        :param X_last_p: cvx parameter for last states
        :param U_last_p: cvx parameter for last inputs
        :return: A cvx objective function.
        """
        return cvx.Minimize(1e5 * cvx.pnorm(self.s_prime_eq, 1))

    def get_constraints(self, X_v, U_v, \
                        D_eq, E_eq, r_eq, \
                        D_ineq, E_ineq, r_ineq):
        # initial and terminal constraints
        constraints = [
            X_v[:, 0] == self.x_init,
            X_v[:, -1] == self.x_final
        ]

        # variable boundary constraints
        constraints += [
            # constrain the state
            X_v <= cvx.reshape(self.ubx, (self.n_x, 1)),
            -X_v <= -cvx.reshape(self.lbx, (self.n_x, 1)),

            # constain the input
            U_v <= cvx.reshape(self.ubu, (self.n_u, 1)),
            -U_v <= -cvx.reshape(self.lbu, (self.n_u, 1)),
        ]

        # convex equality constraints
        constraints += []

        # convex inequality constraints
        constraints += [
            cvx.reshape(cvx.norm(U_v[self.f_b_idx, :][0:2, :], axis=0) - self.mu_p * U_v[self.f_b_idx, :][2, :], (K, 1)) <= 0,
            U_v[self.k_idx, :] + self.eps <= 0,
        ]

        # linearized equality constraints
        lhs_eq = [cvx.reshape(D_eq[:, k], (self.n_eq, self.n_x)) * X_v[:, k]
                  + cvx.reshape(E_eq[:, k], (self.n_eq, self.n_u)) * U_v[:, k]
                  + r_eq[:, k]
                  for k in range(K)]
        constraints += [
            cvx.reshape(cvx.hstack(lhs_eq), (K*self.n_eq, 1)) == cvx.reshape(self.s_prime_eq, (K*self.n_eq, 1)),
        ]

        return constraints
    
    def calculate_constraint_linearization(self, X_last_p, U_last_p):
        D_eq = np.empty((self.n_eq * self.n_x, K))
        E_eq = np.empty((self.n_eq * self.n_u, K))
        r_eq = np.empty((self.n_eq * 1, K))

        D_ineq = np.empty((self.n_ineq * self.n_x, K))
        E_ineq = np.empty((self.n_ineq * self.n_u, K))
        r_ineq = np.empty((self.n_ineq * 1, K))

        # linearized equality constraints
        if self.n_eq > 0:
            for k in range(K):
                D_eq[:, k] = self.D_eq_func(X_last_p[:, k], U_last_p[:, k]).reshape((self.n_eq, self.n_x)).flatten(order='F')
                E_eq[:, k] = self.E_eq_func(X_last_p[:, k], U_last_p[:, k]).reshape((self.n_eq, self.n_u)).flatten(order='F')
                r_eq[:, k] = self.r_eq_func(X_last_p[:, k], U_last_p[:, k]).flatten(order='F')
        else:
            D_eq, E_eq, r_eq = None, None, None

        # linearized inequality constraints
        if self.n_ineq > 0:
            for k in range(K):
                D_ineq[:, k] = self.D_ineq_func(X_last_p[:, k], U_last_p[:, k]).reshape((self.n_ineq, self.n_x)).flatten(order='F')
                E_ineq[:, k] = self.E_ineq_func(X_last_p[:, k], U_last_p[:, k]).reshape((self.n_ineq, self.n_u)).flatten(order='F')
                r_ineq[:, k] = self.r_ineq_func(X_last_p[:, k], U_last_p[:, k]).flatten(order='F')
        else:
            D_ineq, E_ineq, r_ineq = None, None, None

        return D_eq, E_eq, r_eq, D_ineq, E_ineq, r_ineq

    def get_linear_cost(self):
        cost = np.linalg.norm(self.s_prime_eq.value.flatten(), ord=1)
        return cost

    def get_nonlinear_cost(self, X=None, U=None):
        cost = 0.0
        # equality constraints
        for k in range(K):
            eq_eval = self.eq_func(X[:, k], U[:, k]).flatten()
            cost += np.linalg.norm(eq_eval, ord=1)
        return cost
