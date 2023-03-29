import sympy as sp
import numpy as np
import cvxpy as cvx

from global_parameters import K

def rotmat_anticlk(a):
    ca, sa = sp.cos(a), sp.sin(a)
    return sp.Matrix([
        [ca, -sa],
        [sa, ca],
    ])

class Model:
    # dimensions
    n_x = 8
    n_u = 9
    # n_z = 2

    n_eq = 5
    n_ineq = 1

    # system parameters
    m = 0.1
    l = 0.2
    h = 0.01
    r = 0.02
    v_max = 2.0
    omega_max = 2.0
    a_max = 2.0
    beta_max = 2.0
    f_max = 1.0

    # environment parameters
    g = 9.81
    mu_g = 0.2
    mu_p = 0.1

    # TO parameters
    r0 = np.array([0.0, 0.0, 0.0, 0.0])
    rf = np.array([0.5, 0.5, -0.25*np.pi, 0.0])
    v0 = np.array([0.0, 0.0, 0.0, 0.0])
    vf = np.array([0.0, 0.0, 0.0, 0.0])
    
    t_f_guess = 3.0

    x_init = np.concatenate((r0, v0))
    x_final = np.concatenate((rf, vf))

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

        x = sp.Matrix(sp.symbols('x_board y_board theta_board x_mass dx_board dy_board dtheta_board dx_mass', real=True))
        u = sp.Matrix(sp.symbols('f_norm f_tan fx_ground fy_ground dx_mass+ dx_mass- d2x_board d2y_board d2theta_board', real=True))

        velBoard2G = x[4:6, 0]
        rotMatB2G = rotmat_anticlk(x[2, 0])
        velMass2G = velBoard2G + rotMatB2G * sp.Matrix([x[7, 0], x[6, 0]*x[3, 0]])
        fricGround2G = u[2:4, 0]
        fricGround2B = rotMatB2G.T * fricGround2G
        inerBoard2B = sp.Matrix([self.m*(x[6, 0]**2)*x[3, 0], -2*self.m*x[6, 0]*x[7, 0]]) - \
                        self.m * rotMatB2G.T * u[6:8, 0]

        f[0:4, 0] = x[4:8, 0]
        f[4:7, 0] = u[6:9, 0]
        f[7, 0] = (1/self.m) * (u[1, 0] + fricGround2B[0, 0] + inerBoard2B[0, 0])

        f = sp.simplify(f)
        A = sp.simplify(f.jacobian(x))
        B = sp.simplify(f.jacobian(u))

        f_func = sp.lambdify((x, u), f, 'numpy')
        A_func = sp.lambdify((x, u), A, 'numpy')
        B_func = sp.lambdify((x, u), B, 'numpy')

        # equality functions (linearized)
        eq = sp.zeros(self.n_eq, 1)
        eq[0, 0] = (1/self.m) * (u[0, 0] + fricGround2B[1, 0] + inerBoard2B[1, 0])
        eq[1, 0] = fricGround2G[0, 0] * velMass2G[1, 0] - fricGround2G[1, 0] * velMass2G[0, 0]
        eq[2, 0] = fricGround2G.norm(ord=2) - (self.mu_g * self.m * self.g)
        # eq3 = x[7, :] - (u[4, :] - u[5, :])
        eq[3, 0] = (self.mu_p * u[0, 0] - u[1, 0]) * u[5, :]
        eq[4, 0] = (self.mu_p * u[0, 0] + u[1, 0]) * u[4, :]

        eq = sp.simplify(eq)
        D_eq = sp.simplify(eq.jacobian(x))
        E_eq = sp.simplify(eq.jacobian(u))
        r_eq = sp.simplify(eq - D_eq * x - E_eq * u)

        self.eq_func = sp.lambdify((x, u), eq, 'numpy')
        self.D_eq_func = sp.lambdify((x, u), D_eq, 'numpy')
        self.E_eq_func = sp.lambdify((x, u), E_eq, 'numpy')
        self.r_eq_func = sp.lambdify((x, u), r_eq, 'numpy')

        # inequality functions
        ineq = sp.zeros(self.n_ineq, 1)
        # ineq0 = -(self.mu_p * u[0, 0] - u[1, 0])
        # ineq1 = -(self.mu_p * u[0, 0] + u[1, 0])
        ineq[0, 0] = u[2, 0] * velMass2G[0, 0] + u[3, 0] * velBoard2G[1, 0]

        ineq = sp.simplify(ineq)
        D_ineq = sp.simplify(ineq.jacobian(x))
        E_ineq = sp.simplify(ineq.jacobian(u))
        r_ineq = sp.simplify(ineq - D_ineq * x - E_ineq * u)

        self.ineq_func = sp.lambdify((x, u), ineq, 'numpy')
        self.D_ineq_func = sp.lambdify((x, u), D_ineq, 'numpy')
        self.E_ineq_func = sp.lambdify((x, u), E_ineq, 'numpy')
        self.r_ineq_func = sp.lambdify((x, u), r_ineq, 'numpy')

        self.s_prime_eq = cvx.Variable((self.n_eq, K), nonneg=False)
        self.s_prime_ineq = cvx.Variable((self.n_ineq, K), nonneg=True)

        # other function evaluations (all refered to ground)
        self.f_iner_b_func = sp.lambdify((x, u), sp.simplify(rotMatB2G * inerBoard2B), 'numpy')
        self.f_fric_b_func = sp.lambdify((x, u), sp.simplify(rotMatB2G * sp.Matrix([u[1, 0], u[0, 0]])), 'numpy')
        self.f_fric_g_func = sp.lambdify((x, u), sp.simplify(fricGround2G), 'numpy')
        self.v_mass_g_func = sp.lambdify((x, u), sp.simplify(velMass2G), 'numpy')

        return f_func, A_func, B_func

    def initialize_trajectory(self, X, U):
        for k in range(K):
            alpha1 = (K - k) / K
            alpha2 = k / K

            r_board = alpha1 * self.x_init[0:3] + alpha2 * self.x_final[0:3]
            r_mass = alpha1 * self.x_init[3] + alpha2 * self.x_final[3]
            v_board = self.x_final[0:3] - self.x_init[0:3]
            v_mass = self.x_final[3] - self.x_init[3]

            f_norm = self.mu_g * self.m * self.g
            f_ground = - f_norm * (v_board[0:2] / np.linalg.norm(v_board[0:2]))
            
            X[:, k] = np.concatenate((r_board, [r_mass], v_board, [v_mass]))
            U[:, k] = np.concatenate(([f_norm], [0.0], f_ground, [0.0, 0.0, 0.0, 0.0, 0.0]))

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
        return cvx.Minimize(1e5 * (cvx.pnorm(self.s_prime_eq, 1) + cvx.sum(self.s_prime_ineq)))

    def get_constraints(self, X_v, U_v, \
                        D_eq, E_eq, r_eq, \
                        D_ineq, E_ineq, r_ineq):
        # initial and terminal constraints
        constraints = [
            X_v[:, 0] == self.x_init,
            # X_v[0:3, -1] == self.x_final[0:3],
            # X_v[4:, -1] == self.x_final[4:]
            X_v[:, -1] == self.x_final
        ]

        # variable boundary constraints
        constraints += [
            # keep the mass on board
            X_v[3, :] <= self.l/2,
            -X_v[3, :] <= self.l/2,
            # constrain the velocity
            X_v[4:6, :] <= self.v_max,
            -X_v[4:6, :] <= self.v_max,
            X_v[6, :] <= self.omega_max,
            -X_v[6, :] <= self.omega_max,
            X_v[7, :] <= self.v_max,
            -X_v[7, :] <= self.v_max,

            # constain the input
            -U_v[0, :] <= 0.0,
            U_v[0:4, :] <= self.f_max,
            -U_v[1:4, :] <= self.f_max,
            U_v[4:6, :] <= self.v_max,
            -U_v[4:6, :] <= 0.0,
            U_v[6:8, :] <= self.a_max,
            -U_v[6:8, :] <= self.a_max,
            U_v[8, :] <= self.beta_max,
            -U_v[8, :] <= self.beta_max,
        ]

        # convex equality constraints
        constraints += [
            X_v[7, :] - (U_v[4, :] - U_v[5, :]) == 0,
        ]

        # convex inequality constraints
        constraints += [
            -(self.mu_p * U_v[0, :] - U_v[1, :]) <= 0,
            -(self.mu_p * U_v[0, :] + U_v[1, :]) <= 0,
        ]

        # linearized equality constraints
        lhs_eq = [cvx.reshape(D_eq[:, k], (self.n_eq, self.n_x)) * X_v[:, k]
                  + cvx.reshape(E_eq[:, k], (self.n_eq, self.n_u)) * U_v[:, k]
                  + r_eq[:, k]
                  for k in range(K)]
        constraints += [
            cvx.reshape(cvx.hstack(lhs_eq), (K*self.n_eq, 1)) == cvx.reshape(self.s_prime_eq, (K*self.n_eq, 1)),
        ]

        # linearized inequality constraints
        lhs_ineq = [cvx.reshape(D_ineq[:, k], (self.n_ineq, self.n_x)) * X_v[:, k]
                    + cvx.reshape(E_ineq[:, k], (self.n_ineq, self.n_u)) * U_v[:, k]
                    + r_ineq[:, k]
                    for k in range(K)]
        constraints += [
            cvx.reshape(cvx.hstack(lhs_ineq), (K*self.n_ineq, 1)) <= cvx.reshape(self.s_prime_ineq, (K*self.n_ineq, 1)),
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
        for k in range(K):
            D_eq[:, k] = self.D_eq_func(X_last_p[:, k], U_last_p[:, k]).reshape((self.n_eq, self.n_x)).flatten(order='F')
            E_eq[:, k] = self.E_eq_func(X_last_p[:, k], U_last_p[:, k]).reshape((self.n_eq, self.n_u)).flatten(order='F')
            r_eq[:, k] = self.r_eq_func(X_last_p[:, k], U_last_p[:, k]).flatten(order='F')

        # linearized inequality constraints
        for k in range(K):
            D_ineq[:, k] = self.D_ineq_func(X_last_p[:, k], U_last_p[:, k]).reshape((self.n_ineq, self.n_x)).flatten(order='F')
            E_ineq[:, k] = self.E_ineq_func(X_last_p[:, k], U_last_p[:, k]).reshape((self.n_ineq, self.n_u)).flatten(order='F')
            r_ineq[:, k] = self.r_ineq_func(X_last_p[:, k], U_last_p[:, k]).flatten(order='F')

        return D_eq, E_eq, r_eq, D_ineq, E_ineq, r_ineq

    def get_linear_cost(self):
        cost = np.linalg.norm(self.s_prime_eq.value.flatten(), ord=1) + np.sum(self.s_prime_ineq.value)
        return cost

    def get_nonlinear_cost(self, X=None, U=None):
        cost = 0.0
        # equality constraints
        for k in range(K):
            eq_eval = self.eq_func(X[:, k], U[:, k]).flatten()
            cost += np.linalg.norm(eq_eval, ord=1)
        # inequality constraints
        for k in range(K):
            ineq_eval = self.ineq_func(X[:, k], U[:, k]).flatten()
            cost += np.maximum(ineq_eval, np.zeros_like(ineq_eval)).sum()
        return cost
