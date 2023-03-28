"""
    1D Non-prehensile Dynamic Pushing
"""
import casadi as cs
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

class DynPusherToyProblem():
    m = None
    mu_g = None
    g = None

    v_max = None
    a_max = None
    f_max = None

    ri = None
    vi = None
    rf = None
    vf = None
    tf_min = None
    tf_max = None

    lbx = None
    ubx = None
    lbu = None
    ubu = None
    lbp = None
    ubp = None

    dyn_con = None
    eq_con = None
    ibd_con = None
    fbd_con = None
    phi_cst = None
    gamma_cst = None

    var_x = None
    var_u = None
    var_p = None

    opt_x0 = None
    opt_x = None
    opt_f = None
    opt_g = None
    opt_p = None
    opt_lbg = None
    opt_ubg = None
    opt_lbx = None
    opt_ubx = None
    opt_discrete = None

    opt_param = None
    opt_cfg = None

def set_dynamics(pbm: DynPusherToyProblem):
    # constants
    __kMass = pbm.m
    __kMuGround = pbm.mu_g
    __kGravity = pbm.g
    __kPosInit = pbm.ri
    __kVelInit = pbm.vi
    __kPosFinish = pbm.rf
    __kVelFinish = pbm.vf
    __kMinFinishTime = pbm.tf_min
    __kMaxFinishTime = pbm.tf_max
    __kMaxLinVel = pbm.v_max
    __kMaxLinAcc = pbm.a_max
    __kMaxForce = pbm.f_max

    # variables
    x = cs.SX.sym('x', 2)
    u = cs.SX.sym('u', 2)
    p = cs.SX.sym('p', 1)

    pbm.var_x = x
    pbm.var_u = u
    pbm.var_p = p

    __x_board = cs.SX.sym('x_board')
    __dx_board = cs.SX.sym('dx_board')
    __r = __x_board
    __v = __dx_board
    __x = cs.veccat(__r, __v)

    __d2x_board = cs.SX.sym('d2x_board')
    __f_norm = cs.SX.sym('f_norm')
    __u = cs.veccat(__d2x_board, __f_norm)

    __t_finish = cs.SX.sym('t_finish')
    __p = cs.veccat(__t_finish)

    # bound on variables
    pbm.lbx = [-cs.inf, -__kMaxLinVel]
    pbm.ubx = [cs.inf, __kMaxLinVel]

    pbm.lbu = [-__kMaxLinAcc, 0.0]
    pbm.ubu = [__kMaxLinAcc, __kMaxForce]

    pbm.lbp = [__kMinFinishTime]
    pbm.ubp = [__kMaxFinishTime]

    # dynamics
    __f = cs.vertcat(__dx_board, __d2x_board)

    # equality constraint
    __eq0 = __d2x_board - (1/__kMass)*(__f_norm - __kMuGround * __kMass * __kGravity)
    __h = cs.vertcat(__eq0)

    # boundary constraint
    __hi = cs.vertcat(__r - __kPosInit, __v - __kVelInit)
    __hf = cs.vertcat(__r - __kPosFinish, __v - __kVelFinish)

    # cost
    __phi = (__t_finish / __kMaxFinishTime) ** 2
    __gamma = cs.norm_2(__u / pbm.ubu) ** 2

    # set problem config
    pbm.dyn_con = cs.Function('f', [__x, __u, __p], [__f], ['x', 'u', 'p'], ['f'])
    pbm.eq_con = cs.Function('h', [__x, __u, __p], [__h], ['x', 'u', 'p'], ['h'])
    pbm.ibd_con = cs.Function('hi', [__x, __u, __p], [__hi], ['x', 'u', 'p'], ['hi'])
    pbm.fbd_con = cs.Function('hf', [__x, __u, __p], [__hf], ['x', 'u', 'p'], ['hf'])
    pbm.phi_cst = cs.Function('phi', [__x, __u, __p], [__phi], ['x', 'u', 'p'], ['phi'])
    pbm.gamma_cst = cs.Function('gamma', [__x, __u, __p], [__gamma], ['x', 'u', 'p'], ['gamma'])

def set_discretization(pbm: DynPusherToyProblem, N):
    nx = pbm.var_x.shape[0]
    nu = pbm.var_u.shape[0]
    np = pbm.var_p.shape[0]

    X = cs.SX.sym('X', nx, N)
    U = cs.SX.sym('U', nu, N)
    P = cs.SX.sym('P', np)

    dt = P[0] / N

    pbm.opt_x = []
    pbm.opt_f = []
    pbm.opt_g = []
    pbm.opt_p = []

    pbm.opt_lbg = []
    pbm.opt_ubg = []
    pbm.opt_lbx = []
    pbm.opt_ubx = []

    pbm.opt_discrete = []

    pbm.opt_f = 1.0 * pbm.phi_cst(X[:, -1], U[:, -1], P)
    for i in range(N-1):
        pbm.opt_f += (dt/2) * (pbm.gamma_cst(X[:, i], U[:, i], P) + \
                               pbm.gamma_cst(X[:, i+1], U[:, i+1], P))

    for i in range(N-1):
        pbm.opt_g += ((dt/2) * (pbm.dyn_con(X[:, i], U[:, i], P) + \
                                pbm.dyn_con(X[:, i+1], U[:, i+1], P)) - \
                      (X[:, i+1] - X[:, i])).elements()
    pbm.opt_lbg += [0.0] * (N-1) * pbm.dyn_con.size_out('f')[0]
    pbm.opt_ubg += [0.0] * (N-1) * pbm.dyn_con.size_out('f')[0]

    for i in range(N-1):
        pbm.opt_g += pbm.eq_con(X[:, i], U[:, i], P).elements()
    pbm.opt_lbg += [0.0] * (N-1) * pbm.eq_con.size_out('h')[0]
    pbm.opt_ubg += [0.0] * (N-1) * pbm.eq_con.size_out('h')[0]

    pbm.opt_g += pbm.ibd_con(X[:, 0], U[:, 0], P).elements()
    pbm.opt_lbg += [0.0] * pbm.ibd_con.size_out('hi')[0]
    pbm.opt_ubg += [0.0] * pbm.ibd_con.size_out('hi')[0]
    pbm.opt_g += pbm.fbd_con(X[:, -1], U[:, -1], P).elements()
    pbm.opt_lbg += [0.0] * pbm.fbd_con.size_out('hf')[0]
    pbm.opt_ubg += [0.0] * pbm.fbd_con.size_out('hf')[0]

    for i in range(N):
        pbm.opt_x += X[:, i].elements()
    pbm.opt_lbx += pbm.lbx * N
    pbm.opt_ubx += pbm.ubx * N
    pbm.opt_discrete += [False] * nx * N

    for i in range(N):
        pbm.opt_x += U[:, i].elements()
    pbm.opt_lbx += pbm.lbu * N
    pbm.opt_ubx += pbm.ubu * N
    pbm.opt_discrete += [False] * nu * N

    ## fixed final time
    pbm.opt_p += P.elements()

    ## free final time
    # pbm.opt_x += P.elements()
    # pbm.opt_lbx += pbm.lbp
    # pbm.opt_ubx += pbm.ubp
    # pbm.opt_discrete += [False]

def set_solver_configuration(pbm: DynPusherToyProblem):
    pbm.opt_cfg = {}
    pbm.opt_cfg['print_time'] = 0
    pbm.opt_cfg['ipopt.print_level'] = 5
    pbm.opt_cfg['ipopt.max_iter'] = 3000
    pbm.opt_cfg['ipopt.jac_d_constant'] = 'no'
    pbm.opt_cfg['ipopt.warm_start_init_point'] = 'yes'
    pbm.opt_cfg['ipopt.hessian_constant'] = 'no'
    pbm.opt_cfg['discrete'] = pbm.opt_discrete

def set_init_guess(pbm: DynPusherToyProblem, N):
    ri = pbm.ri
    rf = pbm.rf
    r_interp = interp1d([0, N-1], np.c_[ri, rf], 'linear')(np.arange(0, N, 1))
    
    tf = (pbm.tf_min + pbm.tf_max)/2
    v_traj = ((np.array(rf) - np.array(ri)) / tf).reshape(-1, 1).repeat(N, -1)

    x_traj = np.concatenate((r_interp, v_traj), axis=0)
    assert x_traj.shape[0] == pbm.var_x.shape[0]

    u_traj = np.zeros((pbm.var_u.shape[0], N))
    u_traj[1, :] = np.ones(N) * pbm.mu_g * pbm.m * pbm.g

    x0 = []
    for i in range(N):
        x0 += x_traj[:, i].tolist()
    for i in range(N):
        x0 += u_traj[:, i].tolist()
    
    ## free final time
    # x0 += [tf]

    pbm.opt_x0 = x0

def get_solution(pbm: DynPusherToyProblem, solver_name, tf):
    prob = {
            'f': pbm.opt_f,
            'x': cs.vertcat(*pbm.opt_x),
            'g': cs.vertcat(*pbm.opt_g),
            'p': cs.vertcat(*pbm.opt_p)
            }
    
    param = []

    ## fixed final time
    param += [tf]
    ## free final time
    # pass

    pbm.opt_param = param
    
    solver = cs.nlpsol('solver', solver_name, prob, pbm.opt_cfg)

    sol = solver(x0=pbm.opt_x0,
                lbx=pbm.opt_lbx, ubx=pbm.opt_ubx,
                lbg=pbm.opt_lbg, ubg=pbm.opt_ubg,
                p=param)
    
    return sol, solver.stats()

def plot_solution(pbm: DynPusherToyProblem, sol, N, tf):
    nx = pbm.var_x.shape[0]
    nu = pbm.var_u.shape[0]

    var_sol = sol['x']  # variables
    con_sol = sol['g']  # constraints

    x = []
    x_sol = var_sol[:N*nx]
    for i in range(N):
        x.append(x_sol[i*nx:(i+1)*nx].toarray().flatten().tolist())

    u = []
    u_sol = var_sol[N*nx:N*nx+N*nu]
    for i in range(N):
        u.append(u_sol[i*nx:(i+1)*nx].toarray().flatten().tolist())

    tf_sol = var_sol[-1].toarray().flatten()[0]

    x = np.array(x)
    u = np.array(u)
    t_grid = np.linspace(0, tf_sol, N)

    # plot optimization variables

    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(t_grid, x[:, 0], label='x0')
    ax.set_xlabel('t(s)')
    ax.set_ylabel('x0(m)')
    ax.legend(); ax.grid('on')

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t_grid, x[:, 1], label='x1')
    ax.set_xlabel('t(s)')
    ax.set_ylabel('x1(m)')
    ax.legend(); ax.grid('on')

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(t_grid, u[:, 0], label='u0')
    ax.set_xlabel('t(s)')
    ax.set_ylabel('u0(m/s^2)')
    ax.legend(); ax.grid('on')

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(t_grid, u[:, 1], label='u1')
    ax.set_xlabel('t(s)')
    ax.set_ylabel('u1(N)')
    ax.legend(); ax.grid('on')

    # plot constraints
    p = pbm.opt_param
    dt = tf_sol/N
    g0 = []
    for i in range(N-1):
        eval = (dt/2) * (pbm.dyn_con(x[i], u[i], p) + \
                         pbm.dyn_con(x[i+1], u[i+1], p)) - \
                        (x[i+1] - x[i])
        g0.append(eval.toarray().flatten().tolist())
    g1 = []
    for i in range(N-1):
        eval = pbm.eq_con(x[i], u[i], p).elements()
        g1.append(eval)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(t_grid[:-1], np.array(g0)[:, 0], label='dyn_0')
    ax.set_xlabel('t(s)')
    ax.set_ylabel('dyn_0(m/s)')
    ax.legend(); ax.grid('on')

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t_grid[:-1], np.array(g0)[:, 1], label='dyn_1')
    ax.set_xlabel('t(s)')
    ax.set_ylabel('dyn_1(m/s^2)')
    ax.legend(); ax.grid('on')

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(t_grid[:-1], np.array(g1)[:, 0], label='eq_0')
    ax.set_xlabel('t(s)')
    ax.set_ylabel('eq_0(m/s^2)')
    ax.legend(); ax.grid('on')

    plt.show()

def main(N=100, tf=2.5):
    pbm = DynPusherToyProblem()
    pbm.m = 0.1
    pbm.mu_g = 0.2
    pbm.g = 9.81

    pbm.v_max = 1.0
    pbm.a_max = 2.0
    pbm.f_max = 0.6

    pbm.ri = [0.0]
    pbm.vi = [0.0]
    pbm.rf = [1.0]
    pbm.vf = [0.0]
    pbm.tf_min = 2.0
    pbm.tf_max = 3.0

    set_dynamics(pbm)
    set_discretization(pbm, N)
    set_init_guess(pbm, N)
    set_solver_configuration(pbm)
    sol, stats = get_solution(pbm, 'ipopt', tf)
    plot_solution(pbm, sol, N, tf)

if __name__ == '__main__':
    main(N=200, tf=3.0)
