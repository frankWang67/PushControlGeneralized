import casadi as cs
import dyn_sliding_pack


def make_discretize_variables(pbm:dyn_sliding_pack.params.DynPusherOptimizationConfig, N):
    # dimensions
    nx = pbm.x.shape[0]
    nu = pbm.u.shape[0]
    np = pbm.p.shape[0]
    nz = pbm.z.shape[0]

    # symbols
    X = cs.SX.sym('X', nx, N)
    U = cs.SX.sym('U', nu, N)
    P = cs.SX.sym('P', np)
    Z = cs.SX.sym('Z', nz, N)

    return (X, U, P, Z)

def discretize(pbm:dyn_sliding_pack.params.DynPusherOptimizationConfig, N):
    # get discretization variables
    X, U, P, Z = make_discretize_variables(pbm, N)

    dt = P[0] / N

    casadi_nlpopt = dyn_sliding_pack.params.CasADiNLPOptions()
    casadi_nlpopt.f = []
    casadi_nlpopt.x = []
    casadi_nlpopt.p = []
    casadi_nlpopt.g = []

    casadi_nlpopt.lbx = []
    casadi_nlpopt.ubx = []
    casadi_nlpopt.lbg = []
    casadi_nlpopt.ubg = []

    # discretize cost
    casadi_nlpopt.f = pbm.phi(X[:, -1], P)
    for i in range(N-1):
        casadi_nlpopt.f += (dt/2) * (pbm.gamma(X[:, i], U[:, i], P) + \
                                     pbm.gamma(X[:, i+1], U[:, i+1], P))

    # discretize dynamics
    for i in range(N-1):
        casadi_nlpopt.g += ((dt/2) * (pbm.f(X[:, i], U[:, i], P) + \
                                      pbm.f(X[:, i+1], U[:, i+1], P)) - \
                            (X[:, i+1] - X[:, i])).elements()
    casadi_nlpopt.lbg += [0.0] * (N-1)
    casadi_nlpopt.ubg += [0.0] * (N-1)

    # equality path constraints
    for i in range(N):
        casadi_nlpopt.g += pbm.h(X[:, i], U[:, i], P, Z[:, i])
    casadi_nlpopt.lbg += [0.0] * N
    casadi_nlpopt.ubg += [0.0] * N

    # unequality path constraints
    for i in range(N):
        casadi_nlpopt.g += pbm.g(X[:, i], U[:, i], P, Z[:, i])
    casadi_nlpopt.lbg += [-cs.inf] * N
    casadi_nlpopt.ubg += [0.0] * N

    # boundary constraints
    casadi_nlpopt.g += pbm.hi(X[:, 0], U[:, 0], P)
    casadi_nlpopt.lbg += [0.0]
    casadi_nlpopt.ubg += [0.0]
    casadi_nlpopt.g += pbm.ht(X[:, -1], U[:, -1], P)
    casadi_nlpopt.lbg += [0.0]
    casadi_nlpopt.ubg += [0.0]

    # variable constraints
    for i in range(N):
        casadi_nlpopt.x += X[:, i].elements()
    casadi_nlpopt.lbx += pbm.lbx * N
    casadi_nlpopt.ubx += pbm.ubx * N
    casadi_nlpopt.discrete += [False] * N

    for i in range(N):
        casadi_nlpopt.x += U[:, i].elements()
    casadi_nlpopt.lbx += pbm.lbu * N
    casadi_nlpopt.ubx += pbm.ubu * N
    casadi_nlpopt.discrete += [False] * N

    for i in range(N):
        casadi_nlpopt.x += Z[:, i].elements()
    casadi_nlpopt.lbx += pbm.lbz * N
    casadi_nlpopt.ubx += pbm.ubz * N
    casadi_nlpopt.discrete += [False] * N

    # terminal time
    casadi_nlpopt.p += P.elements()

    return casadi_nlpopt

