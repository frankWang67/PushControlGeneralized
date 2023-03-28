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
    nx = X.shape[0]
    nu = U.shape[0]
    np = P.shape[0]
    nz = Z.shape[0]

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

    casadi_nlpopt.discrete = []

    # discretize cost
    # casadi_nlpopt.f = pbm.phi(X[:, -1], P)
    casadi_nlpopt.f = cs.SX(0)
    for i in range(N-1):
        casadi_nlpopt.f += (dt/2) * (pbm.gamma(X[:, i], U[:, i], P, Z[:, i]) + \
                                     pbm.gamma(X[:, i+1], U[:, i+1], P, Z[:, i+1]))

    # discretize dynamics
    for i in range(N-1):
        casadi_nlpopt.g += ((dt/2) * (pbm.f(X[:, i], U[:, i], P, Z[:, i]) + \
                                      pbm.f(X[:, i+1], U[:, i+1], P, Z[:, i+1])) - \
                            (X[:, i+1] - X[:, i])).elements()
    casadi_nlpopt.lbg += [0.0] * (N-1) * pbm.f.size_out('f')[0]
    casadi_nlpopt.ubg += [0.0] * (N-1) * pbm.f.size_out('f')[0]

    # equality path constraints
    for i in range(N):
        casadi_nlpopt.g += pbm.h(X[:, i], U[:, i], P, Z[:, i]).elements()
    casadi_nlpopt.lbg += [0.0] * N * pbm.h.size_out('h')[0]
    casadi_nlpopt.ubg += [0.0] * N * pbm.h.size_out('h')[0]

    # unequality path constraints
    for i in range(N):
        casadi_nlpopt.g += pbm.g(X[:, i], U[:, i], P, Z[:, i]).elements()
    casadi_nlpopt.lbg += [-cs.inf] * N * pbm.g.size_out('g')[0]
    casadi_nlpopt.ubg += [0.0] * N * pbm.g.size_out('g')[0]

    # boundary constraints
    casadi_nlpopt.g += pbm.hi(X[:, 0], U[:, 0], P).elements()
    casadi_nlpopt.lbg += [0.0] * pbm.hi.size_out('hi')[0]
    casadi_nlpopt.ubg += [0.0] * pbm.hi.size_out('hi')[0]
    casadi_nlpopt.g += pbm.ht(X[:, -1], U[:, -1], P).elements()
    casadi_nlpopt.lbg += [0.0] * pbm.ht.size_out('ht')[0]
    casadi_nlpopt.ubg += [0.0] * pbm.ht.size_out('ht')[0]

    # variable constraints
    for i in range(N):
        casadi_nlpopt.x += X[:, i].elements()
    casadi_nlpopt.lbx += pbm.lbx * N
    casadi_nlpopt.ubx += pbm.ubx * N
    casadi_nlpopt.discrete += [False] * N * nx

    for i in range(N):
        casadi_nlpopt.x += U[:, i].elements()
    casadi_nlpopt.lbx += pbm.lbu * N
    casadi_nlpopt.ubx += pbm.ubu * N
    casadi_nlpopt.discrete += [False] * N * nu

    for i in range(N):
        casadi_nlpopt.x += Z[:, i].elements()
    casadi_nlpopt.lbx += pbm.lbz * N
    casadi_nlpopt.ubx += pbm.ubz * N
    casadi_nlpopt.discrete += [False] * N * nz

    # terminal time
    casadi_nlpopt.p += P.elements()

    return casadi_nlpopt

