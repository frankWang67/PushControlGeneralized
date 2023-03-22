import casadi as cs
import dyn_sliding_pack


def configure(pbm:dyn_sliding_pack.params.CasADiNLPOptions):
    pbm.cfg = {}
    pbm.cfg['print_time'] = 0
    pbm.cfg['ipopt.print_level'] = 0
    pbm.cfg['ipopt.max_iter'] = 3000
    pbm.cfg['ipopt.jac_d_constant'] = 'yes'
    pbm.cfg['ipopt.warm_start_init_point'] = 'yes'
    pbm.cfg['ipopt.hessian_constant'] = 'yes'
    pbm.cfg['discrete'] = pbm.discrete

def solve(pbm:dyn_sliding_pack.params.CasADiNLPOptions, solver_name='ipopt', tf=2.0):
    if pbm.cfg is None:
        configure(pbm)
    
    prob = {
            'f': pbm.f,
            'x': cs.vertcat(*pbm.x),
            'g': cs.vertcat(*pbm.g),
            'p': cs.vertcat(*pbm.p)
            }
    
    param = []
    param += [tf]
    
    solver = cs.nlpsol('solver', solver_name, prob, pbm.cfg)

    sol = solver(
                lbx=pbm.lbx, ubx=pbm.ubx,
                lbg=pbm.lbg, ubg=pbm.ubg,
                p=param)
