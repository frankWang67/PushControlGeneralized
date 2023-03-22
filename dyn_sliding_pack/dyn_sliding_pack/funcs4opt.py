import casadi as cs


def make_rotation_matrix_symbol(theta:cs.SX.sym, dim):
    __ctheta = cs.cos(theta)
    __stheta = cs.sin(theta)
    if dim == 2:
        __R = cs.SX(2, 2)
        __R[0,0] = __ctheta
        __R[0,1] = -__stheta
        __R[1,0] = __stheta
        __R[1,1] = __ctheta
    elif dim == 3:
        __R = cs.SX(3, 3)
        __R[0,0] = __ctheta
        __R[0,1] = -__stheta
        __R[1,0] = __stheta
        __R[1,1] = __ctheta
        __R[2,2] = 1.0
    else:
        raise NotImplementedError('make_rotation_matrix_symbol: \
                                   only support dimension 2 or 3, but %d is given'.format(dim))

    return __R

def make_limit_surface_matrix_symbol(fx_max, fy_max, m_max):
    __L = cs.SX(3 ,3)
    __L[0,0] = 2/(fx_max**2)
    __L[1,1] = 2/(fy_max**2)
    __L[2,2] = 2/(m_max**2)

    return __L
