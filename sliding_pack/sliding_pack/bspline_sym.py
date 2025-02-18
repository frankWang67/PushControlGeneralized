import casadi as cs

# CasADi does not support scipy splines directly, so we manually implement the spline basis functions
def bspline_basis(k, i, u, knots):
    """
    B spline basis function recursive definition.

    Parameters
    ----------
    `k` : `int`
        The degree of the spline
    `i` : `int`
        The index of the knot
    `u` : `float`
        The parameter variable
    `knots` : `np.ndarray`
        The knot vector

    Returns
    -------
    `float`
        The value of the B-spline basis function
    """
    if k == 0:
        return cs.if_else(cs.logic_and(u >= knots[i], u < knots[i + 1]), 1.0, 0.0)
    else:
        denom1 = knots[i + k] - knots[i]
        denom2 = knots[i + k + 1] - knots[i + 1]

        term1 = 0 if denom1 == 0 else (u - knots[i]) / denom1 * bspline_basis(k - 1, i, u, knots)
        term2 = 0 if denom2 == 0 else (knots[i + k + 1] - u) / denom2 * bspline_basis(k - 1, i + 1, u, knots)

        return term1 + term2

def get_bspline_curve(t, knots, coeffs, degree):
    """
    The expression of the curve interpolation based on B-spline.

    Parameters
    ----------
    `t` : `cs.MX`
        The parameter variable
    `knots` : `np.ndarray`
        The knot vector
    `coeffs` : `list` of `cs.MX`
        The control points coefficients matrix, each column is a dimension
    `degree` : `int`
        The degree of the spline
    
    Returns
    -------
    `cs.MX`
        The coordinates of the interpolated points
    """
    dims = len(coeffs)  # number of dimensions
    result = [0] * dims
    for d in range(dims):
        for i in range(len(knots) - degree - 1):
            result[d] += coeffs[d][i] * bspline_basis(degree, i, t, knots)
    return cs.vertcat(*result)

def get_bspline_func(t, knots, coeffs, degree):
    """
    Get the B-spline function.

    Parameters
    ----------
    `t` : `cs.MX`
        The parameter variable
    `knots` : `np.ndarray`
        The knot vector
    `coeffs` : `list` of `cs.MX`
        The control points coefficients matrix, each column is a dimension
    `degree` : `int`
        The degree of the spline
    
    Returns
    -------
    `cs.Function`
        The B-spline function
    """
    return cs.Function('bspline', [t], [get_bspline_curve(t, knots, coeffs, degree)])

def get_tangent_normal(t, knots, coeffs, degree):
    """
    Get the tangent and normal vectors of the B-spline curve at the given parameter.

    Parameters
    ----------
    `t` : `cs.MX`
        The parameter variable
    `knots` : `np.ndarray`
        The knot vector
    `coeffs` : `list` of `cs.MX`
        The control points coefficients matrix, each column is a dimension
    `degree` : `int`
        The degree of the spline
    
    Returns
    -------
    `cs.MX`, `cs.MX`
        The tangent and normal vectors
    """
    curve = get_bspline_curve(t, knots, coeffs, degree)
    tangent = cs.jacobian(curve, t)
    tangent /= cs.norm_2(tangent)
    normal = cs.vertcat(-tangent[1], tangent[0])
    return tangent, normal

def get_tangent_normal_func(t, knots, coeffs, degree):
    """
    Get the tangent and normal vectors of the B-spline curve at the given parameter.

    Parameters
    ----------
    `t` : `cs.MX`
        The parameter variable
    `knots` : `np.ndarray`
        The knot vector
    `coeffs` : `list` of `cs.MX`
        The control points coefficients matrix, each column is a dimension
    `degree` : `int`
        The degree of the spline
    
    Returns
    -------
    `cs.Function`, `cs.Function`
        The tangent and normal functions
    """
    tangent, normal = get_tangent_normal(t, knots, coeffs, degree)
    return cs.Function('tangent', [t], [tangent]), cs.Function('normal', [t], [normal])

def psic_to_t(psic):
    """
    Convert azimuth angle to parameter.

    Parameters
    ----------
    `psic` : `cs.MX`
        The azimuth angle
    
    Returns
    -------
    `cs.MX`
        The parameter
    """
    psic = cs.fmod(psic, 2 * cs.pi)
    psic = cs.if_else(cs.le(psic, 0), psic + 2 * cs.pi, psic)
    return psic / (2 * cs.pi)

if __name__ == "__main__":
    import numpy as np
    from scipy.interpolate import splprep
    import matplotlib.pyplot as plt

    # control_points = np.array([
    #     [2, 0], [2, 0.25], [2, 0.5], [2, 0.75], [2, 1], [1.5, 1], [1, 1], [0.5, 1], [0, 1],
    #     [-0.5, 1], [-1, 1], [-1.5, 1], [-2, 1], [-2, 0.75], [-2, 0.5], [-2, 0.25], [-2, 0],
    #     [-2, -0.25], [-2, -0.5], [-2, -0.75], [-2, -1], [-1.5, -1], [-1, -1], [-0.5, -1], [0, -1],
    #     [0.5, -1], [1, -1], [1.5, -1], [2, -1], [2, -0.75], [2, -0.5], [2, -0.25]
    # ])
    control_points = np.array([
        [0.061, 0.035], [0.041, 0.035], [0.020, 0.035], [0.000, 0.035], [-0.020, 0.035], [-0.041, 0.035], [-0.061, 0.035], [-0.061, 0.023], [-0.061, 0.012], [-0.061, 0.000], [-0.061, -0.012], [-0.061, -0.023], [-0.061, -0.035], [-0.041, -0.035], [-0.020, -0.035], [0.000, -0.035], [0.020, -0.035], [0.041, -0.035], [0.061, -0.035], [0.061, -0.023], [0.061, -0.012], [0.061, 0.000], [0.061, 0.012], [0.061, 0.023]
    ])
    # control_points = np.array([
    #     [2, 0], [2, 1], [0, 1], [-2, 1], [-2, 0], [-2, -1], [0, -1], [2, -1]
    # ])
    # control_points = np.array([
    #     [1, 0], [0, 1], [-1, 0], [0, -1]
    # ])
    control_points = np.vstack([control_points, control_points[0]])

    tck, _ = splprep(control_points.T, s=0, per=True)
    knots = tck[0] 
    coeffs = tck[1] 
    degree = tck[2] 
    print(f"knots: {knots}")
    print(f"coeffs: {coeffs}")
    print(f"degree: {degree}")

    t = cs.MX.sym('t')
    bspline_func = get_bspline_func(t, knots, coeffs, degree)
    tangent_func, normal_func = get_tangent_normal_func(t, knots, coeffs, degree)
    
    t_vals = np.linspace(0, 1, 100)
    pt_vals = np.array([bspline_func(t_val) for t_val in t_vals])


    psic = 3.853564251100438
    t_ = psic_to_t(psic)
    pt = bspline_func(t_)
    tangent = tangent_func(t_)
    normal = normal_func(t_)
    pt = np.array(pt).reshape(-1)
    tangent = np.array(tangent).reshape(-1)
    normal = np.array(normal).reshape(-1)
    print(f"psic: {np.arctan2(pt[1], pt[0])+2*np.pi}")

    plt.plot(control_points[:, 0], control_points[:, 1], 'ro')
    plt.plot(pt_vals[:, 0], pt_vals[:, 1], 'b-')
    plt.plot(pt[0], pt[1], 'go')
    plt.quiver(pt[0], pt[1], tangent[0], tangent[1], color='r')
    plt.quiver(pt[0], pt[1], normal[0], normal[1], color='b')
    plt.axis('equal')
    plt.show()
