# Author: Joao Moura
# Contact: jpousad@ed.ac.uk
# Date: 15/12/2020
# -------------------------------------------------------------------
# Description:
# 
# Integration functions based on scipy integrate and symbolic casadi
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------
import numpy as np
from scipy import integrate
import casadi as cs

# -------------------------------------------------------------------
# python integration lambda functions (1D and 2D)
# -------------------------------------------------------------------
square_np = lambda sq_side: integrate.dblquad(lambda x, y: np.sqrt((x**2)
    + (y**2)), - sq_side/2, sq_side/2, -sq_side/2, sq_side/2)[0]
quad_np = lambda sq_side: integrate.quad(lambda var: var**2,
    - sq_side/2, sq_side/2)[0]
# -------------------------------------------------------------------
# casadi auxiliary variables
# -------------------------------------------------------------------
# Fixed step Runge-Kutta 4 integrator
M = 4  # RK4 steps per interval
N = 4  # number of control intervals
sLenght = cs.SX.sym('sLenght')
xLenght = cs.SX.sym('xLenght')
yLenght = cs.SX.sym('yLenght')
x = cs.SX.sym('x')
y = cs.SX.sym('y')
DX = xLenght/(N*M)
DY = yLenght/(N*M)
# -------------------------------------------------------------------
# 1D casadi integration of g
# integrand y'=f(x)
# integrate x^2 dx for x=-xL/2..xL/2
# h equals to DX and DY for x and y separately
# cost function
g = cs.Function('g_ext', [x], [DX, (x**2)*DX])
Q = 0  # initialize cost
xx = -xLenght/2  # initialize initial cond
for n in range(N):
    for m in range(M):
        k1, k1_q = g(xx)
        k2, k2_q = g(xx + k1/2)
        k3, k3_q = g(xx + k2/2)
        k4, k4_q = g(xx + k3)
        Q += (k1_q + 2*k2_q + 2*k3_q + k4_q)/6
        xx += (k1 + 2*k2 + 2*k3 + k4)/6
quad_cs = cs.Function('quad_cs', [xLenght], [Q])
# -------------------------------------------------------------------
# 2D casadi integration of g
# integrand g'=f(x, y)
# integrate sqrt(x^2+y^2) dxdy for x=-xL/2..xL/2, y=-yL/2..yL/2
# h equals to DX and DY for x and y separately
g = cs.Function('h_ext', [x, y], [DX, DY, (cs.sqrt((x**2)+(y**2)))*DX*DY])
Q = 0  # initialize cost
yy = -yLenght/2  # initialize initial cond
for ny in range(N):
    for my in range(M):
        xx = -xLenght/2
        for nx in range(N):
            for mx in range(M):
                k1_x, k1_y, k1_q = g(xx, yy)
                k2_x, k2_y, k2_q = g(xx + k1_x/2, yy + k1_y/2)
                k3_x, k3_y, k3_q = g(xx + k2_x/2, yy + k2_y/2)
                k4_x, k4_y, k4_q = g(xx + k3_x, yy + k3_y)
                Q += (k1_q + 2*k2_q + 2*k3_q + k4_q)/6
                xx += (k1_x + 2*k2_x + 2*k3_x + k4_x)/6
        yy += (k1_y + 2*k2_y + 2*k3_y + k4_y)/6
rect_cs = cs.Function('rect_cs', [xLenght, yLenght], [Q])
square_cs = cs.Function('square_cs', [sLenght], [rect_cs(sLenght, sLenght)])

# -------------------------------------------------------------------
# 2D casadi integration of g
# integrand g'=f(x, y)
# integrate sqrt(x^2+y^2) dxdy for (x, y) in a B-spline
# h equals to DX and DY for x and y separately
def in_contour(x, y, spline_func):
    """
    Check if a point is inside a B-spline using CasADi's SX symbols

    Parameters
    ----------
    x : cs.SX
        The x coordinate of the point
    y : cs.SX
        The y coordinate of the point
    spline_func : cs.Function
        The B-spline function

    Returns
    -------
    cs.SX
        1 if the point is inside the B-spline, 0 otherwise
    """
    theta = cs.atan2(y, x)  # Calculate the angle of the point
    negative = cs.if_else(theta < 0, 1, 0)  # Check if the angle is negative
    theta = cs.if_else(negative, theta + 2*cs.pi, theta)  # Normalize the angle
    t = theta/(2*cs.pi)  # Normalize the angle to [0, 1]
    pt = spline_func(t)  # Evaluate the B-spline at the normalized angle
    inside = cs.le(cs.sqrt(x ** 2 + y ** 2), cs.norm_2(pt))  # Check if the point is inside the B-spline
    
    return inside

def RungeKutta4_Integrator(g, x_len, y_len, return_func_name):
    """
    Runge-Kutta 4 integrator inside an area

    Parameters
    ----------
    g : cs.Function
        The integrand function
    pts : cs.SX
        The vertices of the polygon
    x_len : cs.SX
        The length of the x axis
    y_len : cs.SX
        The length of the y axis
    return_func_name : str
        The name of the function to return

    Returns
    ----------
    cs.Function
        The integration function
    """
    Q = 0  # initialize cost
    yy = -y_len/2  # initialize initial cond
    for ny in range(N):
        for my in range(M):
            xx = -x_len/2
            for nx in range(N):
                for mx in range(M):
                    k1_x, k1_y, k1_q = g(xx, yy)
                    k2_x, k2_y, k2_q = g(xx + k1_x/2, yy + k1_y/2)
                    k3_x, k3_y, k3_q = g(xx + k2_x/2, yy + k2_y/2)
                    k4_x, k4_y, k4_q = g(xx + k3_x, yy + k3_y)
                    Q += (k1_q + 2*k2_q + 2*k3_q + k4_q)/6
                    xx += (k1_x + 2*k2_x + 2*k3_x + k4_x)/6
            yy += (k1_y + 2*k2_y + 2*k3_y + k4_y)/6
    
    integ_func = cs.Function(return_func_name, [x_len, y_len], [Q])
    return integ_func

def spline_area(x_len, y_len, spline_func):
    DX = x_len/(N*M)
    DY = y_len/(N*M)
    g = cs.Function('h_ext', [x, y], [DX, DY, in_contour(x, y, spline_func)*DX*DY])

    return RungeKutta4_Integrator(g, x_len, y_len, 'poly_area')

def spline_cs(x_len, y_len, spline_func):
    DX = x_len/(N*M)
    DY = y_len/(N*M)
    g = cs.Function('h_ext', [x, y], [DX, DY, in_contour(x, y, spline_func)*cs.sqrt((x**2)+(y**2))*DX*DY])

    return RungeKutta4_Integrator(g, x_len, y_len, 'poly_cs')
