import casadi as cs
import numpy as np
import scipy.interpolate as spi
from scipy.integrate import quad
import matplotlib.pyplot as plt

class bspline_curve:
    def __init__(self, control_points):
        """
        Initialize a B-spline curve object with the given control points.

        Parameters
        ----------
        `control_points`: `np.ndarray` of shape `(n, 2)`
        """
        tck, _ = spi.splprep([control_points[:, 0], control_points[:, 1]], s=0, per=True)
        self.tck = tck
        self.knots = tck[0]
        self.coeffs = tck[1]
        self.degree = tck[2]

        t = cs.MX.sym('t')
        coeffs_matrix = cs.horzcat(*self.coeffs).T
        curve = cs.bspline(t, coeffs_matrix, [self.knots.tolist()], [self.degree], 2, {})
        self.curve_func = cs.Function('curve_func', [t], [curve])
        tangent = cs.jacobian(curve, t)
        tangent /= cs.norm_2(tangent)
        normal = cs.vertcat(-tangent[1], tangent[0])
        self.tangent_func = cs.Function('tangent_func', [t], [tangent])
        self.normal_func = cs.Function('normal_func', [t], [normal])

        self.lim_surf_A = np.diag([1.0, 1.0, self.get_curvature()])

    def psic_to_t(self, psic):
        """
        Convert the azimuth angle to the parameter of the B-spline curve.

        Parameters
        ----------
        `psic`: `float`
            The azimuth angle. Unit: rad

        Returns
        -------
        `float`
            The parameter of the B-spline curve.
        """
        psic = cs.fmod(psic, 2 * cs.pi)
        psic = cs.if_else(cs.le(psic, 0), psic + 2 * cs.pi, psic)
        return psic / (2 * cs.pi)
    
    def integrate(self, f, N=1000, M=1000):
        """
        Integrate the given function in the area enclosed by the B-spline curve, using Green's theorem.

        Parameters
        ----------
        `f`: `function`
            The integrand function, such as:
            ```
            def f(x, y):
                return np.sqrt(x ** 2 + y ** 2)
            ```

        Returns
        -------
        `float`
            The integral value.
        """
        def t_to_xy(t):
            pts = np.array(spi.splev(t, self.tck))
            return pts[0, :], pts[1, :]
        
        def t_to_dxdy(t):
            pts = np.array(spi.splev(t, self.tck, der=1))
            return pts[0, :], pts[1, :]
        
        t_samples = np.linspace(0, 1, N)
        x_samples, y_samples = t_to_xy(t_samples)
        dx_samples, dy_samples = t_to_dxdy(t_samples)

        s_samples = np.linspace(0, 1, M)
        t_grid, s_grid = np.meshgrid(t_samples, s_samples, indexing='ij')

        x_t = x_samples.reshape(-1, 1)
        y_t = y_samples.reshape(-1, 1)
        
        x_points = s_grid * x_t
        y_points = s_grid * y_t
        
        f_values = f(x_points, y_points)
        jacobian = s_grid * (x_t * dy_samples.reshape(-1, 1) - y_t * dx_samples.reshape(-1, 1))
        integrand = f_values * jacobian
        integral = np.trapz(np.trapz(integrand, s_samples, axis=1), t_samples)
        
        return integral
    
    def get_curvature(self):
        """
        Get the curvature squared of the B-spline curve.

        Returns
        -------
        `float`
            The curvature squared value.
        """
        area = self.integrate(lambda x, y: 1)
        integral = self.integrate(lambda x, y: np.sqrt(x ** 2 + y ** 2))
        c = integral / area
        return 1.0 / (c ** 2)

if __name__ == "__main__":
    # control_points = np.array([
    #     [0.061, 0.000], [0.061, 0.012], [0.061, 0.023], [0.061, 0.035], [0.041, 0.035], [0.020, 0.035], [0.000, 0.035], [-0.020, 0.035], [-0.041, 0.035], [-0.061, 0.035], [-0.061, 0.023], [-0.061, 0.012], [-0.061, 0.000], [-0.061, -0.012], [-0.061, -0.023], [-0.061, -0.035], [-0.041, -0.035], [-0.020, -0.035], [0.000, -0.035], [0.020, -0.035], [0.041, -0.035], [0.061, -0.035], [0.061, -0.023], [0.061, -0.012]
    # ])
    # control_points = np.vstack([control_points, control_points[0]])
    control_points = np.array([
        [0.070, 0.000], [0.061, 0.035], [0.035, 0.061], [0.000, 0.070], [-0.035, 0.061], [-0.061, 0.035], [-0.070, 0.000], [-0.061, -0.035], [-0.035, -0.061], [-0.000, -0.070], [0.035, -0.061], [0.061, -0.035], [0.070, -0.000]
    ])
    curve = bspline_curve(control_points)

    t_vals = np.linspace(0, 1, 1000)
    pt_vals = np.array([curve.curve_func(t) for t in t_vals])

    psic = np.pi*2
    t = curve.psic_to_t(psic)
    pt = np.array(curve.curve_func(t)).reshape(-1)
    print(f"psic: {np.arctan2(pt[1], pt[0])}, error: {np.arctan2(pt[1], pt[0]) - psic}")
    tangent = np.array(curve.tangent_func(t)).reshape(-1)
    normal = np.array(curve.normal_func(t)).reshape(-1)
    
    # t_sym = cs.MX.sym('t')
    # pt_sym = curve.curve_func(t_sym)
    # tangent_sym = curve.tangent_func(t_sym)
    # normal_sym = curve.normal_func(t_sym)
    # print(f"pt_sym: {pt_sym}")
    # print(f"tangent_sym: {tangent_sym}")
    # print(f"normal_sym: {normal_sym}")

    print(f"Curvature squared: {curve.get_curvature()}")

    plt.plot(pt_vals[:, 0], pt_vals[:, 1], 'b-')
    plt.plot(control_points[:, 0], control_points[:, 1], 'ro')
    plt.plot(pt[0], pt[1], 'go')
    plt.quiver(pt[0], pt[1], tangent[0], tangent[1], color='r')
    plt.quiver(pt[0], pt[1], normal[0], normal[1], color='b')
    plt.axis('equal')
    plt.show()
