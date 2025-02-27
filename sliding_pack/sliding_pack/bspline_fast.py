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
        self.tck, _ = spi.splprep([control_points[:, 0], control_points[:, 1]], s=0, per=True)
        self.knots = self.tck[0]
        self.coeffs = self.tck[1]
        self.degree = self.tck[2]

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

        self.t_samples = np.linspace(0, 1, 100)
        self.pt_samples = np.array(spi.splev(self.t_samples, self.tck))
        self.psic_samples = np.arctan2(self.pt_samples[1, :], self.pt_samples[0, :])
        permute_idx = np.argsort(self.psic_samples)
        self.t_samples = self.t_samples[permute_idx]
        self.psic_samples = self.psic_samples[permute_idx]
        deduplicate_idx = np.where(np.diff(self.psic_samples) > 0)[0]
        self.t_samples = self.t_samples[deduplicate_idx]
        self.psic_samples = self.psic_samples[deduplicate_idx]
        tck_psic2t = spi.splrep(self.psic_samples, self.t_samples, s=0.1, per=True)
        psic = cs.MX.sym('psic')
        psic2t = cs.bspline(psic, cs.horzcat(*tck_psic2t[1]).T, [tck_psic2t[0].tolist()], [tck_psic2t[2]], 1, {})
        self.psic_to_t_func = cs.Function('psic_to_t_func', [psic], [psic2t])

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
        # psic = cs.fmod(psic, 2 * cs.pi)
        # psic = cs.if_else(cs.le(psic, 0), psic + 2 * cs.pi, psic)
        # return psic / (2 * cs.pi)
        return self.psic_to_t_func(psic)
    
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
    # control_points = np.array([
    #     [-0.01466997, -0.03942101],
    #     [ 0.00497099, -0.0384836 ],
    #     [ 0.01840592, -0.03466345],
    #     [ 0.03326383, -0.03802705],
    #     [ 0.06135208, -0.04166347],
    #     [ 0.07260709, -0.03428812],
    #     [ 0.07324306, -0.02225373],
    #     [ 0.07334644, -0.00442549],
    #     [ 0.06607522,  0.02440813],
    #     [ 0.05607435,  0.02384318],
    #     [ 0.0239411 ,  0.01806577],
    #     [ 0.00337937,  0.02222508],
    #     [-0.01601585,  0.02468296],
    #     [-0.03114276,  0.01117183],
    #     [-0.03957137,  0.00858912],
    #     [-0.06999464,  0.00032966],
    #     [-0.07060502, -0.01424038],
    #     [-0.05220167, -0.0239637 ],
    #     [-0.03423954, -0.02900642],
    #     [-0.01520698, -0.03894434]
    # ])
    control_points = np.array([[0.0744672435728953, 0.047023460493674535], [0.05016979251064657, 0.027878515493950276], [0.02321525338304571, 0.019204752072111517], [-0.0013090182566544636, 0.025097089830710267], [-0.0367663982036114, 0.012300017709359692], [-0.03773590173234673, -0.013117090696594662], [-0.011763929784634308, -0.024844159241380795], [0.02955074829676256, -0.020034528375129652], [0.008998486068468946, -0.031205003676214302], [0.04505697101856897, -0.037803217796939015], [0.07479637262604298, -0.05647440327757139], [0.07431412288532566, -0.05014418638293433], [0.08254684185629932, -0.05706848965842848], [0.04357892738225819, -0.025227231391774102], [0.01952298998291236, -0.015242923291244409], [0.012478492658411153, -0.00923721790068388], [0.02134984867705076, -0.0027273808139399378], [0.009399604028468934, 0.0059263271878978036], [0.05097863813480291, 0.021924184982029887], [0.07579311648911306, 0.04692819411663153], [0.0744672435728953, 0.047023460493674535]])
    # control_points = np.vstack([control_points, control_points[0]])
    # control_points = np.array([
    #     [0.070, 0.000], [0.061, 0.035], [0.035, 0.061], [0.000, 0.070], [-0.035, 0.061], [-0.061, 0.035], [-0.070, 0.000], [-0.061, -0.035], [-0.035, -0.061], [-0.000, -0.070], [0.035, -0.061], [0.061, -0.035], [0.070, -0.000]
    # ])
    curve = bspline_curve(control_points)

    t_vals = np.linspace(0, 1, 1000)
    pt_vals = np.array([curve.curve_func(t) for t in t_vals])

    psic = np.pi/4
    t = curve.psic_to_t(psic)
    print(f"{t=}")
    pt = np.array(curve.curve_func(t)).reshape(-1)
    print(f"psic: {np.arctan2(pt[1], pt[0])}, error: {np.arctan2(pt[1], pt[0]) - psic}")
    tangent = np.array(curve.tangent_func(t)).reshape(-1)
    normal = np.array(curve.normal_func(t)).reshape(-1)

    plt.plot(pt_vals[:, 0], pt_vals[:, 1], 'b-')
    plt.plot(control_points[:, 0], control_points[:, 1], 'ro')
    plt.plot(pt[0], pt[1], 'go')
    plt.quiver(pt[0], pt[1], tangent[0], tangent[1], color='r')
    plt.quiver(pt[0], pt[1], normal[0], normal[1], color='b')
    plt.axis('equal')
    plt.show()
