import numpy as np
import scipy.interpolate as spi

class ShapeContour:
    def __init__(self, control_points):
        """
        Initialize a shape contour object with the given control points.

        Parameters
        ----------
        `control_points`: `np.ndarray` of shape `(n, 2)`
        """
        assert (isinstance(control_points, np.ndarray) and control_points.ndim == 2 and control_points.shape[1] == 2), \
            "control_points must be a 2D numpy array with shape (n, 2)"
        assert np.all(control_points[0] == control_points[-1]), "control_points must be closed"
        
        self.control_points = control_points
        self.tck, _ = spi.splprep([control_points[:, 0], control_points[:, 1]], s=0, per=True)

        self.t_samples = np.linspace(0, 1, 10000)
        self.pt_samples = np.array(spi.splev(self.t_samples, self.tck)).T
        self.psic_samples = np.arctan2(self.pt_samples[:, 1], self.pt_samples[:, 0])
        permute_idx = np.argsort(self.psic_samples)
        self.t_samples = self.t_samples[permute_idx]
        self.pt_samples = self.pt_samples[permute_idx, :]
        self.psic_samples = self.psic_samples[permute_idx]
        deduplicate_idx = np.where(np.diff(self.psic_samples) > 0)[0]
        self.t_samples = self.t_samples[deduplicate_idx]
        self.pt_samples = self.pt_samples[deduplicate_idx, :]
        self.psic_samples = self.psic_samples[deduplicate_idx]
        self.psic2t_tck = spi.splrep(self.psic_samples, self.t_samples, s=0.1, per=True)

        self.lim_surf_A = np.diag([1.0, 1.0, self.get_curvature()])

    def psic_to_t(self, psic):
        return np.array(spi.splev(psic, self.psic2t_tck))

    def get_point_xy(self, psic):
        """
        Get the point(s) on the shape contour at the given azimuth angle(s).

        Parameters
        ----------
        `psic`: `float` | `np.ndarray`
            The azimuth angle(s) of the point(s) to evaluate the shape contour at. Unit: rad
        
        Returns
        -------
        `np.ndarray` of shape `(n, 2)`
            The `(x, y)` point(s) on the shape contour at the given azimuth angle(s).
        """
        t = self.psic_to_t(psic)
        pt = np.array(spi.splev(t, self.tck))
        if pt.ndim == 1:
            pt = pt.reshape(-1, 2)
        else:
            pt = pt.T

        return pt
    
    def get_tangent_and_normal(self, psic):
        """
        Get the tangent and normal vectors of the shape contour at the given azimuth angle(s).

        Parameters
        ----------
        `psic`: `float` | `np.ndarray`
            The azimuth angle(s) of the point(s) to evaluate the shape contour at. Unit: rad
        
        Returns
        -------
        `tangent`, `normal`: `tuple` of two `np.ndarray` of shape `(n, 2)`
            The tangent and normal vectors of the shape contour at the given azimuth angle(s).
        """
        t = self.psic_to_t(psic)
        tangent = np.array(spi.splev(t, self.tck, der=1))
        if tangent.ndim == 1:
            tangent = tangent.reshape(-1, 2)
        else:
            tangent = tangent.T
        tangent = tangent / np.linalg.norm(tangent, axis=1, keepdims=True)
        normal = np.array([-tangent[:, 1], tangent[:, 0]])

        return tangent, normal
    
    def get_contour_points(self, num_points=100):
        """
        Get the points on the shape contour.

        Parameters
        ----------
        `num_points`: `int`
            The number of points to sample on the shape contour.
        
        Returns
        -------
        `np.ndarray` of shape `(num_points, 2)`
            The points on the shape contour.
        """
        t = np.linspace(0, 1, num_points)
        pts = np.array(spi.splev(t, self.tck)).T

        return pts
    
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
    import matplotlib.pyplot as plt
    import time

    control_points = np.array([[-0.0878332 ,  0.00894283],
                               [-0.06927735, -0.00348606],
                               [-0.05296399, -0.01158518],
                               [-0.04714031, -0.01441735],
                               [-0.041185  , -0.01692681],
                               [-0.0356633 , -0.02305051],
                               [-0.02967586, -0.02596084],
                               [-0.02266738, -0.02777073],
                               [-0.00996478, -0.02773122],
                               [-0.00041986, -0.02406837],
                               [ 0.0044366 , -0.02229121],
                               [ 0.0168373 , -0.02315491],
                               [ 0.03243288, -0.02825957],
                               [ 0.0445449 , -0.02781803],
                               [ 0.06018997, -0.02223057],
                               [ 0.06313941, -0.01095734],
                               [ 0.063856  ,  0.00187829],
                               [ 0.06498363,  0.00934821],
                               [ 0.06430348,  0.02594651],
                               [ 0.06331416,  0.0364197 ],
                               [ 0.04723743,  0.04102006],
                               [ 0.03250469,  0.04007967],
                               [ 0.0155726 ,  0.03512453],
                               [ 0.00858525,  0.03455442],
                               [ 0.00335342,  0.035297  ],
                               [-0.00586487,  0.0390531 ],
                               [-0.01101431,  0.04052551],
                               [-0.02407005,  0.04040656],
                               [-0.02948088,  0.03901881],
                               [-0.03477284,  0.03622134],
                               [-0.04096227,  0.03262363],
                               [-0.04923069,  0.0288909 ],
                               [-0.06533611,  0.02522149],
                               [-0.07393795,  0.0175605 ],
                               [-0.0878332 ,  0.00894283]])
    curve = ShapeContour(control_points)

    pt_samples = curve.get_contour_points()

    psic = -2.027722210361019
    t = curve.psic_to_t(psic)
    print(f"{t=}")
    pt = curve.get_point_xy(psic).reshape(-1)
    print(f"{pt=}")
    print(f"psic: {np.arctan2(pt[1], pt[0])}, error: {np.arctan2(pt[1], pt[0]) - psic}")
    # tangent = np.array(curve.tangent_func(t)).reshape(-1)
    # normal = np.array(curve.normal_func(t)).reshape(-1)
    tangent, normal = curve.get_tangent_and_normal(psic)
    tangent = tangent.reshape(-1)
    normal = normal.reshape(-1)

    plt.plot(0.0, 0.0, 'ro')
    plt.plot(pt_samples[:, 0], pt_samples[:, 1], 'b-')
    plt.plot(control_points[:, 0], control_points[:, 1], 'ro')
    plt.plot(pt[0], pt[1], 'go')
    plt.quiver(pt[0], pt[1], tangent[0], tangent[1], color='r')
    plt.quiver(pt[0], pt[1], normal[0], normal[1], color='b')
    plt.axis('equal')
    plt.show()
