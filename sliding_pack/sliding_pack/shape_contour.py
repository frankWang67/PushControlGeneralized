import numpy as np
import scipy.interpolate as spi
from matplotlib.path import Path

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
        assert control_points[0] == control_points[-1], "control_points must be closed"
        assert control_points[0, 1] == 0, "control_points must start at the x-axis"
        
        self.control_points = control_points
        self.tck, _ = spi.splprep([control_points[:, 0], control_points[:, 1]], s=0, per=True)

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
        psic = psic % (2 * np.pi)
        t = psic / (2 * np.pi)

        pt = np.array(spi.splev(t, self.tck)).reshape(-1, 2)

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
        psic = psic % (2 * np.pi)
        t = psic / (2 * np.pi)

        tangent = np.array(spi.splev(t, self.tck, der=1)).reshape(-1, 2)
        normal = np.array([-tangent[:, 1], tangent[:, 0]]).T

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
    
    def integrate_in_area(self, f, resolute=1000):
        """
        Integrate the given function over the area enclosed by the shape contour.

        Parameters
        ----------
        `f`: `function`
            The function to integrate. The function must take two arguments `x` and `y`.
        `resolute`: `int`
            The resolute of the integration. The area will be divided into `resolute` x `resolute` grids.
        
        Returns
        -------
        `float`
            The integral of the function over the area enclosed by the shape contour.

        Example
        -------
        ```python
        slider = ShapeContour(control_points)
        def f(x, y):
            return x ** 2 + y ** 2
        integral = slider.integrate_in_area(f)
        """
        pts = self.get_contour_points()
        path = Path(pts)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        x_range = np.linspace(x_min, x_max, resolute)
        y_range = np.linspace(y_min, y_max, resolute)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        points = np.vstack([x_mesh.ravel(), y_mesh.ravel()]).T
        mask = path.contains_points(points)
        mask = np.array(mask).reshape(x_mesh.shape)
        integral = np.sum(f(x_mesh, y_mesh) * mask) * (x_max - x_min) * (y_max - y_min) / mask.sum()
        return integral
